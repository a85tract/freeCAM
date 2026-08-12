"""Serializable NumPy expressions evaluated inside each persistent MPI rank."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from .errors import PICAMStateError
from .state import active_field_mask


_ARRAY_FUNCTIONS = {
    "clip": np.clip,
    "flip": np.flip,
    "gradient": np.gradient,
    "moveaxis": np.moveaxis,
    "nan_to_num": np.nan_to_num,
    "roll": np.roll,
    "swapaxes": np.swapaxes,
    "transpose": np.transpose,
    "where": np.where,
}


def _literal(value: Any) -> Mapping[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return {"type": "literal", "value": value}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": tuple(_literal(item) for item in value)}
    if isinstance(value, list):
        return {"type": "list", "items": tuple(_literal(item) for item in value)}
    raise TypeError(
        "distributed NumPy expressions accept fields, scalar values, and "
        "integer tuple/list arguments; full local arrays must be declared as fields"
    )


def _operand_payload(value: Any, session: Any) -> Mapping[str, Any]:
    if isinstance(value, DistributedOperand):
        if value._expression_session is not session:
            raise ValueError("a distributed expression cannot mix different models")
        return value._expression_payload
    return _literal(value)


class DistributedOperand(NDArrayOperatorsMixin):
    """Mixin that builds a lazy rank-local expression tree."""

    __array_priority__ = 1000

    @property
    def _expression_session(self) -> Any:
        raise NotImplementedError

    @property
    def _expression_payload(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> "DistributedExpression":
        if method != "__call__":
            raise TypeError(
                f"distributed field ufunc method {method!r} is unsupported; "
                "use field.mean/min/max for reductions"
            )
        if kwargs.get("out") is not None:
            raise TypeError("NumPy out= is managed by the remote StatePool")
        kwargs = {key: value for key, value in kwargs.items() if key != "out"}
        session = self._expression_session
        payload = {
            "type": "ufunc",
            "name": ufunc.__name__,
            "args": tuple(_operand_payload(value, session) for value in inputs),
            "kwargs": {
                str(key): _operand_payload(value, session)
                for key, value in kwargs.items()
            },
        }
        return DistributedExpression(session, payload)

    def __array_function__(
        self,
        function: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> "DistributedExpression":
        del types
        name = getattr(function, "__name__", "")
        if name not in _ARRAY_FUNCTIONS:
            raise TypeError(
                f"np.{name or '<unknown>'} is not supported by distributed fields; "
                "supported functions are " + ", ".join(sorted(_ARRAY_FUNCTIONS))
            )
        session = self._expression_session
        payload = {
            "type": "function",
            "name": name,
            "args": tuple(_operand_payload(value, session) for value in args),
            "kwargs": {
                str(key): _operand_payload(value, session)
                for key, value in kwargs.items()
            },
        }
        return DistributedExpression(session, payload)


class DistributedExpression(DistributedOperand):
    """One immutable expression waiting to execute on every target MPI rank."""

    def __init__(self, session: Any, payload: Mapping[str, Any]) -> None:
        self.session = session
        self.payload = dict(payload)

    @property
    def _expression_session(self) -> Any:
        return self.session

    @property
    def _expression_payload(self) -> Mapping[str, Any]:
        return self.payload

    def compute(self, *, rank: int = 0) -> np.ndarray:
        """Evaluate this expression on one rank and return a copy."""

        return np.asarray(self.session.evaluate_expression(self.payload, rank=rank))

    values = compute

    def __getitem__(self, index: Any) -> "DistributedExpression":
        # The session owns index validation so the wire format stays identical
        # to a direct field slice.
        from .session import _normalize_field_selection

        return DistributedExpression(
            self.session,
            {
                "type": "getitem",
                "value": self.payload,
                "selection": _normalize_field_selection(index),
            },
        )

    def __repr__(self) -> str:
        return f"DistributedExpression({self.payload.get('type', 'unknown')})"


def evaluate_expression(pool: Any, payload: Mapping[str, Any]) -> Any:
    """Evaluate a validated expression against one rank's StatePool."""

    kind = str(payload.get("type"))
    if kind == "field":
        values = pool[str(payload["name"])]
        selection = payload.get("selection")
        return values if selection is None else values[tuple(selection)]
    if kind == "literal":
        return payload.get("value")
    if kind in {"tuple", "list"}:
        values = [evaluate_expression(pool, item) for item in payload.get("items", ())]
        return tuple(values) if kind == "tuple" else values
    if kind == "getitem":
        values = evaluate_expression(pool, payload["value"])
        return values[tuple(payload["selection"])]
    if kind in {"ufunc", "function"}:
        name = str(payload["name"])
        if kind == "ufunc":
            function = getattr(np, name, None)
            if not isinstance(function, np.ufunc):
                raise PICAMStateError(f"unknown distributed NumPy ufunc {name!r}")
        else:
            try:
                function = _ARRAY_FUNCTIONS[name]
            except KeyError as exc:
                raise PICAMStateError(
                    f"unknown distributed NumPy function {name!r}"
                ) from exc
        args = tuple(
            evaluate_expression(pool, item) for item in payload.get("args", ())
        )
        kwargs = {
            str(key): evaluate_expression(pool, value)
            for key, value in payload.get("kwargs", {}).items()
        }
        with np.errstate(all="ignore"):
            return function(*args, **kwargs)
    raise PICAMStateError(f"unknown distributed expression node {kind!r}")


def assign_expression(
    pool: Any,
    name: str,
    payload: Mapping[str, Any],
    *,
    selection: object = Ellipsis,
) -> int:
    """Evaluate and copy one expression into active entries of a field slice."""

    canonical = pool.canonical_name(name)
    contract = pool.contract(canonical)
    if not contract.writable:
        raise PICAMStateError(f"StatePool field {canonical!r} is read-only")
    target = pool[canonical]
    target_view = np.asarray(target[selection])
    active = np.asarray(active_field_mask(pool, canonical)[selection], dtype=np.bool_)
    count = int(active.sum())
    if count == 0:
        raise PICAMStateError("field selection contains no active CAM values")
    result = np.asarray(evaluate_expression(pool, payload))
    try:
        broadcast = np.broadcast_to(result, target_view.shape)
    except ValueError as exc:
        raise PICAMStateError(
            f"expression shape {result.shape} cannot broadcast to target "
            f"shape {target_view.shape}"
        ) from exc
    updated = np.array(target_view, copy=True, order="K")
    try:
        np.copyto(updated, broadcast, where=active, casting="same_kind")
    except TypeError as exc:
        raise PICAMStateError(
            f"expression dtype {result.dtype} cannot be assigned to {target.dtype}"
        ) from exc
    target[selection] = updated
    return count


__all__ = [
    "DistributedExpression",
    "DistributedOperand",
    "assign_expression",
    "evaluate_expression",
]
