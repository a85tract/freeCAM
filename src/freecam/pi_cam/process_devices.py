"""Lazy runtime access to the generated per-process CAM devices.

The default CAM timestep keeps using the untouched BFB image.  An explicit
``driver.cam.physics.<process>.run()`` call promotes that image to
``RTLD_GLOBAL`` and loads only the source device containing the requested
procedure.  Every call still receives rank-local Python-owned storage through
the common pointer-table ABI.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from freecam.core.fortran_adapter import PointerTableAdapter

from .errors import NativeCAMError


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PICAMProcessDevice:
    """One generated process ABI and the shared object that exports it."""

    name: str
    qualified_name: str
    source: str
    symbol: str
    library: Path
    library_sha256: str
    arguments: tuple[Mapping[str, Any], ...]
    result: Mapping[str, Any] | None

    @property
    def abi_arguments(self) -> tuple[Mapping[str, Any], ...]:
        result: list[Mapping[str, Any]] = [
            argument
            for argument in self.arguments
            if not bool(argument.get("procedure", False))
        ]
        if self.result is not None:
            result.append(self.result)
        return tuple(result)

    def operation(self) -> Mapping[str, Any]:
        return {
            "symbol": self.symbol,
            "action_id": 0,
            "arguments": [
                {
                    "field": f"process_context.{self.name}.{argument['name']}",
                    "dtype": (
                        None
                        if str(argument.get("dtype")) == "character"
                        else argument.get("dtype")
                    ),
                    "rank": int(argument.get("rank", 0)),
                    "intent": (
                        "out"
                        if self.result is not None and argument is self.result
                        else str(argument.get("intent") or "inout")
                    ),
                    "contiguous": "F",
                    "character": str(argument.get("fortran_type", "")).startswith(
                        "character"
                    ),
                }
                for argument in self.abi_arguments
            ],
        }


class PICAMProcessDeviceCatalog:
    """Validated adapter inventory shared by every MPI rank."""

    def __init__(
        self,
        devices: Mapping[str, PICAMProcessDevice],
        *,
        loadable: frozenset[str] | None = None,
    ) -> None:
        self._devices = dict(devices)
        self._loadable = (
            frozenset(self._devices) if loadable is None else frozenset(loadable)
        )

    @staticmethod
    def identity(qualified_name: str, source: str) -> str:
        return f"{qualified_name}@{source}"

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._devices))

    @property
    def loadable_names(self) -> tuple[str, ...]:
        """Process identities proven loadable in the selected CAM image."""

        return tuple(sorted(self._loadable))

    def device(self, qualified_name: str, source: str) -> PICAMProcessDevice:
        try:
            return self._devices[self.identity(qualified_name, source)]
        except KeyError as exc:
            raise NativeCAMError(
                f"physical process {qualified_name!r} from {source!r} has no "
                "compiled source device"
            ) from exc

    def has(self, qualified_name: str, source: str) -> bool:
        return self.identity(qualified_name, source) in self._loadable

    def generated(self, qualified_name: str, source: str) -> bool:
        """Whether an adapter exists, independent of case compatibility."""

        return self.identity(qualified_name, source) in self._devices

    @classmethod
    def from_reports(
        cls,
        generation_path: str | Path,
        validation_path: str | Path,
        loading_path: str | Path | None = None,
    ) -> "PICAMProcessDeviceCatalog":
        generation = json.loads(Path(generation_path).read_text())
        validation = json.loads(Path(validation_path).read_text())
        if int(generation.get("schema_version", 0)) != 1:
            raise NativeCAMError("unsupported process adapter generation schema")
        if int(validation.get("schema_version", 0)) != 1:
            raise NativeCAMError("unsupported process adapter validation schema")
        libraries: dict[str, tuple[Path, str]] = {}
        for source in validation.get("sources", ()):
            if source.get("status") != "passed":
                continue
            library = source.get("library")
            digest = source.get("library_sha256")
            if library is None or digest is None:
                for attempt in reversed(tuple(source.get("attempts", ()))):
                    if attempt.get("status") == "passed":
                        library = attempt.get("library")
                        digest = attempt.get("library_sha256")
                        break
            if library is not None and digest is not None:
                libraries[str(source["source"])] = (Path(str(library)), str(digest))
        devices: dict[str, PICAMProcessDevice] = {}
        for record in generation.get("generated", ()):
            source = str(record["original_source"])
            if source not in libraries:
                continue
            library, digest = libraries[source]
            qualified = str(record["name"])
            # The committed physics catalog resolves collisions in the same
            # deterministic way.  Runtime registration later rekeys this
            # record to the public catalog name when needed.
            name = str(record["original_routine"])
            devices[cls.identity(qualified, source)] = PICAMProcessDevice(
                name=name,
                qualified_name=qualified,
                source=source,
                symbol=str(record["symbol"]),
                library=library,
                library_sha256=digest,
                arguments=tuple(dict(item) for item in record.get("arguments", ())),
                result=(
                    None
                    if not isinstance(record.get("result"), Mapping)
                    else dict(record["result"])
                ),
            )
        loadable: frozenset[str] | None = None
        if loading_path is not None and Path(loading_path).is_file():
            loading = json.loads(Path(loading_path).read_text())
            if int(loading.get("schema_version", 0)) != 1:
                raise NativeCAMError("unsupported process device loading schema")
            loadable = frozenset(
                str(record["name"])
                for record in loading.get("records", ())
                if bool(record.get("loaded")) and str(record.get("name")) in devices
            )
        return cls(devices, loadable=loadable)


class PICAMProcessDeviceRuntime:
    """Load and call validated source devices without changing default steps."""

    def __init__(
        self,
        catalog: PICAMProcessDeviceCatalog,
        *,
        main_library: Path,
    ) -> None:
        self.catalog = catalog
        self.main_library = Path(main_library)
        self._main_global: ctypes.CDLL | None = None
        self._libraries: dict[Path, ctypes.CDLL] = {}
        self._adapters: dict[str, PointerTableAdapter] = {}

    @property
    def qualified_names(self) -> tuple[str, ...]:
        return self.catalog.names

    def _adapter(
        self, qualified_name: str, source: str, public_name: str
    ) -> PointerTableAdapter:
        key = self.catalog.identity(qualified_name, source)
        cached = self._adapters.get(key)
        if cached is not None:
            return cached
        device = self.catalog.device(qualified_name, source)
        if not device.library.is_file():
            raise NativeCAMError(
                f"compiled process device is missing: {device.library}"
            )
        actual_hash = _sha256(device.library)
        if actual_hash != device.library_sha256:
            raise NativeCAMError(
                f"compiled process device hash differs: {device.library}"
            )
        if self._main_global is None:
            self._main_global = ctypes.CDLL(
                str(self.main_library), mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL
            )
        library = self._libraries.get(device.library)
        if library is None:
            library = ctypes.CDLL(str(device.library), mode=ctypes.RTLD_LOCAL)
            self._libraries[device.library] = library
        operation = device.operation()
        operation = {
            **operation,
            "arguments": [
                {
                    **argument,
                    "field": str(argument["field"]).replace(
                        f"process_context.{device.name}.",
                        f"process_context.{public_name}.",
                        1,
                    ),
                }
                for argument in operation["arguments"]
            ],
        }
        adapter = PointerTableAdapter(
            library,
            {public_name: operation},
            library_name=str(device.library),
        )
        self._adapters[key] = adapter
        return adapter

    def call(
        self,
        qualified_name: str,
        source: str,
        public_name: str,
        pool: Mapping[str, np.ndarray],
        *,
        fcomm: int,
    ) -> None:
        self._adapter(qualified_name, source, public_name).call(
            public_name, pool, fcomm=fcomm
        )
