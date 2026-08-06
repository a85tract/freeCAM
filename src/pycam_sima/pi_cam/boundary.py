"""Replace the CESM coupler with an explicit CAM boundary provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .errors import BoundaryReplayError
from .state import PICAMStatePool


def _array_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return sha256(memoryview(contiguous).cast("B")).hexdigest()


def _first_difference(candidate: np.ndarray, reference: np.ndarray) -> str:
    """Return one compact, reproducible element-level BFB diagnostic."""

    candidate_bytes = np.ascontiguousarray(candidate).view(np.uint8).reshape(-1)
    reference_bytes = np.ascontiguousarray(reference).view(np.uint8).reshape(-1)
    differing_bytes = np.flatnonzero(candidate_bytes != reference_bytes)
    if differing_bytes.size == 0:
        return "logical values differ but contiguous bytes match"
    itemsize = candidate.dtype.itemsize
    flat_index = int(differing_bytes[0]) // itemsize
    index = np.unravel_index(flat_index, candidate.shape, order="C")
    actual = np.asarray(candidate[index])
    expected = np.asarray(reference[index])
    return (
        f"first index {tuple(int(value) for value in index)}: "
        f"{actual.item()!r} [0x{actual.tobytes().hex()}] != "
        f"{expected.item()!r} [0x{expected.tobytes().hex()}]"
    )


class CAMBoundaryProvider(ABC):
    """One source of surface/coupler imports and one sink for CAM exports."""

    def initialize(self, *, rank: int, size: int, config_fingerprint: str) -> None:
        del rank, size, config_fingerprint

    @abstractmethod
    def import_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        """Populate the rank-local ``cam_in.*`` fields for one coupling step."""

    @abstractmethod
    def export_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        """Consume or validate the rank-local ``cam_out.*`` fields."""

    def finalize(self) -> None:
        return None

    def has_fresh_import(self, step: int, rank: int) -> bool:
        """Whether source CAM called ``atm_import`` at this action boundary."""

        del step, rank
        return True


@dataclass(frozen=True, slots=True)
class BoundaryManifest:
    rank_count: int
    step_count: int
    config_fingerprint: str | None = None
    schema_version: int = 1
    file_pattern: str = "step-{step:06d}/rank-{rank:04d}-{direction}.npz"
    storage: str = "per_step_v1"
    held_import_steps: tuple[int, ...] = ()

    @classmethod
    def from_path(cls, path: Path) -> "BoundaryManifest":
        payload = json.loads(path.read_text())
        if int(payload.get("schema_version", 1)) != 1:
            raise BoundaryReplayError("unsupported CAM boundary manifest schema")
        return cls(
            rank_count=int(payload["rank_count"]),
            step_count=int(payload["step_count"]),
            config_fingerprint=payload.get("config_fingerprint"),
            schema_version=1,
            file_pattern=str(
                payload.get(
                    "file_pattern",
                    "step-{step:06d}/rank-{rank:04d}-{direction}.npz",
                )
            ),
            storage=str(payload.get("storage", "per_step_v1")),
            held_import_steps=tuple(
                int(step) for step in payload.get("held_import_steps", ())
            ),
        )


class ReplayBoundaryProvider(CAMBoundaryProvider):
    """Strictly replay per-step, per-rank imports captured from PI-atm."""

    def __init__(self, root: str | Path, *, verify_exports: bool = True) -> None:
        self.root = Path(root)
        self.verify_exports = bool(verify_exports)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise BoundaryReplayError(f"missing boundary manifest {manifest_path}")
        self.manifest = BoundaryManifest.from_path(manifest_path)
        self._rank: int | None = None
        self._bundle: dict[str, np.ndarray] | None = None

    def initialize(self, *, rank: int, size: int, config_fingerprint: str) -> None:
        if size != self.manifest.rank_count:
            raise BoundaryReplayError(
                f"boundary capture has {self.manifest.rank_count} ranks, runtime has {size}"
            )
        if self.manifest.config_fingerprint not in {None, config_fingerprint}:
            raise BoundaryReplayError("boundary capture belongs to a different config")
        if not 0 <= rank < size:
            raise BoundaryReplayError(f"invalid runtime rank {rank}")
        self._rank = rank
        if self.manifest.storage == "rank_bundle_v1":
            path = self.root / self.manifest.file_pattern.format(rank=rank)
            if not path.is_file():
                raise BoundaryReplayError(f"missing boundary rank bundle {path}")
            self._bundle = self._load(path)
            expected = self.manifest.step_count
            if any(values.shape[0] != expected for values in self._bundle.values()):
                raise BoundaryReplayError(
                    f"boundary rank bundle {path} does not contain {expected} steps"
                )
        elif self.manifest.storage != "per_step_v1":
            raise BoundaryReplayError(
                f"unsupported boundary storage {self.manifest.storage!r}"
            )

    def _path(self, step: int, rank: int, direction: str) -> Path:
        if not 0 <= step < self.manifest.step_count:
            raise BoundaryReplayError(
                f"boundary step {step} is outside 0..{self.manifest.step_count - 1}"
            )
        path = self.root / self.manifest.file_pattern.format(
            step=step, rank=rank, direction=direction
        )
        if not path.is_file():
            raise BoundaryReplayError(f"missing boundary payload {path}")
        return path

    @staticmethod
    def _load(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name].copy(order="F") for name in payload.files}

    def _fields(self, step: int, rank: int, direction: str) -> dict[str, np.ndarray]:
        if not 0 <= step < self.manifest.step_count:
            raise BoundaryReplayError(
                f"boundary step {step} is outside 0..{self.manifest.step_count - 1}"
            )
        if self.manifest.storage == "rank_bundle_v1":
            if self._bundle is None or self._rank != rank:
                raise BoundaryReplayError("boundary rank bundle is not initialized")
            name = "x2a_rattr" if direction == "import" else "a2x_rattr"
            try:
                values = self._bundle[name][step]
            except KeyError as exc:
                raise BoundaryReplayError(
                    f"boundary rank bundle lacks {name!r}"
                ) from exc
            return {name: values}
        return self._load(self._path(step, rank, direction))

    def import_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        if self._rank != rank:
            raise BoundaryReplayError("boundary provider was initialized for another rank")
        fields = self._fields(step, rank, "import")
        for name, values in fields.items():
            canonical = name if name.startswith("cam_in.") else f"cam_in.{name}"
            pool.ensure_from_array(canonical, values, category="boundary_import")
        # Allocate the candidate export array from the oracle contract before
        # the native atm_export kernel writes it.  Its reference bytes are not
        # copied, so the later comparison still detects every differing bit.
        expected_exports = self._fields(step, rank, "export")
        for name, values in expected_exports.items():
            canonical = name if name.startswith("cam_out.") else f"cam_out.{name}"
            if canonical not in pool:
                pool.ensure_from_array(
                    canonical,
                    np.zeros_like(values, order="F"),
                    category="boundary_export",
                )

    def has_fresh_import(self, step: int, rank: int) -> bool:
        if self._rank != rank:
            raise BoundaryReplayError("boundary provider was initialized for another rank")
        if not 0 <= step < self.manifest.step_count:
            raise BoundaryReplayError(
                f"boundary step {step} is outside 0..{self.manifest.step_count - 1}"
            )
        return step not in self.manifest.held_import_steps

    def export_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        if not self.verify_exports:
            return
        expected = self._fields(step, rank, "export")
        differences: list[str] = []
        for name, reference in expected.items():
            canonical = name if name.startswith("cam_out.") else f"cam_out.{name}"
            try:
                candidate = pool[canonical]
            except KeyError:
                differences.append(f"{canonical}: missing")
                continue
            if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
                differences.append(
                    f"{canonical}: {candidate.shape}/{candidate.dtype} != "
                    f"{reference.shape}/{reference.dtype}"
                )
            elif not np.array_equal(candidate, reference):
                differences.append(
                    f"{canonical}: {_array_hash(candidate)} != {_array_hash(reference)}; "
                    + _first_difference(candidate, reference)
                )
        if differences:
            raise BoundaryReplayError(
                f"rank {rank} step {step} CAM export is not bitwise identical: "
                + "; ".join(differences[:8])
            )


class InMemoryBoundaryProvider(CAMBoundaryProvider):
    """Programmatic boundary provider for a future coupler or experiment."""

    def __init__(
        self,
        imports: Mapping[tuple[int, int], Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        self.imports = {
            (int(step), int(rank)): {
                name: np.asfortranarray(values).copy(order="F")
                for name, values in fields.items()
            }
            for (step, rank), fields in (imports or {}).items()
        }
        self.exports: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._size: int | None = None

    def initialize(self, *, rank: int, size: int, config_fingerprint: str) -> None:
        del rank, config_fingerprint
        self._size = size

    def set_imports(
        self, step: int, rank: int, fields: Mapping[str, np.ndarray]
    ) -> None:
        self.imports[(int(step), int(rank))] = {
            name: np.asfortranarray(values).copy(order="F")
            for name, values in fields.items()
        }

    def import_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        try:
            fields = self.imports[(step, rank)]
        except KeyError as exc:
            raise BoundaryReplayError(
                f"no in-memory CAM imports for step {step}, rank {rank}"
            ) from exc
        for name, values in fields.items():
            canonical = name if name.startswith("cam_in.") else f"cam_in.{name}"
            pool.ensure_from_array(canonical, values, category="boundary_import")

    def export_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        self.exports[(step, rank)] = {
            name: values.copy(order="F")
            for name, values in pool.items()
            if name.startswith("cam_out.")
        }


def write_boundary_payload(
    root: str | Path,
    *,
    step: int,
    rank: int,
    direction: str,
    fields: Mapping[str, np.ndarray],
) -> Path:
    """Write one atomic capture payload used by source instrumentation."""

    if direction not in {"import", "export"}:
        raise BoundaryReplayError("direction must be import or export")
    root_path = Path(root)
    destination = root_path / f"step-{step:06d}" / f"rank-{rank:04d}-{direction}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **{name: np.asfortranarray(value) for name, value in fields.items()})
    temporary.replace(destination)
    return destination
