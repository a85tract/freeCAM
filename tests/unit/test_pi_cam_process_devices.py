from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from freecam.pi_cam.native import _ChunkProcessStatePool
from freecam.pi_cam.process_devices import PICAMProcessDeviceCatalog
from freecam.pi_cam.state import PICAMFieldContract, PICAMStatePool


def test_process_device_catalog_joins_generation_to_compiled_library(
    tmp_path: Path,
) -> None:
    library = tmp_path / "process.so"
    library.write_bytes(b"device")
    import hashlib

    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    generation = tmp_path / "generation.json"
    generation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated": [
                    {
                        "name": "sample_module::sample",
                        "original_source": "sample.F90",
                        "original_routine": "sample",
                        "symbol": "freecam_sample_v1",
                        "arguments": [
                            {
                                "name": "field",
                                "dtype": "float64",
                                "rank": 1,
                                "intent": "inout",
                                "fortran_type": "real",
                            }
                        ],
                        "result": None,
                    }
                ],
            }
        )
    )
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source": "sample.F90",
                        "status": "passed",
                        "library": str(library),
                        "library_sha256": digest,
                    }
                ],
            }
        )
    )

    catalog = PICAMProcessDeviceCatalog.from_reports(generation, validation)

    assert catalog.names == ("sample_module::sample@sample.F90",)
    device = catalog.device("sample_module::sample", "sample.F90")
    assert device.library == library
    assert device.operation()["arguments"][0]["rank"] == 1


def test_process_device_catalog_separates_compiled_from_case_loadable(
    tmp_path: Path,
) -> None:
    library = tmp_path / "process.so"
    library.write_bytes(b"device")
    import hashlib

    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    generation = tmp_path / "generation.json"
    generation.write_text(json.dumps({
        "schema_version": 1,
        "generated": [{
            "name": "sample_module::sample",
            "original_source": "sample.F90",
            "original_routine": "sample",
            "symbol": "freecam_sample_v1",
            "arguments": [],
            "result": None,
        }],
    }))
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({
        "schema_version": 1,
        "sources": [{
            "source": "sample.F90",
            "status": "passed",
            "library": str(library),
            "library_sha256": digest,
        }],
    }))
    loading = tmp_path / "loading.json"
    loading.write_text(json.dumps({
        "schema_version": 1,
        "records": [{
            "name": "sample_module::sample@sample.F90",
            "loaded": False,
        }],
    }))

    catalog = PICAMProcessDeviceCatalog.from_reports(
        generation, validation, loading
    )

    assert catalog.generated("sample_module::sample", "sample.F90")
    assert not catalog.has("sample_module::sample", "sample.F90")
    assert catalog.loadable_names == ()


def test_chunk_process_pool_returns_zero_copy_chunk_and_derived_record() -> None:
    pool = PICAMStatePool({"column": 3, "chunks": 2, "owner_bytes": 16})
    values = pool.create(
        PICAMFieldContract("temperature", ("column", "chunks"), "float64")
    )
    values[...] = np.arange(6).reshape(3, 2, order="F")
    owner = pool.create(
        PICAMFieldContract("__native_owner.phys_state", ("owner_bytes",), "uint8")
    )
    owner[...] = np.arange(16, dtype=np.uint8)
    bindings = {
        "process_context.sample.field": "temperature",
        "process_context.sample.state": "__native_owner.phys_state",
    }
    arguments = {
        "process_context.sample.field": {"rank": 1, "fortran_type": "real"},
        "process_context.sample.state": {
            "rank": 0,
            "fortran_type": "type:physics_state",
        },
    }

    chunk = _ChunkProcessStatePool(pool, bindings, arguments, 1)

    field = chunk["process_context.sample.field"]
    state = chunk["process_context.sample.state"]
    assert np.shares_memory(field, values)
    assert field.tolist() == values[:, 1].tolist()
    assert state.shape == ()
    assert state.dtype.itemsize == 8
    assert int(state.ctypes.data) == int(owner.ctypes.data) + 8


def test_chunk_process_pool_accepts_explicit_opaque_record_arrays() -> None:
    pool = PICAMStatePool({"records": 3, "chunks": 2})
    opaque = pool.create(
        PICAMFieldContract("opaque", ("records", "chunks"), "V16")
    )
    bindings = {"process_context.sample.objects": "opaque"}
    arguments = {
        "process_context.sample.objects": {
            "rank": 1,
            "fortran_type": "type:external_object",
        }
    }

    selected = _ChunkProcessStatePool(pool, bindings, arguments, 1)[
        "process_context.sample.objects"
    ]

    assert selected.shape == (3,)
    assert selected.dtype == np.dtype("V16")
    assert np.shares_memory(selected, opaque)
