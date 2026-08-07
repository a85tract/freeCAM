from __future__ import annotations

from pathlib import Path

import pytest

from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.kernel_codegen import (
    generate_direct_kernel_module,
    load_direct_kernels,
)


def test_direct_kernel_descriptor_generates_chunked_pointer_adapter(tmp_path: Path) -> None:
    descriptor = tmp_path / "kernels.yaml"
    descriptor.write_text(
        """
schema_version: 1
kernels:
  - name: sample
    routine: original_sample
    symbol: pycam_sample_v1
    modules:
      grid_mod: [nlev, ntracer]
    arguments:
      - field: state.ncol
        dtype: int32
        rank: 1
        intent: in
        chunk_axis: 1
        extents: [chunks]
      - field: state.q
        dtype: float64
        rank: 4
        intent: inout
        chunk_axis: 4
        extents: [16, nlev, ntracer, chunks]
        fixed_indices:
          3: 1
"""
    )

    kernels = load_direct_kernels(descriptor)
    source = generate_direct_kernel_module(kernels)

    assert len(kernels) == 1
    assert kernels[0].operation_name == "direct_kernel.sample"
    assert kernels[0].operation_payload()["arguments"][1] == {
        "field": "state.q",
        "dtype": "float64",
        "rank": 4,
        "intent": "inout",
    }
    assert "bind(C, name='pycam_sample_v1')" in source
    assert "call pycam_pi_cam_set_fp_environment_v1()" in source
    assert "use grid_mod, only: nlev, ntracer" in source
    assert "/= ntracer" in source
    assert "call c_f_pointer(pointers(2), field_2" in source
    assert "call original_sample( &" in source
    assert "field_1(chunk), &" in source
    assert "field_2(:,:,1,chunk))" in source


def test_direct_kernel_descriptor_rejects_chunk_axis_as_fixed_index(tmp_path: Path) -> None:
    descriptor = tmp_path / "kernels.yaml"
    descriptor.write_text(
        """
schema_version: 1
kernels:
  - name: invalid
    arguments:
      - field: state.t
        dtype: float64
        rank: 3
        chunk_axis: 3
        fixed_indices: {3: 1}
"""
    )

    with pytest.raises(PICAMConfigurationError, match="fixed index"):
        load_direct_kernels(descriptor)


def test_direct_kernel_descriptor_rejects_fortran_extent_expressions(
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "kernels.yaml"
    descriptor.write_text(
        """
schema_version: 1
kernels:
  - name: invalid
    arguments:
      - field: state.t
        dtype: float64
        rank: 1
        extents: ["pcols); call unsafe()"]
"""
    )

    with pytest.raises(PICAMConfigurationError, match="extents accept"):
        load_direct_kernels(descriptor)
