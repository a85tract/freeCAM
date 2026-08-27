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


def _pointer_descriptor(tmp_path: Path, **overrides: object) -> Path:
    fields = {
        "rank": 3,
        "chunk_axis": 3,
        "intent": "in",
        "extra": "",
    }
    fields.update(overrides)  # type: ignore[arg-type]
    descriptor = tmp_path / "kernels.yaml"
    descriptor.write_text(
        f"""
schema_version: 1
kernels:
  - name: sample
    routine: original_sample
    symbol: pycam_sample_v1
    arguments:
      - field: state.tke
        dtype: float64
        rank: {fields["rank"]}
        intent: {fields["intent"]}
        chunk_axis: {fields["chunk_axis"]}
        pointer: true
{fields["extra"]}
      - field: state.do_ice
        dtype: int32
        rank: 1
        intent: in
        chunk_axis: 1
        fortran_type: logical
"""
    )
    return descriptor


def test_pointer_dummy_receives_a_pointer_and_a_logical_is_converted(
    tmp_path: Path,
) -> None:
    kernels = load_direct_kernels(_pointer_descriptor(tmp_path))
    source = generate_direct_kernel_module(kernels)

    # A POINTER dummy cannot take an array section, so the wrapper keeps a
    # pointer of the dummy's own rank and associates it with this chunk.
    assert "real(c_double), pointer :: field_1_chunk(:,:)" in source
    assert "field_1_chunk => field_1(:,:,chunk)" in source
    # A Fortran logical travels as int32 and is converted at the call.
    assert "logical :: field_2_value" in source
    assert "field_2_value = (field_2(chunk) /= 0_c_int32_t)" in source
    assert "call original_sample( &" in source
    assert "field_1_chunk, &" in source
    assert "field_2_value)" in source


def test_pointer_dummy_requires_the_chunk_axis_to_be_last(tmp_path: Path) -> None:
    # A pointer is associated with a contiguous slice, so chunks must be slowest.
    descriptor = _pointer_descriptor(tmp_path, chunk_axis=1)
    with pytest.raises(PICAMConfigurationError, match="chunk axis must be last"):
        load_direct_kernels(descriptor)


def test_pointer_dummy_cannot_also_fix_an_axis(tmp_path: Path) -> None:
    descriptor = _pointer_descriptor(
        tmp_path, rank=4, chunk_axis=4, extra="        fixed_indices: {3: 1}"
    )
    with pytest.raises(PICAMConfigurationError, match="cannot fix an axis"):
        load_direct_kernels(descriptor)


def test_logical_argument_must_be_one_int32_per_chunk(tmp_path: Path) -> None:
    descriptor = tmp_path / "kernels.yaml"
    descriptor.write_text(
        """
schema_version: 1
kernels:
  - name: invalid
    arguments:
      - field: state.flag
        dtype: float64
        rank: 1
        intent: in
        chunk_axis: 1
        fortran_type: logical
"""
    )

    with pytest.raises(PICAMConfigurationError, match="one int32 value per chunk"):
        load_direct_kernels(descriptor)


def test_logical_argument_cannot_be_written_back(tmp_path: Path) -> None:
    descriptor = tmp_path / "kernels.yaml"
    descriptor.write_text(
        """
schema_version: 1
kernels:
  - name: invalid
    arguments:
      - field: state.flag
        dtype: int32
        rank: 1
        intent: inout
        chunk_axis: 1
        fortran_type: logical
"""
    )

    with pytest.raises(PICAMConfigurationError, match="intent\\(in\\)"):
        load_direct_kernels(descriptor)


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


def test_a_character_error_channel_is_read_back_and_reported() -> None:
    """A routine whose last dummy is a character(len=*) intent(out) reports
    failure the way CAM's callers read it: blank means it succeeded.  The
    wrapper holds the string, hands the text to the caller through the error
    message it already carries, and leaves one int32 per chunk."""

    import numpy as np

    from freecam.pi_cam.kernel_codegen import (
        CHARACTER_LENGTH, DirectKernel, DirectKernelArgument, generate_direct_kernel_module,
    )

    arguments = (
        DirectKernelArgument(field="f.ncol", dtype=np.dtype("int32").str, rank=1,
                             intent="in", chunk_axis=1, extents=("chunks",)),
        DirectKernelArgument(field="f.errstring", dtype=np.dtype("int32").str, rank=1,
                             intent="out", chunk_axis=1, extents=("chunks",),
                             fortran_type="character"),
    )
    kernel = DirectKernel(name="demo", routine="demo_routine", symbol="pycam_demo_v1",
                          action_id=0, modules=(), arguments=arguments)
    text = generate_direct_kernel_module((kernel,))
    assert f"character(len={CHARACTER_LENGTH}) :: field_2_value" in text
    assert "    field_2_value = ' '" in text                     # blank before the call
    assert "         field_2_value)" in text                     # passed, not a slice
    assert "    if (field_2_value /= ' ') then" in text           # read back after it
    assert "      status = 70_c_int" in text
    assert "errmsg(index) = field_2_value(index:index)" in text


def test_a_character_carrier_must_be_one_int32_per_chunk_and_intent_out() -> None:
    from freecam.pi_cam.errors import PICAMConfigurationError
    from freecam.pi_cam.kernel_codegen import DirectKernelArgument

    payload = {"field": "f.errstring", "dtype": "int32", "rank": 1, "intent": "out",
               "chunk_axis": 1, "extents": ["chunks"], "fortran_type": "character"}
    DirectKernelArgument.from_payload(payload)
    with pytest.raises(PICAMConfigurationError, match="intent"):
        DirectKernelArgument.from_payload({**payload, "intent": "inout"})
    with pytest.raises(PICAMConfigurationError, match="one int32 value per chunk"):
        DirectKernelArgument.from_payload({**payload, "dtype": "float64"})
