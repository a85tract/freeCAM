from __future__ import annotations

from pathlib import Path

from pycam_sima.pi_cam.state_codegen import (
    generate_fortran_include,
    generate_owner_binder,
    instrument_cam_comp,
    load_state_bridge,
    pointerize_owner_type,
    split_cam_init_for_python_state,
)


def test_state_bridge_discovers_inline_pointer_and_allocatable_fields(tmp_path: Path) -> None:
    source = tmp_path / "state.F90"
    source.write_text(
        """
module state_mod
  type cam_in_t
    integer :: ncol
    real(8) :: temperature(pcols,pver)
    real(8), pointer, dimension(:) :: optional_flux
  end type cam_in_t
  type physics_state
    real(8), dimension(:,:), allocatable :: t, u
  end type physics_state
end module state_mod
"""
    )
    description = tmp_path / "bridge.yaml"
    description.write_text(
        """
schema_version: 1
owners:
  - name: cam_in
    type: cam_in_t
    source: state.F90
  - name: phys_state
    type: physics_state
    source: state.F90
dimension_defaults:
  pcols: 16
  pver: 30
dimension_bindings:
  cam_in.optional_flux: [pcols]
  phys_state.t: [pcols, pver]
  phys_state.u: [pcols, pver]
inactive_fields:
  - cam_in.optional_flux
exclude: []
"""
    )

    bridge = load_state_bridge(description, tmp_path)

    assert [field.name for field in bridge.fields] == [
        "cam_in.ncol",
        "cam_in.temperature",
        "cam_in.optional_flux",
        "phys_state.t",
        "phys_state.u",
    ]
    assert bridge.fields[1].source_dimensions == ("pcols", "pver")
    assert bridge.fields[2].pointer
    assert bridge.fields[2].python_dimensions == ("pcols",)
    assert not bridge.fields[2].active_by_default
    assert bridge.fields[3].allocatable
    assert bridge.fields[3].python_dimensions == ("pcols", "pver")
    manifest = bridge.manifest()
    assert manifest["dimension_defaults"] == {"pcols": 16, "pver": 30}
    assert manifest["fields"][3]["dimensions"] == ["pcols", "pver", "chunks"]
    assert manifest["owners"][0]["raw_field"] == "__native_owner.cam_in"
    assert manifest["owners"][1]["layout_symbol"] == (
        "pycam_pi_cam_layout_phys_state_v1"
    )
    generated = generate_fortran_include(bridge)
    assert "integer function cam_python_state_count()" in generated
    assert "count = 5" in generated
    assert "associated(cam_in(c)%optional_flux)" in generated
    assert "allocated(phys_state(c)%t)" in generated
    assert "reshape(phys_state(c)%t" in generated

    state_owner = bridge.owners[1]
    state_fields = tuple(field for field in bridge.fields if field.owner == state_owner)
    shell = pointerize_owner_type(source.read_text(), state_owner, state_fields)
    binder = generate_owner_binder(state_owner, state_fields, 2)
    assert "real(r8), pointer, contiguous :: t(:,:) => null()" in shell
    assert "real(r8), pointer, contiguous :: pycam_field_4(:,:,:)" in binder
    assert "phys_state(begchunk_in:endchunk_in) => pycam_owner_records" in binder
    assert "pycam_pi_cam_layout_phys_state_v1" in binder


def test_cam_comp_instrumentation_adds_public_generated_bridge() -> None:
    source = """
module cam_comp
  private
   public cam_final     ! CAM Finalization
contains
subroutine cam_final()
end subroutine
end module cam_comp
"""

    result = instrument_cam_comp(source, "generated_state.inc")

    assert "public cam_python_state_count" in result
    assert "include 'generated_state.inc'" in result
    assert result.index("include 'generated_state.inc'") < result.index("end module")


def test_cam_init_split_returns_to_python_before_state_allocation() -> None:
    source = """
module cam_comp
contains
subroutine cam_init(cam_out, cam_in, mpicom_atm, start_ymd, start_tod, &
                    ref_ymd, ref_tod, stop_ymd, stop_tod, perpetual_run, &
                    perpetual_ymd, calendar)
  integer :: mpicom_atm, start_ymd, start_tod, ref_ymd, ref_tod
  integer :: stop_ymd, stop_tod, perpetual_ymd
  logical :: perpetual_run
  character(len=*) :: calendar
  integer :: cam_out, cam_in
  character(len=8) :: filein
  etamid = nan
  filein = 'atm_in'
  if ( nsrest == 0 )then
     call cam_initfiles_open()
     call cam_initial(dyn_in, dyn_out, NLFileName=filein)
     ! Allocate and setup surface exchange data
     call atm2hub_alloc(cam_out)
     call hub2atm_alloc(cam_in)
  else
     call cam_read_restart()
  end if
  call phys_init( phys_state, phys_tend, pbuf2d, cam_out )
  dtime = get_step_size()
end subroutine cam_init
end module cam_comp
"""

    result = split_cam_init_for_python_state(source)

    prepare = result[result.index("subroutine cam_init_prepare") :]
    prepare = prepare[: prepare.index("end subroutine cam_init_prepare")]
    finish = result[result.index("subroutine cam_init_finish") :]
    finish = finish[: finish.index("end subroutine cam_init_finish")]
    assert "call cam_initial" in prepare
    assert "hub2atm_alloc" not in prepare
    assert "call hub2atm_alloc" in finish
    assert "call phys_init" in finish
