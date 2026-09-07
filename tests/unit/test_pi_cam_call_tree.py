"""The call-tree analysis: scoping, preprocessing, guards and contexts on toy sources."""
from pathlib import Path
import shutil

import pytest

from freecam.pi_cam.call_tree import (
    CallTree,
    cpp_conditions,
    expand_templates,
    fortran_files,
    preprocess_source,
    scan_file,
    select_sources,
)

pytestmark = pytest.mark.skipif(shutil.which("cpp") is None, reason="the C preprocessor is not on PATH")


KINDS = """
module shr_kind_mod
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
end module shr_kind_mod
"""

HELPERS = """
module helpers
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  private
  public :: qsat, table, gen_scale, r8, op_t
  real(r8) :: table(10)
  logical, parameter, public :: fixed_flag = .false.
  type :: op_t
    real(r8) :: a
  contains
    procedure :: apply => apply_op
    procedure :: scalar_add
    generic :: add => scalar_add
  end type op_t
  interface gen_scale
    module procedure scale_r, scale_i
  end interface
contains
  real(r8) function qsat(t)
    real(r8), intent(in) :: t
    qsat = t * 2.0_r8
  end function qsat
  subroutine scale_r(x); real(r8) :: x; x = x * 2; end subroutine scale_r
  subroutine scale_i(x); integer :: x; x = x * 2; end subroutine scale_i
  subroutine hidden(x); real(r8) :: x; x = 0; end subroutine hidden
  subroutine apply_op(self, x); class(op_t) :: self; real(r8) :: x; x = self%a * x; end subroutine apply_op
  subroutine scalar_add(self, x); class(op_t) :: self; real(r8) :: x; x = x + self%a; end subroutine scalar_add
end module helpers
"""

DRIVER = """
module driver
  use helpers
  implicit none
  character(len=16) :: deep_scheme = 'ZM'
contains
  subroutine drv_tend(state, ncol, pf)
    real(r8), intent(inout) :: state(:,:)
    integer, intent(in) :: ncol
    procedure(real), optional :: pf
    external :: legacy_fun
    real(r8) :: legacy_fun, loc(4)
    type(op_t) :: op
    integer :: i, k
    if (.not. present(pf)) return
    do i = 1, ncol
      do k = 1, 4
        state(i,k) = qsat(state(i,k)) + table(k) + loc(k) + legacy_fun(state(i,k)) + sqrt(state(i,k))
      end do
      if (deep_scheme == 'ZM') then
        call zm_core(state(i,:), inner(i))
      else if (deep_scheme == 'UNICON') then
        call unicon_core(state(i,:))
      else
        call other_core(state(i,:))
      end if
      select case (trim(deep_scheme))
      case ('ZM')
        call gen_scale(state(i,1))
      case default
        call gen_scale(i)
      end select
      call op%apply(state(i,1))
      associate (t => state(i,:))
        t(1) = t(2)
      end associate
#ifdef SPMD
      call mpi_barrier(0, k)
#endif
#ifndef SPMD
      call never_compiled(state)
#endif
      if (present(pf)) state(i,1) = pf(state(i,1))
    end do
  contains
    real(r8) function inner(j)
      integer :: j
      inner = j * qsat(1.0_r8)
    end function inner
  end subroutine drv_tend
  subroutine zm_core(col, w)
    real(r8) :: col(:), w
    col = col * w
  end subroutine zm_core
  subroutine unicon_core(col)
    real(r8) :: col(:)
    col = col - 1
  end subroutine unicon_core
  subroutine other_core(col)
    real(r8) :: col(:)
    call hidden(col(1))
  end subroutine other_core
end module driver
real(8) function legacy_fun(x)
  real(8) :: x
  legacy_fun = x
end function legacy_fun
"""


@pytest.fixture
def toy_tree(tmp_path: Path) -> CallTree:
    src = tmp_path / "src"
    src.mkdir()
    (src / "kinds.F90").write_text(KINDS)
    (src / "helpers.F90").write_text(HELPERS)
    (src / "driver.F90").write_text(DRIVER)
    return CallTree.scan(fortran_files([src]), source_root=tmp_path, macros=["SPMD"], workers=1)


def _sites(tree: CallTree, qualified: str):
    return {(site.name, site.line): site for site in tree.sites[qualified]}


def test_functions_in_expressions_are_resolved_and_arrays_are_not(toy_tree: CallTree) -> None:
    sites = _sites(toy_tree, "driver::drv_tend")
    names = {name for name, _ in sites}
    assert "qsat" in names and "legacy_fun" in names
    # arrays from the local scope, the host module and a wildcard import are not calls
    assert not names & {"state", "loc", "table", "sqrt"}
    qsat = next(site for (name, _), site in sites.items() if name == "qsat")
    assert qsat.kind == "function" and qsat.target == "helpers::qsat"
    assert qsat.in_expression and qsat.loop_depth == 2
    legacy = next(site for (name, _), site in sites.items() if name == "legacy_fun")
    assert legacy.kind == "external" and legacy.target == "driver::legacy_fun"


def test_internal_procedure_generic_and_dummy_procedure(toy_tree: CallTree) -> None:
    sites = _sites(toy_tree, "driver::drv_tend")
    inner = next(site for (name, _), site in sites.items() if name == "inner")
    assert inner.target == "driver::drv_tend.inner" and inner.as_argument
    generics = [site for (name, _), site in sites.items() if name == "gen_scale"]
    assert len(generics) == 2
    assert set(generics[0].candidates) == {"helpers::scale_r", "helpers::scale_i"}
    pf = next(site for (name, _), site in sites.items() if name == "pf")
    assert pf.kind == "dummy_procedure"


def test_guards_follow_if_else_and_select_branches(toy_tree: CallTree) -> None:
    sites = _sites(toy_tree, "driver::drv_tend")
    other = next(site for (name, _), site in sites.items() if name == "other_core")
    assert [(part.text, part.negate) for part in other.guards] == [
        ("deep_scheme == 'ZM'", True),
        ("deep_scheme == 'UNICON'", True),
    ]
    default_case = [site for (name, _), site in sites.items() if name == "gen_scale"][1]
    assert default_case.guards[0].kind == "case" and default_case.guards[0].negate
    assert default_case.guards[0].values == ("'ZM'",)


def test_preprocessor_conditions_and_removed_lines(toy_tree: CallTree) -> None:
    sites = _sites(toy_tree, "driver::drv_tend")
    barrier = next(site for (name, _), site in sites.items() if name == "mpi_barrier")
    assert barrier.active and barrier.cpp_condition == "(defined(SPMD))"
    assert barrier.kind == "unresolved"
    assert all(name != "never_compiled" for name, _ in sites)


def test_type_bound_call_and_associate_alias(toy_tree: CallTree) -> None:
    sites = _sites(toy_tree, "driver::drv_tend")
    bound = next(site for (name, _), site in sites.items() if name == "op%apply")
    assert bound.kind == "call" and bound.target == "helpers::apply_op"
    assert all(name != "t" for name, _ in sites)


def test_private_procedure_is_not_visible_through_a_wildcard_use(toy_tree: CallTree) -> None:
    hidden = toy_tree.sites["driver::other_core"]
    assert [(site.name, site.kind) for site in hidden] == [("hidden", "unresolved")]


def test_entry_guard_and_literal_parameter_are_recorded(toy_tree: CallTree) -> None:
    assert toy_tree.procedures["driver::drv_tend"].entry_guards == (".NOT. PRESENT(pf)",)
    assert toy_tree.modules["helpers"].parameters == {"fixed_flag": ".FALSE."}


def test_reach_stops_at_guards_the_caller_rules_out(toy_tree: CallTree) -> None:
    everything = toy_tree.reach(["driver::drv_tend"])
    assert "driver::unicon_core" in everything
    without_unicon = toy_tree.reach(
        ["driver::drv_tend"], decide=lambda site: site.name != "unicon_core"
    )
    assert "driver::unicon_core" not in without_unicon and "driver::zm_core" in without_unicon


def test_cpp_conditions_track_elif_and_else() -> None:
    text = "#ifdef A\nx\n#elif defined(B)\ny\n#else\nz\n#endif\nw\n"
    conditions = cpp_conditions(text)
    assert conditions[2] == "(defined(A))"
    assert conditions[4] == "!(defined(A)) && (defined(B))"
    assert conditions[6] == "!(defined(A)) && !(defined(B))"
    assert 8 not in conditions


def test_preprocess_maps_lines_back_after_removed_blocks(tmp_path: Path) -> None:
    path = tmp_path / "a.F90"
    path.write_text("subroutine a\n#ifdef GONE\n  call gone()\n#endif\n  call kept()\nend subroutine a\n")
    pre = preprocess_source(path, macros=[])
    kept = next(index for index, line in enumerate(pre.text.splitlines(), start=1) if "kept" in line)
    assert pre.origin[kept - 1] == 5
    assert 3 not in pre.active_lines and 5 in pre.active_lines


def test_select_sources_gives_the_first_directory_precedence(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "shared.F90").write_text("module shared\nend module shared\n")
    (second / "shared.F90").write_text("module shared\nend module shared\n")
    (second / "only_here.F90").write_text("module only_here\nend module only_here\n")
    chosen = select_sources([first, second])
    assert chosen["shared.f90"] == first / "shared.F90"
    assert chosen["only_here.f90"] == second / "only_here.F90"


def test_parse_failure_is_reported_not_hidden(tmp_path: Path) -> None:
    path = tmp_path / "bad.F90"
    path.write_text("subroutine bad(\n  real :: x\nend subroutine bad\n")
    scan = scan_file(path, source_root=tmp_path, macros=[], include_dirs=[])
    assert scan.failure is not None and scan.failure["error_type"]
    assert scan.procedures == []


@pytest.mark.skipif(shutil.which("perl") is None, reason="genf90 needs perl")
def test_expand_templates_runs_genf90_when_available(tmp_path: Path) -> None:
    genf90 = Path("external/iCESM1.3.1_fzhu/cime/src/externals/genf90/genf90.pl")
    if not genf90.is_file():
        pytest.skip("pinned source not checked out")
    template = tmp_path / "toy.F90.in"
    template.write_text("module toy\n! TYPE int,double\nsubroutine s_{TYPE}(x)\n{VTYPE} :: x\nend subroutine s_{TYPE}\nend module toy\n")
    produced = expand_templates([template], genf90=genf90, output_dir=tmp_path / "gen")
    text = produced[str(template)].read_text()
    assert "s_int" in text and "s_double" in text
