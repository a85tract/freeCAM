"""The state copy into kept storage: physics_state_copy's statements, less its allocation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/physics_types.F90"
MODULE = REPO / "native/pi_cam/support/pycam_state_copy.F90"

pinned = pytest.mark.skipif(not PINNED.exists(), reason="the pinned iCESM source is not checked out")


@pinned
def test_the_committed_module_is_what_the_generator_writes() -> None:
    import generate_pi_cam_state_copy as gen

    assert MODULE.read_text() == gen.render()


@pinned
def test_the_routine_is_the_pinned_copy_statement_for_statement_without_the_allocation() -> None:
    import generate_pi_cam_state_copy as gen

    first, lines = gen.routine_lines()
    def code_only(items):
        return [s for s in (x.strip() for x in items) if s and not s.startswith("!")]
    expected = code_only(lines[1:-1])
    expected = [s for s in expected if not s.startswith(("use ", "implicit none", "type(physics_state)"))]
    assert sum(s.startswith(gen.ALLOCATION) for s in expected) == 1
    expected = [s for s in expected if not s.startswith(gen.ALLOCATION)]
    module = MODULE.read_text()
    body = module[module.index("subroutine pycam_state_copy_into("):module.index("end subroutine pycam_state_copy_into")]
    carried = code_only(body.splitlines())
    carried = [s for s in carried if not s.startswith(("subroutine", "use ", "type(physics_state)"))]
    assert carried == expected
    assert f"physics_types.F90:{first}-{first + len(lines) - 1}" in module


def test_the_module_moves_memory_only_and_never_allocates() -> None:
    code = "\n".join(l for l in MODULE.read_text().splitlines() if not l.strip().startswith("!"))
    assert not re.search(r"\ballocate\s*\(", code, re.I)
    assert "physics_state_alloc" not in code
    # every executable statement is an assignment of a component, a loop, or the column count
    for line in code.splitlines():
        s = line.strip()
        if not s or s.startswith(("module", "end", "use ", "implicit", "private", "public", "contains",
                                  "subroutine", "type(", "integer", "do ", "end do")):
            continue
        assert re.match(r"^(state_out%\w+\([^)]*\)|state_out%\w+|ncol)\s*=", s), s


def test_the_handles_keep_the_copy_and_use_the_module() -> None:
    macro = (REPO / "native/pi_cam/support/pycam_macro_handles.F90").read_text()
    micro = (REPO / "native/pi_cam/support/pycam_micro_handles.F90").read_text()
    for code in (macro, micro):
        assert "use pycam_state_copy, only: pycam_state_copy_into" in code
        assert "state_kept" in code
    # the macro copy: allocated once per chunk, rewritten afterwards; its dealloc entry keeps the storage
    copy = macro.split("function pycam_macro_state_copy_v1", 1)[1].split("end function", 1)[0]
    assert "pycam_state_copy_into(host_state(lchnk), macro_state_loc(lchnk))" in copy
    assert "physics_state_copy(host_state(lchnk), macro_state_loc(lchnk))" in copy
    dealloc = macro.split("function pycam_macro_state_dealloc_v1", 1)[1].split("end function", 1)[0]
    assert "physics_state_dealloc" not in dealloc.replace("! the driver deallocates here (physics_state_dealloc)", "")
    # the micro copy the same; the runner frees a kept copy before its own head allocates one
    begin = micro.split("function pycam_micro_begin_v1", 1)[1].split("end function", 1)[0]
    assert "pycam_state_copy_into(state, state_loc)" in begin and "physics_state_copy(state, state_loc)" in begin
    bind = micro.split("subroutine micro_runner_bind", 1)[1].split("end subroutine", 1)[0]
    assert "if (state_live .or. state_kept) call physics_state_dealloc(state_loc)" in bind
    # registered for the source tree and the image, ahead of the handles that use it
    sources = (REPO / "tools/apply_pi_cam_source_patches.py").read_text()
    modules = (REPO / "tools/build_pi_cam_devices.py").read_text()
    assert sources.index("pycam_state_copy.F90") < sources.index("pycam_macro_handles.F90")
    assert modules.index('"pycam_state_copy.F90"') < modules.index('"pycam_macro_handles.F90"')
