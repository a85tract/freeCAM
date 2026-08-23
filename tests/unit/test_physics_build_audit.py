"""Pure logic of the standalone-image builder: descriptors, stubs, audits."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools"))

import build_pi_cam_standalone_function as builder  # noqa: E402
from freecam.physics.spec import load_function_spec  # noqa: E402
from freecam.pi_cam.kernel_codegen import generate_direct_kernel_module  # noqa: E402


def test_standalone_kernel_declares_every_extent_in_fortran_order() -> None:
    spec = load_function_spec("dadadj")
    kernel = builder.standalone_kernel(spec)

    assert kernel.symbol == "freecam_standalone_dadadj_v1"
    assert [item.field for item in kernel.arguments] == [
        f"dadadj.{name}" for name in ("lchnk", "ncol", "pmid", "pint", "pdel", "t", "q")
    ]
    # Scalars become one value per chunk; arrays gain the chunk axis last, and
    # every axis is named so the wrapper checks pcols/pver/pverp at the call.
    assert kernel.arguments[0].extents == ("chunks",)
    assert kernel.arguments[3].extents == ("pcols", "pverp", "chunks")
    assert kernel.arguments[3].chunk_axis == 3
    # An external routine is called directly; only ppgrid is used.
    assert kernel.modules == (("ppgrid", ("pcols", "pver", "pverp")),)
    source = generate_direct_kernel_module((kernel,), module_name="probe")
    assert "call dadadj( &" in source
    assert source.count("/= pcols") == 5 and source.count("/= pverp") == 1


def test_stub_list_instantiates_each_class_and_rejects_unknown_inert() -> None:
    spec = load_function_spec("dadadj")
    text = builder.stub_list(spec)
    assert "FREECAM_INERT_DATA_INT32(spmd_utils_mp_masterproc_)" in text
    assert "FREECAM_FAIL_CLOSED(phys_grid_mp_get_lat_p_)" in text
    assert "FREECAM_ABORT(shr_sys_mod_mp_shr_sys_abort_)" in text

    stubs = dict(spec.image.stubs)
    stubs["inert"] = (*stubs["inert"], "mystery_mp_thing_")
    patched = type(spec.image)(spec.image.archive_members, stubs, spec.image.base_address)
    with pytest.raises(RuntimeError, match="no known shape"):
        builder.stub_list(replace(spec, image=patched))


def test_stub_set_audit_requires_exact_agreement(monkeypatch) -> None:
    spec = load_function_spec("dadadj")
    declared = set(spec.image.stub_symbols)
    runtime = {"for_write_seq_lis", "_intel_fast_memcpy", "exp", "f_ldnint_val", "__svml_pow2"}

    monkeypatch.setattr(builder, "undefined_symbols", lambda objects: declared | runtime)
    audit = builder.audit_stub_set(spec, [])
    assert set(audit["cesm_symbols_needed"]) == declared

    monkeypatch.setattr(builder, "undefined_symbols", lambda objects: declared | {"cam_history_mp_outfld_"})
    with pytest.raises(RuntimeError, match="missing=\\['cam_history_mp_outfld_'\\]"):
        builder.audit_stub_set(spec, [])

    monkeypatch.setattr(builder, "undefined_symbols", lambda objects: declared - {"mpibcast_"})
    with pytest.raises(RuntimeError, match="unused=\\['mpibcast_'\\]"):
        builder.audit_stub_set(spec, [])


def test_link_trace_rejects_cesm_archives() -> None:
    own = {Path("/work/objects/wrapper.o"), Path("/work/objects/dadadj.o")}
    trace = textwrap.dedent(
        """
        /work/objects/wrapper.o
        /work/objects/dadadj.o
        /opt/cray/pe/mpich/8.1.25/ofi/intel/19.0/lib/libmpifort_intel.so
        /glade/u/apps/common/23.04/spack/opt/spack/intel-oneapi-compilers/2023.0.0/compiler/2023.0.0/linux/compiler/lib/intel64_lin/libifcoremt.so
        """
    )
    parsed = builder.parse_link_trace(trace, own)
    assert len(parsed["own_objects"]) == 2 and len(parsed["runtime_inputs"]) == 2

    with pytest.raises(RuntimeError, match="forbidden input"):
        builder.parse_link_trace(trace + "/scratch/bld/lib/libatm.a(cam_history.o)\n", own)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_stub_c_behaves_as_declared(tmp_path: Path) -> None:
    """Compile cam_stubs.c with a small list and exercise each stub class."""

    (tmp_path / "stub_list.h").write_text(
        "FREECAM_INERT_FALSE(probe_false_)\n"
        "FREECAM_INERT_VOID(probe_void_)\n"
        "FREECAM_INERT_DATA_INT32(probe_data_)\n"
        "FREECAM_FAIL_CLOSED(probe_closed_)\n"
        "FREECAM_ABORT(probe_abort_)\n"
    )
    library = tmp_path / "libstubs.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-O2", f"-I{tmp_path}", str(PROJECT / "native/pi_cam/standalone/cam_stubs.c"), "-o", str(library)],
        check=True, capture_output=True, text=True,
    )
    script = textwrap.dedent(
        """
        import ctypes, sys
        lib = ctypes.CDLL(sys.argv[1])
        mode = sys.argv[2]
        lib.probe_void_()
        assert lib.probe_false_() == 0
        assert ctypes.c_int32.in_dll(lib, "probe_data_").value == 0
        if mode == "closed":
            lib.probe_closed_()
        if mode == "abort":
            message = b"Impossible case1 in instratus_condensate   "
            lib.probe_abort_.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
            lib.probe_abort_(message, None, len(message))
        print("inert ok")
        """
    )
    env = {key: value for key, value in os.environ.items() if key != "LD_PRELOAD"}
    inert = subprocess.run([sys.executable, "-c", script, str(library), "inert"], capture_output=True, text=True, env=env)
    assert inert.returncode == 0 and "inert ok" in inert.stdout
    closed = subprocess.run([sys.executable, "-c", script, str(library), "closed"], capture_output=True, text=True, env=env)
    assert closed.returncode == builder.EXIT_STUB_CALLED
    assert "FREECAM_STUB_CALLED: probe_closed_" in closed.stderr
    aborted = subprocess.run([sys.executable, "-c", script, str(library), "abort"], capture_output=True, text=True, env=env)
    assert aborted.returncode == builder.EXIT_ABORT
    # Trailing Fortran blanks are trimmed from the reported message.
    assert "FREECAM_FORTRAN_ABORT: Impossible case1 in instratus_condensate\n" in aborted.stderr
