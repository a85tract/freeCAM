"""The count-only trampolines: selection, rendering, and real forwarding through a compiler.

The compiled test is the small proof the counting design rests on: a tail-jump
trampoline forwards register and stack arguments untouched, returns scalar
function values unchanged (a subroutine wrapper is never passed off as a
function wrapper), and counts once per call under the current context slot.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "kcount_trampolines", REPO / "tools/generate_pi_cam_kcount_trampolines.py")
gen = importlib.util.module_from_spec(spec)
sys.modules["kcount_trampolines"] = gen
spec.loader.exec_module(gen)

OBSERVABILITY = REPO / "validation/pi_cam_kernel_observability.json"


def _fake_record():
    def kernel(index, qualified, routine, symbol, processes, coverage="full", blind=()):
        return {"index": index, "qualified": qualified, "routine": routine, "symbol": symbol,
                "entry": "trampoline", "defining_object": "m.o", "coverage": coverage,
                "blind_spots": list(blind), "processes": processes}
    return {
        "schema_version": 1, "content_hash": "f" * 64,
        "table": {"slots": 64},
        "kernels": [
            kernel(3, "m::fsum", "fsum", "m_mp_fsum_", ["cam_run1.a"]),
            kernel(7, "m::vfill", "vfill", "m_mp_vfill_", ["cam_run1.a", "cam_run1.b"]),
            {"index": 9, "qualified": "m::inlined", "routine": "inlined", "symbol": "m_mp_inlined_",
             "entry": "none", "defining_object": "m.o", "coverage": "none", "blind_spots": [],
             "processes": ["cam_run1.b"]},
        ],
    }


def test_scope_selection_and_rendering() -> None:
    record = _fake_record()
    assert [k["index"] for k in gen.select_kernels(record, "all")] == [3, 7]
    assert [k["routine"] for k in gen.select_kernels(record, "cam_run1.b")] == ["vfill"]
    assert [k["routine"] for k in gen.select_kernels(record, "fsum")] == ["fsum"]
    text = gen.render(record, gen.select_kernels(record, "all"), "all")
    assert "pycam_kcount_table+1536(,%r11,8)" in text            # index 3 * 64 slots * 8 bytes
    assert "jmp pycam_orig_m_mp_fsum_" in text
    assert ".long 8" in text                                     # highest index + 1
    table = gen.trampoline_table(gen.select_kernels(record, "all"))
    assert table[0]["alias"] == "pycam_orig_m_mp_fsum_" and table[1]["object"] == "m.o"


def test_the_committed_observability_record_drives_the_generator() -> None:
    record = gen.load_observability()
    kernels = gen.select_kernels(record, "all")
    assert len(kernels) == record["summary"]["by_entry"]["trampoline"]
    indexes = [k["index"] for k in kernels]
    assert len(set(indexes)) == len(indexes) and max(indexes) < 1024
    fice = next(k for k in kernels if k["qualified"] == "cloud_fraction::cldfrc_fice")
    assert fice["symbol"] == "cloud_fraction_mp_cldfrc_fice_" and fice["coverage"] == "full"


_CC = shutil.which("gcc") or shutil.which("cc")


@pytest.mark.skipif(_CC is None or platform.machine() != "x86_64",
                    reason="needs an x86-64 C compiler")
def test_trampolines_forward_arguments_return_values_and_count(tmp_path: Path) -> None:
    record = _fake_record()
    (tmp_path / "tramp.S").write_text(gen.render(record, gen.select_kernels(record, "all"), "all"))
    # the originals live under the alias names, exactly as the build's weaken-and-alias produces
    (tmp_path / "orig.c").write_text(r"""
#include <stdint.h>
/* a scalar function with register and stack arguments in both classes */
double pycam_orig_m_mp_fsum_(long a, long b, long c, long d, long e, long f, long g,
                             double x0, double x1, double x2, double x3, double x4,
                             double x5, double x6, double x7, double x8) {
    return (double)(a + 2*b + 3*c + 4*d + 5*e + 6*f + 7*g)
         + x0 + 2*x1 + 3*x2 + 4*x3 + 5*x4 + 6*x5 + 7*x6 + 8*x7 + 9*x8;
}
/* a subroutine writing through Fortran-style reference arguments */
void pycam_orig_m_mp_vfill_(double *out, const long *n, const double *scale) {
    for (long i = 0; i < *n; ++i) out[i] = (double)(i + 1) * *scale;
}
double m_mp_fsum_(long, long, long, long, long, long, long,
                  double, double, double, double, double, double, double, double, double);
void m_mp_vfill_(double *, const long *, const double *);
""")
    (tmp_path / "main.c").write_text(r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern int64_t pycam_kcount_table[];
int32_t pycam_kcount_context_set_v1(int32_t);
int32_t pycam_kcount_info_v1(int32_t *, int32_t *, int32_t *);
void pycam_kcount_reset_v1(void);
double pycam_orig_m_mp_fsum_(long, long, long, long, long, long, long,
                             double, double, double, double, double, double, double, double, double);
void pycam_orig_m_mp_vfill_(double *, const long *, const double *);
double m_mp_fsum_(long, long, long, long, long, long, long,
                  double, double, double, double, double, double, double, double, double);
void m_mp_vfill_(double *, const long *, const double *);

int main(void) {
    pycam_kcount_reset_v1();
    int32_t kernels, slots, max_kernels;
    if (pycam_kcount_info_v1(&kernels, &slots, &max_kernels) != 0) return 10;
    if (kernels != 8 || slots != 64) return 11;             /* strong count from the trampoline object */

    pycam_kcount_context_set_v1(3);
    double through = m_mp_fsum_(1, 2, 3, 4, 5, 6, 7, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5);
    double direct = pycam_orig_m_mp_fsum_(1, 2, 3, 4, 5, 6, 7, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5);
    if (memcmp(&through, &direct, sizeof through) != 0) return 12;   /* bitwise, not approximate */

    long n = 4; double scale = 2.5, got[4] = {0}, want[4] = {0};
    m_mp_vfill_(got, &n, &scale);
    pycam_kcount_context_set_v1(5);
    m_mp_vfill_(got, &n, &scale);
    pycam_orig_m_mp_vfill_(want, &n, &scale);
    if (memcmp(got, want, sizeof got) != 0) return 13;

    if (pycam_kcount_table[3 * 64 + 3] != 1) return 14;     /* fsum counted once, in slot 3 */
    if (pycam_kcount_table[7 * 64 + 3] != 1) return 15;     /* first vfill call, slot 3 */
    if (pycam_kcount_table[7 * 64 + 5] != 1) return 16;     /* second vfill call, slot 5 */
    if (pycam_kcount_table[3 * 64 + 5] != 0) return 17;     /* nothing invented */
    pycam_kcount_context_set_v1(9999);                       /* out of range falls back to 0 */
    m_mp_vfill_(got, &n, &scale);
    if (pycam_kcount_table[7 * 64 + 0] != 1) return 18;
    puts("trampolines ok");
    return 0;
}
""")
    binary = tmp_path / "kcount_test"
    compile_cmd = [_CC, "-O2", "-no-pie", "-o", str(binary),
                   str(tmp_path / "main.c"), str(tmp_path / "orig.c"), str(tmp_path / "tramp.S"),
                   str(REPO / "native/pi_cam/pycam_kcount.c")]
    build = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, f"exit {run.returncode}: {run.stdout} {run.stderr}"
    assert "trampolines ok" in run.stdout
