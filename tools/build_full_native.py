#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


REQUIRED_SYMBOLS = (
    "pycam_full_initialize",
    "pycam_full_timestep_init",
    "pycam_full_run1",
    "pycam_full_run2",
    "pycam_full_run3",
    "pycam_full_run4",
    "pycam_full_timestep_final",
    "pycam_full_advance_timestep",
    "pycam_full_get_field",
    "pycam_full_finalize",
)


def output(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/libpycam_sima_full.so"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    case = args.case_root.resolve()
    target = args.output if args.output.is_absolute() else (repo / args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    macros = (case / "Macros.make").read_text()
    if "FFLAGS" not in macros or "-fPIC" not in macros:
        raise SystemExit(
            "the CAM-SIMA case was not built with -fPIC; add -fPIC to CFLAGS, "
            "CXXFLAGS, and FFLAGS in the case cmake macros, clean-all, and rebuild"
        )

    build = Path(output(["./xmlquery", "EXEROOT", "--value"], cwd=case))
    wrapper = repo / "native/full_driver/cam_sima_full_abi.F90"
    obj = target.with_suffix(".o")
    shell = f"""
set -euo pipefail
source {shlex.quote(str(case / '.env_mach_specific.sh'))} >/dev/null 2>&1
B={shlex.quote(str(build))}
ftn -c -fPIC -O2 -ffp-contract=off -ffree-line-length-none \
  -I$B/atm/obj -I$B/gnu/mpich/nodebug/nothreads/include \
  -I$PIO/include -I$NCAR_ROOT_ESMF/include \
  {shlex.quote(str(wrapper))} -o {shlex.quote(str(obj))}
ftn -shared -o {shlex.quote(str(target))} {shlex.quote(str(obj))} \
  -L$B/lib -latm \
  -L$B/gnu/mpich/nodebug/nothreads/CDEPS/dshr -ldshr \
  -L$B/gnu/mpich/nodebug/nothreads/CDEPS/streams -lstreams \
  -L$B/gnu/mpich/nodebug/nothreads/lib -lcsm_share -lpiof -lpioc -lgptl \
  -L$B/gnu/mpich/nodebug/nothreads/CDEPS/fox/lib \
  -lFoX_dom -lFoX_sax -lFoX_utils -lFoX_fsys -lFoX_wxml -lFoX_common \
  -lpnetcdf -lesmf -lrt -lstdc++ -ldl -lsci_gnu -lnetcdf -lnetcdff -lm \
  -Wl,-rpath,$NCAR_ROOT_ESMF/lib
"""
    subprocess.run(["bash", "-c", shell], cwd=repo, check=True)

    symbols = output(["nm", "-D", "--defined-only", str(target)], cwd=repo)
    missing = [symbol for symbol in REQUIRED_SYMBOLS if symbol not in symbols]
    if missing:
        raise SystemExit(f"full CAM-SIMA library is missing ABI symbols: {missing}")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
