#!/usr/bin/env python3
"""Compile the walk's Python glue in place with Cython -- a trial, not part of the install.

    tools/build_glue_trial.py            # build the .so files beside the .py files (gitignored)
    tools/build_glue_trial.py --clean    # remove them: the pure Python path is back

Two layers.  The modules the walk runs through (the stage runtime, the
Fortran adapter, the physics buffer, the four walks) are compiled as they
are, in Cython's pure-Python mode, so their statements run as C instead of
bytecode.  ``freecam/core/_glue.pyx`` adds direct callers for the image's
hottest entries -- the bound kernel call, the view probes, the buffer
accessors, the history call -- which the Python modules use when the
extension is importable and fall back from otherwise.  Needs Cython and a C
compiler; nothing here changes what is computed, and the gate decides.
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
MODULES = ("freecam/physics/stage.py", "freecam/core/fortran_adapter.py", "freecam/pi_cam/pbuf.py",
           "freecam/physics/macrophysics.py", "freecam/physics/microphysics.py",
           "freecam/physics/microp_aero.py", "freecam/physics/cloud_macro_microphysics.py")
DIRECT = "freecam/core/_glue.pyx"


def clean() -> None:
    for module in (*MODULES, DIRECT):
        stem = str(SRC / module).rsplit(".", 1)[0]
        for path in glob.glob(f"{stem}.cpython-*.so") + glob.glob(f"{stem}.c"):
            os.remove(path)
            print("removed", os.path.relpath(path, REPO))


def build() -> None:
    from setuptools import Extension, setup
    from Cython.Build import cythonize

    # The site's compiler wrappers link Cray MPI, PMI and a libfabric into
    # anything they build; an extension imported before mpi4py initialises
    # MPI then brings a second MPI into the process and MPI_Init fails on
    # scattered ranks (sixteen gates on 2026-09-15).  Build with the bare
    # compiler: the modules must need libc alone.
    compiler = os.environ.get("FREECAM_GLUE_CC") or ("/usr/bin/gcc" if os.path.exists("/usr/bin/gcc") else "gcc")
    os.environ["CC"] = compiler
    os.environ["LDSHARED"] = f"{compiler} -shared"
    os.environ.pop("LDFLAGS", None)
    os.environ.pop("CFLAGS", None)
    temp = tempfile.mkdtemp(prefix="freecam-glue-")
    extensions = [Extension(module.replace("/", ".")[:-3], [str(SRC / module)], extra_compile_args=["-O2"])
                  for module in MODULES]
    extensions.append(Extension(DIRECT.replace("/", ".")[:-4], [str(SRC / DIRECT)], extra_compile_args=["-O2"]))
    os.chdir(SRC)
    setup(name="freecam_glue_trial",
          script_args=["build_ext", "--inplace", "--build-temp", temp, "-j", "4"],
          ext_modules=cythonize(extensions, build_dir=temp, quiet=True,
                                compiler_directives={"language_level": "3", "annotation_typing": False}))
    stray = SRC / "build"
    if stray.is_dir():
        import shutil
        shutil.rmtree(stray)
    for module in (*MODULES, DIRECT):
        stem = str(SRC / module).rsplit(".", 1)[0]
        for path in glob.glob(f"{stem}.cpython-*.so"):
            needed = os.popen(f"readelf -d {path}").read()
            if "mpi" in needed.lower() or "pmi" in needed.lower() or "RPATH" in needed or "RUNPATH" in needed:
                raise SystemExit(f"{os.path.relpath(path, REPO)} links more than libc: refusing (see the docstring)")
    print("built; every module needs libc alone")


if __name__ == "__main__":
    if "--clean" in sys.argv[1:]:
        clean()
    else:
        build()
