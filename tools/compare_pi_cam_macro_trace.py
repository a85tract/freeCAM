#!/usr/bin/env python3
"""Deprecated name for tools/compare_pi_cam_stage_trace.py.

The comparison is not specific to macrophysics; it reads any stage's kernel
trace.  This alias keeps the Gate B2 job working while it names the old path.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("compare_pi_cam_stage_trace.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
