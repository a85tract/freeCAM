"""Command-line entry point.

``freecam ui`` serves the Workflow Builder page; every other invocation is
the MPI rank command line of :mod:`freecam.pi_cam.cli`.
"""

from __future__ import annotations

import sys

from .pi_cam.cli import main as rank_main


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "ui":
        from .pi_cam.workflow_builder.ui import main as ui_main

        return ui_main(arguments[1:])
    return rank_main(argv)


__all__ = ["main"]
