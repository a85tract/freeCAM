"""Command-line entry point.

``freecam ui`` serves the Workflow Builder page; ``freecam timeline DIR`` serves (or
writes, with ``--html``) the viewer of a run's action timeline; ``freecam globe DIR``
the globe viewer of a run's state snapshots, ``freecam trace DIR`` one column through one of its
action steps; ``freecam build`` makes the model for compile-time options
(:mod:`freecam.pi_cam.build`); every other invocation is the MPI rank command line of
:mod:`freecam.pi_cam.cli`.
"""

from __future__ import annotations

import sys

from .pi_cam.cli import main as rank_main


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "ui":
        from .pi_cam.workflow_builder.ui import main as ui_main

        return ui_main(arguments[1:])
    if arguments and arguments[0] == "build":
        from .pi_cam.build import main as build_main

        return build_main(arguments[1:])
    if arguments and arguments[0] == "timeline":
        from .pi_cam.timeline_view import main as timeline_main

        return timeline_main(arguments[1:])
    if arguments and arguments[0] == "globe":
        from .pi_cam.state_view import main as globe_main

        return globe_main(arguments[1:])
    if arguments and arguments[0] == "trace":
        from .pi_cam.state_view import trace_main

        return trace_main(arguments[1:])
    if arguments and arguments[0] == "anomalies":
        from .pi_cam.state_view import anomalies_main

        return anomalies_main(arguments[1:])
    return rank_main(argv)


__all__ = ["main"]
