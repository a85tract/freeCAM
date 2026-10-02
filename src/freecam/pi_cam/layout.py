"""How the PI-atm components are laid out over a run's MPI ranks.

CESM reads its process layout from ``drv_in`` at run time: each component's
task count and first rank.  freeCAM runs CAM on every rank, so CAM and the
coupler take all of them; the others are placed as the admitted 512-rank case
places them, scaled to the count:

* CLM on ``min(N/2, N - 160)`` ranks from rank 0 (all of them below 192);
* CICE on 128 ranks after CLM -- its decomposition is compiled for 128 tasks;
* DOCN on 32 ranks after CICE;
* RTM, GLC and WAV on ``min(128, CLM)`` ranks from rank 0.

At 512 this is the admitted layout.  Any other count is an exploration: its
answers are not bit-for-bit with the 512-rank oracle (CESM's own are not either).

    python -m freecam.pi_cam.layout DRV_IN RANKS [--steps N] [--config IN --derived OUT]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Mapping, Sequence

#: the admitted case's rank count
ADMITTED_RANKS = 512
#: CICE's compiled decomposition
CICE_TASKS = 128
#: DOCN's tasks
DOCN_TASKS = 32
#: the drv_in task and first-rank entries a layout sets
LAYOUT_NAMES = ("atm_ntasks", "cpl_ntasks", "lnd_ntasks", "ice_ntasks", "ice_rootpe", "ocn_ntasks",
                "ocn_rootpe", "rof_ntasks", "glc_ntasks", "wav_ntasks")


class LayoutError(ValueError):
    """A rank count the components cannot be laid out over, or a drv_in that lacks an entry."""


def component_layout(ranks: int) -> dict[str, int]:
    """The drv_in task counts and first ranks for a run on ``ranks`` MPI ranks."""

    n = int(ranks)
    if n < CICE_TASKS:
        raise LayoutError(f"{n} ranks: CICE's decomposition is compiled for {CICE_TASKS} tasks, so a run needs "
                          f"at least {CICE_TASKS}")
    if n >= 192:
        land = min(n // 2, n - 160)
        ice_root, ocn_root = land, land + CICE_TASKS
    else:
        land, ice_root, ocn_root = n, 0, 0
    other = min(128, land)
    return {"atm_ntasks": n, "cpl_ntasks": n, "lnd_ntasks": land, "ice_ntasks": CICE_TASKS,
            "ice_rootpe": ice_root, "ocn_ntasks": DOCN_TASKS, "ocn_rootpe": ocn_root,
            "rof_ntasks": other, "glc_ntasks": other, "wav_ntasks": other}


def rewrite_drv_in(text: str, values: Mapping[str, int]) -> str:
    """``drv_in`` with each named integer entry set; an entry it does not hold exactly once is refused."""

    for name, value in values.items():
        text, count = re.subn(rf"(\b{re.escape(name)}\s*=\s*)-?\d+", rf"\g<1>{int(value)}", text)
        if count != 1:
            raise LayoutError(f"drv_in holds {count} entries {name}; a layout sets exactly one")
    return text


def read_layout(text: str) -> dict[str, int]:
    """The task counts and first ranks a ``drv_in`` gives."""

    return {name: int(value) for name, value in re.findall(r"\b(\w+_(?:ntasks|rootpe))\s*=\s*(-?\d+)", text)}


def lay_out(drv_in: str | Path, ranks: int, *, steps: int | None = None) -> dict[str, int]:
    """Rewrite the ``drv_in`` file for ``ranks`` ranks (and a run of ``steps`` steps); the values set."""

    path = Path(drv_in)
    values = component_layout(ranks)
    if steps is not None:
        values.update(stop_n=int(steps), restart_n=int(steps))
    path.write_text(rewrite_drv_in(path.read_text(), values))
    return values


def derive_config(source: str | Path, destination: str | Path, *, mpi_size: int | None = None,
                  stop_n: int | None = None) -> Path:
    """A copy of a case configuration with its rank count or length changed, the rest kept as written."""

    text = Path(source).read_text()
    for name, value in (("mpi_size", mpi_size), ("stop_n", stop_n)):
        if value is None:
            continue
        text, count = re.subn(rf"^{name}: \d+$", f"{name}: {int(value)}", text, flags=re.M)
        if count != 1:
            raise LayoutError(f"{source} holds {count} lines '{name}: N'; a derived configuration sets one")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("drv_in", type=Path)
    parser.add_argument("ranks", type=int)
    parser.add_argument("--steps", type=int, help="also set the run's length (stop_n, restart_n)")
    parser.add_argument("--config", type=Path, help="a case configuration to derive a copy of")
    parser.add_argument("--derived", type=Path, help="where the derived configuration goes")
    arguments = parser.parse_args(argv)
    values = lay_out(arguments.drv_in, arguments.ranks, steps=arguments.steps)
    if arguments.config is not None:
        if arguments.derived is None:
            parser.error("--config needs --derived")
        derive_config(arguments.config, arguments.derived, mpi_size=arguments.ranks, stop_n=arguments.steps)
    print("layout:", values)
    return 0


__all__ = ["ADMITTED_RANKS", "CICE_TASKS", "DOCN_TASKS", "LAYOUT_NAMES", "LayoutError", "component_layout",
           "derive_config", "lay_out", "read_layout", "rewrite_drv_in"]


if __name__ == "__main__":
    raise SystemExit(main())
