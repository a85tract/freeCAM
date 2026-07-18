# Jupyter Notebook interface

`NotebookSession` lets one ordinary Jupyter kernel control the validated
24-rank CAM-SIMA configuration. It starts a separate MPI worker and exposes a
synchronous Python API over an authenticated socket on the same node.

The Notebook must run inside a compute allocation that can launch 24 MPI
processes. Prepare a fresh run directory containing `atm_in`; do not point a
new session at a directory containing results that must be preserved.
When Jupyter does not expose `PBS_NODEFILE`, `NotebookSession` automatically
passes the current compute hostname to Cray PALS. It refuses to launch on a
Derecho login node.

Open the ready-made Notebook for a real cell-by-cell test:

```text
/glade/work/ruitong/pycam-sima/examples/try_notebook_session.ipynb
```

It keeps the MPI session open between cells, so `model.step()` and
`model.get_field()` are genuinely interactive. The companion `.py` file is
only a command-line or `%run` smoke test.

```python
from pathlib import Path

from pycam_sima import NotebookSession

repo = Path("/glade/work/ruitong/pycam-sima")
run_dir = Path(
    "/glade/derecho/scratch/ruitong/pycam-sima/experiments/notebook01/run"
)
case = repo / "reference/cases/FKESSLER_ne3pg3_gnu_24x50"

run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "atm_in").write_bytes(
    Path(
        "/glade/derecho/scratch/ruitong/pycam-sima/"
        "FKESSLER_ne3pg3_gnu_24x50/FKESSLER_ne3pg3_gnu_24x50/run/atm_in"
    ).read_bytes()
)

model = NotebookSession(
    repo / "configs/fkessler_ne3pg3.yaml",
    run_dir=run_dir,
    env_script=case / ".env_mach_specific.sh",
)
model.start()
```

Inspect the available rank-zero fields before or after a step:

```python
model.field_names
model.field_info("air_temperature")

model.step()
temperature = model.get_field("air_temperature", rank=0)
temperature.shape, temperature.min(), temperature.max()
```

Get one value without transferring every rank-local array:

```python
statistics = model.get_field_stats("air_temperature", rank="all")
statistics[0]
```

`rank="all"` returns one local result per MPI rank. It does not reconstruct SE
global-column order. History NetCDF remains the authoritative globally ordered
output.

Interactive sessions may modify live CAM memory. The following change is
applied on rank zero and is consumed by the next model step:

```python
temperature = model.get_field("air_temperature", rank=0)
temperature[0, 0] += 0.01
model.set_field("air_temperature", temperature, rank=0)
model.step()
```

Such changes intentionally break BFB. Close the worker to run CAM finalization
and release all MPI processes:

```python
model.close()
```

A context manager closes the model if a Notebook cell raises an exception:

```python
with NotebookSession(
    repo / "configs/fkessler_ne3pg3.yaml",
    run_dir=run_dir,
    env_script=case / ".env_mach_specific.sh",
) as model:
    for _ in range(10):
        model.step()
        print(model.get_field_stats("air_temperature", rank=0))
```
