# freeCAM

freeCAM runs the CAM atmosphere component of iCESM1.3.1 under a Python
control layer. Python owns the model workflow, the clock, the coupling
decisions and the rank-local state; the original Fortran remains the numerical
source of truth and is called through generated C-interoperable adapters. There
is no shadow model and no Fortran-to-Python callback path.

The supported scientific configuration is the `ne16` PI-atm case with CAM5
physics, SE dynamics and 512 MPI ranks on NCAR Derecho, with the original CESM
surface components and coupler running live beside CAM.

```text
Python / Jupyter
    |
    |  Driver, workflow, clock, state
    v
512 persistent MPI Python ranks
    |
    +-- CAM numerical kernels ---------> original iCESM Fortran, as a shared library
    |
    +-- online coupling provider ------> CLM-SP, CICE%PRES, DOCN%DOM, RTM
                                        and the CESM mapping and flux kernels
```

Each MPI rank owns its local NumPy arrays; Python chooses which operation
runs next, and Fortran performs it.

## Features

- **One persistent model.** Constructing a `Driver` starts nothing; the first
  live operation launches one MPI session that every later call reuses, from a
  script or a Jupyter notebook.
- **An editable workflow.** The step is an ordered list of processes that can
  be enabled, disabled, reordered, run on their own, or joined by Python
  physics without rebuilding CAM.
- **Observable, writable state.** Every CAM field is a distributed NumPy
  array with rank-local zero-copy views, global statistics and in-place
  mutation; Python-defined fields join CAM's own history files.
- **Live parameters.** Namelist tunables are validated against the pinned
  source before launch, and an audited subset can be changed while the model
  runs.
- **Schemes as functions.** A physics routine can be called on a single
  column as `y = f(x, p)`, with no model session, and sampled into datasets.
- **A workflow page.** `driver.ui()` serves a browser editor for the step
  that generates the freeCAM code it describes and runs it on the model; the
  same page is published as a preview that edits without running.

## Installation

freeCAM is installed from source. The Python package alone runs the unit
tests; the model itself also needs the pinned iCESM source, a native image
built from a configured CESM case, and the case's input data on Derecho. See
[docs/installation.md](docs/installation.md) for the complete recipe.

```bash
git clone git@github.com:a85tract/freeCAM.git
cd freeCAM
git submodule update --init external/iCESM1.3.1_fzhu
uv sync --extra notebook --extra test
uv run python tools/prepare_pi_cam_source.py --check   # check out the pinned externals
```

Take the interpreter from the system or a conda environment rather than
letting `uv` download one: the native image is mapped at a fixed address, and
a CPython linked at a fixed low address can grow its heap into it. Pass the
interpreter explicitly, for example `uv sync -p /path/to/python3.13 ...`.

The site is described once, in `site.env` at the repository root; the
allocation is the only required entry. The preflight reports what the checkout
resolves to and what it still lacks:

```bash
cp site.env.example site.env     # then set FREECAM_ACCOUNT
uv run python -m freecam.site
```

## Quick start

```python
import freecam as fc

# Nothing starts here: the run directory is prepared and the MPI session
# launched by the first live operation, driver.initialize().
with fc.Driver(case="PI-atm", nsteps=2) as driver:
    driver.initialize()

    temperature = driver.cam.state.T          # a distributed field
    print(temperature.stats(rank="global"))   # min, max, mean over all ranks
    print(temperature.get(rank=0).shape)      # rank 0's own array

    result = driver.run(progress=True)        # the two steps declared above
    print(result)
    print(driver.cam.history.latest())        # the history file CAM wrote
```

The first `initialize()` requests the compute resources the case needs: 512
MPI ranks on four Derecho nodes, through PBS, charged to `FREECAM_ACCOUNT`.
Leaving the `with` block, or calling `driver.close()`, releases them.

## Documentation and examples

- [docs/installation.md](docs/installation.md): site configuration, the
  interpreter requirement, external data, reusing an existing installation,
  and building the native image.
- [docs/usage.md](docs/usage.md): the workflow, dynamic fields and Python
  processes, parameters, online and offline runs, history output, timing
  reports, and the single-column function and dataset interfaces.
- [docs/validation.md](docs/validation.md): what bit-for-bit means here, the
  gates that have been run, and where the evidence and performance records
  live.
- Notebooks under [examples/](examples/): [try_pi_cam.ipynb](examples/try_pi_cam.ipynb)
  is the maintained walkthrough; [macro_microphysics.ipynb](examples/macro_microphysics.ipynb),
  [physics_function.ipynb](examples/physics_function.ipynb) and
  [kernel_surrogate.ipynb](examples/kernel_surrogate.ipynb) cover one stage as
  a Python class, a scheme as a function, and a trained kernel in a scheme's
  place.
- [validation/performance_overhead.md](validation/performance_overhead.md):
  the measured time and memory cost of the Python control layer.
- The Workflow Builder preview at https://a85tract.github.io/freeCAM/ (once
  Pages is enabled for the repository); how to use it is in
  [docs/usage.md](docs/usage.md#the-workflow-builder).

## Development

```bash
uv sync --extra notebook --extra test
uv run pytest -q                    # the local unit suite; no MPI, no model
uv run freecam --help
git diff --check
```

The unit suite checks the control layer against fakes and runs anywhere. The
scientific gates run under PBS on Derecho with 512 ranks and compare CAM's
output with the original model byte for byte; they are submitted with
`validation/jobs/submit.sh`, which supplies the allocation from `site.env`.
Any change to the numerical runtime needs a gate before it is merged, and its
record belongs under `validation/`. [AGENTS.md](AGENTS.md) holds the
repository guidelines.

## License

freeCAM is licensed under the Apache License, Version 2.0
([LICENSE.txt](LICENSE.txt)). The repository builds against iCESM/CESM source
and other third-party material whose terms are collected in
[LICENSES/](LICENSES/) and summarised in [NOTICE](NOTICE).
