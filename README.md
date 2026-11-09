# freeCAM

The documentation, with the overview, the architecture, and the manual, is at
**https://a85tract.github.io/freeCAM/**.

freeCAM runs the CAM atmosphere component of CESM step by step from Python. A
notebook drives the model on the NSF NCAR Derecho Supercomputer: Python decides
which process runs next, reads and writes the model state as NumPy arrays, and
can put Python code or a trained network in place of a process or a single
kernel. The original Fortran computes every number, so with nothing replaced a
run is bit-for-bit the original model.

The supported configuration is the `ne16` PI-atm case: CAM5 physics, SE
dynamics, 512 MPI ranks, with the CESM land, sea-ice, ocean and river
components (CLM-SP, CICE%PRES, DOCN%DOM, RTM) and the coupler running live
beside CAM. Other rank counts run as explorations, and other compile-time
options (today `pcols`) are rebuilt by `fc.build` and checked against their own
original.

## Installation

freeCAM is installed from source. The Python package alone runs the unit
tests; the model also needs the pinned iCESM source, a native image built from
a configured CESM case, and the case's input data on Derecho
([docs/installation.md](docs/installation.md) has the full recipe).

```bash
git clone git@github.com:a85tract/freeCAM.git
cd freeCAM
git submodule update --init external/iCESM1.3.1_fzhu
uv sync --extra notebook --extra test
uv run python tools/prepare_pi_cam_source.py --check   # check out the pinned externals
```

Take the interpreter from the system or a conda environment rather than
letting `uv` download one (`uv sync -p /path/to/python3.13 ...`): the native
image is mapped at a fixed address, and a CPython linked at a fixed low address
can grow its heap into it.

The site is described once, in `site.env`; the allocation is the only required
entry, and the preflight reports what is still missing:

```bash
cp site.env.example site.env     # then set FREECAM_ACCOUNT
uv run python -m freecam.site
```

## Quick start

```python
import freecam as fc

# Nothing starts here: the MPI job is launched by driver.initialize().
with fc.Driver(case="PI-atm", nsteps=2) as driver:
    driver.initialize()

    temperature = driver.cam.state.T          # a distributed field
    print(temperature.stats(rank="global"))   # min, max, mean over all ranks
    print(temperature.get(rank=0).shape)      # rank 0's own array

    result = driver.run(progress=True)        # the two steps declared above
    print(driver.cam.history.latest())        # the history file CAM wrote
```

`initialize()` requests 512 MPI ranks on four Derecho nodes through PBS,
charged to `FREECAM_ACCOUNT` (`ntasks=N` takes N/128 nodes, rounded up).
Leaving the `with` block, or `driver.close()`, releases them.

From a notebook the model is one MPI job that stays up between cells; each
rank keeps its part of the globe, and the notebook gets back only what it
asks for:

![A Jupyter notebook starts one 512-rank MPI job on the NSF NCAR Derecho Supercomputer with initialize() and then sends it commands; on every rank, freeCAM in Python decides what runs and CAM in the original Fortran computes it](web/site/docs/figures/notebook-hpc.svg)

## Documentation

- [Installation](docs/installation.md): the site file, the native image, the
  input data, rebuilding the CESM cases.
- [Usage](docs/usage.md): the model, state, the workflow, replacing a process
  or a kernel, parameters, the Workflow Builder.
- [Examples](examples/): start with `try_pi_cam.ipynb`; `replace_process`,
  `replace_kernel` and `kernel_surrogate` put Python or a network in CAM's
  place.
- [Validation](docs/validation.md): what bit-for-bit means here and the gates
  behind it; [performance_overhead.md](validation/performance_overhead.md):
  what the Python layer costs.
- [How the slots are built](docs/physics_kernel_decoupling.md): kernel and
  process slots, and what answering each one costs.
- The site, https://a85tract.github.io/freeCAM/: these figures and the pages
  on a trained network in the shallow-convection kernel's slot, on CPU and
  GPU.

## Development

```bash
uv run pytest -q                    # the unit suite; no MPI, no model
git diff --check
```

The unit suite checks the control layer against fakes and runs anywhere. The
scientific gates run under PBS on Derecho with 512 ranks and compare CAM's
output with the original byte for byte (`validation/jobs/submit.sh`); a change
to the numerical runtime needs one before it is merged.
[AGENTS.md](AGENTS.md) holds the repository guidelines.

## License

freeCAM is licensed under the Apache License, Version 2.0
([LICENSE.txt](LICENSE.txt)). It builds against iCESM/CESM source and other
third-party material whose terms are collected in [LICENSES/](LICENSES/) and
summarised in [NOTICE](NOTICE).
