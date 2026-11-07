# freeCAM

freeCAM runs the CAM atmosphere of iCESM1.3.1 step by step from Python. A
notebook drives the model on NCAR Derecho: Python decides which process runs
next, reads and writes the model state as NumPy arrays, and can put Python code
or a trained network in place of a process or a single kernel. The original
Fortran computes every number, so with nothing replaced a run is bit-for-bit
the original model.

The supported configuration is the `ne16` PI-atm case: CAM5 physics, SE
dynamics, 512 MPI ranks, with the CESM land, sea-ice, ocean and river
components (CLM-SP, CICE%PRES, DOCN%DOM, RTM) and the coupler running live
beside CAM. Other rank counts run as explorations, and other compile-time
options (today `pcols`) are rebuilt by `fc.build` and checked against their own
original.

![freeCAM in layers: notebooks, scripts, the Workflow Builder, batch jobs and viewers drive a Python control layer on every MPI rank; generated adapters call the original iCESM Fortran, built from the pinned source and checked bit-for-bit against the original](web/site/docs/figures/architecture.svg)

## Features

- **An editable step.** A step is an ordered list of processes; any of them,
  the dynamical core included, can be switched off, moved, run alone or
  replaced by Python, and new physics written in Python can join it, without
  rebuilding CAM.
- **Process and kernel slots.** Answer a whole process (radiation, the cloud
  macro/microphysics stage) or one kernel inside it (`compute_uwshcu_inv`,
  `mmacro_pcond`) with Python, a replay or a trained network; the
  shallow-convection network runs on the CPU or the GPU.
- **Live state.** Every CAM field is a distributed NumPy array: rank-local
  zero-copy views, global statistics, in-place edits, and Python fields
  written into CAM's own history files.
- **Schemes as functions.** Call a physics routine on one column as
  `y = f(x, p)`, with no model running, and sample it into training data.
- **Checked against the original.** The default path is bit-for-bit the
  original iCESM in 512-rank online gates; the records are under
  [validation/](validation/).
- **One persistent model.** The first live call starts one MPI job; every
  later notebook cell reuses it. `driver.ui()` opens a browser editor for the
  step, and `fc.build(pcols=32)` or `fc.Driver(ntasks=N, exploratory=True)`
  change the build or the rank count.

![One model step is an ordered list of processes; the list can be edited, a whole process or one kernel inside it can be answered by a network, and with nothing replaced the run is bit-for-bit the original](web/site/docs/figures/step.svg)

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

![A Jupyter notebook submits one 512-rank MPI job on Derecho and sends it commands; every rank runs freeCAM's Python driver over its own NumPy state, and the original iCESM Fortran computes every number](web/site/docs/figures/notebook-hpc.svg)

## Examples

Read the running model from a notebook
([try_pi_cam.ipynb](examples/try_pi_cam.ipynb)):

```python
driver.cam.state.plot(rank=0, variables=('T', 'u', 'v', 'q', 'omega', 'pmid'))
```

![Rank 0's mean profiles of temperature, both winds, water vapour, vertical velocity and pressure, read live from the model](web/site/docs/figures/notebook-profiles.png)

### The Workflow Builder and its globe

A page in the browser on the same model: it edits the step and generates its
freeCAM code, runs it, and shows the state on a globe
([docs/usage.md](docs/usage.md#the-workflow-builder)). Start it from
Python; the ranks keep each step's state in memory for the globe:

```bash
uv run python -i -c '
import freecam as fc
driver = fc.Driver(case="PI-atm", nsteps=1488,
                   record_state={"store": "memory", "keep_steps": 1500})
ui = driver.ui(port=8793)
print(ui.url)
'
```

Open the URL. Without `record_state`, `ui()` still keeps the state in memory,
for the last 1000 steps.

Every process can be moved, switched off, removed or replaced. A change that
leaves the validated default is marked experimental, and Generate writes the
step as freeCAM code: a complete program, a notebook, or the setup for a
running driver. Here shallow convection moves ahead of deep convection:

![The Workflow Builder: shallow convection is moved ahead of deep convection, the check asks for Experimental, and Generate writes the code, in which shallow_convection now comes before deep_convection](web/site/docs/figures/workflow-builder.gif)

Run starts the model, and the *Globe* tab follows it. Inside one step it shows
what each process alone did to the state; here the seven processes that change
the temperature at step 24:

<p align="center"><img src="web/site/docs/figures/globe-processes.gif" width="560" alt="Temperature at 848 hPa at the start of a step, then the change made by each process in turn: vertical diffusion, gravity-wave drag, dynamics, deep convection, shallow convection, cloud macro/microphysics and radiation"></p>

Step by step, over one model day of the same run: the air over land warms in
daylight and cools at night, by 5 to 7 K over the Sahara and the Kalahari,
while over the ocean it changes by less than 1 K.

<p align="center"><img src="web/site/docs/figures/globe-steps.gif" width="520" alt="Temperature at the lowest model level through 50 steps, one model day: the land warms by day and cools by night, the ocean barely changes"></p>

A click picks a column: its profile, and the latitude-pressure section along
its meridian. A run recorded to disk (`record_state=True`) opens later with
`freecam globe <run>/state`.

![The globe viewer: temperature at 848 hPa over the globe, the column picked at 23.5N 12.8E, and the latitude-pressure section along its meridian](web/site/docs/figures/globe.png)

### A trained network in a kernel slot

A distilled network took the place of the shallow-convection kernel,
`compute_uwshcu_inv`, for half a model year at 512 ranks, with land, sea ice,
ocean and rivers running live
([docs/usage.md](docs/usage.md#replacing-one-kernel-inside-a-process)):

```python
shallow = driver.processes["shallow_convection"]
shallow.kernels["compute_uwshcu_inv"] = fc.NativeModel("uwshcu.pt")   # TorchScript, on the CPU
driver.advance(8688)                                                  # half a model year
```

Its mean precipitation keeps the original's pattern (correlation 0.95) and
global mean (3.05 and 3.06 mm/day); low cloud is 2.7% more and cloud liquid
water 5% more. On CPU the half year took 1.19 times as long as the original
Fortran.

![Half-year mean total precipitation from the original Fortran and from the run with a network in compute_uwshcu_inv, and their zonal means: the patterns agree, global means 3.05 and 3.06 mm/day](web/site/docs/figures/shallow-network-halfyear.png)

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
