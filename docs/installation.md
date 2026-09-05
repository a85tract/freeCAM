# Installing freeCAM

freeCAM is installed from source. The Python package is only the control
layer: running the model also needs the pinned iCESM source, a native image
built from a configured CESM case, the online coupler library, and the case's
input data. All of that lives on NCAR Derecho, where the supported PI-atm
configuration runs on 512 MPI ranks.

## The Python package

```bash
git clone git@github.com:a85tract/freeCAM.git
cd freeCAM
git submodule update --init external/iCESM1.3.1_fzhu
uv sync --extra notebook --extra test
uv run python tools/prepare_pi_cam_source.py --check
```

The submodule is only the iCESM shell: its `components/` are managed
externals, and the last command checks them out and verifies all seven pinned
revisions. Tests that read the pinned source skip until it is there. Python
3.11 or later is required (`pyproject.toml` is authoritative); the `notebook`
extra adds Jupyter and plotting, the `test` extra adds pytest.

### The interpreter has to be position-independent

Python 3.11 through 3.13 all work, but not every build of them does. The
native image is linked non-PIC at a fixed address and mapped there, over
whatever is in the way. A position-independent interpreter is loaded high and
its heap grows from there, clear of the window. One linked at a fixed low
address, which is how `uv`'s own downloaded CPython is built from 3.12 on,
starts its heap low enough to grow into the window, and the ranks whose heap
lands there die inside glibc or on a fault with no Python left to say why.
Because the heap base is randomised, only some ranks die, and a green run
proves nothing.

Take the interpreter from the system or a conda environment rather than
letting `uv` download one, and let the preflight read its load address:

```bash
uv sync -p /path/to/a/python3.13 --extra notebook --extra test
uv run python -m freecam.site        # the `interpreter` check reads the load address
```

## Site configuration

Nothing in this repository names a user or an allocation. A site describes
itself once, in `site.env` at the repository root:

```bash
cp site.env.example site.env
$EDITOR site.env          # FREECAM_ACCOUNT is the only required entry
```

Both readers use that one file, `freecam.site` from Python and
`validation/jobs/common.sh` from every PBS job, so a notebook and a job cannot
disagree about where the model lives. Anything set in the environment wins
over the file, and an explicit `Driver(..., account=...)` wins over both.

| setting | meaning | default |
| --- | --- | --- |
| `FREECAM_ACCOUNT` | PBS allocation every job is charged to | none: it must be given |
| `FREECAM_SCRATCH` | run directories and generated data | `$SCRATCH`, then `/glade/derecho/scratch/$USER` |
| `FREECAM_CASES` | directory holding the configured CESM cases | `<repository>/../CESM_cases` |
| `FREECAM_REFERENCE_CASE` | the case supplying the machine environment | `$FREECAM_CASES/<case name>` |
| `FREECAM_REFERENCE_RUN` | the oracle run supplying `atm_in` and the initial state | `$FREECAM_SCRATCH/pyCAM/PI-cam/<case name>/run` |
| `FREECAM_QUEUE` | PBS queue for interactive sessions | `develop` |
| `FREECAM_NATIVE_MANIFEST` | an existing native image to run against | this checkout's `build/pi_cam_promoted/` |
| `FREECAM_CESM_PROVIDER_LIBRARY` | the online surface components and coupler | this checkout's `build/cesm/pi_atm/production-components/` |
| `FREECAM_CESM_PROVIDER_SEED` | the completed CESM run the provider seeds from | under `$FREECAM_SCRATCH` |

[`site.env.example`](../site.env.example) documents every entry. Check what a
checkout resolves to, and what it is still missing, before launching anything:

```bash
uv run python -m freecam.site
```

## What a clone cannot bring with it

Three things are external to this repository and have to exist before the
512-rank configuration runs. `python -m freecam.site` names whichever is
absent, and what produces it.

1. **Derecho**, its module environment, and a PBS allocation.
2. **Two builds** under `build/`: the native image (`pi_cam_promoted/`, see
   below) and the online CESM provider library
   (`cesm/pi_atm/production-components/`, built by
   [`validation/jobs/pi_cam_online_coupler_build.pbs`](../validation/jobs/pi_cam_online_coupler_build.pbs)),
   which the default case runs live. Both are compiled in place and are
   pointed at, not copied.
3. **A configured CESM case and two completed runs**: the oracle's, for the
   machine environment, `atm_in`, the initial state and the input data they
   name; and one of the original coupled model, which the online provider
   seeds its surface components from.

### Reusing an existing installation

To use an installation someone else has built rather than repeat the build,
point `FREECAM_NATIVE_MANIFEST`, `FREECAM_CESM_PROVIDER_LIBRARY`,
`FREECAM_CESM_PROVIDER_SEED`, `FREECAM_REFERENCE_CASE` and
`FREECAM_REFERENCE_RUN` at its owner's paths and keep your own
`FREECAM_SCRATCH` and `FREECAM_ACCOUNT`: run directories are created under
your scratch, and your allocation is charged. `site.env.example` carries this
recipe.

## Building the native image

The `.so` freeCAM loads is not a recompilation of the pinned source. It is the
oracle's own machine code with three control surfaces replaced and the ELF
type changed from `ET_EXEC` to `ET_DYN`, so Python can `dlopen` it. Rebuilding
CAM as position-independent changes register allocation and fails the PI-atm
bitwise gate, which is why the pipeline preserves the oracle's objects rather
than producing its own.

1. `git submodule update --init external/iCESM1.3.1_fzhu`.
   [`tools/prepare_pi_cam_source.py`](../tools/prepare_pi_cam_source.py)
   refuses a revision mismatch in any of the seven pinned components.
2. Two CESM cases, both built:
   * the **oracle** case (`FREECAM_REFERENCE_CASE`): its `bld/lib/libatm.a`
     supplies the numerical objects the image links, unchanged;
   * the **python-state** case (`FREECAM_STATE_CASE`): supplies `.mod` files
     and the control shells, its `SourceMods/src.cam` written by
     [`tools/generate_pi_cam_python_state_source.py`](../tools/generate_pi_cam_python_state_source.py).
3. One PBS job does the rest, so the parts cannot drift apart:

   ```bash
   validation/jobs/submit.sh validation/jobs/pi_cam_promoted_statepool_build.pbs
   ```

   * `prepare_pi_cam_source.py` copies the pinned tree to
     `build/iCESM1.3.1_PI_cam_only` and applies the patches and support
     modules this repository owns. The tree is deleted and rebuilt every
     time: a stale one silently drops a patch added since it was written.
     The patches applied and the modules installed are recorded with their
     hashes in `.pycam-source.json` inside the prepared tree. Every patch
     adds a Python control point or a capture hook; none edits a numerical
     routine.
   * `build_pi_cam_promoted_kernels.py` regenerates the descriptor of the
     direct kernels reached from Python.
   * `build_pi_cam_devices.py` generates the adapters, compiles them non-PIC,
     links the fixed-address image, retypes it, and writes
     `native_cam_manifest.json`: every compile and link command, and the
     sha256 of what they produced.

That the pipeline reproduces the image in use is checked rather than assumed:
[`validation/pi_cam_native_image_rebuild.json`](../validation/pi_cam_native_image_rebuild.json)
records a rebuild into a scratch image root, compared command by command,
object by object, and symbol by symbol against the image this checkout runs.

Generated libraries and compiler products belong under `build/`, PBS output
under `logs/`; neither is committed.

## Submitting jobs

Every PBS job in [`validation/jobs/`](../validation/jobs/) is submitted
through one wrapper:

```bash
validation/jobs/submit.sh validation/jobs/<job>.pbs [qsub arguments...]
```

`submit.sh` supplies `-A $FREECAM_ACCOUNT` on the command line. No job carries
a `#PBS -A` directive, because `qsub` does not expand variables in directives
and a working one would have to name a project in a shared file. Every job
resolves its paths through `validation/jobs/common.sh`, and so runs from any
checkout.
