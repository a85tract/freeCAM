# Fortran device authoring

PyCAM-SIMA treats a numerical scheme as a source-preserving device:

```text
unmodified Fortran + CCPP metadata + device.yaml
                         │
                         ▼
        generated bind(C) adapter + device.json + isolated .so
                         │
                         ▼
          DeviceRegistry ↔ Python-owned StatePool
```

The device is the original scheme, not a translated copy of its numerical
body. Source-derived descriptors belong under `devices/generated/`; adapters,
manifests, and libraries belong under `build/devices/` and are reproducible
from those descriptors.

The same descriptor can be an external runtime plugin source.
`model.install_physics(...)` uses the pinned CCPP parser, generates the adapter
in `PYCAM_SIMA_PLUGIN_CACHE`, and loads it collectively. A production plugin
may instead ship only its generated `device.json` and matching `.so`.

An installed Python package can advertise a device through:

```toml
[project.entry-points."pycam_sima.physics"]
my_microphysics = "my_package:device_path"
```

The entry point must resolve to a descriptor, manifest, or directory
containing one. Runtime loading never edits `devices/generated/`.

## Generated descriptor and policy overrides

Every active suite scheme has exactly one descriptor under
`devices/generated/<scheme>/device.yaml`. Do not edit these generated files
directly. Source-independent host policy that CCPP metadata cannot express is
kept in `devices/overrides.yaml`. Kessler is the reference override:

```yaml
schema_version: 1
schemes:
  kessler:
    state_policy: reinitialize_each_run
    initialize_entrypoint: initialize
    bindings:
      vertical_index_at_surface_adjacent_layer:
        source: dimension
        name: pver
      vertical_index_at_top_adjacent_layer:
        source: literal
        value: 1
```

Most array/scalar bindings do not appear in this file. The generator carries
the metadata `standard_name` into `device.json`, and StatePool resolves it at
runtime. Overrides are restricted to host policy that metadata cannot
determine, such as lifecycle state policy, vertical orientation, or a
literal. Sources, entrypoints, providers, and ABI arguments remain
source-derived and cannot be replaced by the override.

## Build pipeline

Run:

```bash
uv run pycam-sima generate-devices --clean
uv run pycam-sima build-device devices/generated/kessler/device.yaml
```

The builder:

1. loads and validates descriptor schema version 1;
2. invokes CAM-SIMA's pinned CCPP parser;
3. checks the `.meta` declarations against the associated Fortran source;
4. parses every `use` statement and resolves it to an intrinsic module,
   source module, or declared portable provider;
5. creates explicit C-interoperable dimensions for assumed-shape arguments;
6. hides CCPP `errmsg`, `errflg`, and `scheme_name` behind a uniform status
   return and C error buffer;
7. emits one versioned C symbol for every selected lifecycle table;
8. compiles the provider sources, original scheme sources, and generated
   adapter in a clean environment;
9. limits exported symbols with a linker version script;
10. rejects forbidden ELF dependencies, undefined framework symbols, and
    RPATH/RUNPATH.

The output is:

```text
build/devices/kessler/
  device.json
  libpycam_device_kessler.so
  generated/
    kessler_adapter.F90
    kessler.map
  mod/
```

`device.json` contains a hash of the descriptor, source, metadata, and provider
files. That hash appears in `DeviceRegistry.describe()`.

## StatePool contract

A persistent field may expose a CCPP name:

```python
FieldContract(
    standard_name="potential_temperature",
    ccpp_standard_name="air_potential_temperature",
    dtype="float64",
    dimensions=("nphys_local", "pver"),
    units="K",
    intent="inout",
    category="physics",
)
```

A zero-copy constituent slice can expose its own CCPP name through
`AliasRule.ccpp_standard_name`. Runtime connection verifies:

- the CCPP standard name exists exactly once;
- NumPy dtype equals the manifest dtype;
- rank and shape equal the metadata dimensions after dimension binding;
- units match;
- output and inout buffers are writable;
- every array is aligned and Fortran contiguous;
- all StatePool pointers are unchanged after the call.

No array copy is performed by `DeviceRegistry`.

## State policies

`stateless`

: The process entrypoint is called directly.

`reinitialize_each_run`

: The initialize entrypoint is called with current StatePool values before
  every process call. This is the preferred compatibility policy for a small
  original scheme module that stores configuration variables.

`initialize_once`

: Initialization is performed once in each Python worker. The generated
  manifest reports `persistent_native_state: true`. A model using this policy
  must define how that state is reconstructed for checkpoint/restart before it
  can claim complete restartability.

Generated adapters never retain NumPy pointers.

## ABI v1 support and fail-closed behavior

Device ABI v1 supports:

- `real(kind_phys)` mapped to `float64`;
- default integer/`c_int` mapped to `int32`;
- `logical(c_bool)` and default Fortran logical through a generated bridge;
- fixed-width character scalars and rank-one arrays;
- shaped primitive fields marked allocatable by CCPP metadata, with allocation
  remaining in Python/NumPy;
- non-allocatable derived-type scalars/arrays as opaque process-state handles;
- scalar input values;
- scalar output/inout references;
- explicit-shape arrays of supported intrinsic types;
- CCPP error code/message and scheme-name outputs;
- injected ABI-only dimensions for assumed-shape routines that do not receive
  the extent as a scheme argument;
- Python-owned legacy physical constants injected through portable setter
  services.

The generator rejects optional arguments, allocatable derived objects or
fields without a concrete shape, unsupported kinds, ambiguous dimensions,
undeclared modules, and unimplemented host dependencies. That rejection is
intentional. A new ABI rule or a portable host provider must be designed
before such a scheme becomes a numerical device.

For an opaque derived type, the generated library exports create/destroy
symbols in addition to the scheme entrypoints. StatePool owns the lifetime
record, verifies the Fortran type and shape on every use, and never interprets
the object layout. Opaque state cannot be checkpointed until that type has an
explicit serializer.

MPI, ESMF, PIO, NetCDF, CAM history, and full CAM control are outside a
numerical device. MPI communication remains in the Python host through
mpi4py.

Portable providers must state their scientific scope. For example,
`native/devices/support/ref_pres.F90` implements only the low-top
`ntop_eddy = 1` behavior explicitly documented by the source interstitial; it
must not be reused as a WACCM-X pressure-coordinate implementation.

## Add a new scheme

1. Keep the upstream `.F90` and `.meta` files unchanged.
2. Regenerate `devices/generated/<name>/device.yaml` from the suite catalog;
   put an intentional policy exception in `devices/overrides.yaml` rather
   than creating a second descriptor directory.
3. Add portable support modules under `native/devices/support/` only when they
   do not import the CAM runtime.
4. Add CCPP standard-name providers to `FieldContract` or zero-copy
   `AliasRule` entries.
5. Build the descriptor independently.
6. Add a registry/contract/error-path test.
7. Add a source-level comparison if an old implementation exists.
8. Register the process in the scientific scheme plan.
9. Run the fixed 24-rank, 50-step BFB gate when the process participates in
   the validated model.

Do not place a second implementation of the scheme's formulas under
`native/kernels/`.
