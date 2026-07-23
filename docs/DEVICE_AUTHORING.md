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
body. Generated files belong under `build/devices/` and are reproducible from
the descriptor.

## Descriptor

`devices/kessler/device.yaml` is the reference descriptor:

```yaml
schema_version: 1
name: kessler
fortran_module: kessler

sources:
  - external/CAM-SIMA/src/physics/ncar_ccpp/schemes/kessler/kessler.F90
metadata:
  - external/CAM-SIMA/src/physics/ncar_ccpp/schemes/kessler/kessler.meta

source_modules: [kessler]
providers:
  ccpp_kinds: native/devices/support/ccpp_kinds.F90

state_policy: reinitialize_each_run
initialize_entrypoint: initialize

dimension_bindings:
  horizontal_loop_extent: nphys_local
  vertical_layer_dimension: pver

bindings:
  vertical_index_at_surface_adjacent_layer:
    source: dimension
    name: pver
  vertical_index_at_top_adjacent_layer:
    source: literal
    value: 1

entrypoints:
  initialize:
    table: kessler_init
  run:
    table: kessler_run

processes:
  kessler: run
```

Most array/scalar bindings do not appear in this file. The generator carries
the metadata `standard_name` into `device.json`, and StatePool resolves it at
runtime. Descriptor bindings are required only for host policy that metadata
cannot determine, such as vertical orientation, a literal, or an intentional
field-name override.

## Build pipeline

Run:

```bash
uv run pycam-sima build-device devices/kessler/device.yaml
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
- `logical(c_bool)`;
- scalar input values;
- scalar output/inout references;
- explicit-shape arrays of supported intrinsic types;
- CCPP error code/message and scheme-name outputs;
- injected ABI-only dimensions for assumed-shape routines that do not receive
  the extent as a scheme argument.

The generator rejects optional arguments, derived types, unsupported kinds,
ambiguous dimension expressions, undeclared modules, and host dependencies.
That rejection is intentional. A new ABI rule or a small portable dependency
provider must be designed before such a scheme becomes a device.

MPI, ESMF, PIO, NetCDF, CAM history, and full CAM control are outside a
numerical device. MPI communication remains in the Python host through
mpi4py.

## Add a new scheme

1. Keep the upstream `.F90` and `.meta` files unchanged.
2. Add `devices/<name>/device.yaml`.
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
