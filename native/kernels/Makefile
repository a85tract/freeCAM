# Stateless numerical kernels used by the Python model.
SHELL := /bin/bash
PROJECT_ROOT := $(abspath ../..)
FC := /opt/cray/pe/gcc/12.2.0/bin/gfortran
FFLAGS ?= -O2 -march=znver3 -fPIC -ffp-contract=off -fno-fast-math -ffree-line-length-none -cpp
LDFLAGS ?= -Wl,--as-needed -Wl,--no-undefined
TARGET ?= $(PROJECT_ROOT)/build/libpycam_sima_kernels.so
CAM_SE := $(PROJECT_ROOT)/external/CAM-SIMA/src/dynamics/se/dycore
FVM_SOURCE := .build/fvm_consistent_se_cslam.F90
KERNEL_CONFIG := .build/kernel_config.F90
ABI_MAP := .build/abi-v2.map
ABI_SYMBOLS := .build/abi-v2.symbols
ABI_MANIFEST := abi-v2.json
PYTHON ?= python3

.PHONY: all clean check-symbols
all: $(TARGET) check-symbols

$(FVM_SOURCE): prepare_fvm_source.py $(CAM_SE)/fvm_consistent_se_cslam.F90
	$(PYTHON) prepare_fvm_source.py $(CAM_SE)/fvm_consistent_se_cslam.F90 $@

$(KERNEL_CONFIG): prepare_kernel_config.py $(PROJECT_ROOT)/configs/fkessler_model.yaml
	$(PYTHON) prepare_kernel_config.py $(PROJECT_ROOT)/configs/fkessler_model.yaml $@

$(ABI_MAP) $(ABI_SYMBOLS): $(ABI_MANIFEST) prepare_abi_exports.py
	@mkdir -p .build
	$(PYTHON) prepare_abi_exports.py $(ABI_MANIFEST) $(ABI_MAP) $(ABI_SYMBOLS)

$(TARGET): kessler_kernel.F90 physics_diagnostics_kernel.F90 se_startup_kernel.F90 remap_stubs.F90 remap_support.F90 fvm_kernel_stubs.F90 fvm_transport_kernel.F90 fvm_displacement_kernel.F90 $(CAM_SE)/fv_mapz.F90 $(CAM_SE)/fvm_reconstruction_mod.F90 $(CAM_SE)/fvm_analytic_mod.F90 $(FVM_SOURCE) $(KERNEL_CONFIG) $(ABI_MAP)
	@mkdir -p $(dir $@)
	env -i PATH=/usr/bin:/bin HOME=$$HOME $(FC) $(FFLAGS) -shared $(LDFLAGS) -Wl,--version-script=$(ABI_MAP) -o $@ $(KERNEL_CONFIG) kessler_kernel.F90 physics_diagnostics_kernel.F90 se_startup_kernel.F90 remap_stubs.F90 $(CAM_SE)/fv_mapz.F90 remap_support.F90 fvm_kernel_stubs.F90 $(CAM_SE)/fvm_reconstruction_mod.F90 $(CAM_SE)/fvm_analytic_mod.F90 $(FVM_SOURCE) fvm_transport_kernel.F90 fvm_displacement_kernel.F90

check-symbols: $(TARGET)
	@nm -D --defined-only $(TARGET) | awk '{print $$3}' | sed 's/@@PYCAM_SIMA_[0-9.]*$$//' | grep '^pycam_sima_' | sort > .defined-symbols
	@cmp -s $(ABI_SYMBOLS) .defined-symbols
	@! grep -Eq 'cam_init|cam_run|time_manager|pio_|esmf_' .defined-symbols
	@! nm -u $(TARGET) | grep -Eiq 'mpi_|pio_|esmf_|cam_init|cam_run|time_manager'
	@! readelf -d $(TARGET) | grep -Eiq 'NEEDED.*(mpi|pmi|pals|esmf|pio|netcdf|hdf5|sci)|RPATH|RUNPATH'

clean:
	rm -rf $(TARGET) *.mod .defined-symbols .build
