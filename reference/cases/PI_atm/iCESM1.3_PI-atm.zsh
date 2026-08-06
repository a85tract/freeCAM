#!/bin/zsh
set -euo pipefail

# Reproducible 50-coupling-step oracle used to extract the PI-CAM boundary.
# The reference itself remains fully coupled: its only purpose is to provide
# scientifically correct CAM imports/exports for the later CAM-only replay.
# This is intentionally self-contained and does not source personal helpers.

info() { print -r -- "[PI-atm] $*"; }
die() { print -ru2 -- "[PI-atm] ERROR: $*"; exit 1; }

repo_root=${0:A:h:h:h:h}
# Do not inherit an unrelated interactive-shell PROJECT value implicitly.
account=${PYCESM_PROJECT:-UCUB0188}
priority=${PRIORITY:-regular}
queue=${QUEUE:-develop}
walltime=${WALLTIME:-02:00:00}
source_root=${ICESM_SOURCE_ROOT:-/glade/work/ruitong/iCESM1.3.1_PI_cam_only}
case_root=${CESM_CASE_ROOT:-/glade/work/ruitong/CESM_cases}
output_root=${CESM_OUTPUT_ROOT:-/glade/derecho/scratch/ruitong/pyCAM/PI-cam}
mapping_root=${PI_ATM_MAPPING_ROOT:-/glade/work/fengzhu/Projects/pyCESM/test_cases/PI/mappings}
casetag=${CASE_TAG:-PI-cam-oracle.50step}
resolution=ne16_g16
compset=1850_CAM50_CLM40%SP_CICE%PRES_DOCN%DOM_RTM_SGLC_SWAV
casename=f.e13.F1850C5.${resolution}.icesm131_ihesp.${casetag}
case_dir=${case_root}/${casename}
run_dir=${output_root}/${casename}/run
archive_dir=${output_root}/${casename}/archive

[[ -x ${source_root}/cime/scripts/create_newcase ]] || \
  die "missing create_newcase below ${source_root}"
[[ -d ${mapping_root} ]] || die "missing mapping directory ${mapping_root}"

if [[ -e ${case_dir} ]]; then
  if [[ ${RECREATE:-0} == 1 ]]; then
    info "RECREATE=1: removing ${case_dir}"
    rm -rf -- ${case_dir}
    rm -rf -- ${output_root}/${casename}
  else
    die "case already exists: ${case_dir}; set RECREATE=1 to replace it"
  fi
fi

mkdir -p -- ${case_root} ${output_root}
export PROJECT=${account}
export PRIORITY=${priority}

info "source: ${source_root}"
info "case: ${case_dir}"
info "run: ${run_dir}"
info "project/queue: ${account}/${queue}"

${source_root}/cime/scripts/create_newcase \
  --case ${case_dir} \
  --res ${resolution} \
  --compset ${compset} \
  --mach derecho \
  --run-unsupported

cd ${case_dir}

./xmlchange RUN_TYPE=startup,GET_REFCASE=FALSE,CLM_FORCE_COLDSTART=on
./xmlchange CLM_BLDNML_OPTS=-ignore_warnings --append
./xmlchange CIME_OUTPUT_ROOT=${output_root}
./xmlchange JOB_QUEUE=${queue} --force
./xmlchange JOB_WALLCLOCK_TIME=${walltime}
./xmlchange PROJECT=${account}

# Preserve the original PI-atm PE layout. Component PE sets overlap inside
# one 512-rank global MPI world.
./xmlchange NTASKS_CPL=512,NTHRDS_CPL=1,ROOTPE_CPL=0
./xmlchange NTASKS_ATM=512,NTHRDS_ATM=1,ROOTPE_ATM=0
./xmlchange NTASKS_LND=256,NTHRDS_LND=1,ROOTPE_LND=0
./xmlchange NTASKS_ICE=128,NTHRDS_ICE=1,ROOTPE_ICE=256
./xmlchange NTASKS_OCN=32,NTHRDS_OCN=1,ROOTPE_OCN=384
./xmlchange NTASKS_ROF=128,NTHRDS_ROF=1,ROOTPE_ROF=0

./xmlchange ATM_DOMAIN_PATH=${mapping_root},ATM_DOMAIN_FILE=domain.lnd.ne16np4_gx1v6.231103.nc
./xmlchange LND_DOMAIN_PATH=${mapping_root},LND_DOMAIN_FILE=domain.lnd.ne16np4_gx1v6.231103.nc
./xmlchange OCN_DOMAIN_PATH=${mapping_root},OCN_DOMAIN_FILE=domain.ocn.gx1v6.231103.nc
./xmlchange ICE_DOMAIN_PATH=${mapping_root},ICE_DOMAIN_FILE=domain.ocn.gx1v6.231103.nc
./xmlchange ATM2OCN_FMAPNAME=${mapping_root}/map_ne16np4_TO_gx1v6_aave.231103.nc
./xmlchange ATM2OCN_SMAPNAME=${mapping_root}/map_ne16np4_TO_gx1v6_blin.231103.nc
./xmlchange ATM2OCN_VMAPNAME=${mapping_root}/map_ne16np4_TO_gx1v6_patc.231103.nc
./xmlchange OCN2ATM_FMAPNAME=${mapping_root}/map_gx1v6_TO_ne16np4_aave.231103.nc
./xmlchange OCN2ATM_SMAPNAME=${mapping_root}/map_gx1v6_TO_ne16np4_aave.231103.nc
./xmlchange LND2ROF_FMAPNAME=${mapping_root}/map_ne16np4_TO_r05_nomask_aave.231103.nc
./xmlchange ROF2LND_FMAPNAME=${mapping_root}/map_r05_nomask_TO_ne16np4_aave.231103.nc
./xmlchange ROF2OCN_FMAPNAME=${mapping_root}/map_r05_nomask_TO_gx1v6_aave.231103.nc
./xmlchange ROF2OCN_RMAPNAME=${mapping_root}/map_r05_nomask_to_gx1v6_nnsm_e1000r300.231103.nc

./case.setup --reset

cat >> user_nl_cam <<'EOF'
cldfrc_rhminl  = 0.870D0
micro_mg_dcs   = 400.D-6
dust_emis_fact = 0.95D0
wtrc_limiter_phis_crit = 100000.0
nhtfrq = -50
mfilt = 1
EOF

cat >> user_nl_clm <<'EOF'
hist_nhtfrq = -50
hist_mfilt = 1
EOF

cat >> user_nl_rtm <<'EOF'
wiso_runoff = .true.
EOF

cat >> user_nl_cice <<'EOF'
ice_ic  = 'default'
tr_pond = .false.
tr_iso  = .false.
EOF

./preview_namelists
./case.build

./xmlchange STOP_OPTION=nsteps,STOP_N=50
./xmlchange REST_OPTION=nsteps,REST_N=50
./xmlchange RESUBMIT=0,DOUT_S=FALSE
./xmlchange DOUT_S_ROOT=${archive_dir}

info "configured 50-step reference case"
info "run directory: ${run_dir}"

if [[ ${CREATE_ONLY:-0} == 1 ]]; then
  info "CREATE_ONLY=1: case built but not submitted"
  exit 0
fi

if [[ ${queue} == develop ]]; then
  # cpudev is capped at 256 allocated CPUs, while the scientific PE layout
  # requires 512 MPI ranks. Keep the layout and place two ranks per CPU.
  qsub_variables="ARGS_FOR_SCRIPT=--resubmit"
  if [[ -n ${PYCAM_BOUNDARY_CAPTURE:-} ]]; then
    mkdir -p -- ${PYCAM_BOUNDARY_CAPTURE:h}
    qsub_variables="${qsub_variables},PYCAM_BOUNDARY_CAPTURE=${PYCAM_BOUNDARY_CAPTURE}"
  fi
  job_id=$(qsub \
    -q develop \
    -l select=4:ncpus=64:mpiprocs=128:ompthreads=1:mem=64GB \
    -l walltime=${walltime} \
    -A ${account} \
    -v ${qsub_variables} \
    .case.run)
  info "submitted ${casename} to cpudev as ${job_id}"
else
  ./case.submit
  info "submitted ${casename}"
fi
