# Site resolution shared by every PBS job in this directory.
#
# Sourced immediately after `set -euo pipefail`:
#
#   source "${FREECAM_REPO:-${PBS_O_WORKDIR:-$PWD}}/validation/jobs/common.sh"
#
# Nothing here names a user or a project.  Each value comes from the
# environment, then from site.env at the repository root -- the same file
# freecam.site reads, so a notebook and a job cannot disagree about where the
# model lives -- then from a default derived from where this checkout sits.
# Jobs are submitted from the repository root, which is what PBS_O_WORKDIR
# holds.
#
# Defined, and exported:
#
#   FREECAM_REPO             this checkout
#   FREECAM_PYTHON           its virtual environment's interpreter
#   FREECAM_SCRATCH          root for run directories and generated data
#   FREECAM_CASES            directory holding the configured CESM cases
#   FREECAM_REFERENCE_CASE   the oracle case: machine environment, namelist
#   FREECAM_REFERENCE_RUN    the oracle run: atm_in and the initial state
#   FREECAM_STATE_CASE       the case the native image was built from
#   FREECAM_PYCESM_CASE      the original-Fortran reference case
#   FREECAM_ICESM_SOURCE     the pinned iCESM source tree the coupler
#                            build reads
#   FREECAM_LD_PRELOAD       the Intel maths library the image is linked to
#
# The allocation is deliberately not a PBS directive: qsub does not expand
# variables in `#PBS` lines and rejects `#PBS -A $ANYTHING`.  It travels on
# the command line instead -- validation/jobs/submit.sh does that.

FREECAM_REPO=${FREECAM_REPO:-${PBS_O_WORKDIR:-$PWD}}
if [ ! -f "${FREECAM_REPO}/pyproject.toml" ]; then
  echo "not a freeCAM checkout: ${FREECAM_REPO}" >&2
  echo "submit from the repository root, or set FREECAM_REPO" >&2
  exit 1
fi
FREECAM_REPO=$(cd "${FREECAM_REPO}" && pwd)

# The site's own answers, where it has given any.
if [ -f "${FREECAM_REPO}/site.env" ]; then
  # shellcheck disable=SC1091
  . "${FREECAM_REPO}/site.env"
fi

# Case names are a property of the admitted configuration, not of a user.
FREECAM_REFERENCE_CASE_NAME=${FREECAM_REFERENCE_CASE_NAME:-f.e13.F1850C5.ne16_g16.icesm131_ihesp.PI-cam-oracle.50step}
FREECAM_STATE_CASE_NAME=${FREECAM_STATE_CASE_NAME:-f.e13.F1850C5.ne16_g16.icesm131_ihesp.PI-cam-python-state}
FREECAM_PYCESM_CASE_NAME=${FREECAM_PYCESM_CASE_NAME:-f.e13.F1850C5.ne16_g16.icesm131_ihesp.PI-atm.pycesm-ref.50step}

FREECAM_PYTHON=${FREECAM_PYTHON:-${FREECAM_REPO}/.venv/bin/python}
FREECAM_SCRATCH=${FREECAM_SCRATCH:-${SCRATCH:-/glade/derecho/scratch/${USER}}}
FREECAM_CASES=${FREECAM_CASES:-$(dirname "${FREECAM_REPO}")/CESM_cases}
FREECAM_REFERENCE_CASE=${FREECAM_REFERENCE_CASE:-${FREECAM_CASES}/${FREECAM_REFERENCE_CASE_NAME}}
FREECAM_REFERENCE_RUN=${FREECAM_REFERENCE_RUN:-${FREECAM_SCRATCH}/pyCAM/PI-cam/${FREECAM_REFERENCE_CASE_NAME}/run}
FREECAM_STATE_CASE=${FREECAM_STATE_CASE:-${FREECAM_CASES}/${FREECAM_STATE_CASE_NAME}}
FREECAM_PYCESM_CASE=${FREECAM_PYCESM_CASE:-${FREECAM_CASES}/${FREECAM_PYCESM_CASE_NAME}}
FREECAM_ICESM_SOURCE=${FREECAM_ICESM_SOURCE:-$(dirname "${FREECAM_REPO}")/iCESM1.3.1_PI_atm_pycesm}
FREECAM_LD_PRELOAD=${FREECAM_LD_PRELOAD:-/glade/u/apps/common/23.04/spack/opt/spack/intel-oneapi-compilers/2023.0.0/compiler/2023.0.0/linux/compiler/lib/intel64_lin/libimf.so}

export FREECAM_REPO FREECAM_PYTHON FREECAM_SCRATCH FREECAM_CASES \
  FREECAM_REFERENCE_CASE FREECAM_REFERENCE_RUN FREECAM_STATE_CASE \
  FREECAM_PYCESM_CASE FREECAM_ICESM_SOURCE FREECAM_LD_PRELOAD
