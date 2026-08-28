#!/bin/bash
# Submit a freeCAM job with the site's allocation.
#
#   validation/jobs/submit.sh validation/jobs/<name>.pbs [qsub arguments...]
#
# The account is not a `#PBS -A` directive because qsub does not expand
# variables in directives, and a literal one would name a project in a file
# everybody shares.  It is read here from FREECAM_ACCOUNT in the environment
# or in site.env, and passed on the command line, where it also overrides
# whatever a job's own directives say.

set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -f "${repo}/site.env" ]; then
  # shellcheck disable=SC1091
  . "${repo}/site.env"
fi
account=${FREECAM_ACCOUNT:-}

if [ -z "${account}" ]; then
  cat >&2 <<'MESSAGE'
no allocation: set FREECAM_ACCOUNT in site.env at the repository root.

  cp site.env.example site.env    # then fill in FREECAM_ACCOUNT

`qsub` lists the projects you may charge if you submit without one.
MESSAGE
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo "usage: $(basename "$0") <job.pbs> [qsub arguments...]" >&2
  exit 2
fi

job=$1
shift
cd "${repo}"
mkdir -p logs
exec qsub -A "${account}" "$@" "${job}"
