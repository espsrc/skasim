#!/usr/bin/env bash
# Activate the conda environment (runs its activation scripts: GDAL, PROJ, ...)
# then dispatch:
#   no arguments or first argument starting with "-"  ->  skasim <args>
#   anything else (bash, python, wsclean, ...)        ->  run it as given
set -euo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate base

if [ "$#" -eq 0 ] || [ "${1#-}" != "$1" ]; then
    set -- skasim "$@"
fi
exec "$@"
