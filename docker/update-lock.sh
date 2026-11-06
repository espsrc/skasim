#!/usr/bin/env bash
# Regenerate docker/conda-linux-64.lock and docker/requirements-pip.lock from
# environment.yml and pyproject.toml. Requires Docker; nothing is installed on
# the host. Commit the resulting lockfiles together with the change that
# motivated them.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
image="${MICROMAMBA_IMAGE:-mambaorg/micromamba:2.0.5}"
casa_version="${CASA_VERSION:-6.7.6.14}"

docker run --rm --platform linux/amd64 --user root \
    -e CONDA_OVERRIDE_CUDA=12.9 \
    -e CASA_VERSION="$casa_version" \
    -v "$repo:/repo:ro" \
    -v "$here:/out" \
    "$image" bash /repo/docker/lock/resolve.sh

echo "Wrote $here/conda-linux-64.lock and $here/requirements-pip.lock"
