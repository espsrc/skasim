#!/usr/bin/env bash
# Smoke-test a built skasim image before it is published.
#
#   docker/smoke-test.sh IMAGE SKASIM_SOURCE_DIR
#
# SKASIM_SOURCE_DIR is the checkout of the git ref the image was built from;
# its tests/ are mounted into the container (nothing is baked into the image).
# Everything runs as the calling uid, as users are told to run the image.
# Extra pytest options (e.g. --deselect for known defects of an old tag) are
# passed through PYTEST_ADDOPTS.
set -euo pipefail

image="${1:?usage: smoke-test.sh IMAGE SKASIM_SOURCE_DIR}"
src="$(cd "${2:?usage: smoke-test.sh IMAGE SKASIM_SOURCE_DIR}" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

run() {
    docker run --rm --user "$(id -u):$(id -g)" -v "$work:/work" "$@"
}

echo "== CLI and external tools"
run "$image" --help > /dev/null
run "$image" wsclean --version | head -n 3

echo "== Unit tests of the source ref"
run -v "$src:/src:ro" -w /src -e PYTHONDONTWRITEBYTECODE=1 -e PYTEST_ADDOPTS "$image" bash -c '
    python -m pip install -q --user --no-warn-script-location --retries 10 --timeout 60 pytest
    python -m pytest -q -p no:cacheprovider tests
'

echo "== CASA in-process (importfits, imhead)"
run -i "$image" python - <<'PY'
import numpy as np
from astropy.io import fits
from casatasks import imhead, importfits

hdr = fits.Header()
for key, value in {
    "CTYPE1": "RA---SIN", "CRVAL1": 150.0, "CDELT1": -1e-4, "CRPIX1": 16, "CUNIT1": "deg",
    "CTYPE2": "DEC--SIN", "CRVAL2": 2.0, "CDELT2": 1e-4, "CRPIX2": 16, "CUNIT2": "deg",
    "BUNIT": "Jy/pixel",
}.items():
    hdr[key] = value
fits.writeto("/work/smoke.fits", np.zeros((32, 32), dtype="f4"), hdr)
importfits(fitsimage="/work/smoke.fits", imagename="/work/smoke.image", overwrite=True)
assert imhead("/work/smoke.image", mode="get", hdkey="crval1")["value"] != 0
print("CASA ok")
PY

echo "== Tiny CPU simulation imaged with WSClean"
run "$image" \
    --output-dir smoke_wsclean \
    --telescope MeerKAT \
    --observation-time 60 \
    --frequency-mhz 1300 \
    --bandwidth-mhz 20 \
    --n-channels 2 \
    --pixels 256 \
    --flux-density 1.0 \
    --imager wsclean \
    --clean-iterations 50
find "$work/smoke_wsclean" -maxdepth 3 \( -name '*.fits' -o -iname '*.ms' -o -name '*.html' \) | sort
find "$work/smoke_wsclean" -type d -iname '*.ms' | grep -q . \
    || { echo "no measurement set produced" >&2; exit 1; }
find "$work/smoke_wsclean" -name '*.fits' | grep -q . \
    || { echo "no FITS image produced" >&2; exit 1; }

echo "Smoke test passed for $image"
