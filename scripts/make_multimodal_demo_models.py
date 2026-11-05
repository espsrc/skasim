"""Write the small FITS models used by the multimodal (catalogue + 2 images) demo.

Two continuum I+alpha image models, each holding one compact source, so the
cluster check (``scripts/check_multimodal_run.py``) can compare dirty-image
peaks with the beam-attenuated fluxes skasim reports:

* ``mm_target_a``: SIN grid tangent at the phase centre (placed directly with
  ``wsclean -shift``), 50 mJy source 40" east / 20" north of the phase centre.
* ``mm_target_b``: TAN grid tangent at its own centre, 6' east / 4' north of
  the phase centre, so skasim must regrid it onto the WSClean predict grid;
  80 mJy source at its centre, alpha = -0.7.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

PHASE_CENTRE = SkyCoord("15h31m25.36s", "35d33m40.2s")
PIXELS = 64
CELL_ARCSEC = 2.0


def _write(
    path: Path, data: np.ndarray, centre: SkyCoord, bunit: str, projection: str = "SIN"
) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    wcs.wcs.crval = [centre.ra.deg, centre.dec.deg]
    wcs.wcs.crpix = [PIXELS / 2 + 1, PIXELS / 2 + 1]
    wcs.wcs.cdelt = [-CELL_ARCSEC / 3600.0, CELL_ARCSEC / 3600.0]
    header = wcs.to_header()
    header["BUNIT"] = bunit
    fits.writeto(path, data.astype(np.float32), header, overwrite=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="configs/fits_model/generated",
        help="Directory for the generated FITS files",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Target A: tangent at the phase centre; source 40" E, 20" N.
    a = np.zeros((PIXELS, PIXELS))
    # NumPy [y, x]; RA increases towards lower x (CDELT1 < 0): 20 px east, 10 px north
    a[PIXELS // 2 + 10, PIXELS // 2 - 20] = 0.05
    _write(out / "mm_target_a_i.fits", a, PHASE_CENTRE, "Jy/pixel")
    _write(out / "mm_target_a_alpha.fits", np.zeros_like(a), PHASE_CENTRE, "")

    # Target B: TAN grid tangent 6' E, 4' N; source at the centre pixel.
    centre_b = PHASE_CENTRE.spherical_offsets_by(6 * u.arcmin, 4 * u.arcmin)
    b = np.zeros((PIXELS, PIXELS))
    b[PIXELS // 2, PIXELS // 2] = 0.08
    _write(out / "mm_target_b_i.fits", b, centre_b, "Jy/pixel", "TAN")
    _write(out / "mm_target_b_alpha.fits", np.full_like(b, -0.7), centre_b, "", "TAN")

    for name in sorted(out.glob("mm_target_*.fits")):
        print(name)


if __name__ == "__main__":
    main()
