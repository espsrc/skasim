"""Write the small FITS models used by the catalogue + image + HI cube demo.

The demo (``configs/fits_model/vla_c_catalog_image_hi_cube_quick.json``)
places a compact HI source on top of a compact continuum source, 6' east and
4' south of the phase centre, so the cluster check
(``scripts/check_multimodal_run.py``) can test both the continuum image and
the continuum-subtracted line:

* ``mm_hi_cube``: TAN grid tangent at the HI position (so skasim regrids it),
  64 channels x 50 kHz aligned with the demo's MS channels, a Gaussian line
  (peak 20 mJy, FWHM 300 kHz) in the centre pixel, RESTFRQ = HI 21 cm,
  SPECSYS = BARYCENT (recorded, not converted).
* ``mm_hi_continuum``: TAN grid on the same position, a 10 mJy continuum
  source with alpha = -0.7 in the centre pixel.

Offsets are multiples of the 2" model and 3" demo image pixels.
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
HI_REST_HZ = 1420.405752e6

# MS grid of the demo config: 96 channels x 50 kHz centred on 1420.0 MHz.
MS_CENTRE_HZ = 1420.0e6
MS_CHANNEL_HZ = 50e3
MS_CHANNELS = 96
CUBE_FIRST_MS_CHANNEL = 16
CUBE_CHANNELS = 64
LINE_PEAK_JY = 0.020
LINE_FWHM_HZ = 300e3
CONTINUUM_JY = 0.010


def _ms_channel_hz(index: int) -> float:
    return MS_CENTRE_HZ + (index - (MS_CHANNELS - 1) / 2.0) * MS_CHANNEL_HZ


def _celestial_wcs(centre: SkyCoord, naxis: int) -> WCS:
    wcs = WCS(naxis=naxis)
    ctype = ["RA---TAN", "DEC--TAN"]
    crval = [centre.ra.deg, centre.dec.deg]
    crpix = [PIXELS / 2 + 1, PIXELS / 2 + 1]
    cdelt = [-CELL_ARCSEC / 3600.0, CELL_ARCSEC / 3600.0]
    if naxis == 3:
        ctype.append("FREQ")
        crval.append(_ms_channel_hz(CUBE_FIRST_MS_CHANNEL))
        crpix.append(1.0)
        cdelt.append(MS_CHANNEL_HZ)
    wcs.wcs.ctype = ctype
    wcs.wcs.crval = crval
    wcs.wcs.crpix = crpix
    wcs.wcs.cdelt = cdelt
    if naxis == 3:
        wcs.wcs.cunit = ["deg", "deg", "Hz"]
    return wcs


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

    centre = PHASE_CENTRE.spherical_offsets_by(6 * u.arcmin, -4 * u.arcmin)
    y = x = PIXELS // 2  # the TAN reference pixel (1-based PIXELS/2 + 1)

    # HI cube: Gaussian line, exactly zero beyond 3 sigma so the line channels
    # (non-zero emission) are well defined for continuum subtraction.
    freqs = _ms_channel_hz(CUBE_FIRST_MS_CHANNEL) + np.arange(CUBE_CHANNELS) * (
        MS_CHANNEL_HZ
    )
    line_centre = _ms_channel_hz(48)
    sigma = LINE_FWHM_HZ / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    profile = LINE_PEAK_JY * np.exp(-0.5 * ((freqs - line_centre) / sigma) ** 2)
    profile[np.abs(freqs - line_centre) > 3.0 * sigma] = 0.0
    cube = np.zeros((CUBE_CHANNELS, PIXELS, PIXELS), dtype=np.float32)
    cube[:, y, x] = profile
    header = _celestial_wcs(centre, 3).to_header()
    header["BUNIT"] = "Jy/pixel"
    header["RESTFRQ"] = HI_REST_HZ
    header["SPECSYS"] = "BARYCENT"
    fits.writeto(out / "mm_hi_cube.fits", cube, header, overwrite=True)

    image = np.zeros((PIXELS, PIXELS), dtype=np.float32)
    image[y, x] = CONTINUUM_JY
    header = _celestial_wcs(centre, 2).to_header()
    header["BUNIT"] = "Jy/pixel"
    fits.writeto(out / "mm_hi_continuum_i.fits", image, header, overwrite=True)
    fits.writeto(
        out / "mm_hi_continuum_alpha.fits",
        np.full_like(image, -0.7),
        _celestial_wcs(centre, 2).to_header(),
        overwrite=True,
    )

    dv = 299792.458 * MS_CHANNEL_HZ / HI_REST_HZ
    print(
        f"line: peak {LINE_PEAK_JY * 1e3:.1f} mJy, integrated "
        f"{profile.sum() * dv:.4f} Jy km/s (intrinsic), channels "
        f"{int(np.flatnonzero(profile)[0]) + CUBE_FIRST_MS_CHANNEL}-"
        f"{int(np.flatnonzero(profile)[-1]) + CUBE_FIRST_MS_CHANNEL} of the MS"
    )
    for name in sorted(out.glob("mm_hi_*.fits")):
        print(name)


if __name__ == "__main__":
    main()
