"""WSClean -predict injection for spectral-cube models.

WSClean predicts visibilities from a set of per-channel model images and writes
them into the MODEL_DATA column of a Measurement Set.  This module splits a
resampled 3D spectral cube into the naming convention WSClean expects and
calls ``wsclean -predict``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from loguru import logger

from ..config import ImgConfig
from ..primary_beam import angular_separation_deg

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

    from ..primary_beam import GaussianStationBeam


def write_per_channel_model_fits(
    cube_data: np.ndarray,
    header_template: fits.Header,
    output_dir: Path,
    prefix: str,
    freq_axis: int = 3,
) -> list[Path]:
    """Split a 3D spectral cube into WSClean per-channel model images.

    Parameters
    ----------
    cube_data
        Numpy array in axis order (freq, dec, ra) after any reordering.
    header_template
        FITS header describing the full cube.  Must contain CTYPE1=RA,
        CTYPE2=DEC and a frequency axis.
    output_dir
        Directory where ``<prefix>-NNNN-model.fits`` files will be written.
    prefix
        WSClean prefix, e.g. ``model_entry_01_spectral_cube``.
    freq_axis
        1-based FITS axis index for frequency in ``header_template``.

    Returns
    -------
    paths
        Ordered list of per-channel FITS paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_freq = cube_data.shape[0]
    crpix = float(header_template.get(f"CRPIX{freq_axis}", 1.0))  # type: ignore[arg-type]
    crval_key = f"CRVAL{freq_axis}"
    cdelt_key = f"CDELT{freq_axis}"
    cunit_key = f"CUNIT{freq_axis}"

    if crval_key not in header_template or cdelt_key not in header_template:
        raise ValueError(
            f"header_template missing {crval_key} or {cdelt_key}; "
            "cannot build per-channel WSClean model headers."
        )

    crval = float(header_template[crval_key])  # type: ignore[arg-type]
    cdelt = float(header_template[cdelt_key])  # type: ignore[arg-type]
    cunit = str(header_template.get(cunit_key) or "").strip().lower()
    if cunit == "mhz":
        crval *= 1e6
        cdelt *= 1e6

    out_paths: list[Path] = []
    for i in range(n_freq):
        hdr = header_template.copy()
        # strip old NAXIS* keywords and rebuild a 3D single-plane header
        for key in list(hdr.keys()):
            if key.startswith("NAXIS") and key != "NAXIS":
                hdr.remove(key)
        hdr["NAXIS"] = 3
        hdr["NAXIS1"] = cube_data.shape[2]
        hdr["NAXIS2"] = cube_data.shape[1]
        hdr["NAXIS3"] = 1

        for k in (
            "CTYPE1",
            "CRPIX1",
            "CRVAL1",
            "CDELT1",
            "CUNIT1",
            "CTYPE2",
            "CRPIX2",
            "CRVAL2",
            "CDELT2",
            "CUNIT2",
        ):
            if k in header_template:
                hdr[k] = header_template[k]

        hdr["CTYPE3"] = "FREQ"
        hdr["CRPIX3"] = 1.0
        hdr["CRVAL3"] = crval + (i - (crpix - 1.0)) * cdelt
        hdr["CDELT3"] = cdelt
        hdr["CUNIT3"] = "Hz"
        hdr["BUNIT"] = header_template.get("BUNIT", "Jy/pixel")

        plane = cube_data[i, :, :][np.newaxis, :, :]  # shape (1, dec, ra)
        path = output_dir / f"{prefix}-{i:04d}-model.fits"
        fits.writeto(path, plane, hdr, overwrite=True)
        out_paths.append(path)

    logger.info(
        f"wrote {len(out_paths)} per-channel model FITS for WSClean predict "
        f"under {output_dir}"
    )
    return out_paths


def _imaging_cellsize(
    fov_deg: float | None, pixels: int, cell_size_arcsec: float | None = None
) -> u.Quantity:
    """Return angular pixel size from explicit geometry, never a fabricated default.

    If ``cell_size_arcsec`` is supplied it is returned directly.  Otherwise the
    pixel size is computed from ``fov_deg`` and ``pixels``.  If neither are
    available, raise an error rather than silently substituting a 1.0° field.
    """
    if cell_size_arcsec is not None:
        return cell_size_arcsec * u.arcsec
    if fov_deg is None:
        raise ValueError(
            "cannot derive wsclean-predict pixel size: fov_deg and "
            "cell_size_arcsec are both unset"
        )
    fov = float(fov_deg) * u.deg
    return fov / pixels


# Maximum tolerated difference, in model pixels, between where the model FITS
# WCS puts a pixel and where WSClean predicts it. WSClean applies the same 0.1
# pixel tolerance to pixel-scale mismatches between -scale and the FITS header.
PLACEMENT_TOLERANCE_PIX = 0.1

# Offsets below this fraction of a pixel are treated as "already centred".
_SHIFT_EPSILON_PIX = 1e-6


def format_wsclean_shift(position: "SkyCoord") -> Tuple[str, str]:
    """Format a position for ``wsclean -shift`` (aocommon ``RaDecCoord`` syntax).

    WSClean parses ``HHhMMmSS.Ss`` and ``±DDdMMmSS.Ss``; the sign is taken from
    a leading ``-``, so declinations in (-1, 0) degrees keep their sign.
    """
    icrs = position.icrs
    ra = icrs.ra.to_string(unit=u.hour, sep="hms", precision=8, pad=True)
    dec = icrs.dec.to_string(unit=u.deg, sep="dms", precision=7, pad=True)
    return ra, dec


def _wsclean_implied_wcs(
    phase_center: "SkyCoord",
    shift: "SkyCoord",
    n_pixels: int,
    pixel_size_deg: float,
) -> WCS:
    """Return the WCS WSClean assigns to a predict model image.

    Transcribed from aocommon ``FitsWriter``: a SIN projection tangent at the
    phase centre, ``CDELT = (-d, d)`` and
    ``CRPIX = (N/2 + 1 + l/d, N/2 + 1 - m/d)`` with ``(l, m)`` the direction
    cosines of the ``-shift`` position (``ImageCoordinates::RaDecToLM``).
    """
    pc = phase_center.icrs
    ra0, dec0 = pc.ra.rad, pc.dec.rad
    ra, dec = shift.icrs.ra.rad, shift.icrs.dec.rad
    dra = ra - ra0
    l_shift = np.cos(dec) * np.sin(dra)
    m_shift = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
    d_rad = np.radians(pixel_size_deg)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.crval = [pc.ra.deg, pc.dec.deg]
    wcs.wcs.cdelt = [-pixel_size_deg, pixel_size_deg]
    wcs.wcs.crpix = [
        n_pixels / 2.0 + 1.0 + l_shift / d_rad,
        n_pixels / 2.0 + 1.0 - m_shift / d_rad,
    ]
    return wcs


def _celestial_pixel_scales_deg(wcs: WCS) -> Tuple[float, float]:
    # celestial WCS units are degrees (astropy normalises CUNIT)
    scale_x, scale_y = (float(v) for v in np.abs(proj_plane_pixel_scales(wcs)))
    return scale_x, scale_y


def wsclean_predict_placement(
    header: fits.Header, shape: tuple, phase_center: "SkyCoord"
) -> dict:
    """Decide whether WSClean can predict a model image directly via ``-shift``.

    WSClean ignores the model's ``CRVAL``: it treats the image as a SIN
    projection tangent at the phase centre whose centre pixel (1-based
    ``N/2 + 1``) lies at the ``-shift`` position. skasim takes the model's own
    world position of that pixel as the shift and compares the model WCS with
    the WCS WSClean implies over the whole image.

    Returns a record with ``direct_placement_ok``. Direct placement requires an
    RA/Dec SIN projection, a square image with square pixels, and a maximum
    placement error within :data:`PLACEMENT_TOLERANCE_PIX`; otherwise
    ``reason`` says why, and the caller resamples (:func:`prepare_predict_grid`).
    """
    wcs = WCS(header).celestial
    if wcs.naxis != 2:
        raise ValueError("model image has no celestial WCS; cannot place it on sky")
    ny, nx = int(shape[-2]), int(shape[-1])
    scale_x, scale_y = _celestial_pixel_scales_deg(wcs)
    centre = wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    record = {
        "model_centre_ra_deg": float(centre.icrs.ra.deg),
        "model_centre_dec_deg": float(centre.icrs.dec.deg),
        "offset_arcsec": float(centre.separation(phase_center).arcsec),
        "direct_placement_ok": False,
        "direct_placement_error_pix": None,
        "placement_tolerance_pix": PLACEMENT_TOLERANCE_PIX,
        "reason": None,
    }

    ctypes = [str(c).upper() for c in wcs.wcs.ctype]
    if not (ctypes[0].startswith("RA---SIN") and ctypes[1].startswith("DEC--SIN")):
        record["reason"] = f"projection {ctypes} is not RA/Dec SIN"
        return record
    if nx != ny:
        record["reason"] = f"image is not square ({nx}x{ny} pixels)"
        return record
    if not np.isclose(scale_x, scale_y, rtol=1e-6, atol=0.0):
        record["reason"] = (
            f"pixels are not square ({scale_x * 3600:.6g} x "
            f"{scale_y * 3600:.6g} arcsec)"
        )
        return record

    pixel_size_deg = scale_x
    n_pixels = nx
    centre_pix = n_pixels / 2.0  # 0-based index of WSClean's centre pixel
    shift = wcs.pixel_to_world(centre_pix, centre_pix)
    implied = _wsclean_implied_wcs(phase_center, shift, n_pixels, pixel_size_deg)
    samples = np.linspace(0.0, n_pixels - 1.0, 9)
    xx, yy = np.meshgrid(samples, samples)
    model_ra, model_dec = wcs.all_pix2world(xx, yy, 0)
    wsc_ra, wsc_dec = implied.all_pix2world(xx, yy, 0)
    error_deg = angular_separation_deg(model_ra, model_dec, wsc_ra, wsc_dec)
    error_pix = float(np.nanmax(error_deg) / pixel_size_deg)
    record["direct_placement_error_pix"] = error_pix
    if not np.isfinite(error_pix) or error_pix > PLACEMENT_TOLERANCE_PIX:
        record["reason"] = (
            f"maximum placement error {error_pix:.3g} pixels exceeds "
            f"{PLACEMENT_TOLERANCE_PIX} pixels (tangent point or orientation "
            "differs from the grid WSClean assumes)"
        )
        return record

    record["direct_placement_ok"] = True
    record.update(
        _shift_record(shift, phase_center, pixel_size_deg),
        n_pixels=n_pixels,
        pixel_size_arcsec=pixel_size_deg * 3600.0,
        placement_error_pix=error_pix,
    )
    return record


def _shift_record(
    shift: "SkyCoord", phase_center: "SkyCoord", pixel_size_deg: float
) -> dict:
    offset_deg = float(shift.separation(phase_center).deg)
    applied = offset_deg > _SHIFT_EPSILON_PIX * pixel_size_deg
    ra_text, dec_text = format_wsclean_shift(shift)
    return {
        "shift_applied": bool(applied),
        "shift_ra_deg": float(shift.icrs.ra.deg),
        "shift_dec_deg": float(shift.icrs.dec.deg),
        "shift_wsclean": [ra_text, dec_text] if applied else None,
    }


def _resampled_grid(
    wcs_in: WCS, shape: tuple, phase_center: "SkyCoord"
) -> Tuple[WCS, int, float, "SkyCoord"]:
    """Return the WSClean-native grid covering the model footprint.

    The pixel size is the geometric mean of the input pixel scales; the grid is
    even-sized, centred on the model centre, and tangent at the phase centre.
    """
    ny, nx = int(shape[-2]), int(shape[-1])
    scale_x, scale_y = _celestial_pixel_scales_deg(wcs_in)
    pixel_size_deg = float(np.sqrt(scale_x * scale_y))
    centre = wcs_in.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)

    # Probe a grid with the centre pixel on the model centre, then size it so
    # the model's pixel-edge footprint fits with a one-pixel margin.
    probe = _wsclean_implied_wcs(phase_center, centre, 2, pixel_size_deg)
    edge_x = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    edge_y = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    ra, dec = wcs_in.all_pix2world(edge_x, edge_y, 0)
    px, py = probe.all_world2pix(ra, dec, 0)
    centre_px = 1.0  # 0-based N/2 for the N=2 probe
    half_extent = float(np.max(np.abs(np.concatenate([px, py]) - centre_px)))
    n_pixels = 2 * int(np.ceil(half_extent + 1.0))
    target = _wsclean_implied_wcs(phase_center, centre, n_pixels, pixel_size_deg)
    return target, n_pixels, pixel_size_deg, centre


def _pixel_solid_angle_sr(wcs: WCS, shape: tuple) -> np.ndarray:
    """Per-pixel solid angle of a celestial grid (steradian).

    Projected pixel area divided by ``n = cos(theta)`` from the tangent point,
    exact for SIN and the leading-order correction for other zenithal
    projections near their reference point.
    """
    scale_x, scale_y = _celestial_pixel_scales_deg(wcs)
    plane_area = float(np.radians(scale_x) * np.radians(scale_y))
    ny, nx = int(shape[-2]), int(shape[-1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    theta = angular_separation_deg(ra, dec, wcs.wcs.crval[0], wcs.wcs.crval[1])
    return plane_area / np.cos(np.radians(theta))


def prepare_predict_grid(
    planes: "list[np.ndarray]",
    header: fits.Header,
    phase_center: "SkyCoord",
) -> Tuple["list[np.ndarray]", fits.Header, dict]:
    """Put 2D Jy/pixel model planes on a grid WSClean predicts correctly.

    Planes that WSClean can place directly (see
    :func:`wsclean_predict_placement`) are returned unchanged. Otherwise every
    plane is regridded with ``reproject_exact`` onto a SIN grid tangent at the
    phase centre, with the WSClean centre-pixel convention. Flux is conserved
    by converting Jy/pixel to surface brightness and back with the input and
    output pixel solid angles. Returns ``(planes, header, placement)``.
    """
    shape = planes[0].shape
    placement = wsclean_predict_placement(header, shape, phase_center)
    if placement["direct_placement_ok"]:
        placement["resampled"] = False
        return planes, header, placement

    try:
        from reproject import reproject_exact
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise RuntimeError(
            "reproject is required to regrid image models for WSClean predict "
            f"({placement['reason']})."
        ) from exc

    wcs_in = WCS(header).celestial
    target, n_pixels, pixel_size_deg, centre = _resampled_grid(
        wcs_in, shape, phase_center
    )
    omega_in = _pixel_solid_angle_sr(wcs_in, shape)
    omega_out = _pixel_solid_angle_sr(target, (n_pixels, n_pixels))
    out_planes = []
    for plane in planes:
        # Jy/pixel -> Jy/sr; reproject_exact then returns the overlap-weighted
        # mean surface brightness and the covered fraction of each output
        # pixel, whose product times the output solid angle conserves flux,
        # including partly covered edge pixels.
        surface_brightness = np.asarray(plane, dtype=float) / omega_in
        regridded, coverage = reproject_exact(
            (surface_brightness, wcs_in),
            target,
            shape_out=(n_pixels, n_pixels),
        )
        out_planes.append(np.nan_to_num(regridded * coverage, nan=0.0) * omega_out)

    flux_in = float(np.nansum(planes[0]))
    flux_out = float(np.sum(out_planes[0]))
    if flux_in != 0.0 and abs(flux_out - flux_in) > 1e-3 * abs(flux_in):
        logger.warning(
            "regridded model flux differs from the input by {:.3g}% "
            "({:.6g} -> {:.6g} Jy).",
            100.0 * (flux_out - flux_in) / flux_in,
            flux_in,
            flux_out,
        )

    out_header = header.copy()
    for key in list(out_header.keys()):
        if key[:5] in ("CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT", "CROTA") and (
            key[5:] in ("1", "2")
        ):
            del out_header[key]
        elif key[:2] in ("PC", "CD") and key[2:] in ("1_1", "1_2", "2_1", "2_2"):
            del out_header[key]
    out_header.update(target.to_header())
    out_header["NAXIS1"] = n_pixels
    out_header["NAXIS2"] = n_pixels
    out_header["HISTORY"] = (
        "regridded by skasim onto the WSClean predict grid (reproject_exact)"
    )

    placement.update(
        resampled=True,
        resample_method="reproject_exact",
        flux_before_jy=flux_in,
        flux_after_jy=flux_out,
        n_pixels=n_pixels,
        pixel_size_arcsec=pixel_size_deg * 3600.0,
        placement_error_pix=0.0,
        **_shift_record(centre, phase_center, pixel_size_deg),
    )
    return out_planes, out_header, placement


def build_wsclean_predict_argv(
    wsclean_command: str,
    visibility_path: Path,
    img_config: ImgConfig,
    prefix: str,
    n_channels: int,
    pixel_size_arcsec: float | None = None,
    n_pixels: int | None = None,
    shift: Optional[Tuple[str, str]] = None,
) -> list[str]:
    """Build argv for ``wsclean -predict``.

    Only the executable from ``wsclean_command`` is kept; imaging/deconvolution
    flags (``-niter``, ``-mgain``, ``-auto-mask``, etc.) are stripped because
    they are invalid in predict mode.  Use ``wsclean_predict_command`` if you
    need custom predict-only flags.

    When ``pixel_size_arcsec`` or ``n_pixels`` are not supplied, they fall back
    to the values in ``img_config``.  Callers that already know the exact model
    image geometry (e.g. spectral-cube predict) should pass them explicitly so
    that the predict image size matches the model images regardless of any
    auto-derived imaging geometry.

    ``shift`` is an ``(ra, dec)`` pair already formatted for ``-shift`` (see
    :func:`format_wsclean_shift`).
    """
    if pixel_size_arcsec is None or n_pixels is None:
        pixel_size_arcsec = img_config.cell_size_arcsec
        if pixel_size_arcsec is None:
            imaging_cellsize = _imaging_cellsize(
                img_config.fov_deg, img_config.pixels, img_config.cell_size_arcsec
            )
            pixel_size_arcsec = imaging_cellsize.to(u.arcsec).value
        n_pixels = img_config.pixels

    # Keep only the executable from the supplied command line; imaging flags
    # are not applicable to ``wsclean -predict`` and make it fail.
    tokens = shlex.split(wsclean_command)
    executable = tokens[0] if tokens else "wsclean"

    argv = [executable] + [
        "-predict",
        "-gridder",
        "wgridder",
        "-wgridder-accuracy",
        "1e-5",
        "-size",
        str(n_pixels),
        str(n_pixels),
        "-scale",
        f"{pixel_size_arcsec:.6f}asec",
        "-channels-out",
        str(n_channels),
        "-name",
        prefix,
    ]
    if shift is not None:
        argv += ["-shift", shift[0], shift[1]]
    if img_config.threads is not None:
        argv += ["-j", str(img_config.threads)]
    argv.append(str(visibility_path))
    return argv


def run_wsclean_predict(
    wsclean_command: str,
    visibility_path: Path,
    img_config: ImgConfig,
    prefix: str,
    n_channels: int,
    work_dir: Path,
    pixel_size_arcsec: float | None = None,
    n_pixels: int | None = None,
    shift: Optional[Tuple[str, str]] = None,
) -> None:
    """Run WSClean in predict mode to fill MODEL_DATA of the MS."""
    argv = build_wsclean_predict_argv(
        wsclean_command,
        visibility_path,
        img_config,
        prefix,
        n_channels,
        pixel_size_arcsec=pixel_size_arcsec,
        n_pixels=n_pixels,
        shift=shift,
    )
    logger.info(f"WSClean predict command: {argv}")

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"

    lines: list[str] = []
    with subprocess.Popen(
        argv,
        shell=False,
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        if proc.stdout is None:
            raise RuntimeError("subprocess.Popen returned None stdout with PIPE")
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            logger.info("[wsclean-predict] {}", stripped)
            lines.append(stripped)
        proc.wait()
    combined = "\n".join(lines)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            argv,
            output=combined,
            stderr=None,
        )


def pixel_separation_deg(
    header: fits.Header, shape: tuple, phase_center: "SkyCoord"
) -> np.ndarray:
    """Return each pixel's angular distance from ``phase_center`` (degrees).

    ``shape`` is the NumPy ``(ny, nx)`` spatial shape; the FITS celestial WCS is
    evaluated with 0-based pixel indices (FITS axis 1 is NumPy's last axis).
    """
    wcs = WCS(header).celestial
    ny, nx = int(shape[-2]), int(shape[-1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    icrs = phase_center.icrs
    return angular_separation_deg(ra, dec, icrs.ra.deg, icrs.dec.deg)


def _beam_flux_report(
    ref_plane: np.ndarray,
    separation_deg: Optional[np.ndarray],
    beam: "Optional[GaussianStationBeam]",
    ref_freq_hz: float,
) -> dict:
    """Summarise intrinsic and beam-attenuated flux at the band centre."""
    intrinsic = float(np.nansum(ref_plane))
    if beam is None or separation_deg is None:
        return {
            "primary_beam_applied": False,
            "intrinsic_flux_jy_at_ref": intrinsic,
            "attenuated_flux_jy_at_ref": intrinsic,
        }
    attenuated = float(
        np.nansum(ref_plane * beam.power_response(separation_deg, ref_freq_hz))
    )
    return {
        "primary_beam_applied": True,
        "intrinsic_flux_jy_at_ref": intrinsic,
        "attenuated_flux_jy_at_ref": attenuated,
        "beam_power_fwhm_deg_at_ref": beam.power_fwhm_deg(ref_freq_hz),
        "beam_ref_freq_hz": ref_freq_hz,
    }


def _prepare_sky_placement(
    planes: "list[np.ndarray]",
    header: fits.Header,
    phase_center: "Optional[SkyCoord]",
    beam: "Optional[GaussianStationBeam]",
    label: str,
) -> Tuple["list[np.ndarray]", fits.Header, Optional[np.ndarray], Optional[dict]]:
    """Place model planes on sky and return ``(planes, header, sep, placement)``.

    With a phase centre, the planes are put on a grid WSClean predicts at the
    position given by their FITS WCS (directly through ``-shift``, or after
    regridding; see :func:`prepare_predict_grid`) and per-pixel separations
    from the phase centre are returned for the primary beam. Without a phase
    centre (direct API callers) the legacy behaviour applies: the image is
    predicted centred on the MS phase centre, unattenuated.
    """
    if phase_center is None:
        if beam is not None:
            raise ValueError(
                f"{label}: a phase centre is required to apply the primary beam."
            )
        logger.warning(
            "{}: no phase centre supplied; the model is predicted centred on the "
            "MS phase centre and without primary-beam attenuation.",
            label,
        )
        return planes, header, None, None
    planes, header, placement = prepare_predict_grid(planes, header, phase_center)
    if placement["resampled"]:
        logger.info(
            "{}: regridded onto the WSClean predict grid ({}); {} px at {:.4g} "
            "arcsec, flux {:.6g} -> {:.6g} Jy.",
            label,
            placement["reason"],
            placement["n_pixels"],
            placement["pixel_size_arcsec"],
            placement["flux_before_jy"],
            placement["flux_after_jy"],
        )
    if placement["shift_applied"]:
        logger.info(
            "{}: model centre is {:.3f} arcsec from the phase centre; WSClean "
            "-shift {} {}.",
            label,
            placement["offset_arcsec"],
            *placement["shift_wsclean"],
        )
    separation = None
    if beam is None:
        logger.warning(
            "{}: no primary beam supplied; the model is predicted unattenuated.",
            label,
        )
    else:
        separation = pixel_separation_deg(header, planes[0].shape, phase_center)
    return planes, header, separation, placement


def _shift_arg(placement: Optional[dict]) -> Optional[Tuple[str, str]]:
    if placement is None or not placement["shift_applied"]:
        return None
    ra_text, dec_text = placement["shift_wsclean"]
    return ra_text, dec_text


def _placement_report(placement: Optional[dict]) -> dict:
    if placement is None:
        return {"wcs_placement": "phase_centre_assumed"}
    return {"wcs_placement": "fits_wcs", **placement}


def merge_model_data_into_data(visibility_path: Path) -> None:
    """Add image-model MODEL_DATA into the delivered DATA column.

    This is a thin compatibility wrapper around the canonical implementation in
    ``src/skasim/loaders/image_models.py``.  It exists so that legacy imports
    from ``skasim.loaders.wsclean_predict`` keep working; new code should import
    from ``skasim.loaders`` or ``skasim.loaders.image_models`` instead.
    """
    from .image_models import merge_model_data_into_data as _canonical_merge

    return _canonical_merge(visibility_path)


def inject_spectral_cube_with_wsclean_predict(
    ctx,
    entry,
    index: int,
    visibility_path: Path,
    img_config: ImgConfig,
    cube_data: np.ndarray,
    header: fits.Header,
    freq_axis: int,
) -> dict:
    """Write per-channel FITS and run ``wsclean -predict`` for a spectral cube."""
    prefix = f"model_entry_{index + 1:02d}_spectral_cube"
    model_dir = ctx.work_dir / f"{prefix}_predict_models"

    model_paths = write_per_channel_model_fits(
        cube_data, header, model_dir, prefix, freq_axis=freq_axis
    )

    # Use the cube's own spatial geometry for predict; this keeps spectral-cube
    # models valid even when the imaging geometry was auto-derived from fov_deg.
    n_pixels = max(int(header["NAXIS1"]), int(header["NAXIS2"]))
    cdelt1 = float(header["CDELT1"])
    cunit1 = str(header.get("CUNIT1") or "").strip().lower()
    if cunit1 == "deg":
        pixel_size_arcsec = abs(cdelt1) * 3600.0
    elif cunit1 in {"asec", "arcsec"}:
        pixel_size_arcsec = abs(cdelt1)
    elif cunit1 == "rad":
        pixel_size_arcsec = abs(np.degrees(cdelt1)) * 3600.0
    else:
        raise ValueError(f"unsupported CUNIT1={cunit1!r} for spectral-cube predict")

    run_wsclean_predict(
        wsclean_command=img_config.wsclean_command,
        visibility_path=visibility_path,
        img_config=img_config,
        prefix=prefix,
        n_channels=cube_data.shape[0],
        work_dir=model_dir,
        pixel_size_arcsec=pixel_size_arcsec,
        n_pixels=n_pixels,
    )

    return {
        "model_entry_index": index,
        "model_type": entry.type,
        "backend": "wsclean_predict",
        "n_channels": len(model_paths),
        "prefix": prefix,
        "model_dir": str(model_dir),
    }


def inject_static_stokes_i_with_wsclean_predict(
    ctx,
    entry,
    index: int,
    visibility_path: Path,
    img_config: ImgConfig,
    phase_center: "Optional[SkyCoord]" = None,
    beam: "Optional[GaussianStationBeam]" = None,
) -> dict:
    """Inject a static Stokes I model using ``wsclean -predict``.

    The input is a single 2D spatial FITS image in Jy/pixel. It is converted
    into per-channel model images following WSClean's ``<prefix>-NNNN-model.fits``
    convention. Each channel is a spectrally-flat copy of the input image with
    frequency metadata taken from the observation spectral window.
    ``wsclean -predict`` then fills the MODEL_DATA column of the MS.

    When ``beam`` is given, every channel plane is multiplied by the shared
    Gaussian power beam centred on ``phase_center`` (see
    ``skasim.primary_beam``) so image models and catalogue sources share one
    attenuation.
    """
    prefix = f"model_entry_{index + 1:02d}_static_stokes"
    model_dir = ctx.work_dir / f"{prefix}_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    stokes_path = Path(entry.stokes_i).expanduser().resolve()
    with fits.open(stokes_path) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data, dtype=float).squeeze()  # type: ignore[union-attr]
        header_template = hdu.header.copy()  # type: ignore[union-attr]

    if data.ndim != 2:
        raise ValueError(
            f"static_stokes_maps stokes_i must be 2D; got shape {data.shape}"
        )
    (data,), header_template, separation, placement = _prepare_sky_placement(
        [data], header_template, phase_center, beam, "static_stokes_maps"
    )

    cdelt1 = abs(float(header_template.get("CDELT1", 0.0)))
    cdelt2 = abs(float(header_template.get("CDELT2", 0.0)))
    if cdelt1 <= 0.0 or cdelt2 <= 0.0:
        raise ValueError(
            "model header missing CDELT1/CDELT2; cannot determine pixel size"
        )
    pixel_size_deg = 0.5 * (cdelt1 + cdelt2)
    pixel_size_arcsec = pixel_size_deg * 3600.0

    ny, nx = data.shape
    if ny != nx:
        logger.warning(
            "static_stokes_maps model image is not square ({}×{}); WSClean predict "
            "will use {} pixels for both dimensions, which may distort the model.",
            nx,
            ny,
            nx,
        )
    n_pixels = nx

    obs = ctx.config.observation
    n_channels = obs.n_channels
    center_hz = obs.frequency_mhz * 1e6
    bandwidth_hz = obs.bandwidth_mhz * 1e6
    if n_channels < 1:
        raise ValueError(f"observation.n_channels must be >= 1, got {n_channels}")
    chan_width_hz = bandwidth_hz / n_channels
    start_hz = center_hz - bandwidth_hz / 2.0

    model_paths: list[Path] = []
    for i in range(n_channels):
        freq_hz = start_hz + (i + 0.5) * chan_width_hz

        hdr = header_template.copy()
        for key in list(hdr.keys()):
            if key.startswith("NAXIS") and key != "NAXIS":
                hdr.remove(key)
        hdr["NAXIS"] = 3
        hdr["NAXIS1"] = nx
        hdr["NAXIS2"] = ny
        hdr["NAXIS3"] = 1

        for k in (
            "CTYPE1",
            "CRPIX1",
            "CRVAL1",
            "CDELT1",
            "CUNIT1",
            "CTYPE2",
            "CRPIX2",
            "CRVAL2",
            "CDELT2",
            "CUNIT2",
        ):
            if k in header_template:
                hdr[k] = header_template[k]

        hdr["CTYPE3"] = "FREQ"
        hdr["CRPIX3"] = 1.0
        hdr["CRVAL3"] = freq_hz
        hdr["CDELT3"] = chan_width_hz
        hdr["CUNIT3"] = "Hz"
        hdr["BUNIT"] = header_template.get("BUNIT", "Jy/pixel")

        plane = data
        if beam is not None and separation is not None:
            plane = data * beam.power_response(separation, freq_hz)

        path = model_dir / f"{prefix}-{i:04d}-model.fits"
        fits.writeto(path, plane[np.newaxis, :, :], hdr, overwrite=True)
        model_paths.append(path)

    logger.info(
        f"wrote {len(model_paths)} per-channel model FITS for static Stokes I "
        f"WSClean predict under {model_dir}"
    )

    wsclean_command = img_config.wsclean_predict_command or img_config.wsclean_command

    run_wsclean_predict(
        wsclean_command=wsclean_command,
        visibility_path=visibility_path,
        img_config=img_config,
        prefix=prefix,
        n_channels=n_channels,
        work_dir=model_dir,
        pixel_size_arcsec=pixel_size_arcsec,
        n_pixels=n_pixels,
        shift=_shift_arg(placement),
    )

    return {
        "model_entry_index": index,
        "model_type": entry.type,
        "backend": "wsclean_predict",
        "n_channels": len(model_paths),
        "prefix": prefix,
        "model_dir": str(model_dir),
        "model_paths": [str(p) for p in model_paths],
        "pixel_size_arcsec": pixel_size_arcsec,
        "n_pixels": n_pixels,
        "wsclean_command": wsclean_command,
        **_beam_flux_report(data, separation, beam, center_hz),
        **_placement_report(placement),
    }


def inject_continuum_i_alpha_with_wsclean_predict(
    ctx,
    entry,
    index: int,
    visibility_path: Path,
    img_config: ImgConfig,
    product,
    phase_center: "Optional[SkyCoord]" = None,
    beam: "Optional[GaussianStationBeam]" = None,
) -> dict:
    """Inject a continuum I+alpha model using ``wsclean -predict``.

    The CASA Taylor-term images (tt0, tt1) produced for the model are exported to
    per-channel FITS images following WSClean's ``<prefix>-NNNN-model.fits``
    convention.  Each channel is computed from the first-order Taylor expansion
    around the observation reference frequency:

        I(ν) = tt0 + tt1 * (ν - ν0) / ν0

    where ν0 is the observation band centre.  ``wsclean -predict`` then fills
    the MODEL_DATA column of the visibility MS.

    When ``beam`` is given, every channel plane is multiplied by the shared
    Gaussian power beam centred on ``phase_center`` (see
    ``skasim.primary_beam``).
    """
    if product.nterms != 2:
        raise ValueError(
            "inject_continuum_i_alpha_with_wsclean_predict requires a 2-term "
            "Taylor product (tt0 and tt1)."
        )

    prefix = f"model_entry_{index + 1:02d}_continuum_predict"
    model_dir = ctx.work_dir / f"{prefix}_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use the intermediate FITS files (4D with degenerate STOKES/FREQ axes) so we
    # can read tt0/tt1 without requiring a CASA image export step.
    fits_paths = getattr(product, "intermediates", None)
    if not fits_paths or len(fits_paths) != 2:
        raise ValueError(
            "continuum_i_alpha product must expose two intermediate FITS paths "
            "(tt0.fits and tt1.fits)."
        )

    tt0_fits = fits_paths[0]
    tt1_fits = fits_paths[1]

    with fits.open(tt0_fits) as hdul0, fits.open(tt1_fits) as hdul1:
        tt0_data = np.asarray(hdul0[0].data, dtype=float).squeeze()
        tt1_data = np.asarray(hdul1[0].data, dtype=float).squeeze()
        header_template = hdul0[0].header.copy()

    if tt0_data.ndim != 2 or tt1_data.ndim != 2:
        raise ValueError(
            f"intermediate tt0 and tt1 must squeeze to 2D spatial arrays; "
            f"got tt0 {tt0_data.shape}, tt1 {tt1_data.shape}"
        )

    if tt0_data.shape != tt1_data.shape:
        raise ValueError(
            f"tt0 and tt1 shapes differ: {tt0_data.shape} vs {tt1_data.shape}"
        )
    (tt0_data, tt1_data), header_template, separation, placement = (
        _prepare_sky_placement(
            [tt0_data, tt1_data],
            header_template,
            phase_center,
            beam,
            "continuum_i_alpha",
        )
    )

    # Derive the pixel geometry from the model header so WSClean predict matches.
    cdelt1 = abs(float(header_template.get("CDELT1", 0.0)))
    cdelt2 = abs(float(header_template.get("CDELT2", 0.0)))
    if cdelt1 <= 0.0 or cdelt2 <= 0.0:
        raise ValueError(
            "model header missing CDELT1/CDELT2; cannot determine pixel size"
        )
    # Average pixel size in arcseconds (FITS CDELT is in degrees).
    pixel_size_deg = 0.5 * (cdelt1 + cdelt2)
    pixel_size_arcsec = pixel_size_deg * 3600.0

    ny, nx = tt0_data.shape
    if ny != nx:
        logger.warning(
            "continuum_i_alpha model image is not square ({}×{}); WSClean predict "
            "will use {} pixels for both dimensions, which may distort the model.",
            nx,
            ny,
            nx,
        )
    n_pixels = nx

    obs = ctx.config.observation
    n_channels = obs.n_channels
    center_hz = obs.frequency_mhz * 1e6
    bandwidth_hz = obs.bandwidth_mhz * 1e6
    chan_width_hz = bandwidth_hz / n_channels
    start_hz = center_hz - bandwidth_hz / 2.0

    model_paths: list[Path] = []
    for i in range(n_channels):
        freq_hz = start_hz + (i + 0.5) * chan_width_hz
        # First-order Taylor expansion around the observation reference.
        relative = (freq_hz - center_hz) / center_hz
        plane = tt0_data + tt1_data * relative
        if beam is not None and separation is not None:
            plane = plane * beam.power_response(separation, freq_hz)

        hdr = header_template.copy()
        # Strip old NAXIS* keywords and rebuild a 3D single-plane header.
        for key in list(hdr.keys()):
            if key.startswith("NAXIS") and key != "NAXIS":
                hdr.remove(key)
        hdr["NAXIS"] = 3
        hdr["NAXIS1"] = plane.shape[-1]
        hdr["NAXIS2"] = plane.shape[-2]
        hdr["NAXIS3"] = 1

        for k in (
            "CTYPE1",
            "CRPIX1",
            "CRVAL1",
            "CDELT1",
            "CUNIT1",
            "CTYPE2",
            "CRPIX2",
            "CRVAL2",
            "CDELT2",
            "CUNIT2",
        ):
            if k in header_template:
                hdr[k] = header_template[k]

        hdr["CTYPE3"] = "FREQ"
        hdr["CRPIX3"] = 1.0
        hdr["CRVAL3"] = freq_hz
        hdr["CDELT3"] = chan_width_hz
        hdr["CUNIT3"] = "Hz"
        hdr["BUNIT"] = header_template.get("BUNIT", "Jy/pixel")

        path = model_dir / f"{prefix}-{i:04d}-model.fits"
        fits.writeto(path, plane[np.newaxis, :, :], hdr, overwrite=True)
        model_paths.append(path)

    logger.info(
        f"wrote {len(model_paths)} per-channel model FITS for continuum I+alpha "
        f"WSClean predict under {model_dir}"
    )

    wsclean_command = img_config.wsclean_predict_command or img_config.wsclean_command

    run_wsclean_predict(
        wsclean_command=wsclean_command,
        visibility_path=visibility_path,
        img_config=img_config,
        prefix=prefix,
        n_channels=n_channels,
        work_dir=model_dir,
        pixel_size_arcsec=pixel_size_arcsec,
        n_pixels=n_pixels,
        shift=_shift_arg(placement),
    )

    return {
        "model_entry_index": index,
        "model_type": entry.type,
        "backend": "wsclean_predict",
        "n_channels": len(model_paths),
        "prefix": prefix,
        "model_dir": str(model_dir),
        "pixel_size_arcsec": pixel_size_arcsec,
        "n_pixels": n_pixels,
        "wsclean_command": wsclean_command,
        **_beam_flux_report(tt0_data, separation, beam, center_hz),
        **_placement_report(placement),
    }


__all__ = [
    "PLACEMENT_TOLERANCE_PIX",
    "build_wsclean_predict_argv",
    "format_wsclean_shift",
    "inject_continuum_i_alpha_with_wsclean_predict",
    "inject_static_stokes_i_with_wsclean_predict",
    "inject_spectral_cube_with_wsclean_predict",
    "merge_model_data_into_data",
    "pixel_separation_deg",
    "prepare_predict_grid",
    "run_wsclean_predict",
    "write_per_channel_model_fits",
    "wsclean_predict_placement",
]
