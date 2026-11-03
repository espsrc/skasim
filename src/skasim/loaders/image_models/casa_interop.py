"""CASA image-model preparation and batch execution."""

from __future__ import annotations

import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from loguru import logger

from ...config import (
    CasaTaylorTermsModelEntry,
    ContinuumIAlphaModelEntry,
    SpectralCubeModelEntry,
)
from ...manifest import RunContext
from ...runtime import require_casacore
from .fits_io import (
    _ACCEPTED_JY_PER_PIXEL_UNITS,
    _find_frequency_axis,
    _fits_axis_to_numpy,
    _freq_axis_centres,
    _reorder_cube_axes,
    _squeeze_degenerate_axes,
)

# suppress fits formatting fixes
warnings.simplefilter("ignore", category=FITSFixedWarning)
# suppress polar motion fallback warnings
warnings.filterwarnings("ignore", message=".*polar motions.*")


@dataclass(frozen=True)
class CasaModelProduct:
    """CASA-ready model product generated from one model entry."""

    model_paths: list[Path]
    nterms: int
    reffreq: str
    intermediates: list[Path]
    cube_data: np.ndarray | None = None
    header: fits.Header | None = None
    freq_axis: int | None = None
    model_dir: Path | None = None


def _read_ms_phase_center(visibility_path: Path) -> tuple[float, float] | None:
    """Return (ra_deg, dec_deg) from the MS FIELD table, or None if unavailable."""
    try:
        table = require_casacore()
    except Exception as exc:
        logger.debug(
            f"_read_ms_phase_center: casacore not available for {visibility_path}: {exc}"
        )
        return None
    try:
        with table(str(visibility_path / "FIELD"), ack=False) as tb:
            phase_dir = np.asarray(tb.getcol("PHASE_DIR"))
            # PHASE_DIR is stored as (nrows, npoly, 2) per the MS v2 spec.
            # Some older tables or single-field data may appear as (2, nrows) or
            # (2, nrows, npoly); only the standard layout is accepted here.
            if phase_dir.ndim not in (2, 3):
                logger.warning(
                    f"PHASE_DIR in {visibility_path}/FIELD has unexpected ndim={phase_dir.ndim}; "
                    "expected 2 or 3."
                )
                return None
            if phase_dir.ndim == 2:
                if phase_dir.shape[1] != 2:
                    logger.warning(
                        f"PHASE_DIR in {visibility_path}/FIELD has shape {phase_dir.shape}; "
                        "expected (nrows, 2)."
                    )
                    return None
                ra_rad = float(phase_dir[0, 0])
                dec_rad = float(phase_dir[0, 1])
            else:
                if phase_dir.shape[2] != 2:
                    logger.warning(
                        f"PHASE_DIR in {visibility_path}/FIELD has shape {phase_dir.shape}; "
                        "expected (nrows, npoly, 2)."
                    )
                    return None
                ra_rad = float(phase_dir[0, 0, 0])
                dec_rad = float(phase_dir[0, 0, 1])
            return float(np.degrees(ra_rad)), float(np.degrees(dec_rad))
    except Exception as exc:
        logger.debug(
            f"_read_ms_phase_center: failed to read PHASE_DIR from {visibility_path}: {exc}"
        )
        return None


def _read_ms_spectral_window(visibility_path: Path) -> tuple[np.ndarray, float] | None:
    """Return (chan_freqs_hz, chan_width_hz) from the MS SPECTRAL_WINDOW table."""
    try:
        table = require_casacore()
    except Exception as exc:
        logger.debug(
            f"_read_ms_spectral_window: casacore not available for {visibility_path}: {exc}"
        )
        return None
    try:
        with table(str(visibility_path / "SPECTRAL_WINDOW"), ack=False) as tb:
            num_chan = np.asarray(tb.getcol("NUM_CHAN"))
            n_spw = len(num_chan)
            n_chan_first = int(num_chan[0])

            chan_freq = np.asarray(tb.getcol("CHAN_FREQ"))
            chan_width = np.asarray(tb.getcol("CHAN_WIDTH"))

            # CHAN_FREQ shape is usually (n_spw, n_chan) but can be transposed.
            # Choose orientation that matches n_chan of the first SPW.
            if chan_freq.shape[0] == n_spw and chan_freq.shape[1] == n_chan_first:
                freqs = chan_freq[0, :]
                widths = chan_width[0, :]
            elif chan_freq.shape[1] == n_spw and chan_freq.shape[0] == n_chan_first:
                freqs = chan_freq[:, 0]
                widths = chan_width[:, 0]
            else:
                # Fallback: flatten and take the first n_chan_first values
                freqs = chan_freq.flatten()[:n_chan_first]
                widths = chan_width.flatten()[:n_chan_first]

            return freqs, float(np.mean(widths))
    except Exception as exc:
        logger.debug(
            f"_read_ms_spectral_window: failed to read SPECTRAL_WINDOW from {visibility_path}: {exc}"
        )
        return None


def _resample_spectral_axis_to_ms_channels(
    freqs_in_hz: np.ndarray,
    data_in: np.ndarray,
    freqs_out_hz: np.ndarray,
    df_out_hz: float,
) -> np.ndarray:
    """Resample a 3D spectral cube to the MS channel grid conserving integrated flux.

    data_in shape is (n_freq, n_y, n_x).  For each output channel, integrate the
    input pixels that overlap the output channel width, then divide by the output
    channel width.  This preserves total flux when the input channels are finer
    than (or misaligned with) the output channels.
    """
    n_in, n_y, n_x = data_in.shape
    n_out = len(freqs_out_hz)
    half_out = abs(df_out_hz) / 2.0

    data_in_r = data_in.reshape(n_in, n_y * n_x)
    data_out_r = np.zeros((n_out, n_y * n_x), dtype=np.float32)

    df_in = np.diff(freqs_in_hz)
    if df_in.size == 0:
        df_in = np.array([abs(df_out_hz)])
    edges_in = np.empty(n_in + 1, dtype=freqs_in_hz.dtype)
    edges_in[0] = freqs_in_hz[0] - df_in[0] / 2.0
    edges_in[1:-1] = freqs_in_hz[:-1] + df_in / 2.0
    edges_in[-1] = freqs_in_hz[-1] + df_in[-1] / 2.0

    for out_idx in range(n_out):
        out_lo = freqs_out_hz[out_idx] - half_out
        out_hi = freqs_out_hz[out_idx] + half_out

        in_lo_idx = max(0, np.searchsorted(edges_in, out_lo, side="right") - 1)
        in_hi_idx = min(n_in, np.searchsorted(edges_in, out_hi, side="right"))
        if in_lo_idx >= in_hi_idx:
            continue

        overlap_lo = np.maximum(edges_in[in_lo_idx:in_hi_idx], out_lo)
        overlap_hi = np.minimum(edges_in[in_lo_idx + 1 : in_hi_idx + 1], out_hi)
        weights = np.clip(overlap_hi - overlap_lo, 0.0, None)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            continue

        band = data_in_r[in_lo_idx:in_hi_idx, :]
        weighted_sum = np.average(band, axis=0, weights=weights) * total_weight
        data_out_r[out_idx, :] = weighted_sum / abs(df_out_hz)

    return data_out_r.reshape(n_out, n_y, n_x)


def prepare_spectral_cube_for_casa(
    ctx: RunContext,
    entry: SpectralCubeModelEntry,
    index: int,
    report: dict,
) -> CasaModelProduct:
    """Resample a 3D FITS spectral cube to the MS spectral grid.

    Returns the resampled cube array in numpy order (freq, dec, ra) together
    with a 3D FITS header that describes the per-channel model images.  The
    caller is responsible for splitting this into WSClean per-channel model
    images and running ``wsclean -predict``.
    """
    source_path = Path(entry.cube).expanduser().resolve()
    prefix = f"model_entry_{index + 1:02d}_spectral_cube"
    cube_fits = ctx.work_dir / f"{prefix}.fits"

    obs = ctx.config.observation
    if obs.n_channels is None:
        raise ValueError("observation n_channels is required; cannot resample cube")

    with fits.open(source_path) as hdul:
        hdu = hdul[0]
        if hdu.data is None:
            raise ValueError(f"{source_path} has no image data")
        data_in = np.asarray(hdu.data, dtype=np.float32)
        header_in = hdu.header.copy()
        data_in, header_in = _squeeze_degenerate_axes(data_in, header_in)
        if data_in.ndim != 3:
            raise ValueError(f"spectral_cube must be 3D, got ndim={data_in.ndim}")

    freq_axis = _find_frequency_axis(header_in)
    spatial_axes = [i for i in range(1, 4) if i != freq_axis]
    n_freq_in = int(header_in[f"NAXIS{freq_axis}"])
    n_dec_in = int(header_in[f"NAXIS{spatial_axes[1]}"])
    n_ra_in = int(header_in[f"NAXIS{spatial_axes[0]}"])

    freqs_in_hz = _freq_axis_centres(header_in, n_freq_in, axis=freq_axis)
    cdelt_freq = float(header_in[f"CDELT{freq_axis}"])
    cunit_freq = str(header_in.get(f"CUNIT{freq_axis}") or "").strip().lower()
    if cunit_freq == "mhz":
        cdelt_freq *= 1e6
    if cdelt_freq < 0:
        sort_idx = np.argsort(freqs_in_hz)
        slicer = [slice(None)] * 3
        slicer[_fits_axis_to_numpy(freq_axis, data_in.ndim)] = sort_idx
        freqs_in_hz = freqs_in_hz[sort_idx]
        data_in = data_in[tuple(slicer)]

    visibility_path = ctx.visibility_path
    ms_grid = None
    if visibility_path is not None:
        ms_grid = _read_ms_spectral_window(visibility_path)
    if ms_grid is not None:
        freqs_out_hz, obs_df_hz = ms_grid
        n_freq_out = len(freqs_out_hz)
    else:
        if obs.channel_width_mhz is None or obs.frequency_mhz is None:
            raise ValueError("Cannot determine output spectral grid from MS or config")
        n_freq_out = obs.n_channels
        obs_df_hz = obs.channel_width_mhz * 1e6
        obs_center_hz = obs.frequency_mhz * 1e6
        freqs_out_hz = (
            obs_center_hz + (np.arange(n_freq_out) - (n_freq_out - 1) / 2.0) * obs_df_hz
        )

    if obs_df_hz == 0:
        raise ValueError("output channel width is 0; cannot build output grid")

    reverse_output = False
    if obs_df_hz < 0:
        sort_idx = np.argsort(freqs_out_hz)
        freqs_out_hz = freqs_out_hz[sort_idx]
        reverse_output = True

    cube_min_hz, cube_max_hz = float(freqs_in_hz[0]), float(freqs_in_hz[-1])
    ms_min_hz, ms_max_hz = float(freqs_out_hz[0]), float(freqs_out_hz[-1])
    logger.info(
        f"spectral cube input: {n_freq_in} channels, "
        f"[{cube_min_hz / 1e6:.6f}, {cube_max_hz / 1e6:.6f}] MHz, "
        f"data min={float(data_in.min()):.3e} max={float(data_in.max()):.3e} mean={float(data_in.mean()):.3e}"
    )
    logger.info(
        f"MS output grid: {n_freq_out} channels x {obs_df_hz / 1e6:.6f} MHz, "
        f"[{ms_min_hz / 1e6:.6f}, {ms_max_hz / 1e6:.6f}] MHz"
    )
    if cube_max_hz < ms_min_hz or cube_min_hz > ms_max_hz:
        raise ValueError(
            f"spectral cube [{cube_min_hz / 1e6:.3f}, {cube_max_hz / 1e6:.3f}] MHz "
            f"does not overlap MS grid [{ms_min_hz / 1e6:.3f}, {ms_max_hz / 1e6:.3f}] MHz"
        )

    data_ordered = _reorder_cube_axes(data_in, header_in)

    logger.info(f"resampling spectral cube from {n_freq_in} to {n_freq_out} channels")
    data_out = _resample_spectral_axis_to_ms_channels(
        freqs_in_hz, data_ordered, freqs_out_hz, obs_df_hz
    )
    if reverse_output:
        data_out = data_out[::-1, :, :]
        obs_df_hz = abs(obs_df_hz)
        freqs_out_hz = freqs_out_hz[::-1]

    nonzero_out = int(np.count_nonzero(data_out))
    logger.info(
        f"resampled cube: min={float(data_out.min()):.3e} max={float(data_out.max()):.3e} "
        f"mean={float(data_out.mean()):.3e} nonzero_pixels={nonzero_out}/{data_out.size}"
    )

    phase_center = None
    if visibility_path is not None:
        phase_center = _read_ms_phase_center(visibility_path)
    if phase_center is None:
        try:
            wcs = WCS(header_in).celestial
            center_x = (n_ra_in - 1) / 2.0
            center_y = (n_dec_in - 1) / 2.0
            sky = wcs.pixel_to_world(center_x, center_y)
            if isinstance(sky, SkyCoord):
                phase_center = (float(sky.ra.deg), float(sky.dec.deg))
        except Exception as exc:
            logger.debug(
                f"prepare_spectral_cube_for_casa: WCS phase centre failed for {source_path}: {exc}"
            )
            phase_center = None
    if phase_center is None:
        try:
            ra_val = float(header_in.get(f"CRVAL{spatial_axes[0]}", 0.0))
            dec_val = float(header_in.get(f"CRVAL{spatial_axes[1]}", 0.0))
            phase_center = (ra_val, dec_val)
        except Exception as exc:
            raise ValueError(
                f"Cannot determine spatial phase centre for {source_path}"
            ) from exc

    ra_deg, dec_deg = phase_center
    cdelt1 = float(header_in[f"CDELT{spatial_axes[0]}"])
    cdelt2 = float(header_in[f"CDELT{spatial_axes[1]}"])
    cunit1 = str(header_in.get(f"CUNIT{spatial_axes[0]}") or "").strip().lower()
    cunit2 = str(header_in.get(f"CUNIT{spatial_axes[1]}") or "").strip().lower()
    if cunit1 == "rad":
        cdelt1 = np.degrees(cdelt1)
    elif cunit1 in {"asec", "arcsec"}:
        cdelt1 = cdelt1 / 3600.0
    if cunit2 == "rad":
        cdelt2 = np.degrees(cdelt2)
    elif cunit2 in {"asec", "arcsec"}:
        cdelt2 = cdelt2 / 3600.0

    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = n_ra_in
    header["NAXIS2"] = n_dec_in
    header["NAXIS3"] = n_freq_out

    header["CTYPE1"] = "RA---SIN"
    header["CRPIX1"] = float(n_ra_in) / 2.0
    header["CRVAL1"] = float(ra_deg)
    header["CDELT1"] = float(cdelt1)
    header["CUNIT1"] = "deg"

    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX2"] = float(n_dec_in) / 2.0
    header["CRVAL2"] = float(dec_deg)
    header["CDELT2"] = float(cdelt2)
    header["CUNIT2"] = "deg"

    header["CTYPE3"] = "FREQ"
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = float(freqs_out_hz[0])
    header["CDELT3"] = float(obs_df_hz)
    header["CUNIT3"] = "Hz"

    bunit = str(header_in.get("BUNIT") or "Jy/px").strip().lower()
    if bunit in _ACCEPTED_JY_PER_PIXEL_UNITS:
        bunit = "Jy/pixel"
    header["BUNIT"] = bunit
    header["RESTFRQ"] = float(np.mean(freqs_out_hz))
    header["SPECSYS"] = "LSRK"
    header["SSYSOBS"] = "LSRK"
    header["HISTORY"] = (
        "resampled to MS spectral grid by skasim.prepare_spectral_cube_for_casa"
    )

    # wsclean -predict expects one 2D FITS per channel named <prefix>-NNNN-model.fits.
    # We keep the full resampled cube as a reference FITS but do not import to CASA.
    fits.writeto(cube_fits, data_out, header, overwrite=True)

    obs_center_hz = float(np.mean(freqs_out_hz))
    return CasaModelProduct(
        model_paths=[cube_fits],
        nterms=1,
        reffreq=f"{obs_center_hz}Hz",
        intermediates=[cube_fits],
        cube_data=data_out,
        header=header,
        freq_axis=3,
        model_dir=ctx.work_dir,
    )


def validate_casa_taylor_terms(entry: CasaTaylorTermsModelEntry) -> dict:
    """Validate an existing CASA Taylor-term image model entry."""
    model_paths = [
        Path(path).expanduser().resolve()
        for path in (entry.tt0, entry.tt1)
        if path is not None
    ]
    if not model_paths:
        raise ValueError("casa_taylor_terms requires at least tt0.")
    for path in model_paths:
        if not path.is_dir():
            raise ValueError(f"{path} must be a CASA image table directory.")
        if not (path / "table.dat").exists():
            raise ValueError(f"{path} does not look like a CASA image table.")
    return {
        "model_paths": [str(path) for path in model_paths],
        "nterms": len(model_paths),
        "reference_frequency_hz": entry.reference_frequency_hz,
    }


def prepare_casa_taylor_terms(
    ctx: RunContext,
    entry: CasaTaylorTermsModelEntry,
    index: int,
) -> CasaModelProduct:
    """Copy CASA Taylor-term images into the run and align their spectral reference.

    The reference frequency is adjusted to the observation band centre.
    For nterms≥2, tt0 pixel data is scaled:  tt0' = tt0 · (ν_obs / ν_old)^α
    where α = mean(tt1) / mean(tt0).  tt1 pixel data is unchanged.
    For nterms=1, only the spectral CRVAL is updated.
    """
    new_ref_hz = ctx.config.observation.frequency_mhz * 1e6
    old_ref_hz = entry.reference_frequency_hz

    source_paths = [
        Path(path).expanduser().resolve()
        for path in (entry.tt0, entry.tt1)
        if path is not None
    ]
    prefix = f"model_entry_{index + 1:02d}_casa_taylor"
    model_paths = []
    for term_index, source_path in enumerate(source_paths):
        target_path = ctx.work_dir / f"{prefix}.tt{term_index}.image"
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)
        model_paths.append(target_path)

    nterms = len(model_paths)

    if nterms >= 2:
        # alpha map: element-wise tt1/tt0, with safeguard for tt0==0 (those pixels
        # stay zero regardless of spectral index).
        casacore_table = require_casacore()
        with casacore_table(str(model_paths[0]), readonly=True, ack=False) as tbl0:
            tt0_data = np.asarray(tbl0.getcol("map"))
        with casacore_table(str(model_paths[1]), readonly=True, ack=False) as tbl1:
            tt1_data = np.asarray(tbl1.getcol("map"))

        # compute element-wise alpha, safeguarding divide-by-zero
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_map = np.where(
                tt0_data != 0,
                tt1_data / tt0_data,
                0.0,
            )
        alpha_mean = float(np.mean(alpha_map))
        logger.info(
            f"prepare_casa_taylor_terms: alpha_map_mean={alpha_mean:.6f} "
            f"from element-wise tt1/tt0"
        )

        adjust_spectral_reference(
            model_paths[0],
            old_ref_hz,
            new_ref_hz,
            alpha_map=alpha_map,
        )
        # tt1 must be scaled by the same factor as tt0 so that the ratio
        # tt1/tt0 (the per-pixel spectral index) is preserved when the
        # reference frequency changes.
        adjust_spectral_reference(
            model_paths[1],
            old_ref_hz,
            new_ref_hz,
            alpha_map=alpha_map,
        )
    else:
        # nterms=1: spectrally flat, only set the spectral CRVAL
        adjust_spectral_reference(
            model_paths[0],
            old_ref_hz,
            new_ref_hz,
            alpha_map=None,
        )

    ctx.add_milestone(
        "adjusted_spectral_reference",
        "completed",
        details={
            "model_type": "casa_taylor_terms",
            "old_reference_frequency_hz": old_ref_hz,
            "new_reference_frequency_hz": new_ref_hz,
            "nterms": nterms,
        },
    )

    return CasaModelProduct(
        model_paths=model_paths,
        nterms=nterms,
        reffreq=f"{new_ref_hz}Hz",
        intermediates=model_paths,
    )


def prepare_continuum_i_alpha_for_casa(
    ctx: RunContext,
    entry: ContinuumIAlphaModelEntry,
    index: int,
) -> CasaModelProduct:
    """Create CASA image products for a continuum I+alpha model.

    Adjusts the spectral reference to the observation band centre using
    the explicit spectral index from the model entry.
    """
    new_ref_hz = ctx.config.observation.frequency_mhz * 1e6
    old_ref_hz = entry.reference_frequency_hz

    stokes_path = Path(entry.stokes_i).expanduser().resolve()
    alpha_path = Path(entry.alpha).expanduser().resolve()
    prefix = f"model_entry_{index + 1:02d}_continuum"
    tt0_fits = ctx.work_dir / f"{prefix}.tt0.fits"
    tt1_fits = ctx.work_dir / f"{prefix}.tt1.fits"
    tt0_image = ctx.work_dir / f"{prefix}.tt0.image"
    tt1_image = ctx.work_dir / f"{prefix}.tt1.image"

    with fits.open(stokes_path) as stokes_hdul, fits.open(alpha_path) as alpha_hdul:
        stokes_data = np.asarray(stokes_hdul[0].data, dtype=float)
        alpha_data = np.asarray(alpha_hdul[0].data, dtype=float)
        header = stokes_hdul[0].header.copy()
        header["BUNIT"] = stokes_hdul[0].header.get("BUNIT", "Jy/pixel")

        # ensure both model FITS have a degenerate 4D shape (STOKES, FREQ)
        # so CASA importfits creates images with a frequency axis that ft can map.
        stokes_4d, header_4d = _as_4d_image(stokes_data, header, new_ref_hz)
        alpha_4d = _broadcast_to_4d(alpha_data)

        # Scale data to the new reference frequency
        factor = (new_ref_hz / old_ref_hz) ** alpha_4d
        stokes_4d_scaled = stokes_4d * factor
        # CASA Taylor terms: tt0 is I(nu_ref), tt1 is I(nu_ref) * alpha
        tt1_data = stokes_4d_scaled * alpha_4d

        fits.writeto(tt0_fits, stokes_4d_scaled, header=header_4d, overwrite=True)
        fits.writeto(tt1_fits, tt1_data, header=header_4d, overwrite=True)

    for imagename in (tt0_image, tt1_image):
        if imagename.exists():
            shutil.rmtree(imagename)
    try:
        from casatasks import importfits

        importfits(fitsimage=str(tt0_fits), imagename=str(tt0_image), overwrite=True)
        importfits(fitsimage=str(tt1_fits), imagename=str(tt1_image), overwrite=True)
    except Exception as exc:
        logger.debug(
            f"prepare_continuum_i_alpha_for_casa: in-process importfits unavailable: {exc}"
        )
        run_casa_importfits(
            ctx.work_dir,
            [(tt0_fits, tt0_image), (tt1_fits, tt1_image)],
        )

    # Pixel data was already scaled; we just use adjust_spectral_reference
    # with alpha_map=None to log the new frequency and enforce the spectral CRVAL.
    alpha_mean = float(np.mean(alpha_data))
    adjust_spectral_reference(
        tt0_image,
        old_ref_hz,
        new_ref_hz,
        alpha_map=None,
    )
    adjust_spectral_reference(
        tt1_image,
        old_ref_hz,
        new_ref_hz,
        alpha_map=None,
    )

    ctx.add_milestone(
        "adjusted_spectral_reference",
        "completed",
        details={
            "model_type": "continuum_i_alpha",
            "old_reference_frequency_hz": old_ref_hz,
            "new_reference_frequency_hz": new_ref_hz,
            "alpha_mean": alpha_mean,
            "nterms": 2,
        },
    )

    return CasaModelProduct(
        model_paths=[tt0_image, tt1_image],
        nterms=2,
        reffreq=f"{new_ref_hz}Hz",
        intermediates=[tt0_fits, tt1_fits],
    )


def _as_4d_image(
    data: np.ndarray, header: fits.Header, reference_freq_hz: float
) -> tuple[np.ndarray, fits.Header]:
    """Return a 4D view (FREQ, STOKES, Y, X) of a 2D FITS image plus an updated header."""
    # collapse any leading degenerate axes and validate spatial dimensions
    flat = np.asarray(data).squeeze()
    if flat.ndim != 2:
        raise ValueError(
            f"continuum_i_alpha expects a 2D spatial image; got shape {data.shape}"
        )
    ny, nx = flat.shape
    image_4d = flat.reshape(1, 1, ny, nx)

    new_header = header.copy()
    new_header["NAXIS"] = 4
    new_header["NAXIS1"] = nx
    new_header["NAXIS2"] = ny
    new_header["NAXIS3"] = 1
    new_header["NAXIS4"] = 1
    new_header["CTYPE1"] = header.get("CTYPE1", "RA---SIN")
    new_header["CTYPE2"] = header.get("CTYPE2", "DEC--SIN")
    new_header["CTYPE3"] = "STOKES"
    new_header["CTYPE4"] = "FREQ"
    new_header["CRPIX1"] = header.get("CRPIX1", nx / 2.0)
    new_header["CRPIX2"] = header.get("CRPIX2", ny / 2.0)
    new_header["CRPIX3"] = 1.0
    new_header["CRPIX4"] = 1.0
    new_header["CRVAL3"] = 1.0
    new_header["CRVAL4"] = reference_freq_hz
    new_header["CDELT3"] = 1.0
    new_header["CDELT4"] = 1.0
    new_header["CUNIT3"] = ""
    new_header["CUNIT4"] = "Hz"
    return image_4d, new_header


def _broadcast_to_4d(data: np.ndarray) -> np.ndarray:
    """Collapse leading degenerate axes and broadcast a 2D spatial map to 4D."""
    flat = np.asarray(data).squeeze()
    if flat.ndim != 2:
        raise ValueError(
            f"continuum_i_alpha alpha map must be 2D spatial; got shape {data.shape}"
        )
    return flat.reshape(1, 1, *flat.shape)


def adjust_spectral_reference(
    image_path: Path,
    old_ref_hz: float,
    new_ref_hz: float,
    alpha_map: np.ndarray | None = None,
) -> float:
    """Adjust the spectral reference of a CASA image to the observation band centre.

    For nterms=1 (alpha_map is None), set the spectral CRVAL to new_ref_hz only — the model
    is spectrally flat and no pixel-data correction is needed.

    For nterms≥2 (alpha_map provided), correct the pixel data element-wise following
    CASA's Taylor-series convention::

        tt0'(x,y) = tt0(x,y) · (ν_new / ν_old) ^ α(x,y)

    where α(x,y) = tt1(x,y) / tt0(x,y)   for each pixel.  The spectral CRVAL is also set.

    Returns the adjusted reference frequency in Hz (always new_ref_hz).
    """
    if alpha_map is not None:
        casacore_table = require_casacore()

        # element-wise scaling
        factor = (new_ref_hz / old_ref_hz) ** alpha_map
        with casacore_table(str(image_path), readonly=False, ack=False) as tbl:
            data = np.asarray(tbl.getcol("map"))
            corrected = data * factor
            tbl.putcol("map", corrected)
        logger.info(
            f"Adjusted pixel data in {image_path}: ref_freq {old_ref_hz:.3e}Hz → "
            f"{new_ref_hz:.3e}Hz (factor range: [{np.min(factor):.4f}, {np.max(factor):.4f}])"
        )
    else:
        logger.info(
            f"Spectrally flat image {image_path}: updating spectral CRVAL = {new_ref_hz:.3e}Hz "
            "(no pixel-data correction)"
        )

    _set_crval4_via_script(image_path.parent, [image_path], new_ref_hz)
    return new_ref_hz


def _spectral_axis_number(image_path: Path) -> Optional[int]:
    """Return the 1-based spectral pixel axis for imhead's crvalN, or None.

    CASA images usually have RA/Dec/Stokes/Frequency axes, while FITS-derived
    images can have RA/Dec/Frequency/Stokes axes. Read the coordinate mapping
    rather than assuming frequency is on axis 4.
    """
    try:
        table = require_casacore()
    except Exception as exc:
        raise RuntimeError(
            "python-casacore is required to inspect CASA image tables."
        ) from exc

    with table(str(image_path), readonly=True, ack=False) as tbl:
        if "coords" not in tbl.keywordnames():
            return None
        coords = tbl.getkeyword("coords")
    for key in coords:
        if not key.startswith("spectral"):
            continue
        # Coordinate N maps onto pixelmapN; -1 denotes a removed pixel axis.
        pixel_axes = list(coords.get(f"pixelmap{key[len('spectral') :]}", []))
        if len(pixel_axes) == 1 and int(pixel_axes[0]) >= 0:
            return int(pixel_axes[0]) + 1
    return None


def _set_crval4_via_script(
    work_dir: Path,
    image_paths: list[Path],
    frequency_hz: float,
) -> None:
    """Set the spectral CRVAL on CASA images, with a subprocess fallback."""
    # 2D images have no spectral axis; CASA ft will treat them as spectrally flat
    # with reffreq carrying the reference frequency, so no spectral CRVAL is needed.
    axes = [(path, _spectral_axis_number(path)) for path in image_paths]
    axes = [(path, axis) for path, axis in axes if axis is not None]
    if not axes:
        logger.warning(
            "Spectrally flat 2D image(s): skipping spectral CRVAL update; ft reffreq carries the reference frequency"
        )
        return

    try:
        from casatasks import imhead

        for image_path, axis in axes:
            imhead(
                imagename=str(image_path),
                mode="put",
                hdkey=f"crval{axis}",
                hdvalue=f"{frequency_hz}Hz",
            )
        logger.info(
            f"Set spectral CRVAL in-process for {len(axes)} image(s) using casatasks.imhead"
        )
        return
    except Exception as exc:
        logger.debug(f"_set_crval4_via_script: in-process imhead failed: {exc}")
        # fall through to batch mode

    logger.info(
        f"Set spectral CRVAL falling back to CASA batch mode for {len(axes)} image(s)"
    )
    run_casa_set_spectral_coordinate(work_dir, [path for path, _ in axes], frequency_hz)


def require_casa_executable() -> Path:
    """Return a CASA executable for batch-mode fallback."""
    executable = shutil.which("casa")
    if executable is None:
        raise RuntimeError(
            "CASA image-model injection requires either importable casatasks "
            "or a casa executable on PATH."
        )
    return Path(executable)


def run_casa_importfits(
    work_dir: Path,
    images: list[tuple[Path, Path]],
) -> None:
    """Run CASA importfits in batch mode for prepared FITS images."""
    executable = require_casa_executable()
    script_path = work_dir / "skasim_casa_importfits.py"
    lines = [
        "# Auto-generated by skasim; safe to delete after the run.",
        "try:",
        "    from casatasks import importfits",
        "except Exception as _e:",
        "    raise RuntimeError('casatasks.importfits is not available: ' + str(_e))",
    ]
    for fitsimage, imagename in images:
        lines.append(
            "importfits(fitsimage={!r}, imagename={!r}, overwrite=True)".format(
                str(fitsimage),
                str(imagename),
            )
        )
    run_casa_script(executable, script_path, lines)
    # verify side-effects; if not present, casatasks ran but failed silently
    for _fitsimage, imagename in images:
        table_dat = imagename / "table.dat"
        if not table_dat.exists():
            raise RuntimeError(
                f"CASA importfits did not create {imagename} as expected."
            )


def run_casa_exportfits(
    work_dir: Path,
    imagename: Path,
    fitsimage: Path,
) -> None:
    """Run CASA exportfits in batch mode for a CASA image table."""
    executable = require_casa_executable()
    script_path = work_dir / "skasim_casa_exportfits.py"
    lines = [
        "# Auto-generated by skasim; safe to delete after the run.",
        "try:",
        "    from casatasks import exportfits",
        "except Exception:",
        "    pass",
        "exportfits(imagename={!r}, fitsimage={!r}, overwrite=True)".format(
            str(imagename),
            str(fitsimage),
        ),
    ]
    logger.info(
        f"CASA exportfits falling back to batch mode for {imagename} -> {fitsimage}"
    )
    run_casa_script(executable, script_path, lines)


def run_casa_set_spectral_coordinate(
    work_dir: Path,
    image_paths: list[Path],
    frequency_hz: float,
) -> None:
    """Set the single-channel spectral coordinate of CASA images to the run reference."""
    executable = require_casa_executable()
    script_path = work_dir / "skasim_casa_set_spectral_coordinate.py"
    hdvalue = f"{frequency_hz}Hz"
    lines = [
        "# Auto-generated by skasim; safe to delete after the run.",
        "try:",
        "    from casatasks import imhead",
        "except Exception:",
        "    pass",
    ]
    for image_path in image_paths:
        axis = _spectral_axis_number(image_path)
        if axis is None:
            continue
        lines.append(
            "imhead(imagename={!r}, mode='put', hdkey={!r}, hdvalue={!r})".format(
                str(image_path),
                f"crval{axis}",
                hdvalue,
            )
        )
    run_casa_script(executable, script_path, lines)


def run_casa_ft(
    visibility_path: Path,
    model_paths: list[Path],
    nterms: int,
    reffreq: str,
    incremental: bool,
) -> None:
    """Run CASA ft into MODEL_DATA for one prepared model entry."""
    logger.info(
        f"CASA ft model={[str(path) for path in model_paths]} "
        f"nterms={nterms} reffreq={reffreq} incremental={incremental}"
    )
    try:
        from casatasks import ft

        ft(
            vis=str(visibility_path),
            model=[str(path) for path in model_paths],
            nterms=nterms,
            reffreq=reffreq,
            incremental=incremental,
            usescratch=True,
        )
        return
    except Exception as exc:
        logger.debug(f"run_casa_ft: in-process ft unavailable: {exc}")

    executable = require_casa_executable()
    script_path = visibility_path.parent / "skasim_casa_ft.py"
    model_literal = "[" + ", ".join(repr(str(path)) for path in model_paths) + "]"
    lines = [
        "# Auto-generated by skasim; safe to delete after the run.",
        "try:",
        "    from casatasks import ft",
        "except Exception as _e:",
        "    raise RuntimeError('casatasks.ft is not available: ' + str(_e))",
        "ft(",
        f"    vis={str(visibility_path)!r},",
        f"    model={model_literal},",
        f"    nterms={int(nterms)},",
        f"    reffreq={reffreq!r},",
        f"    incremental={bool(incremental)!r},",
        "    usescratch=True,",
        ")",
    ]
    run_casa_script(executable, script_path, lines)


def run_casa_script(
    executable: Path,
    script_path: Path,
    lines: list[str],
    timeout_s: float = 3600.0,
) -> None:
    """Write and execute one CASA batch script, surfacing useful failure output."""
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_path = script_path.with_suffix(".log")
    command = [
        str(executable),
        "--nologger",
        "--nogui",
        "--log2term",
        "-c",
        str(script_path),
    ]
    logger.info(f"CASA batch command: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            result = subprocess.run(
                command,
                cwd=str(script_path.parent),
                text=True,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
            raise RuntimeError(
                f"CASA batch command timed out after {timeout_s} s: {script_path}\n{tail}"
            ) from exc
        if result.returncode != 0:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
            raise RuntimeError(
                f"CASA batch command failed with exit code {result.returncode}: "
                f"{script_path}\n{tail}"
            )

    # CASA tasks can log SEVERE failures yet exit zero; failed ft can leave
    # MODEL_DATA at unity (a spurious 1 Jy point source at the phase centre).
    severe = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "\tSEVERE\t" in line
    ]
    if severe:
        raise RuntimeError(
            f"CASA reported SEVERE errors in {script_path} (see {log_path}):\n"
            + "\n".join(severe[-20:])
        )


def merge_model_data_into_data(
    visibility_path: Path, chunk_bytes: int = 512 * 1024**2
) -> None:
    """Add image-model MODEL_DATA into the delivered DATA column.

    Rows are processed in chunks of roughly ``chunk_bytes`` per column so
    peak memory stays bounded for long observations and wide cubes.
    """
    try:
        table = require_casacore()
    except Exception as exc:
        raise RuntimeError(
            "python-casacore is required to merge MODEL_DATA into DATA."
        ) from exc

    with table(str(visibility_path), readonly=False, ack=False) as ms_table:
        columns = set(ms_table.colnames())
        if "DATA" not in columns or "MODEL_DATA" not in columns:
            raise ValueError(
                f"{visibility_path} must contain DATA and MODEL_DATA columns "
                "after CASA ft injection."
            )
        n_rows = ms_table.nrows()
        if n_rows == 0:
            return
        row_bytes = max(np.asarray(ms_table.getcell("DATA", 0)).nbytes, 1)
        chunk_rows = max(1, chunk_bytes // row_bytes)
        for start in range(0, n_rows, chunk_rows):
            nrow = min(chunk_rows, n_rows - start)
            data = ms_table.getcol("DATA", startrow=start, nrow=nrow)
            data += ms_table.getcol("MODEL_DATA", startrow=start, nrow=nrow)
            ms_table.putcol("DATA", data, startrow=start, nrow=nrow)
