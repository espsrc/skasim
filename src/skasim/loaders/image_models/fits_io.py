"""FITS image-model metadata reading and validation."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from loguru import logger

from ...config import (
    CasaTaylorTermsModelEntry,
    ComponentSkyModelEntry,
    ContinuumIAlphaModelEntry,
    ImgConfig,
    ModelEntry,
    ObsConfig,
    SimConfig,
    SpectralCubeModelEntry,
    StaticStokesMapsModelEntry,
)
from ...image_geometry import MIN_IMAGE_PIXELS
from ...primary_beam import GaussianStationBeam, angular_separation_deg

# suppress fits formatting fixes
warnings.simplefilter("ignore", category=FITSFixedWarning)
# suppress polar motion fallback warnings
warnings.filterwarnings("ignore", message=".*polar motions.*")


def component_model_entries(config: SimConfig) -> list[ComponentSkyModelEntry]:
    return [
        entry for entry in config.models if isinstance(entry, ComponentSkyModelEntry)
    ]


def image_model_entries(config: SimConfig) -> list[ModelEntry]:
    return [
        entry
        for entry in config.models
        if isinstance(
            entry,
            (
                ContinuumIAlphaModelEntry,
                CasaTaylorTermsModelEntry,
                StaticStokesMapsModelEntry,
                SpectralCubeModelEntry,
            ),
        )
    ]


def image_model_center(entries: list[ModelEntry]) -> SkyCoord | None:
    """Return the centre of the first image model with usable celestial WCS."""
    for entry in entries:
        path = primary_model_fits_path(entry)
        if path is None:
            continue
        try:
            info = read_fits_image_info(path)
        except Exception as exc:
            logger.debug(f"image_model_center: failed to read {path}: {exc}")
            continue
        if info.center is not None:
            return info.center
    return None


def primary_model_fits_path(entry: ModelEntry) -> Path | None:
    """Return the representative FITS image for previews and phase-centre inference."""
    if isinstance(entry, ContinuumIAlphaModelEntry):
        return Path(entry.stokes_i).expanduser().resolve()
    if isinstance(entry, StaticStokesMapsModelEntry):
        for value in (entry.stokes_i, entry.stokes_q, entry.stokes_u, entry.stokes_v):
            if value:
                return Path(value).expanduser().resolve()
    return None


def image_model_footprints(
    entries: list[ModelEntry],
    center: SkyCoord,
    beam: GaussianStationBeam,
    field_fov_deg: float,
) -> list[dict]:
    """Return each FITS image model's footprint relative to the phase centre.

    The footprint radius is the largest angular distance of any pixel-edge
    corner from the phase centre. It is compared with the shared beam's
    half-power radius at its reference frequency and with the simulation
    field radius, so heavily attenuated components can be flagged.
    """
    half_power_radius = float(beam.power_fwhm_deg(beam.ref_freq_hz)) / 2.0
    field_radius = float(field_fov_deg) / 2.0
    records = []
    for index, entry in enumerate(entries):
        path = primary_model_fits_path(entry)
        if path is None:
            continue
        try:
            header = fits.getheader(path)
            wcs = WCS(header).celestial
            nx = int(header["NAXIS1"])
            ny = int(header["NAXIS2"])
            xs = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
            ys = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
            ra, dec = wcs.all_pix2world(xs, ys, 0)
            centre_ra, centre_dec = wcs.all_pix2world((nx - 1) / 2.0, (ny - 1) / 2.0, 0)
        except Exception as exc:
            logger.debug(f"image_model_footprints: cannot read {path}: {exc}")
            continue
        icrs = center.icrs
        radius = float(
            np.max(angular_separation_deg(ra, dec, icrs.ra.deg, icrs.dec.deg))
        )
        offset = float(
            angular_separation_deg(centre_ra, centre_dec, icrs.ra.deg, icrs.dec.deg)
        )
        records.append(
            {
                "model_entry_index": index,
                "model_type": entry.type,
                "source": str(path),
                "centre_offset_deg": offset,
                "footprint_radius_deg": radius,
                "beam_half_power_radius_deg": half_power_radius,
                "field_radius_deg": field_radius,
                "beyond_beam_half_power": bool(radius > half_power_radius),
                "beyond_field": bool(radius > field_radius),
            }
        )
    return records


@dataclass(frozen=True)
class FitsCubeInfo:
    """Small summary of an accepted 3D spectral-cube model."""

    path: Path
    shape: tuple[int, int, int]
    spatial_shape: tuple[int, int]
    unit: str
    n_channels: int
    channel_width_hz: float
    start_frequency_hz: float
    reference_frequency_hz: float


@dataclass(frozen=True)
class FitsImageInfo:
    """Small summary of one accepted FITS image model plane."""

    path: Path
    spatial_shape: tuple[int, int]
    unit: str | None
    celestial_header: dict[str, object]
    center: SkyCoord | None


def _select_wsclean_img_config(imaging_configs: list[ImgConfig]) -> ImgConfig:
    """Return the first wsclean imaging config, or the first config if none match."""
    return next(
        (img for img in imaging_configs if img.imager == "wsclean"),
        imaging_configs[0],
    )


def read_fits_image_info(path: Path) -> FitsImageInfo:
    """Read FITS image metadata used by validation and reporting."""
    with fits.open(path) as hdul:
        hdu = hdul[0]
        if hdu.data is None:
            raise ValueError(f"{path} has no image data")
        data = np.asarray(hdu.data).squeeze()
        if data.ndim < 2:
            raise ValueError(f"{path} is not a spatial FITS image")
        spatial_shape = tuple(int(v) for v in data.shape[-2:])
        header = hdu.header.copy()
        unit = header.get("BUNIT")

    try:
        wcs = WCS(header).celestial
        celestial_header = dict(wcs.to_header())
        center_y = (spatial_shape[0] - 1) / 2.0
        center_x = (spatial_shape[1] - 1) / 2.0
        center = wcs.pixel_to_world(center_x, center_y)
        if not isinstance(center, SkyCoord):
            center = None
    except Exception as exc:
        logger.debug(f"read_fits_image_info: WCS construction failed for {path}: {exc}")
        celestial_header = {}
        center = None

    return FitsImageInfo(
        path=path,
        spatial_shape=spatial_shape,
        unit=unit,
        celestial_header=celestial_header,
        center=center,
    )


# canonical aliases accepted for Jy/pixel (per-pixel flux density) inputs.
_ACCEPTED_JY_PER_PIXEL_UNITS = frozenset(
    {
        "jy/pixel",
        "jy pix-1",
        "jy/pix",
        "jy",
        "jy px-1",
        "jy/px",
        "jy pixels-1",
        "jy pixel-1",
        "jy pixel^-1",
        "jy pix^-1",
        "jy px^-1",
    }
)


def validate_static_stokes_maps(
    entry: StaticStokesMapsModelEntry,
    obs: ObsConfig,
    img_config: ImgConfig,
) -> dict:
    """Validate a static Stokes I map and return report metadata.

    Currently only ``stokes_i`` is supported. The image must be a 2D spatial
    FITS file with Jy/pixel-compatible BUNIT.
    """
    stokes_info = read_fits_image_info(Path(entry.stokes_i).expanduser().resolve())
    with fits.open(stokes_info.path) as hdul:
        hdu = hdul[0]
        raw_data = np.asarray(hdu.data)  # type: ignore[union-attr]
        data = raw_data.squeeze()
        # reject genuine 3D input before squeezing degenerate axes
        if raw_data.ndim > 2:
            raise ValueError(
                f"{stokes_info.path} must be a 2D spatial image; "
                f"got shape {raw_data.shape}"
            )
    if data.ndim != 2:
        raise ValueError(
            f"{stokes_info.path} must be a 2D spatial image; got shape {data.shape}"
        )
    unit = (stokes_info.unit or "").strip().lower()
    if unit not in _ACCEPTED_JY_PER_PIXEL_UNITS:
        raise ValueError(
            f"{stokes_info.path} must declare Jy/pixel-compatible BUNIT; "
            f"found {stokes_info.unit!r}"
        )
    if stokes_info.spatial_shape[0] != stokes_info.spatial_shape[1]:
        logger.warning(
            "static_stokes_maps model image is not square ({}x{}); "
            "WSClean predict will use {} pixels for both dimensions.",
            stokes_info.spatial_shape[1],
            stokes_info.spatial_shape[0],
            stokes_info.spatial_shape[1],
        )
    return {
        "stokes_i": str(stokes_info.path),
        "spatial_shape": list(stokes_info.spatial_shape),
        "unit": stokes_info.unit,
        "n_channels": obs.n_channels,
        "bandwidth_mhz": obs.bandwidth_mhz,
        "frequency_mhz": obs.frequency_mhz,
        "imager": img_config.imager,
    }


def validate_continuum_i_alpha(entry: ContinuumIAlphaModelEntry) -> dict:
    """Validate the continuum image contract and return report metadata."""
    stokes_info = read_fits_image_info(Path(entry.stokes_i).expanduser().resolve())
    alpha_info = read_fits_image_info(Path(entry.alpha).expanduser().resolve())
    if stokes_info.spatial_shape != alpha_info.spatial_shape:
        raise ValueError(
            "continuum_i_alpha requires matching spatial dimensions: "
            f"{stokes_info.path} has {stokes_info.spatial_shape}, "
            f"{alpha_info.path} has {alpha_info.spatial_shape}"
        )
    if stokes_info.celestial_header != alpha_info.celestial_header:
        raise ValueError("continuum_i_alpha requires matching celestial WCS.")
    unit = (stokes_info.unit or "").strip().lower()
    if unit not in _ACCEPTED_JY_PER_PIXEL_UNITS:
        raise ValueError(
            f"{stokes_info.path} must declare Jy/pixel-compatible BUNIT; "
            f"found {stokes_info.unit!r}"
        )
    alpha_unit = (alpha_info.unit or "").strip().lower()
    if alpha_unit not in {"", "1", "dimensionless", "none"}:
        raise ValueError(
            f"{alpha_info.path} must be dimensionless; found BUNIT={alpha_info.unit!r}"
        )
    return {
        "stokes_i": str(stokes_info.path),
        "alpha": str(alpha_info.path),
        "spatial_shape": list(stokes_info.spatial_shape),
        "unit": stokes_info.unit,
        "reference_frequency_hz": entry.reference_frequency_hz,
    }


# ---------------------------------------------------------------------------
# spectral cube helpers
# ---------------------------------------------------------------------------


def _find_frequency_axis(header: fits.Header) -> int:
    """Return the 1-based FITS axis index whose CTYPE is FREQ."""
    naxis = int(header.get("NAXIS", 3))
    for axis in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{axis}", "")).strip().upper()
        if ctype.startswith("FREQ"):
            return axis
    raise ValueError("spectral cube has no FREQ axis in CTYPE1..CTYPEn")


def _fits_axis_to_numpy(axis: int, ndim: int = 3) -> int:
    """Return the NumPy axis index for a 1-based FITS axis number.

    FITS axis ``k`` is stored as the ``ndim - k`` NumPy axis (the ``k``-th axis
    from the slowest-varying end).  For a 3D cube this means:
    ``NAXIS1`` -> axis 2, ``NAXIS2`` -> axis 1, ``NAXIS3`` -> axis 0.
    """
    return ndim - axis


def _squeeze_degenerate_axes(
    data: np.ndarray, header: fits.Header
) -> tuple[np.ndarray, fits.Header]:
    """Drop length-1 axes from a FITS cube, keeping the spectral axis intact."""
    if data.ndim == 3:
        return data, header
    if data.ndim < 3 or data.ndim > 4:
        raise ValueError(
            f"spectral_cube must be 3D or 4D with a degenerate axis, got ndim={data.ndim}"
        )

    freq_axis = _find_frequency_axis(header)
    single_axes = [
        axis for axis in range(1, data.ndim + 1) if header[f"NAXIS{axis}"] == 1
    ]

    if not single_axes:
        raise ValueError(
            f"spectral_cube has {data.ndim} dimensions but no degenerate (length-1) axis to squeeze"
        )

    if len(single_axes) > 1:
        raise ValueError(
            f"spectral_cube has multiple degenerate axes {single_axes}; cannot disambiguate"
        )

    squeeze_axis = single_axes[0]
    if squeeze_axis == freq_axis:
        raise ValueError(
            f"spectral_cube frequency axis (FREQ in FITS axis {freq_axis}) has length 1; "
            "cannot squeeze the spectral axis"
        )

    np_axis = _fits_axis_to_numpy(squeeze_axis, data.ndim)
    data = np.squeeze(data, axis=np_axis)

    new_header = header.copy()
    new_header["NAXIS"] = 3

    old_axes = [a for a in range(1, data.ndim + 2) if a != squeeze_axis]
    for new_axis, old_axis in enumerate(old_axes, start=1):
        for key in ("NAXIS", "CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT"):
            old_key = f"{key}{old_axis}"
            new_key = f"{key}{new_axis}"
            if old_key in new_header:
                new_header[new_key] = new_header[old_key]

    for key in list(new_header.keys()):
        match = re.match(r"(NAXIS|CTYPE|CRPIX|CRVAL|CDELT|CUNIT)(\d+)", key)
        if match:
            axis_num = int(match.group(2))
            if axis_num > 3:
                del new_header[key]

    return data, new_header


def read_fits_cube_info(path: Path) -> FitsCubeInfo:
    """Read metadata from a 3D FITS spectral cube."""
    with fits.open(path) as hdul:
        hdu = hdul[0]
        if hdu.data is None:
            raise ValueError(f"{path} has no image data")
        data = np.asarray(hdu.data)
        header = hdu.header.copy()
        data, header = _squeeze_degenerate_axes(data, header)
        if data.ndim != 3:
            raise ValueError(f"spectral_cube must be 3D, got ndim={data.ndim}")
        freq_axis = _find_frequency_axis(header)
        n_freq = int(header[f"NAXIS{freq_axis}"])
        freqs = _freq_axis_centres(header, n_freq, axis=freq_axis)
        channel_width_hz = float(np.diff(freqs).mean()) if n_freq > 1 else 0.0
        unit = str(header.get("BUNIT") or "Jy/px").strip().lower()

        spatial_shape = [None, None]
        for axis in range(1, 4):
            if axis == freq_axis:
                continue
            ctype = str(header.get(f"CTYPE{axis}", "")).strip().upper()
            if ctype.startswith("RA") or ctype.startswith("GLON"):
                spatial_shape[0] = int(header[f"NAXIS{axis}"])
            elif ctype.startswith("DEC") or ctype.startswith("GLAT"):
                spatial_shape[1] = int(header[f"NAXIS{axis}"])
        if None in spatial_shape:
            raise ValueError(
                "spectral cube spatial axes are not labelled as RA/DEC or GLON/GLAT"
            )

    return FitsCubeInfo(
        path=path,
        shape=tuple(int(v) for v in data.shape),
        spatial_shape=tuple(spatial_shape),
        unit=unit,
        n_channels=n_freq,
        channel_width_hz=channel_width_hz,
        start_frequency_hz=float(freqs[0]),
        reference_frequency_hz=float(freqs[n_freq // 2]),
    )


def _freq_axis_centres(
    header: fits.Header, n_channels: int, axis: int = 3
) -> np.ndarray:
    """Return the frequency axis centre positions in Hz for the given 1-based axis."""
    crpix = float(header.get(f"CRPIX{axis}", 1.0))
    crval = float(header[f"CRVAL{axis}"])
    cdelt = float(header[f"CDELT{axis}"])
    cunit = str(header.get(f"CUNIT{axis}") or "").strip().lower()
    if cunit == "mhz":
        crval *= 1e6
        cdelt *= 1e6
    return crval + (np.arange(n_channels) - (crpix - 1.0)) * cdelt


def _reorder_cube_axes(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Reorder a 3D FITS array so its numpy axes become (freq, dec, ra)."""
    ndim = data.ndim
    axis_map: dict[str, int] = {}
    for axis in range(1, ndim + 1):
        ctype = str(header.get(f"CTYPE{axis}", "")).strip().upper()
        if ctype.startswith("FREQ"):
            label = "freq"
        elif ctype.startswith("RA") or ctype.startswith("GLON"):
            label = "ra"
        elif ctype.startswith("DEC") or ctype.startswith("GLAT"):
            label = "dec"
        else:
            raise ValueError(f"spectral cube has unsupported CTYPE{axis}={ctype!r}")
        axis_map[label] = _fits_axis_to_numpy(axis, ndim)

    for label in ("freq", "dec", "ra"):
        if label not in axis_map:
            raise ValueError(f"spectral cube is missing {label} axis")

    target_order = ["freq", "dec", "ra"]
    source_axes = [axis_map[label] for label in target_order]
    return np.moveaxis(data, source_axes, [0, 1, 2])


def _strip_spectral_axis_from_header(header: fits.Header) -> fits.Header:
    """Return a 2D header with the spectral (axis 3) keys removed."""
    out_header = header.copy()
    for key in list(out_header.keys()):
        if key in ("NAXIS", "NAXIS3") or key.startswith("NAXIS3"):
            out_header.remove(key, ignore_missing=True)
        if key in ("CRPIX3", "CRVAL3", "CDELT3", "CTYPE3", "CUNIT3"):
            out_header.remove(key, ignore_missing=True)
    out_header["NAXIS"] = 2
    out_header["NAXIS1"] = header["NAXIS1"]
    out_header["NAXIS2"] = header["NAXIS2"]
    return out_header


def validate_spectral_cube(
    entry: SpectralCubeModelEntry,
    obs: ObsConfig,
    img: ImgConfig,
) -> dict:
    """Validate the spectral-cube contract against observation/imaging config.

    The cube's spatial dimensions are checked against the user's explicit
    ``pixels`` setting when present.  If ``pixels`` was not supplied (e.g. only
    ``fov_deg`` was given and skasim derives the image size automatically), the
    cube dimensions are accepted as the authoritative model geometry and a
    warning is recorded so the run log remains traceable.
    """
    path = Path(entry.cube).expanduser().resolve()
    info = read_fits_cube_info(path)

    if info.unit not in _ACCEPTED_JY_PER_PIXEL_UNITS:
        raise ValueError(
            f"{path} must declare Jy/pixel-compatible BUNIT; found {info.unit!r}"
        )

    spatial_shape = info.spatial_shape
    if spatial_shape[0] != spatial_shape[1]:
        raise ValueError(
            f"spectral_cube spatial dimensions {spatial_shape} must be square"
        )
    if spatial_shape[0] < MIN_IMAGE_PIXELS:
        raise ValueError(
            f"spectral_cube spatial dimension {spatial_shape[0]} is below the "
            f"minimum of {MIN_IMAGE_PIXELS} pixels"
        )

    explicit_pixels = "pixels" in img.model_fields_set
    if explicit_pixels and spatial_shape != (img.pixels, img.pixels):
        raise ValueError(
            f"spectral_cube spatial dimensions {spatial_shape} do not match "
            f"the explicitly requested imaging pixels {img.pixels}"
        )

    warnings: list[str] = []
    if not explicit_pixels:
        warnings.append(
            f"spectral_cube spatial dimensions {spatial_shape} override the "
            f"auto-derived imaging geometry because no explicit 'pixels' value was supplied"
        )

    if (
        obs.bandwidth_mhz is None
        or obs.n_channels is None
        or obs.channel_width_mhz is None
    ):
        raise ValueError("observation spectral grid is incomplete")

    obs_bw_hz = obs.bandwidth_mhz * 1e6
    obs_center_hz = obs.frequency_mhz * 1e6
    obs_min_hz = obs_center_hz - obs_bw_hz / 2.0
    obs_max_hz = obs_center_hz + obs_bw_hz / 2.0

    n_channels = info.n_channels
    channel_width_hz = info.channel_width_hz
    cube_center_hz = info.start_frequency_hz + (n_channels - 1) * channel_width_hz / 2.0
    cube_min_hz = cube_center_hz - n_channels * channel_width_hz / 2.0
    cube_max_hz = cube_center_hz + n_channels * channel_width_hz / 2.0

    edge_tol_hz = 0.01 * obs_bw_hz
    if cube_min_hz < obs_min_hz - edge_tol_hz or cube_max_hz > obs_max_hz + edge_tol_hz:
        raise ValueError(
            f"spectral_cube frequency range [{cube_min_hz:.3e}, {cube_max_hz:.3e}] Hz "
            f"extends beyond the observation band "
            f"[{obs_min_hz:.3e}, {obs_max_hz:.3e}] Hz"
        )

    logger.info(
        f"validate_spectral_cube: cube {n_channels} channels x {channel_width_hz:.3e} Hz "
        f"covering [{cube_min_hz:.3e}, {cube_max_hz:.3e}] Hz inside observation band "
        f"[{obs_min_hz:.3e}, {obs_max_hz:.3e}] Hz"
    )

    return {
        "cube": str(path),
        "shape": info.shape,
        "spatial_shape": list(spatial_shape),
        "unit": info.unit,
        "n_channels": n_channels,
        "channel_width_hz": channel_width_hz,
        "reference_frequency_hz": cube_center_hz,
        "frequency_range_hz": [cube_min_hz, cube_max_hz],
        "warnings": warnings,
    }
