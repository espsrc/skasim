"""pipeline.py — end-to-end interferometric simulation orchestrator."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from loguru import logger

from .config import ImgConfig, SimConfig, has_spectral_cube_model
from .image_geometry import ImageGeometry
from .imaging import run_dirty_imaging, run_wsclean_imaging
from .loaders import (
    FitsCatalogLoader,
    component_model_entries,
    image_model_center,
    image_model_entries,
    image_model_footprints,
    inject_image_models,
    write_image_model_previews,
    write_sky_composition_preview,
)
from .manifest import RunContext, create_run_context
from .primary_beam import GaussianStationBeam, angular_separation_deg, simulation_beam
from .runtime import require_karabo_module
from .sky import SkyModel, Source
from .utils import build_shadems_uv_coverage_argv, get_diameter, run_shadems_command

# --------------------------------------------------------------------------- #
# workdir + logging
# --------------------------------------------------------------------------- #


# (setup_workdir moved to create_run_context in manifest.py)


# --------------------------------------------------------------------------- #
# telescope
# --------------------------------------------------------------------------- #


def build_telescope(ctx: RunContext):
    """return a Karabo Telescope instance."""
    config = ctx.config
    simulator_backend = require_karabo_module("karabo.simulator_backend")
    telescope_module = require_karabo_module("karabo.simulation.telescope")
    kwargs: dict = {"backend": simulator_backend.SimulatorBackend.OSKAR}
    if config.telescope_version is not None:
        kwargs["version"] = resolve_telescope_version(
            telescope_module,
            config.telescope,
            config.telescope_version,
        )
        logger.info(f"Telescope {config.telescope}  version={config.telescope_version}")
    else:
        logger.info(f"Telescope {config.telescope}  (no version)")
    telescope = telescope_module.Telescope.constructor(config.telescope, **kwargs)
    details = {"name": config.telescope, "version": config.telescope_version}
    try:
        if hasattr(telescope, "antennas") and telescope.antennas is not None:
            details["n_stations"] = len(telescope.antennas)
        elif hasattr(telescope, "num_antennas"):
            details["n_stations"] = telescope.num_antennas
        elif hasattr(telescope, "num_stations"):
            details["n_stations"] = telescope.num_stations
        elif hasattr(telescope, "stations") and telescope.stations is not None:
            details["n_stations"] = len(telescope.stations)
    except Exception:
        pass
    ctx.add_milestone(
        "telescope_built",
        "completed",
        details=details,
    )
    return telescope


def resolve_telescope_version(telescope_module, telescope: str, version: str):
    """Resolve a CLI/config telescope version string to Karabo's enum member."""
    version_enum = getattr(telescope_module, "OSKAR_TELESCOPE_TO_VERSIONS", {}).get(
        telescope
    )
    if version_enum is None:
        return version
    if not isinstance(version, str):
        return version
    for candidate in version_enum:
        if version in (candidate.name, candidate.value):
            return candidate
    accepted = ", ".join(candidate.name for candidate in version_enum)
    raise ValueError(
        f"Unsupported version {version!r} for telescope {telescope!r}. "
        f"Accepted versions: {accepted}"
    )


# --------------------------------------------------------------------------- #
# field of view
# --------------------------------------------------------------------------- #


def compute_fov(
    telescope: str,
    fov_deg: Optional[float],
    frequency: u.Quantity,
) -> u.Quantity:
    """return FoV in radians.  If fov_deg is set, use it; else diffraction limit."""
    if fov_deg is not None:
        return (fov_deg * u.deg).to(u.rad)
    wavelength = frequency.to(u.m, equivalencies=u.spectral())
    diameter = get_diameter(telescope.upper())
    fov = (1.25 * wavelength / diameter) * u.rad
    logger.debug(f"Computed FOV is: {fov} radians")
    return fov


def maximum_baseline_m(telescope) -> float:
    """Return Karabo's maximum antenna separation in metres."""
    try:
        baseline_m = float(telescope.max_baseline())
        if not np.isfinite(baseline_m) or baseline_m <= 0:
            raise ValueError(
                f"Karabo returned an invalid maximum baseline: {baseline_m!r} m"
            )
        return baseline_m
    except Exception as exc:
        raise RuntimeError(
            "Could not extract a valid maximum baseline from the telescope; "
            f"cannot resolve image geometry: {exc}"
        ) from exc


def _simulation_fov_deg(
    diffraction_fov_deg: float,
    geometries: Dict[str, ImageGeometry],
) -> float:
    """Return a run-level field covering the primary beam and every image block."""
    return max(
        [diffraction_fov_deg]
        + [geometry.effective_fov_deg for geometry in geometries.values()]
    )


def _resolve_run_geometry(
    ctx: RunContext,
    telescope,
) -> Tuple[float, float, float, Dict[str, ImageGeometry]]:
    """Compute observation-level beam and resolve per-block geometries.

    Returns (diffraction_fov_deg, beam_arcsec, upper_frequency_hz, geometries).
    """
    obs = ctx.config.observation
    if obs.bandwidth_mhz is None:
        raise RuntimeError(
            "Observation bandwidth is not resolved; cannot compute band edges."
        )
    center_frequency = obs.frequency_mhz * u.MHz
    center_frequency_hz = center_frequency.to(u.Hz).value
    baseline_m = maximum_baseline_m(telescope)
    # Both the simulation field of view and the theoretical beam are evaluated at
    # the observing frequency (band centre) per the current PRD.
    wavelength_m = center_frequency.to(u.m, equivalencies=u.spectral()).value
    beam_arcsec = (wavelength_m / baseline_m * u.rad).to(u.arcsec).value
    diffraction_fov_deg = (
        compute_fov(ctx.config.telescope, None, center_frequency).to(u.deg).value
    )

    geometries: Dict[str, ImageGeometry] = {}
    records: Dict[str, dict] = {}
    for img_config in ctx.config.imaging:
        geometry = img_config.resolve_geometry(
            diffraction_fov_deg=diffraction_fov_deg,
            theoretical_beam_arcsec=beam_arcsec,
            reference_frequency_hz=center_frequency_hz,
        )
        geometry_record = geometry.as_dict()
        geometry_record["reference_frequency_label"] = "band centre"
        records[img_config.tag] = geometry_record
        geometries[img_config.tag] = geometry
        logger.info(
            "Image geometry [{}]: requested_fov_deg={requested_fov_deg}, "
            "requested_pixels={requested_pixels}, "
            "requested_cell_size_arcsec={requested_cell_size_arcsec}, "
            "effective={effective_pixels} px, {effective_fov_deg:.6g} deg, "
            "{effective_cell_size_arcsec:.6g} arcsec; "
            "theoretical beam {theoretical_beam_arcsec:.6g} arcsec at {ref:.6g} MHz; "
            "{pixels_per_beam:.4g} pixels/beam; rounded_up={rounded_up}; "
            "legacy_fallback={legacy_fallback}".format(
                img_config.tag,
                requested_fov_deg=geometry.requested_fov_deg,
                requested_pixels=geometry.requested_pixels,
                requested_cell_size_arcsec=geometry.requested_cell_size_arcsec,
                effective_pixels=geometry.effective_pixels,
                effective_fov_deg=geometry.effective_fov_deg,
                effective_cell_size_arcsec=geometry.effective_cell_size_arcsec,
                theoretical_beam_arcsec=geometry.theoretical_beam_arcsec,
                ref=center_frequency.to(u.MHz).value,
                pixels_per_beam=geometry.pixels_per_beam,
                rounded_up=geometry.pixels_rounded_up,
                legacy_fallback=geometry.legacy_fallback,
            )
        )
        for message in geometry.warnings:
            logger.warning("Image geometry [{}]: {}", img_config.tag, message)

    ctx.add_milestone(
        "image_geometry_resolved",
        "completed",
        details={
            "maximum_baseline_m": baseline_m,
            "diffraction_fov_deg": diffraction_fov_deg,
            "diffraction_fov_frequency_label": "band centre",
            "simulation_fov_deg": _simulation_fov_deg(diffraction_fov_deg, geometries),
            "reference_frequency_label": "band centre",
            "blocks": records,
        },
    )
    return diffraction_fov_deg, beam_arcsec, center_frequency_hz, geometries


def _effective_img_config(
    img_config: ImgConfig,
    geometry: ImageGeometry,
) -> ImgConfig:
    """Return an effective config while retaining the user's request in the manifest."""
    return img_config.model_copy(
        update={
            "pixels": geometry.effective_pixels,
            "fov_deg": geometry.effective_fov_deg,
            "cell_size_arcsec": geometry.effective_cell_size_arcsec,
        }
    )


def phase_centre_rule(config: SimConfig) -> str:
    """Name the rule that decides the phase centre for this run.

    Precedence: explicit ``center`` → catalogue (typed component entry or a
    legacy sky file / catalog / FITS catalogue) → first image model with a
    usable WCS → default field (image models without WCS centre) → generated
    sources.
    """
    if config.center is not None:
        return "explicit"
    if component_model_entries(config) or any(
        source is not None
        for source in (config.sky_file, config.catalog, config.fits_image)
    ):
        return "catalogue"
    image_entries = image_model_entries(config)
    if image_entries:
        if image_model_center(image_entries) is not None:
            return "image_model"
        return "default_field"
    return "generated_sources"


def catalogue_source_table(sky_model) -> Optional[Tuple[np.ndarray, ...]]:
    """Return ``(ra_deg, dec_deg, stokes_i_jy, ref_freq_hz, spectral_index)``.

    Columns follow the Karabo/OSKAR source array (``ra, dec, I, Q, U, V,
    ref_freq, spectral_index, ...``); missing spectral columns are zero.
    Returns ``None`` when the model has no readable source array.
    """
    sources = getattr(sky_model, "sources", None)
    if sources is None:
        return None
    try:
        arr = np.asarray(getattr(sources, "values", sources), dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return None
    zeros = np.zeros(arr.shape[0])
    ref = arr[:, 6] if arr.shape[1] > 6 else zeros
    alpha = arr[:, 7] if arr.shape[1] > 7 else zeros
    return arr[:, 0], arr[:, 1], arr[:, 2], ref, alpha


def catalogue_beam_flux(
    table: Tuple[np.ndarray, ...], center: SkyCoord, beam: GaussianStationBeam
) -> dict:
    """Total catalogue Stokes I at the beam reference frequency, with and without beam.

    Sources with a positive reference frequency are scaled with their spectral
    index, ``I (nu0 / nu_ref)^alpha``; others are taken as flat.
    """
    ra, dec, flux, ref, alpha = table
    nu0 = beam.ref_freq_hz
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(ref > 0, (nu0 / np.where(ref > 0, ref, 1.0)) ** alpha, 1.0)
    flux_nu0 = np.where(np.isfinite(flux * scale), flux * scale, 0.0)
    icrs = center.icrs
    sep = angular_separation_deg(ra, dec, icrs.ra.deg, icrs.dec.deg)
    attenuated = flux_nu0 * beam.power_response(sep, nu0)
    return {
        "n_sources": int(flux.size),
        "intrinsic_flux_jy_at_ref": float(np.sum(flux_nu0)),
        "attenuated_flux_jy_at_ref": float(np.sum(attenuated)),
        "max_offset_deg": float(np.nanmax(sep)) if sep.size else 0.0,
        "beam_ref_freq_hz": nu0,
    }


def _record_sky_layout(
    ctx: RunContext,
    center: SkyCoord,
    fov_sim: u.Quantity,
    freq: u.Quantity,
    sky_model=None,
) -> None:
    """Record the phase-centre rule, catalogue flux, and image-model footprints.

    Also draws the combined sky-composition panel for runs with image models.
    """
    config = ctx.config
    beam = run_beam(fov_sim, freq)
    # Image-only runs carry an empty sky model, so there is no source table.
    table = catalogue_source_table(sky_model) if sky_model is not None else None
    if table is not None:
        ctx.add_milestone(
            "catalogue_beam_flux",
            "completed",
            details=catalogue_beam_flux(table, center, beam),
        )
    rule = phase_centre_rule(config)
    ctx.add_milestone(
        "phase_centre_resolved",
        "completed",
        details={
            "rule": rule,
            "ra_deg": float(center.icrs.ra.deg),
            "dec_deg": float(center.icrs.dec.deg),
        },
    )
    logger.info(f"Phase centre rule: {rule}")

    entries = image_model_entries(config)
    if not entries:
        return
    footprints = image_model_footprints(entries, center, beam, fov_sim.to(u.deg).value)
    for record in footprints:
        if record["beyond_beam_half_power"] or record["beyond_field"]:
            logger.warning(
                "Image model {} ({}) reaches {:.4g} deg from the phase centre, "
                "beyond the {} ({:.4g} deg); its outer parts are attenuated by "
                "more than 50%.",
                record["model_entry_index"],
                record["model_type"],
                record["footprint_radius_deg"],
                "simulation field radius"
                if record["beyond_field"]
                else "beam half-power radius",
                record["field_radius_deg"]
                if record["beyond_field"]
                else record["beam_half_power_radius_deg"],
            )
    ctx.add_milestone(
        "image_model_footprints",
        "completed",
        details={"entries": footprints},
    )
    try:
        write_sky_composition_preview(
            ctx,
            center,
            beam,
            fov_sim.to(u.deg).value,
            catalogue=table[:3] if table is not None else None,
            phase_centre_rule=rule,
        )
    except Exception as exc:
        logger.warning(f"Sky composition panel failed: {exc}")
        ctx.add_milestone(
            "sky_composition_preview_failed", "failed", details={"error": str(exc)}
        )


def parse_center(center_str: Optional[str], fallback: SkyCoord) -> SkyCoord:
    if center_str is None:
        return fallback
    try:
        coords_str = center_str.replace(",", " ").replace(":", " ")
        return SkyCoord(coords_str, unit=(u.hourangle, u.deg))
    except Exception:
        return fallback


# --------------------------------------------------------------------------- #
# sky model — file loaders
# --------------------------------------------------------------------------- #


def _load_sky_from_file(
    fpath: str,
    column_mapping: str = "0,1,2,3,4,5,6,7,8,9,10,11,12",
    flux_scale: float = 1.0,
    frequency: Optional[u.Quantity] = None,
) -> SkyModel:
    """Load SkyModel from pickle, fits, json, or karabo.mod."""
    ext = os.path.splitext(fpath)[-1].lower()

    # pickle / karabo model
    if ext in (".pkl", ".pickle", ".kmod", ".karabo.mod"):
        with open(fpath, "rb") as fh:
            sky_model = pickle.load(fh)
        logger.info(f"Loaded pickle model from {fpath}")
        return sky_model

    # json catalog (supports both single JSON array and JSONL: one object per line)
    if ext == ".json":
        with open(fpath, "r") as fh:
            raw = fh.read().strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # JSONL: one JSON object per line
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]
        sources: List[Source] = []
        for item in data:
            src = Source.from_json(item)
            if flux_scale != 1.0:
                src.I *= flux_scale
            if src.ref_freq == 0:
                hz = frequency.to(u.Hz).value if frequency is not None else 0
                src.ref_freq = hz * u.Hz
            sources.append(src)
        if not sources:
            raise ValueError(f"No sources found in JSON {fpath}")
        sky_model = SkyModel()
        arr = np.array([s.to_sky_model(reduced_form=False) for s in sources])
        sky_model.add_point_sources(arr)
        sky_model.get_center()
        logger.info(f"Loaded {len(sources)} sources from JSON {fpath}")
        return sky_model

    # fits table or image
    if ext in (".fits", ".fit"):
        return _load_sky_from_fits(fpath, column_mapping, flux_scale, frequency)

    raise ValueError(f"Unsupported sky-file extension: {ext}")


def _load_sky_from_fits(
    fpath: str,
    column_mapping: str,
    flux_scale: float,
    frequency: Optional[u.Quantity],
) -> SkyModel:
    """try Karabo's get_sky_model_from_fits; fallback to our own loader when columns lack TUNIT."""
    loader = FitsCatalogLoader(
        fpath=fpath,
        column_mapping=column_mapping,
        scale_I=flux_scale,
        ref_freq_hz=None,
        frequency=frequency,
    )
    return loader.load()


# --------------------------------------------------------------------------- #
# sky model — flux/polarization summary helper
# --------------------------------------------------------------------------- #


def _sky_flux_details(sky_model: SkyModel) -> dict:
    """Return best-effort flux-range and polarization presence for a tabular sky model.

    Returns {} when the model has no readable per-source flux column (e.g. an
    empty model, or one backed by something other than a plain source array).
    """
    sources = getattr(sky_model, "sources", None)
    if sources is None:
        return {}
    try:
        values = getattr(sources, "values", sources)
        arr = np.asarray(values, dtype=float)
    except Exception:
        return {}
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return {}
    flux_jy = arr[:, 2]
    finite_flux = flux_jy[np.isfinite(flux_jy)]
    if finite_flux.size == 0:
        return {}
    details: dict = {
        "flux_min_jy": float(np.min(finite_flux)),
        "flux_max_jy": float(np.max(finite_flux)),
    }
    if arr.shape[1] >= 6:
        pol = arr[:, 3:6]
        if bool(np.any(np.isfinite(pol) & (pol != 0.0))):
            details["has_polarization"] = True
    return details


# --------------------------------------------------------------------------- #
# sky model — catalog loader helper
# --------------------------------------------------------------------------- #


def _load_sky_from_catalog(catalog: str) -> tuple[SkyModel, str]:
    """Load one built-in catalog and return (sky_model, format_label)."""
    if catalog == "MIGHTEE":
        logger.info("Loading MIGHTEE catalog")
        if not hasattr(SkyModel, "get_MIGHTEE_Sky"):
            require_karabo_module("karabo.simulation.sky_model")
        return SkyModel.get_MIGHTEE_Sky(), "MIGHTEE"
    if catalog == "GLEAM":
        logger.info("Loading GLEAM catalog")
        if not hasattr(SkyModel, "get_GLEAM_Sky"):
            require_karabo_module("karabo.simulation.sky_model")
        return SkyModel.get_GLEAM_Sky(), "GLEAM"
    if catalog == "SKAMid":
        skamid_path = Path("SKAMid_B1_8h_v3.fits").resolve()
        if skamid_path.exists():
            logger.info(f"Loading SKAMid catalog {skamid_path}")
            return (
                SkyModel.get_sky_model_from_fits(fits_file=str(skamid_path)),
                "SKAMid",
            )
        logger.info(f"SKAMid catalog not found at {skamid_path}")
        raise FileNotFoundError(str(skamid_path))
    raise ValueError(f"Catalog {catalog} not available")


# --------------------------------------------------------------------------- #
# sky model — high-level builder
# --------------------------------------------------------------------------- #


def build_sky_model(
    ctx: RunContext,
    fov: u.Quantity,
) -> tuple[SkyModel, SkyCoord]:
    """Return (sky_model, center)."""
    config = ctx.config
    component_entries = component_model_entries(config)
    image_entries = image_model_entries(config)

    if component_entries:
        entry = component_entries[0]
        component_path = None
        if entry.path is not None:
            component_path = Path(entry.path).expanduser().resolve()
            sky_model = _load_sky_from_file(
                str(component_path),
                column_mapping=entry.column_mapping or "0,1,2,3,4,5,6,7,8,9,10,11,12",
                flux_scale=entry.flux_scale,
                frequency=config.observation.frequency_mhz * u.MHz,
            )
            component_format = entry.sky_format
        else:
            assert entry.catalog is not None
            sky_model, component_format = _load_sky_from_catalog(entry.catalog)
        center = sky_model.get_center()
        n_srcs = len(sky_model.sources) if hasattr(sky_model, "sources") else None
        sky_details = {
            "path": str(component_path) if component_path is not None else None,
            "format": component_format,
            "n_sources": n_srcs,
            "model_entries": len(config.models),
        }
        sky_details.update(_sky_flux_details(sky_model))
        ctx.add_milestone("sky_model_loaded", "completed", details=sky_details)
        if component_path is not None:
            ctx.manifest.add_output(
                "sky_model",
                str(component_path),
                metadata={"format": component_format, "n_sources": n_srcs},
            )
        return sky_model, center

    if image_entries:
        center = image_model_center(image_entries) or Source.from_name("HCG16").coords()
        sky_model = SkyModel()
        sky_model.phase_center = center
        ctx.add_milestone(
            "sky_model_loaded",
            "completed",
            details={
                "format": "image_models",
                "n_sources": 0,
                "model_entries": len(image_entries),
            },
        )
        return sky_model, center

    # 1) file path given?
    if config.sky_file is not None:
        sky_model = _load_sky_from_file(
            str(ctx.sky_file_resolved),
            column_mapping=config.column_mapping or "0,1,2,3,4,5,6,7,8,9,10,11,12",
            flux_scale=config.flux_scale,
            frequency=config.observation.frequency_mhz * u.MHz,
        )
        center = sky_model.get_center()
        n_srcs = len(sky_model.sources) if hasattr(sky_model, "sources") else None
        sky_details = {
            "path": str(ctx.sky_file_resolved),
            "format": config.sky_format,
            "n_sources": n_srcs,
        }
        sky_details.update(_sky_flux_details(sky_model))
        ctx.add_milestone("sky_model_loaded", "completed", details=sky_details)
        ctx.manifest.add_output(
            "sky_model",
            str(ctx.sky_file_resolved),
            metadata={"format": config.sky_format, "n_sources": n_srcs},
        )
        return sky_model, center

    # 2) built-in catalog (legacy path)
    if config.catalog is not None:
        sky_model, fmt = _load_sky_from_catalog(config.catalog)
        center = sky_model.get_center()
        n_srcs = len(sky_model.sources) if hasattr(sky_model, "sources") else None
        sky_details = {"format": fmt, "n_sources": n_srcs}
        sky_details.update(_sky_flux_details(sky_model))
        ctx.add_milestone("sky_model_loaded", "completed", details=sky_details)
        return sky_model, center

    # 3) FITS image ingestion (legacy path)
    if config.fits_image is not None:
        from .loaders import FitsImageLoader

        fpath = Path(config.fits_image)
        if not fpath.is_absolute():
            fpath = Path(os.getcwd()) / fpath
        loader = FitsImageLoader(
            fpath,
            fallback_freq_mhz=config.observation.frequency_mhz,
        )
        sky_model = loader.load()
        center = sky_model.get_center()
        n_srcs = len(sky_model.sources) if hasattr(sky_model, "sources") else None
        sky_details = {
            "path": str(fpath),
            "format": "fits_image",
            "n_sources": n_srcs,
        }
        sky_details.update(_sky_flux_details(sky_model))
        ctx.add_milestone("sky_model_loaded", "completed", details=sky_details)
        ctx.manifest.add_output(
            "sky_model",
            str(fpath),
            metadata={"format": "fits_image", "n_sources": n_srcs},
        )
        return sky_model, center

    # 4) random sources around a reference position
    logger.info("Generating random sources")
    source_ref = Source.from_name("HCG16")
    intensities = [i * u.Jy for i in config.source_flux_jy]
    stokes_q = config.stokes_q_jy or [0.0] * len(intensities)
    stokes_u = config.stokes_u_jy or [0.0] * len(intensities)
    stokes_v = config.stokes_v_jy or [0.0] * len(intensities)
    n_sources = len(intensities)
    sources: List[Source] = []
    for idx in range(n_sources):
        if idx == 0:
            src = source_ref
            src.I = intensities[idx]
            src.Q = stokes_q[idx] * u.Jy
            src.U = stokes_u[idx] * u.Jy
            src.V = stokes_v[idx] * u.Jy
        else:
            x_coord = np.random.uniform(-fov.value / 2, fov.value / 2) * 0.8 * u.rad
            y_coord = np.random.uniform(-fov.value / 2, fov.value / 2) * 0.8 * u.rad
            src = Source(
                source_ref.ra + x_coord,
                source_ref.dec + y_coord,
                intensities[idx],
                Q=stokes_q[idx] * u.Jy,
                U=stokes_u[idx] * u.Jy,
                V=stokes_v[idx] * u.Jy,
            )
        sources.append(src)

    sky_model = SkyModel()
    has_polarization = any(
        value != 0.0 for values in (stokes_q, stokes_u, stokes_v) for value in values
    )
    arr = np.array([s.to_sky_model(reduced_form=not has_polarization) for s in sources])
    sky_model.add_point_sources(arr)
    center = sky_model.get_center()
    sky_details = {"format": "random", "n_sources": n_sources, "reference": "HCG16"}
    sky_details.update(_sky_flux_details(sky_model))
    ctx.add_milestone("sky_model_loaded", "completed", details=sky_details)
    return sky_model, center


def build_observation(
    ctx: RunContext,
    center: SkyCoord,
    telescope,
    sky_center: Optional[SkyCoord] = None,
) -> tuple:
    """Return (observation, frequency, bandwidth, n_channels, delta_freq, start_freq).

    ``sky_center`` is the sky model's own centre, before any ``SimConfig.center``
    override is applied to ``center``. When omitted it defaults to ``center``, so
    a caller that doesn't track the pre-override centre simply reports no drift.
    """
    if sky_center is None:
        sky_center = center
    observation_module = require_karabo_module("karabo.simulation.observation")
    config = ctx.config
    obs = config.observation
    freq = obs.frequency_mhz * u.MHz
    bw_mhz = obs.bandwidth_mhz
    n_channels = obs.n_channels
    df_mhz = obs.channel_width_mhz
    bandwidth = bw_mhz * u.MHz
    delta_freq = df_mhz * u.MHz
    seconds = config.observation.observation_time_s

    start_freq = freq - n_channels * delta_freq / 2
    end_freq = start_freq + n_channels * delta_freq

    # best observation time (culmination)
    obs_time = source_ref_get_best_observation_time(center, telescope)
    start_time_utc = obs_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Observation start: {start_time_utc} UTC")
    n_timesteps = max(1, int(seconds / 7.997))

    observation = observation_module.Observation(
        start_frequency_hz=start_freq.to(u.Hz).value,
        start_date_and_time=obs_time,
        frequency_increment_hz=delta_freq.to(u.Hz).value,
        length=timedelta(seconds=seconds),
        number_of_time_steps=n_timesteps,
        number_of_channels=n_channels,
        phase_centre_ra_deg=center.ra.to(u.deg).value,
        phase_centre_dec_deg=center.dec.to(u.deg).value,
    )

    ctx.add_milestone(
        "observation_configured",
        "completed",
        details={
            "frequency_mhz": obs.frequency_mhz,
            "min_frequency_mhz": start_freq.to(u.MHz).value,
            "max_frequency_mhz": end_freq.to(u.MHz).value,
            "bandwidth_mhz": bw_mhz,
            "n_channels": n_channels,
            "channel_width_mhz": df_mhz,
            "observation_time_s": seconds,
            "start_time_utc": start_time_utc,
            "n_timesteps": n_timesteps,
            "phase_center_ra_deg": center.ra.to(u.deg).value,
            "phase_center_dec_deg": center.dec.to(u.deg).value,
            "sky_model_center_ra_deg": sky_center.ra.to(u.deg).value,
            "sky_model_center_dec_deg": sky_center.dec.to(u.deg).value,
        },
    )
    return observation, freq, bandwidth, n_channels, delta_freq, start_freq


def source_ref_get_best_observation_time(center: SkyCoord, telescope):
    """Wrapper around Source.get_best_observation_time using a dummy Source."""
    src = Source(center.ra, center.dec, 1 * u.Jy)
    return src.get_best_observation_time(telescope=telescope)


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #


def run_beam(fov_sim: u.Quantity, freq: u.Quantity) -> GaussianStationBeam:
    """Return the Gaussian beam OSKAR applies to the catalogue for this run.

    The same beam is applied to every image model before WSClean prediction so
    all sky components share one attenuation (see ``skasim.primary_beam``).
    """
    return simulation_beam(fov_sim.to(u.deg).value, freq.to(u.Hz).value)


def run_simulation(
    ctx: RunContext,
    telescope,
    observation,
    sky_model: SkyModel,
    fov_sim: u.Quantity,
) -> Path:
    """Run InterferometerSimulation and return visibility path."""
    interferometer_module = require_karabo_module("karabo.simulation.interferometer")
    simulator_backend = require_karabo_module("karabo.simulator_backend")
    config = ctx.config
    visibility_path = ctx.visibility_path

    if visibility_path.exists():
        if config.overwrite:
            logger.info(f"Overwriting existing {visibility_path}")
            shutil.rmtree(visibility_path)
        else:
            raise FileExistsError(
                f"{visibility_path} already exists. Use --overwrite to replace it."
            )

    freq = config.observation.frequency_mhz * u.MHz
    delta_freq = config.observation.channel_width_mhz * u.MHz

    beam = run_beam(fov_sim, freq)
    params = {
        "channel_bandwidth_hz": delta_freq.to(u.Hz).value,
        **beam.oskar_params(),
        "use_gpus": False,
    }
    if config.rms or (config.noise_rms_start is not None):
        params["noise_enable"] = True
        # "Observation settings" avoids OSKAR segfault: "Telescope model"
        # triggers oskar_settings_to_telescope which dereferences a null
        # ionosphere screen object when noise_rms="Range".
        params["noise_freq"] = "Observation settings"
        if config.noise_rms_start is not None:
            # numeric station RMS override (calibration recipe)
            start = config.noise_rms_start
            end = config.noise_rms_end if config.noise_rms_end is not None else start
            params["noise_rms"] = "Range"
            params["noise_rms_start"] = start
            params["noise_rms_end"] = end
            logger.info(f"noise_rms override: Range {start:.6e} – {end:.6e} Jy")
        else:
            params["noise_rms"] = "Telescope model"

    simulation = interferometer_module.InterferometerSimulation(**params)
    simulation.run_simulation(
        telescope=telescope,
        observation=observation,
        sky=sky_model,
        visibility_path=str(visibility_path),
        backend=simulator_backend.SimulatorBackend.OSKAR,
    )
    logger.info(f"Visibilities saved in {visibility_path}")
    ctx.manifest.add_output(
        "visibility",
        str(visibility_path.relative_to(ctx.work_dir)),
    )
    return visibility_path


def _run_uv_coverage(ctx: RunContext, visibility_path: Path) -> None:
    """Run shadeMS UV-coverage plot; record milestone on success or failure."""
    config = ctx.config
    ctx.add_milestone("uv_coverage_started", "started")
    try:
        run_id = ctx.work_dir.name
        png_name = f"{run_id}_uvcoverage.png"
        log_name = f"{run_id}_uvcoverage_shadems.log"
        png_path = ctx.work_dir / png_name
        log_path = ctx.work_dir / log_name

        argv = build_shadems_uv_coverage_argv(
            shadems_command=config.shadems_command,
            visibility_path=visibility_path,
            output_dir=ctx.work_dir,
            png_name=png_name,
            title=f"{run_id} uv coverage",
            canvas_size=config.uv_coverage_canvas_size,
        )
        try:
            result = run_shadems_command(argv, ctx.work_dir)
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            log_path.write_text(output, encoding="utf-8")
            ctx.manifest.add_output(
                "log",
                log_name,
                role="uv_coverage",
                metadata={"tool": "shadems", "returncode": exc.returncode},
            )
            raise
        log_path.write_text(
            (result.stdout or "") + (result.stderr or ""), encoding="utf-8"
        )
        if not png_path.exists():
            raise FileNotFoundError(f"shadeMS did not produce {png_path}")
        ctx.manifest.add_output(
            "plot",
            png_name,
            role="uv_coverage",
            metadata={
                "tool": "shadems",
                "xaxis": "u",
                "yaxis": "v",
                "canvas_size": config.uv_coverage_canvas_size,
            },
        )
        ctx.manifest.add_output(
            "log",
            log_name,
            role="uv_coverage",
            metadata={"tool": "shadems"},
        )
        ctx.add_milestone(
            "uv_coverage_completed",
            "completed",
            details={"path": str(png_path.relative_to(ctx.work_dir))},
        )
    except Exception as exc:
        ctx.add_milestone("uv_coverage_failed", "failed", details={"error": str(exc)})
        logger.exception("UV coverage plot failed")


def _run_simulation_phase(
    ctx: RunContext,
    telescope,
    observation,
    sky_model: SkyModel,
    center: SkyCoord,
    fov_sim: u.Quantity,
    imaging_configs: List[ImgConfig],
) -> Path:
    """Run OSKAR simulation + image-model injection.

    Falls back to a zero-flux placeholder source if the initial simulation
    fails and the config only has image models (no component sources).
    Returns the visibility path on success.
    """
    config = ctx.config
    ctx.add_milestone("simulation_started", "started")
    t_phase_a = time.time()
    try:
        try:
            visibility_path = run_simulation(
                ctx, telescope, observation, sky_model, fov_sim
            )
        except Exception:
            if image_model_entries(config) and not component_model_entries(config):
                logger.warning(
                    "Empty base-MS creation failed; retrying with a zero-flux "
                    "placeholder source."
                )
                if ctx.visibility_path.exists():
                    shutil.rmtree(ctx.visibility_path)
                ctx.add_milestone(
                    "base_ms_fallback",
                    "completed",
                    details={"strategy": "zero_flux_placeholder_source"},
                )
                visibility_path = run_simulation(
                    ctx,
                    telescope,
                    observation,
                    build_zero_flux_sky_model(center),
                    fov_sim,
                )
            else:
                raise
        beam = run_beam(fov_sim, config.observation.frequency_mhz * u.MHz)
        ctx.add_milestone(
            "primary_beam_configured", "completed", details=beam.as_dict()
        )
        inject_image_models(
            ctx,
            visibility_path,
            imaging_configs=imaging_configs,
            phase_center=center,
            beam=beam,
        )
        ctx.add_milestone(
            "simulation_completed", "completed", elapsed_s=time.time() - t_phase_a
        )
    except Exception as exc:
        ctx.add_milestone(
            "simulation_failed",
            "failed",
            elapsed_s=time.time() - t_phase_a,
            details={"error": str(exc)},
        )
        raise
    return visibility_path


def build_zero_flux_sky_model(center: SkyCoord) -> SkyModel:
    """Build a one-source zero-flux sky model for base-MS fallback creation."""
    sky_model = SkyModel()
    source = Source(center.ra, center.dec, 0 * u.Jy)
    sky_model.add_point_sources(np.array([source.to_sky_model(reduced_form=True)]))
    sky_model.phase_center = center
    return sky_model


# --------------------------------------------------------------------------- #
# top-level orchestrator
# --------------------------------------------------------------------------- #


def _run_imaging_pass(
    ctx: RunContext,
    visibility_path: Path,
    img_config,
    freq: u.Quantity,
    center: SkyCoord,
) -> None:
    """Run one imaging pass (dirty or wsclean) with milestone tracking."""
    config = ctx.config
    tag = img_config.tag
    sub_dir = ctx.work_dir / tag
    sub_dir.mkdir(parents=True, exist_ok=True)
    fov_i = compute_fov(config.telescope, img_config.fov_deg, freq)

    milestone_prefix = f"imaging_{tag}"
    ctx.add_milestone(f"{milestone_prefix}_started", "started")
    t_b = time.time()
    try:
        if img_config.imager == "oskar-dirty":
            run_dirty_imaging(ctx, visibility_path, fov_i, center, img_config, sub_dir)
        else:
            run_wsclean_imaging(
                ctx,
                visibility_path,
                fov_i,
                img_config,
                sub_dir,
                n_channels=config.observation.n_channels or 1,
            )
        ctx.add_milestone(
            f"{milestone_prefix}_completed",
            "completed",
            elapsed_s=time.time() - t_b,
            details={"imager": img_config.imager, "tag": tag},
        )
    except Exception as exc:
        ctx.add_milestone(
            f"{milestone_prefix}_failed",
            "failed",
            elapsed_s=time.time() - t_b,
            details={
                "error": str(exc),
                "tag": tag,
                "imager": img_config.imager,
            },
        )
        raise


def run(config: SimConfig) -> None:
    """Execute the full simulation pipeline from a SimConfig."""
    import matplotlib

    from .weblog import render_weblog

    matplotlib.use("Agg", force=True)

    t0 = time.time()
    ctx = create_run_context(config)

    try:
        logger.info(f"Telescope : {config.telescope}")
        logger.info(f"Frequency : {config.observation.frequency_mhz} MHz")
        logger.info(f"Bandwidth : {config.observation.bandwidth_mhz} MHz")
        logger.info(f"Channels  : {config.observation.n_channels}")
        logger.info(f"Obs time  : {config.observation.observation_time_s} s")
        logger.info(f"Imager(s) : {', '.join(img.imager for img in config.imaging)}")

        telescope = build_telescope(ctx)
        telescope_png = (
            ctx.work_dir
            / f"{ctx.work_dir.name}_{config.telescope}_{config.telescope_version or ''}_telescope.png"
        )
        try:
            telescope.plot_telescope(file=str(telescope_png))
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
        ctx.manifest.add_output(
            "plot",
            str(telescope_png.relative_to(ctx.work_dir)),
            role="telescope",
        )

        # Resolve per-block image geometry once we have a telescope.
        diffraction_fov_deg, _beam_arcsec, _upper_frequency_hz, geometries = (
            _resolve_run_geometry(ctx, telescope)
        )
        effective_images = [
            _effective_img_config(img, geometries[img.tag])
            for img in ctx.config.imaging
        ]
        freq = config.observation.frequency_mhz * u.MHz
        simulation_fov_deg = _simulation_fov_deg(diffraction_fov_deg, geometries)
        simulation_fov = simulation_fov_deg * u.deg
        logger.info(
            "Simulation FoV: {:.4f} deg (covers diffraction limit and all image blocks)",
            simulation_fov.to(u.deg).value,
        )

        sky_model, sky_center = build_sky_model(ctx, simulation_fov)
        center = parse_center(config.center, sky_center)
        logger.info(f"Centre    : {center.to_string('hmsdms')}")
        _record_sky_layout(ctx, center, simulation_fov, freq, sky_model=sky_model)
        try:
            from .imaging import write_sky_model_previews

            for path, role in write_sky_model_previews(
                sky_model,
                center,
                simulation_fov,
                ctx.work_dir,
                ctx.work_dir.name,
            ):
                ctx.manifest.add_output("plot", path, role=role)
            write_image_model_previews(ctx, simulation_fov, center)
        except Exception as exc:
            logger.warning(f"Sky model previews failed: {exc}")
            logger.exception("Sky model preview traceback")
            ctx.add_milestone(
                "sky_model_previews_failed", "failed", details={"error": str(exc)}
            )

        observation, _, bandwidth, n_channels, delta_freq, start_freq = (
            build_observation(ctx, center, telescope, sky_center=sky_center)
        )
        logger.info(f"StartFreq : {start_freq.to(u.MHz).value:.3f} MHz")
        logger.info(f"DeltaFreq : {delta_freq.to(u.MHz).value:.3f} MHz")
        logger.info(f"N channels: {n_channels}")

        # phase 1: simulation
        visibility_path = _run_simulation_phase(
            ctx,
            telescope,
            observation,
            sky_model,
            center,
            simulation_fov,
            effective_images,
        )

        # UV coverage (shadeMS) — once per run, before imaging
        if config.uv_coverage:
            _run_uv_coverage(ctx, visibility_path)

        # phase 2: batch imaging
        for img_config in effective_images:
            _run_imaging_pass(ctx, visibility_path, img_config, freq, center)

        ctx.manifest.mark_completed()
        ctx.manifest.add_output("weblog", ctx.weblog_path.name)
        ctx.save_manifest()

        render_weblog(ctx.manifest, ctx.work_dir)
        logger.info(f"Weblog written to {ctx.weblog_path}")

        # spectral-cube summary
        if has_spectral_cube_model(config):
            _log_spectral_cube_summary(ctx)

        elapsed = time.time() - t0
        logger.info(f"Done. Elapsed: {elapsed:.1f} s")

    except Exception as exc:
        ctx.manifest.mark_failed(str(exc))
        ctx.manifest.add_output("weblog", ctx.weblog_path.name)
        ctx.save_manifest()
        render_weblog(ctx.manifest, ctx.work_dir)
        logger.info(f"Failure weblog written to {ctx.weblog_path}")
        raise

    elapsed = time.time() - t0
    logger.info(f"Done. Elapsed: {elapsed:.1f} s")


def _log_spectral_cube_summary(ctx: RunContext) -> None:
    """Log a concise post-run summary of spectral-cube products."""
    cube_outputs = [
        out
        for out in ctx.manifest.outputs
        if out.image_product_id is not None
        and out.role is not None
        and "cube" in out.role
    ]
    mom_outputs = [
        out
        for out in ctx.manifest.outputs
        if out.role is not None and out.role.startswith("mom")
    ]
    logger.info(
        f"Spectral-cube products: {len(cube_outputs)} cube outputs, "
        f"{len(mom_outputs)} moment maps"
    )
