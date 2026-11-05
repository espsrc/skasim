"""weblog.py — render weblog.html from a RunManifest via Jinja2."""

from __future__ import annotations

import base64
import json
import shlex
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from jinja2 import Environment, PackageLoader, select_autoescape

from .loaders.image_models import (
    FitsCubeInfo,
    read_fits_cube_info,
)
from .manifest import RunManifest

KNOWN_ANTENNA_COUNTS = {
    "MEERKAT": 64,
}

_IMAGER_LABELS = {
    "wsclean": "WSClean",
    "oskar-dirty": "OSKAR dirty",
}

_STATUS_BADGE_CLASSES = {
    "completed": "success",
    "failed": "danger",
    "running": "running",
}


def _imagers_used(manifest: RunManifest) -> list[str]:
    """Return the distinct imagers configured for this run, in first-use order."""
    return list(
        dict.fromkeys(
            _IMAGER_LABELS.get(img.imager, img.imager)
            for img in manifest.config.imaging
        )
    )


def _humanize_seconds(total_s: float) -> str:
    """convert seconds to '2m 34s' or '1h 2m 3s'."""
    if total_s < 60:
        return f"{total_s:.1f}s"
    minutes, seconds = divmod(int(total_s), 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _find_image_outputs(manifest: RunManifest, work_dir: Path) -> list[dict]:
    """extract image paths, captions, and base64 data for inline embedding."""
    results: list[dict] = []
    for o in manifest.outputs:
        if o.kind == "image_product":
            continue
        output_path = o.path if hasattr(o, "path") else o
        if "telescope" in output_path.lower() or o.role == "telescope":
            continue
        if o.role == "uv_coverage":
            continue
        if "sky_model" in output_path.lower() or o.role in (
            "sky_model",
            "sky_model_fov",
        ):
            continue
        if not output_path.endswith((".png", ".jpg", ".jpeg")):
            continue
        fpath = work_dir / output_path
        if not fpath.exists():
            continue  # skip missing files (e.g. during failed runs)
        caption = _infer_image_caption(Path(output_path).name)
        data_b64 = _file_to_base64_data_uri(fpath)
        results.append({"path": output_path, "caption": caption, "data": data_b64})
    return results


def _find_science_products(manifest: RunManifest, work_dir: Path) -> list[dict]:
    """Group MFS/dirty products plus spectral-cube stack and moment8 into display entries."""
    products: dict[str, dict] = {}
    for output in manifest.outputs:
        if output.kind != "image_product":
            continue

        role = output.role or ""
        # Ignore per-channel WSClean outputs; keep MFS, cube_image, and moment8.
        if output.imager == "wsclean" and "-MFS-" not in output.path:
            if role not in ("cube_image", "moment8"):
                continue

        product_id = output.image_product_id or "imaging"
        tag = (output.metadata or {}).get("tag", "default")
        product = products.setdefault(
            product_id,
            {
                "product_id": product_id,
                "tag": tag,
                "imager": output.imager,
                "model": None,
                "clean": None,
                "residual": None,
                "psf": None,
                "dirty": None,
                "cube": None,
                "moment8": None,
                "stats": None,
            },
        )
        if role.endswith("_preview"):
            continue

        key = None
        if role in ("model",):
            key = "model"
        elif role in ("image",):
            key = "dirty" if output.imager == "oskar-dirty" else "clean"
        elif role in ("dirty", "residual", "psf"):
            key = role
        elif role == "cube_image":
            key = "cube"
        elif role == "moment8":
            key = "moment8"
        elif role == "avg_spectrum_plot":
            key = "avg_spectrum"

        if key is None:
            continue
        fpath = work_dir / output.path
        preview = fpath.with_suffix(".png")
        product[key] = {
            "fits": output.path,
            "preview": str(preview.relative_to(work_dir)) if preview.exists() else None,
            "data": _file_to_base64_data_uri(preview) if preview.exists() else None,
            "beam": _read_fits_beam(fpath)
            if key in ("model", "clean", "residual", "dirty", "psf")
            else None,
            "peak": _read_fits_peak(fpath) if key == "psf" else None,
            "pixel_scale_arcsec": _read_fits_pixel_scale_arcsec(fpath)
            if key == "psf"
            else None,
        }

    for product in products.values():
        clean = product["clean"]
        residual = product["residual"]
        if clean and residual:
            product["stats"] = _calculate_image_stats(
                work_dir / clean["fits"],
                work_dir / residual["fits"],
            )
        product["beam"] = _science_product_beam(product)

    return [
        product
        for product in products.values()
        if any(
            product.get(k)
            for k in (
                "model",
                "clean",
                "residual",
                "dirty",
                "cube",
                "moment8",
                "avg_spectrum",
            )
        )
    ]


def _science_product_beam(product: dict) -> dict | None:
    """Return one representative beam for a science product."""
    for key in ("clean", "dirty", "residual", "model", "psf"):
        item = product.get(key)
        if item and item.get("beam"):
            return item["beam"]
    return None


def _read_fits_beam(path: Path) -> dict | None:
    """Return FITS beam metadata in readable units when present."""
    if not path.exists():
        return None
    try:
        with fits.open(path) as hdul:
            header = hdul[0].header
            bmaj = header.get("BMAJ")
            bmin = header.get("BMIN")
            bpa = header.get("BPA")
    except Exception:
        return None
    if bmaj is None and bmin is None and bpa is None:
        return None
    return {
        "bmaj": _format_angle_deg(bmaj) if bmaj is not None else None,
        "bmin": _format_angle_deg(bmin) if bmin is not None else None,
        "bpa": f"{float(bpa):.2f} deg" if bpa is not None else None,
        "bmaj_arcsec": float(bmaj) * 3600.0 if bmaj is not None else None,
    }


def _format_angle_deg(value_deg: float) -> str:
    """Format an angular size provided in degrees."""
    value = float(value_deg)
    abs_value = abs(value)
    if abs_value >= 1.0:
        return f"{value:.4f} deg"
    if abs_value >= 1.0 / 60.0:
        return f"{value * 60.0:.3f} arcmin"
    return f"{value * 3600.0:.3f} arcsec"


def _read_fits_peak(path: Path) -> float | None:
    """Return peak value of a FITS image, or None on failure."""
    if not path.exists():
        return None
    try:
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data, dtype=float).squeeze()
        finite = np.isfinite(data)
        return float(data[finite].max()) if finite.any() else None
    except Exception:
        return None


def _read_fits_pixel_scale_arcsec(path: Path) -> float | None:
    """Return pixel scale in arcseconds from CDELT1/CD1_1, or None."""
    if not path.exists():
        return None
    try:
        with fits.open(path) as hdul:
            header = hdul[0].header
            cdelt = header.get("CDELT1") or header.get("CD1_1")
        if cdelt is None:
            return None
        return abs(float(cdelt)) * 3600.0
    except Exception:
        return None


def _format_float(value: Optional[float], digits: int = 3) -> Optional[str]:
    """Format a number without distracting trailing zeroes."""
    if value is None:
        return None
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def _observation_summary(manifest: RunManifest) -> dict:
    """Build display rows for the resolved spectral and time setup."""
    obs = manifest.config.observation
    details = _find_milestone_details(manifest, "observation_configured")
    centre = float(obs.frequency_mhz)
    n_channels = int(obs.n_channels or 1)
    channel_width = float(obs.channel_width_mhz or 0.0)
    total_bandwidth = float(obs.bandwidth_mhz or n_channels * channel_width)
    min_freq = float(details.get("min_frequency_mhz", centre - total_bandwidth / 2.0))
    max_freq = float(details.get("max_frequency_mhz", centre + total_bandwidth / 2.0))
    n_timesteps = details.get("n_timesteps")
    integration_time = None
    if n_timesteps:
        integration_time = float(obs.observation_time_s) / float(n_timesteps)
    return {
        "band_edge_min_mhz": _format_float(min_freq),
        "band_edge_max_mhz": _format_float(max_freq),
        "central_frequency_mhz": _format_float(centre),
        "n_channels": n_channels,
        "channel_bandwidth_mhz": _format_float(channel_width),
        "total_bandwidth_mhz": _format_float(total_bandwidth),
        "observation_time_s": _format_float(obs.observation_time_s),
        "start_time_utc": details.get("start_time_utc"),
        "integration_time_s": (
            _format_float(integration_time) if integration_time is not None else None
        ),
        "n_timesteps": n_timesteps,
    }


def _imaging_summary_for_block(
    img_config,
    fov_deg: float | None,
    beam: dict | None,
    geometry: Optional[dict] = None,
) -> dict:
    """Build display rows for imaging geometry and pixelization for a single ImgConfig block."""
    pixel_size_arcsec = None
    if fov_deg is not None:
        pixel_size_arcsec = fov_deg * 3600.0 / img_config.pixels
    summary = {
        "imager": img_config.imager,
        "pixels": img_config.pixels,
        "image_size": f"{img_config.pixels} x {img_config.pixels}",
        "fov_deg": _format_float(fov_deg, 4) if fov_deg is not None else None,
        "pixel_size_arcsec": (
            _format_float(pixel_size_arcsec, 4)
            if pixel_size_arcsec is not None
            else None
        ),
        "beam": _format_beam(beam),
    }
    if geometry is not None:
        summary["requested_geometry"] = geometry
        summary["theoretical_beam_arcsec"] = _format_float(
            geometry.get("theoretical_beam_arcsec"), 4
        )
        summary["pixels_per_beam"] = _format_float(geometry.get("pixels_per_beam"), 3)
        ref_hz = geometry.get("reference_frequency_hz")
        if ref_hz is not None:
            summary["reference_frequency_mhz"] = _format_float(ref_hz / 1e6, 4)
        theoretical = geometry.get("theoretical_beam_arcsec")
        achieved = beam.get("bmaj_arcsec") if beam else None
        if theoretical and achieved:
            summary["beam_vs_theoretical"] = (
                f"{_format_float(achieved / theoretical, 2)}× theoretical"
            )
    return summary


def _format_beam(beam: dict | None) -> str | None:
    """Format one beam dictionary for compact display."""
    if not beam:
        return None
    parts = []
    if beam.get("bmaj"):
        parts.append(f"BMAJ {beam['bmaj']}")
    if beam.get("bmin"):
        parts.append(f"BMIN {beam['bmin']}")
    if beam.get("bpa"):
        parts.append(f"BPA {beam['bpa']}")
    return " · ".join(parts) if parts else None


def _imager_parameter_rows_for_block(
    img_config,
    manifest: RunManifest,
    fov_deg: float | None,
    center: tuple[float, float] | None,
) -> list[tuple[str, str]]:
    """Return the effective OSKAR or WSClean parameters for a single ImgConfig block."""
    config = manifest.config
    n_channels = int(config.observation.n_channels or 1)
    output_prefix = _first_image_product_id(manifest, img_config.imager)
    visibility_input = _first_output_path(manifest, "visibility")
    pixel_size_arcsec = None
    if fov_deg is not None:
        pixel_size_arcsec = fov_deg * 3600.0 / img_config.pixels

    if img_config.imager == "wsclean":
        # reflect any explicit overrides; default values match build_wsclean_argv
        effective_channels_out = (
            img_config.channels_out
            if img_config.channels_out is not None
            else min(n_channels, 8)
        )
        rows = [
            ("Imager", "WSClean"),
            ("Command", img_config.wsclean_command),
            ("Weighting", f"Briggs robust {_format_float(img_config.robust)}"),
            (
                "Multiscale",
                "enabled" if img_config.multiscale is not False else "disabled",
            ),
            ("Image size", f"{img_config.pixels} x {img_config.pixels} pixels"),
            (
                "Pixel scale",
                f"{_format_float(pixel_size_arcsec, 4)} arcsec"
                if pixel_size_arcsec is not None
                else "unknown",
            ),
            ("Clean iterations", str(img_config.clean_iterations)),
            (
                "Major-cycle gain",
                str(img_config.mgain if img_config.mgain is not None else 0.8),
            ),
            (
                "Auto threshold",
                str(
                    img_config.auto_threshold
                    if img_config.auto_threshold is not None
                    else 0.3
                ),
            ),
            (
                "Auto mask",
                str(img_config.auto_mask if img_config.auto_mask is not None else 3.0),
            ),
            ("Channels out", str(effective_channels_out)),
            (
                "Join channels",
                "enabled" if img_config.join_channels is not False else "disabled",
            ),
            (
                "Local RMS",
                "enabled" if img_config.local_rms is not False else "disabled",
            ),
        ]
        if img_config.multiscale_scales:
            rows.append(
                (
                    "Multiscale scales",
                    ", ".join(str(s) for s in img_config.multiscale_scales),
                )
            )
        if img_config.padding is not None:
            rows.append(("Padding", str(img_config.padding)))
        if img_config.threads is not None:
            rows.append(("Threads", str(img_config.threads)))
        if output_prefix is not None:
            rows.append(("Output prefix", output_prefix))
        if visibility_input is not None:
            rows.append(("Visibility input", visibility_input))
        return rows

    rows = [
        ("Imager", "OSKAR dirty"),
        ("Image size", f"{img_config.pixels} x {img_config.pixels} pixels"),
        (
            "Pixel scale",
            f"{_format_float(pixel_size_arcsec, 4)} arcsec"
            if pixel_size_arcsec is not None
            else "unknown",
        ),
        ("Combine frequencies", "enabled"),
    ]
    if center is not None:
        rows.append(
            (
                "Phase centre",
                f"RA {_format_float(center[0], 6)} deg, "
                f"Dec {_format_float(center[1], 6)} deg",
            )
        )
    if visibility_input is not None:
        rows.append(("Visibility input", visibility_input))
    return rows


def _build_imaging_tabs(
    manifest: RunManifest,
    science_products: list[dict],
    fov_deg_default: float | None,
    center: tuple[float, float] | None,
) -> list[dict]:
    """Build one tab dict per imaging block with config summary, parameters, and products."""
    config = manifest.config
    if not config.imaging:
        return []

    geometry_records = _find_milestone_details(manifest, "image_geometry_resolved").get(
        "blocks", {}
    )

    tabs: list[dict] = []

    for img_config in config.imaging:
        tag = img_config.tag
        geometry = geometry_records.get(tag)
        effective_img_config = img_config
        if geometry is not None:
            effective_img_config = img_config.model_copy(
                update={
                    "pixels": geometry["effective_pixels"],
                    "fov_deg": geometry["effective_fov_deg"],
                    "cell_size_arcsec": geometry["effective_cell_size_arcsec"],
                }
            )

        # Compute FoV for this block
        fov_deg = effective_img_config.fov_deg
        if fov_deg is None:
            fov_deg = fov_deg_default

        # Find beam from this tag's products
        tag_products = [p for p in science_products if p.get("tag") == tag]
        beam = None
        for p in tag_products:
            if p.get("beam"):
                beam = p["beam"]
                break

        tabs.append(
            {
                "tag": tag,
                "imaging_summary": _imaging_summary_for_block(
                    effective_img_config, fov_deg, beam, geometry
                ),
                "imager_parameter_rows": _imager_parameter_rows_for_block(
                    effective_img_config,
                    manifest,
                    fov_deg,
                    center,
                ),
                "products": tag_products,
            }
        )

    return tabs


def _first_image_product_id(manifest: RunManifest, imager: str) -> str | None:
    """Return the first image product ID for an imager."""
    for output in manifest.outputs:
        if output.kind == "image_product" and output.imager == imager:
            return output.image_product_id
    return None


def _first_output_path(manifest: RunManifest, kind: str) -> str | None:
    """Return the first output path for a manifest output kind."""
    for output in manifest.outputs:
        if output.kind == kind:
            return output.path
    return None


def _find_milestone_details(manifest: RunManifest, name: str) -> dict:
    """Return details for the first milestone with a matching name."""
    for milestone in manifest.milestones:
        if milestone.name == name:
            return milestone.details or {}
    return {}


def _antenna_count(manifest: RunManifest) -> int | None:
    """Return antenna/station count from the manifest or known telescope metadata."""
    details = _find_milestone_details(manifest, "telescope_built")
    count = details.get("n_stations")
    if count is not None:
        return int(count)
    return KNOWN_ANTENNA_COUNTS.get(manifest.config.telescope.upper())


def _maximum_baseline_m(manifest: RunManifest) -> Optional[str]:
    """Return the resolved maximum baseline for display when the run recorded it."""
    value = _find_milestone_details(manifest, "image_geometry_resolved").get(
        "maximum_baseline_m"
    )
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return None
    return f"{_format_float(value, 3)} m"


def _format_jy_amount(val_jy: float) -> str:
    """Format a plain flux/noise amount using the most suitable unit (no '/beam' suffix)."""
    val_abs = abs(val_jy)
    if val_abs >= 1.0:
        return f"{val_jy:.3f} Jy"
    elif val_abs >= 1e-3:
        return f"{val_jy * 1e3:.3f} mJy"
    elif val_abs >= 1e-6:
        return f"{val_jy * 1e6:.2f} μJy"
    else:
        return f"{val_jy * 1e9:.2f} nJy"


def _noise_summary(manifest: RunManifest) -> dict:
    """Build a display summary for injected thermal noise, when enabled.

    Returns {"enabled": False} when noise was not injected, so the report can
    omit the row rather than imply noise-free runs carry a meaningful zero.
    """
    config = manifest.config
    enabled = bool(config.rms) or config.noise_rms_start is not None
    if not enabled:
        return {"enabled": False}
    if config.noise_rms_start is not None:
        start = config.noise_rms_start
        end = config.noise_rms_end if config.noise_rms_end is not None else start
        if start == end:
            detail = f"Fixed override: {_format_jy_amount(start)} RMS"
        else:
            detail = (
                f"Range override: {_format_jy_amount(start)} – "
                f"{_format_jy_amount(end)} RMS"
            )
    else:
        detail = "Telescope model (per-station SEFD-based RMS)"
    return {"enabled": True, "detail": detail}


_FRIENDLY_MILESTONE_LABELS = {
    "uv_coverage_failed": "UV coverage plot",
    "simulation_failed": "Simulation",
    "sky_model_previews_failed": "Sky model preview plots",
}

PHASE_CENTRE_TOLERANCE_ARCSEC = 10.0


def _phase_centre_warning(manifest: RunManifest) -> Optional[str]:
    """Flag when the resolved phase centre drifts from the sky model's own centre.

    ``SimConfig.center`` can override the observation's phase centre away from
    the sky model's natural centre (catalog centroid, FITS image centre, ...).
    A drift beyond tolerance usually means the imaged field is no longer
    centred on where the injected sources actually are.
    """
    details = _find_milestone_details(manifest, "observation_configured")
    required = (
        "phase_center_ra_deg",
        "phase_center_dec_deg",
        "sky_model_center_ra_deg",
        "sky_model_center_dec_deg",
    )
    if not all(details.get(k) is not None for k in required):
        return None
    effective = SkyCoord(
        ra=details["phase_center_ra_deg"] * u.deg,
        dec=details["phase_center_dec_deg"] * u.deg,
    )
    sky_model_centre = SkyCoord(
        ra=details["sky_model_center_ra_deg"] * u.deg,
        dec=details["sky_model_center_dec_deg"] * u.deg,
    )
    offset_arcsec = effective.separation(sky_model_centre).arcsec
    if offset_arcsec <= PHASE_CENTRE_TOLERANCE_ARCSEC:
        return None
    return (
        f"Phase centre is {offset_arcsec:.1f}\" from the sky model's own centre "
        f'(tolerance {PHASE_CENTRE_TOLERANCE_ARCSEC:.0f}")'
    )


def _report_warnings(manifest: RunManifest) -> list[str]:
    """Collect run-level issues worth flagging near the top of the report.

    Surfaces failed milestones (including ones that a non-fatal `except` may
    have swallowed without aborting the run) and image-geometry warnings,
    both of which otherwise only show up buried in the timeline or in a
    collapsed imaging tab.
    """
    warnings: list[str] = []
    for ms in manifest.milestones:
        if ms.status != "failed":
            continue
        label = _FRIENDLY_MILESTONE_LABELS.get(ms.name)
        if (
            label is None
            and ms.name.startswith("imaging_")
            and ms.name.endswith("_failed")
        ):
            tag = (ms.details or {}).get(
                "tag", ms.name[len("imaging_") : -len("_failed")]
            )
            label = f"Imaging [{tag}]"
        if label is None:
            label = ms.name.replace("_", " ").capitalize()
        message = f"{label} failed"
        error = (ms.details or {}).get("error")
        if error:
            message += f": {error}"
        warnings.append(message)

    phase_centre_warning = _phase_centre_warning(manifest)
    if phase_centre_warning:
        warnings.append(phase_centre_warning)

    geometry = _find_milestone_details(manifest, "image_geometry_resolved")
    for tag, block in (geometry.get("blocks") or {}).items():
        for w in block.get("warnings") or []:
            warnings.append(f"Geometry [{tag}]: {w}")

    warnings.extend(_footprint_warnings(manifest))

    return warnings


def _footprint_warnings(manifest: RunManifest) -> list[str]:
    """Flag image models reaching beyond the beam half-power or field radius."""
    footprints = _find_milestone_details(manifest, "image_model_footprints")
    messages = []
    for record in footprints.get("entries") or []:
        if record.get("beyond_field"):
            limit = "simulation field radius"
            limit_deg = record.get("field_radius_deg")
            effect = "its outer parts are strongly attenuated"
        elif record.get("beyond_beam_half_power"):
            limit = "beam half-power radius"
            limit_deg = record.get("beam_half_power_radius_deg")
            effect = "its outer parts are attenuated by more than 50%"
        else:
            continue
        messages.append(
            f"Image model {record.get('model_entry_index')} "
            f"({record.get('model_type')}) reaches "
            f"{record.get('footprint_radius_deg', float('nan')) * 60:.2f}' from the "
            f"phase centre, beyond the {limit} "
            f"({(limit_deg or float('nan')) * 60:.2f}'); {effect}."
        )
    return messages


def _milestone_detail_text(ms) -> str:
    """Return a concise, human-readable one-liner for a milestone's details."""
    details = ms.details or {}
    if not details:
        return ""
    error = details.get("error")
    if error:
        text = str(error)
        return f"Error: {text[:160]}{'…' if len(text) > 160 else ''}"
    if ms.name == "sky_model_loaded":
        return f"{details.get('n_sources', '?')} sources ({details.get('format', '?')})"
    if ms.name == "telescope_built":
        return f"{details.get('n_stations', '?')} stations"
    if ms.name == "image_geometry_resolved":
        parts = []
        if details.get("maximum_baseline_m") is not None:
            parts.append(
                f"max baseline {_format_float(details['maximum_baseline_m'], 1)} m"
            )
        if details.get("simulation_fov_deg") is not None:
            parts.append(f"sim FoV {_format_float(details['simulation_fov_deg'], 4)}°")
        n_blocks = len(details.get("blocks") or {})
        if n_blocks:
            parts.append(f"{n_blocks} imaging block{'s' if n_blocks != 1 else ''}")
        return " · ".join(parts)
    if ms.name.startswith("imaging_") and ("imager" in details or "tag" in details):
        parts = [str(details[k]) for k in ("imager", "tag") if details.get(k)]
        return " · ".join(parts)
    if ms.name == "uv_coverage_completed":
        return f"plot: {details.get('path', '?')}"
    if ms.name == "base_ms_fallback":
        return f"strategy: {details.get('strategy', '?')}"
    text = json.dumps(details)
    return text if len(text) <= 120 else text[:117] + "..."


def _build_timeline_rows(manifest: RunManifest) -> list[dict]:
    """Build display rows for the timeline table, with friendly detail text."""
    rows = []
    for ms in manifest.milestones:
        duration = None
        if ms.elapsed_s:
            minutes, seconds = divmod(int(ms.elapsed_s), 60)
            duration = f"{minutes}m {seconds}s"
        rows.append(
            {
                "time": ms.timestamp_utc.strftime("%H:%M:%S"),
                "name": ms.name,
                "status": ms.status,
                "duration": duration,
                "detail": _milestone_detail_text(ms),
            }
        )
    return rows


def _humanize_bytes(n: float) -> str:
    """Format a byte count using the most suitable binary unit."""
    value = float(n)
    if value < 1024.0:
        return f"{int(value)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024.0
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TB"


def _run_disk_usage(work_dir: Path) -> Optional[str]:
    """Return the total on-disk footprint of the run directory, formatted for display."""
    total = 0
    try:
        for path in work_dir.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return None
    if total == 0:
        return None
    return _humanize_bytes(total)


def _skasim_git_commit() -> Optional[str]:
    """Return the short git commit of the running skasim checkout, when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _format_invocation(manifest: RunManifest) -> Optional[str]:
    """Return the CLI invocation that produced this run, shell-quoted for display."""
    argv = manifest.invocation
    if not argv:
        return None
    return " ".join(shlex.quote(a) for a in argv)


def _software_versions() -> list[tuple[str, str]]:
    """Return concise runtime software versions for weblog reproducibility."""
    versions = [
        ("skasim", _package_version("skasim")),
        ("Karabo", _package_version("karabo-pipeline")),
        ("OSKAR", _conda_package_version("oskarpy") or _package_version("oskarpy")),
        ("WSClean", _conda_package_version("wsclean") or _package_version("wsclean")),
    ]
    commit = _skasim_git_commit()
    if commit:
        versions.append(("skasim commit", commit))
    casa_ver = _casa_version()
    if casa_ver:
        versions.append(("CASA", casa_ver))
    versions.append(("Python", sys.version.split()[0]))
    return versions


def _package_version(package_name: str) -> str:
    """Return Python package metadata version or an explicit unknown marker."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _casa_version() -> str | None:
    """Detect CASA version from casatasks, python-casacore, or casa binary.

    Returns None when no CASA installation is found (skips the entry entirely).
    """
    # casatasks ships with the CASA monolithic distribution
    try:
        from casatasks import __version__ as casa_ver

        return str(casa_ver)
    except Exception:
        pass
    # python-casacore is the lightweight bindings package
    try:
        import casacore

        return f"casacore {casacore.__version__}"
    except Exception:
        pass
    # fallback: check for casa binary in PATH and scrape version
    import shutil

    casa_bin = shutil.which("casa")
    if casa_bin:
        return "casa (binary)"
    return None


def _conda_package_version(package_name: str) -> str | None:
    """Return Conda package version from the active prefix when available."""
    conda_meta = Path(sys.prefix) / "conda-meta"
    if not conda_meta.exists():
        return None
    for path in sorted(conda_meta.glob(f"{package_name}-*.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if metadata.get("name") == package_name and metadata.get("version"):
            return str(metadata["version"])
    return None


def _calculate_image_stats(clean_path: Path, residual_path: Path) -> dict | None:
    """calculate peak, residual RMS, and peak/RMS from FITS image data."""
    if not clean_path.exists() or not residual_path.exists():
        return None
    try:
        clean_data = _read_fits_image(clean_path)
        residual_data = _read_fits_image(residual_path)
    except Exception:
        return None
    finite_clean = clean_data[np.isfinite(clean_data)]
    finite_residual = residual_data[np.isfinite(residual_data)]
    if finite_clean.size == 0 or finite_residual.size == 0:
        return None

    peak_jy = float(np.nanmax(finite_clean))
    rms_jy = float(np.nanstd(finite_residual))
    snr = peak_jy / rms_jy if rms_jy > 0 else None
    return {
        "peak": _format_flux(peak_jy),
        "rms": _format_flux(rms_jy),
        "snr": f"{snr:.1f}" if snr is not None else None,
    }


def _read_fits_image(path: Path) -> np.ndarray:
    """read a FITS image and reduce singleton axes to a 2D array."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"{path} has no image data")
        data = np.asarray(data).squeeze()
    while data.ndim > 2:
        data = data[0]
    return data


def _format_flux(val_jy: float) -> str:
    """Format flux value dynamically using the most suitable unit (Jy, mJy, uJy, nJy)."""
    val_abs = abs(val_jy)
    if val_abs >= 1.0:
        return f"{val_jy:.3f} Jy/beam"
    elif val_abs >= 1e-3:
        return f"{val_jy * 1e3:.3f} mJy/beam"
    elif val_abs >= 1e-6:
        return f"{val_jy * 1e6:.2f} μJy/beam"
    else:
        return f"{val_jy * 1e9:.2f} nJy/beam"


def _file_to_base64_data_uri(fpath: Path) -> str:
    """read a binary image and return a base64 data URI string."""
    data = fpath.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    # infer mime type from extension
    ext = fpath.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _infer_image_caption(filename: str) -> str:
    """return a human-readable caption from an output filename."""
    fname = Path(filename).stem
    lower = fname.lower()
    if "dirty" in lower:
        if "oskar" in lower or "-dirty" in lower:
            return "Dirty image (OSKAR)"
        return "Dirty image"
    if "psf" in lower:
        return "Point spread function"
    if "residual" in lower:
        return "Residual"
    if "model" in lower:
        return "Component model"
    if "clean" in lower or "image" in lower:
        return "Cleaned image (WSClean)"
    if "telescope" in lower:
        return "Telescope layout"
    return fname  # fallback to filename


_PHASE_CENTRE_RULE_LABELS = {
    "explicit": "explicit center",
    "catalogue": "catalogue centre",
    "image_model": "first image model",
    "default_field": "default field",
    "generated_sources": "generated sources",
}

_COMPONENT_TYPE_LABELS = {
    "continuum_i_alpha": "Continuum I+α",
    "static_stokes_maps": "Static Stokes maps",
    "casa_taylor_terms": "Taylor terms",
    "spectral_cube": "Spectral cube",
}


def _attenuation_percent(intrinsic, attenuated) -> Optional[float]:
    if intrinsic in (None, 0) or attenuated is None:
        return None
    return 100.0 * (1.0 - float(attenuated) / float(intrinsic))


def _flux_display_unit(values) -> tuple:
    """Pick one unit and decimal count so a flux column reads consistently.

    Returns ``(unit, scale, decimals)`` with four significant digits for the
    largest value.
    """
    finite = [abs(float(v)) for v in values if v is not None and np.isfinite(v)]
    peak = max(finite) if finite else 0.0
    for unit, scale in (("Jy", 1.0), ("mJy", 1e3), ("µJy", 1e6)):
        if peak * scale >= 1.0 or unit == "µJy":
            break
    scaled_peak = peak * scale
    magnitude = int(np.floor(np.log10(scaled_peak))) if scaled_peak > 0 else 0
    return unit, scale, max(0, 3 - magnitude)


def _sky_composition(manifest: RunManifest) -> Optional[dict]:
    """Per-component positions and fluxes for combined (multimodal) runs.

    Returns ``None`` unless the run injected at least one image model. The
    catalogue row and every image row use the same beam, so the attenuated
    fluxes are directly comparable.
    """
    injected = [
        m.details or {} for m in manifest.milestones if m.name == "image_model_injected"
    ]
    if not injected:
        return None

    footprints = {
        rec.get("model_entry_index"): rec
        for rec in _find_milestone_details(manifest, "image_model_footprints").get(
            "entries"
        )
        or []
    }
    rows = []
    catalogue = _find_milestone_details(manifest, "catalogue_beam_flux")
    if catalogue:
        rows.append(
            {
                "component": "Catalogue",
                "color": "catalog",
                "source": f"{catalogue.get('n_sources')} sources",
                "offset": (
                    f"≤ {catalogue['max_offset_deg'] * 60:.2f}′"
                    if catalogue.get("max_offset_deg") is not None
                    else "—"
                ),
                "placement": "OSKAR",
                "intrinsic_jy": catalogue.get("intrinsic_flux_jy_at_ref"),
                "attenuated_jy": catalogue.get("attenuated_flux_jy_at_ref"),
            }
        )
    for details in injected:
        index = details.get("model_entry_index")
        footprint = footprints.get(index, {})
        model_type = details.get("model_type", "")
        if details.get("resampled"):
            placement = "regridded + shift"
        elif details.get("shift_applied"):
            placement = "shift"
        elif details.get("wcs_placement") == "fits_wcs":
            placement = "centred"
        elif details.get("backend") == "casa_ft":
            placement = "CASA ft"
        else:
            placement = "phase centre assumed"
        offset = details.get("offset_arcsec")
        if offset is None and footprint.get("centre_offset_deg") is not None:
            offset = footprint["centre_offset_deg"] * 3600.0
        source = footprint.get("source") or details.get("prefix") or ""
        rows.append(
            {
                "component": f"Image {index}",
                "color": "continuum",
                "type": _COMPONENT_TYPE_LABELS.get(model_type, model_type),
                "source": Path(str(source)).name if source else "",
                "offset": f"{offset / 60.0:.2f}′" if offset is not None else "—",
                "placement": placement,
                "intrinsic_jy": details.get("intrinsic_flux_jy_at_ref"),
                "attenuated_jy": details.get("attenuated_flux_jy_at_ref"),
                "beyond_half_power": footprint.get("beyond_beam_half_power"),
            }
        )
    for row in rows:
        row["attenuation_pct"] = _attenuation_percent(
            row.get("intrinsic_jy"), row.get("attenuated_jy")
        )
    flux_unit, flux_scale, flux_decimals = _flux_display_unit(
        [row.get("intrinsic_jy") for row in rows]
        + [row.get("attenuated_jy") for row in rows]
    )
    for row in rows:
        for key in ("intrinsic", "attenuated"):
            value = row.get(f"{key}_jy")
            row[f"{key}_text"] = (
                f"{value * flux_scale:.{flux_decimals}f}" if value is not None else "—"
            )

    rule = _find_milestone_details(manifest, "phase_centre_resolved")
    beam = _find_milestone_details(manifest, "primary_beam_configured")
    completed = _find_milestone_details(manifest, "image_injection_completed")
    ref_hz = beam.get("ref_freq_hz") or next(
        (d.get("beam_ref_freq_hz") for d in injected if d.get("beam_ref_freq_hz")),
        None,
    )
    return {
        "rows": rows,
        "flux_unit": flux_unit,
        "phase_centre_rule": _PHASE_CENTRE_RULE_LABELS.get(
            rule.get("rule"), rule.get("rule")
        ),
        "phase_centre": (
            SkyCoord(rule["ra_deg"] * u.deg, rule["dec_deg"] * u.deg).to_string(
                "hmsdms", precision=2
            )
            if rule.get("ra_deg") is not None
            else None
        ),
        "beam_power_fwhm_arcmin": (
            beam["power_fwhm_deg_at_ref"] * 60.0
            if beam.get("power_fwhm_deg_at_ref") is not None
            else None
        ),
        "ref_freq_mhz": ref_hz / 1e6 if ref_hz else None,
        "model_data_last_entry_only": completed.get("model_data_contents")
        == "last_entry_only",
        "n_merges": completed.get("n_merges"),
    }


def _sky_model_summary(manifest: RunManifest) -> list[dict]:
    """Build a list of sky-model input descriptors for the weblog.

    Each dict has at least 'label' and 'color' (CSS class for the badge),
    and may include 'source' (file path or catalog name), 'n_sources',
    'nterms', 'ref_freq_hz', 'stokes'.
    """
    from .config import SpectralCubeModelEntry

    config = manifest.config
    entries: list[dict] = []
    sky_ms_details = _find_milestone_details(manifest, "sky_model_loaded")

    # typed models first
    for model in config.models:
        mtype = model.type
        if mtype == "component_sky_model":
            if model.catalog:
                entries.append(
                    {
                        "label": "Catalog",
                        "color": "catalog",
                        "source": model.catalog,
                        **_flux_polarization_fields(sky_ms_details),
                    }
                )
            elif model.path:
                entries.append(
                    {
                        "label": "FITS catalog",
                        "color": "fits-catalog",
                        "source": model.path,
                        **_flux_polarization_fields(sky_ms_details),
                    }
                )
        elif mtype == "continuum_i_alpha":
            entries.append(
                {
                    "label": "Continuum I+α",
                    "color": "continuum",
                    "source": model.stokes_i,
                }
            )
        elif mtype == "casa_taylor_terms":
            nterms = 2 if model.tt1 else 1
            entries.append(
                {
                    "label": f"Taylor terms (nterms={nterms})",
                    "color": "taylor",
                    "source": model.tt0,
                    "nterms": nterms,
                    "ref_freq_hz": model.reference_frequency_hz,
                }
            )
        elif mtype == "static_stokes_maps":
            stokes = [
                k.upper()
                for k in ("i", "q", "u", "v")
                if getattr(model, f"stokes_{k}") is not None
            ]
            entries.append(
                {
                    "label": "Static Stokes maps",
                    "color": "stokes-maps",
                    "stokes": stokes,
                }
            )
        elif mtype == "spectral_cube":
            info: FitsCubeInfo | None = None
            if isinstance(model, SpectralCubeModelEntry):
                try:
                    info = read_fits_cube_info(Path(model.cube).expanduser().resolve())
                except Exception:
                    pass
            entry: dict = {
                "label": "Spectral cube",
                "color": "spectral-cube",
                "source": getattr(model, "cube", None),
            }
            if info:
                entry["n_channels"] = info.n_channels
                entry["channel_width_khz"] = info.channel_width_hz / 1e3
                entry["central_frequency_mhz"] = info.reference_frequency_hz / 1e6
                entry["spatial_shape"] = info.spatial_shape
            entries.append(entry)

    # legacy paths
    if not config.models:
        milestone = sky_ms_details
        fmt = milestone.get("format", "?")
        n_sources = milestone.get("n_sources")
        flux_fields = _flux_polarization_fields(milestone)
        if config.sky_file:
            entries.append(
                {
                    "label": f"FITS file ({fmt})",
                    "color": "fits-file",
                    "source": str(milestone.get("path", config.sky_file)),
                    "n_sources": n_sources,
                    **flux_fields,
                }
            )
        elif config.catalog:
            entries.append(
                {
                    "label": "Built-in catalog",
                    "color": "catalog",
                    "source": config.catalog,
                    "n_sources": n_sources,
                    **flux_fields,
                }
            )
        elif config.fits_image:
            entries.append(
                {
                    "label": "FITS image",
                    "color": "fits-file",
                    "source": config.fits_image,
                    "n_sources": n_sources,
                    **flux_fields,
                }
            )
        elif fmt == "random":
            entries.append(
                {
                    "label": "Random sources",
                    "color": "random",
                    "n_sources": n_sources,
                    "reference": milestone.get("reference"),
                    **flux_fields,
                }
            )

    return entries


def _flux_polarization_fields(details: dict) -> dict:
    """Return formatted flux-range/polarization display fields from sky-model details.

    ``details`` is a ``sky_model_loaded`` milestone's details dict; older
    manifests without ``flux_min_jy``/``flux_max_jy``/``has_polarization``
    simply yield {} so the report omits the row rather than showing "None".
    """
    fields: dict = {}
    flux_min = details.get("flux_min_jy")
    flux_max = details.get("flux_max_jy")
    if flux_min is not None and flux_max is not None:
        fields["flux_range"] = (
            f"{_format_jy_amount(flux_min)} – {_format_jy_amount(flux_max)}"
        )
    if "has_polarization" in details:
        fields["polarization_label"] = (
            "polarized (Q/U/V)" if details["has_polarization"] else "Stokes I only"
        )
    return fields


def render_weblog(manifest: RunManifest, work_dir: Path) -> str:
    """generate weblog.html content and write it to disk. returns the html string."""
    env = Environment(
        loader=PackageLoader("skasim", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("weblog.html.j2")

    # compute summary stats
    total_elapsed = None
    if manifest.completed_at and manifest.started_at:
        total_elapsed = (manifest.completed_at - manifest.started_at).total_seconds()

    milestone_lookup = {m.name: m for m in manifest.milestones}
    simulation_duration = None
    if (
        "simulation_started" in milestone_lookup
        and "simulation_completed" in milestone_lookup
    ):
        simulation_duration = (
            milestone_lookup["simulation_completed"].timestamp_utc
            - milestone_lookup["simulation_started"].timestamp_utc
        ).total_seconds()

    # aggregate imaging duration across all tags
    imaging_starts = [
        m.timestamp_utc
        for m in manifest.milestones
        if m.name.endswith("_started") and m.name.startswith("imaging_")
    ]
    imaging_ends = [
        m.timestamp_utc
        for m in manifest.milestones
        if m.name.endswith("_completed") and m.name.startswith("imaging_")
    ]
    imaging_duration = None
    if imaging_starts and imaging_ends:
        imaging_duration = (max(imaging_ends) - min(imaging_starts)).total_seconds()

    # Derived telescope properties
    from .utils import get_diameter

    dish_diameter = None
    derived_fov = None
    fov_deg_value = (
        manifest.config.imaging[0].fov_deg if manifest.config.imaging else None
    )
    telescope_name = manifest.config.telescope
    freq_mhz = manifest.config.observation.frequency_mhz

    try:
        diam = get_diameter(telescope_name)
        dish_diameter = f"{diam.value:.1f} {diam.unit}"
        if freq_mhz:
            wavelength_m = 299.792458 / freq_mhz
            fov_rad = 1.25 * wavelength_m / diam.value
            fov_deg_val = fov_rad * 180.0 / 3.141592653589793
            if fov_deg_value is None:
                fov_deg_value = fov_deg_val
            derived_fov = f"{fov_deg_val:.2f}°"
    except Exception:
        pass

    n_stations = _antenna_count(manifest)
    maximum_baseline = _maximum_baseline_m(manifest)

    # Resolve telescope plot if present
    telescope_plot = None
    for o in manifest.outputs:
        output_path = o.path if hasattr(o, "path") else o
        if "telescope" in output_path.lower() or o.role == "telescope":
            fpath = work_dir / output_path
            if fpath.exists():
                telescope_plot = {
                    "path": output_path,
                    "data": _file_to_base64_data_uri(fpath),
                }
            break

    # Resolve UV coverage plot if present
    uv_coverage_plot = None
    for o in manifest.outputs:
        output_path = o.path if hasattr(o, "path") else o
        role = getattr(o, "role", None)
        if role == "uv_coverage" or "uvcoverage" in output_path.lower():
            fpath = work_dir / output_path
            if fpath.exists() and output_path.endswith((".png", ".jpg", ".jpeg")):
                uv_coverage_plot = {
                    "path": output_path,
                    "data": _file_to_base64_data_uri(fpath),
                }
                break

    # Resolve sky model plot if present
    sky_model_plot = None
    for o in manifest.outputs:
        output_path = o.path if hasattr(o, "path") else o
        role = getattr(o, "role", None)
        if role == "sky_model" or (
            "sky_model" in output_path.lower() and "_fov" not in output_path.lower()
        ):
            fpath = work_dir / output_path
            if fpath.exists():
                sky_model_plot = {
                    "path": output_path,
                    "data": _file_to_base64_data_uri(fpath),
                }
            break

    # Resolve sky model fov plot if present
    sky_model_fov_plot = None
    for o in manifest.outputs:
        output_path = o.path if hasattr(o, "path") else o
        role = getattr(o, "role", None)
        if role == "sky_model_fov" or (
            "sky_model" in output_path.lower() and "_fov" in output_path.lower()
        ):
            fpath = work_dir / output_path
            if fpath.exists():
                sky_model_fov_plot = {
                    "path": output_path,
                    "data": _file_to_base64_data_uri(fpath),
                }
            break

    # Resolve input-cube moment8 plot if present
    input_cube_moment8_plot = None
    for o in manifest.outputs:
        output_path = o.path if hasattr(o, "path") else o
        role = getattr(o, "role", None)
        if role != "input_cube_moment8":
            continue
        if not str(output_path).lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        fpath = work_dir / str(output_path)
        if fpath.exists():
            input_cube_moment8_plot = {
                "path": str(output_path),
                "data": _file_to_base64_data_uri(fpath),
            }
        break

    # Resolve FITS image-model preview if present
    fits_model_plots = []
    for o in manifest.outputs:
        output_path = o.path if hasattr(o, "path") else o
        role = getattr(o, "role", None)
        if role != "fits_model":
            continue
        fpath = work_dir / output_path
        if fpath.exists() and output_path.endswith((".png", ".jpg", ".jpeg")):
            metadata = o.metadata if hasattr(o, "metadata") else {}
            fits_model_plots.append(
                {
                    "path": output_path,
                    "data": _file_to_base64_data_uri(fpath),
                    "model_type": metadata.get("model_type"),
                    "image_size": metadata.get("image_size"),
                    "pixel_scale_arcsec": metadata.get("pixel_scale_arcsec"),
                    "extent_width_deg": metadata.get("extent_width_deg"),
                    "extent_height_deg": metadata.get("extent_height_deg"),
                    "center_ra_deg": metadata.get("center_ra_deg"),
                    "center_dec_deg": metadata.get("center_dec_deg"),
                    "offset_from_center_arcsec": metadata.get(
                        "offset_from_center_arcsec"
                    ),
                    "fov_fraction": metadata.get("fov_fraction"),
                }
            )

    observation_details = _find_milestone_details(manifest, "observation_configured")
    center = None
    if (
        "phase_center_ra_deg" in observation_details
        and "phase_center_dec_deg" in observation_details
    ):
        center = (
            float(observation_details["phase_center_ra_deg"]),
            float(observation_details["phase_center_dec_deg"]),
        )

    sky_model_summary = _sky_model_summary(manifest)
    sky_composition = _sky_composition(manifest)
    sky_composition_plot = None
    for o in manifest.outputs:
        if getattr(o, "role", None) != "sky_composition":
            continue
        fpath = work_dir / o.path
        if fpath.exists():
            sky_composition_plot = {
                "path": o.path,
                "data": _file_to_base64_data_uri(fpath),
            }
        break
    science_products = _find_science_products(manifest, work_dir)
    imaging_tabs = _build_imaging_tabs(
        manifest,
        science_products,
        fov_deg_value,
        center,
    )

    html = template.render(
        manifest=manifest,
        total_elapsed=_humanize_seconds(total_elapsed) if total_elapsed else None,
        simulation_duration=(
            _humanize_seconds(simulation_duration) if simulation_duration else None
        ),
        imaging_duration=(
            _humanize_seconds(imaging_duration) if imaging_duration else None
        ),
        images=_find_image_outputs(manifest, work_dir),
        imaging_tabs=imaging_tabs,
        telescope_plot=telescope_plot,
        uv_coverage_plot=uv_coverage_plot,
        sky_model_plot=sky_model_plot,
        sky_model_fov_plot=sky_model_fov_plot,
        input_cube_moment8_plot=input_cube_moment8_plot,
        sky_model_summary=sky_model_summary,
        sky_composition=sky_composition,
        sky_composition_plot=sky_composition_plot,
        fits_model_plots=fits_model_plots,
        observation_summary=_observation_summary(manifest),
        noise_summary=_noise_summary(manifest),
        software_versions=_software_versions(),
        dish_diameter=dish_diameter,
        derived_fov=derived_fov,
        n_stations=n_stations,
        maximum_baseline=maximum_baseline,
        report_warnings=_report_warnings(manifest),
        timeline_rows=_build_timeline_rows(manifest),
        disk_usage=_run_disk_usage(work_dir),
        invocation_command=_format_invocation(manifest),
        imagers_used=_imagers_used(manifest),
        status_class=_STATUS_BADGE_CLASSES.get(manifest.status, "running"),
    )

    weblog_path = work_dir / "weblog.html"
    weblog_path.write_text(html, encoding="utf-8")
    return html
