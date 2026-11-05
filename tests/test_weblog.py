"""Weblog rendering behavior."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from astropy.io import fits

from skasim.config import ImgConfig, ObsConfig, SimConfig
from skasim.manifest import RunManifest
from skasim.weblog import _software_versions, render_weblog


def test_weblog_renders_structured_outputs(tmp_path):
    """The weblog displays structured output kinds and paths."""
    manifest = RunManifest(
        run_id="example",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    manifest.add_output("visibility", "visibilities.MS")
    manifest.add_output("log", "example.log")

    html = render_weblog(manifest, tmp_path)

    assert "visibility" in html
    assert "visibilities.MS" in html
    assert "log" in html
    assert "example.log" in html


def test_weblog_skips_missing_image_outputs(tmp_path):
    """Failed runs can render even if an image output record points to a missing file."""
    manifest = RunManifest(
        run_id="missing-image",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    manifest.add_output("image_product", "missing.png", imager="wsclean", role="image")
    manifest.mark_failed("imaging failed before writing image")

    html = render_weblog(manifest, tmp_path)

    assert "imaging failed before writing image" in html
    assert "missing.png" in html


def test_weblog_uses_pipeline_milestone_names_for_durations(tmp_path):
    """Weblog duration cards are derived from actual pipeline milestone names."""
    started = datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc)
    manifest = RunManifest(
        run_id="durations",
        started_at=started,
        config=SimConfig(),
    )
    manifest.add_milestone("simulation_started", "started")
    manifest.milestones[-1].timestamp_utc = started
    manifest.add_milestone("simulation_completed", "completed")
    manifest.milestones[-1].timestamp_utc = started + timedelta(seconds=75)
    manifest.add_milestone("imaging_started", "started")
    manifest.milestones[-1].timestamp_utc = started + timedelta(seconds=80)
    manifest.add_milestone("imaging_completed", "completed")
    manifest.milestones[-1].timestamp_utc = started + timedelta(seconds=90)

    html = render_weblog(manifest, tmp_path)

    assert "simulation 1m 15s" in html
    assert "imaging 10.0s" in html


def test_weblog_groups_wsclean_mfs_products_and_stats(tmp_path):
    """MFS products render as model/clean/residual with peak and RMS stats."""
    manifest = RunManifest(
        run_id="science-products",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    prefix = "science-products_wsclean"
    for role in ("dirty", "model", "image", "residual", "psf"):
        stem = f"{prefix}-MFS-{role if role != 'image' else 'image'}"
        fits_path = tmp_path / f"{stem}.fits"
        png_path = tmp_path / f"{stem}.png"
        data = np.array([[1.0e-3, 2.0e-3], [3.0e-3, 4.0e-3]])
        if role == "residual":
            data = np.array([[0.0, 1.0e-6], [-1.0e-6, 0.0]])
        header = fits.Header()
        header["BMAJ"] = 2.0 / 3600.0
        header["BMIN"] = 1.0 / 3600.0
        header["BPA"] = 35.0
        fits.writeto(fits_path, data, header, overwrite=True)
        png_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
            b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        preview_role = role
        manifest.add_output(
            "image_product",
            fits_path.name,
            image_product_id=prefix,
            imager="wsclean",
            role=preview_role,
        )
        manifest.add_output(
            "image_product",
            png_path.name,
            image_product_id=prefix,
            imager="wsclean",
            role=f"{preview_role}_preview",
        )

    html = render_weblog(manifest, tmp_path)

    assert "Science Products" in html
    assert "Model" in html
    assert "Clean" in html
    assert "Residual" in html
    assert "Dirty Image" not in html
    assert "Point Spread Function" in html
    assert "PSF" in html or "x/y profile" in html or "Point Spread Function" in html
    assert f"{prefix}-MFS-psf.png" in html
    assert "Peak:" in html
    assert "4.000 mJy/beam" in html
    assert "RMS:" in html
    assert "BMAJ 2.000 arcsec" in html
    assert "BMIN 1.000 arcsec" in html
    assert "BPA 35.00 deg" in html
    assert "Beam: 2.000 arcsec x 1.000 arcsec" in html
    assert html.count("Beam: 2.000 arcsec x 1.000 arcsec") == 1


def test_weblog_uses_dirty_image_only_when_clean_products_are_absent(tmp_path):
    """OSKAR dirty products render as the fallback science product."""
    manifest = RunManifest(
        run_id="dirty-only",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    fits_path = tmp_path / "dirty-only_dirty.fits"
    png_path = tmp_path / "dirty-only_dirty.png"
    fits.writeto(fits_path, np.ones((2, 2)), overwrite=True)
    png_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    manifest.add_output(
        "image_product",
        png_path.name,
        image_product_id="dirty-only",
        imager="oskar-dirty",
        role="preview",
    )
    manifest.add_output(
        "image_product",
        fits_path.name,
        image_product_id="dirty-only",
        imager="oskar-dirty",
        role="dirty",
    )

    html = render_weblog(manifest, tmp_path)

    assert "Dirty Image" in html


def test_weblog_renders_observation_imaging_and_cleaning_parameters(tmp_path):
    """Resolved setup tables expose spectral, imaging, and imager parameters."""
    manifest = RunManifest(
        run_id="setup-summary",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(
            observation=ObsConfig(
                frequency_mhz=1284.0,
                bandwidth_mhz=856.0,
                n_channels=8,
                observation_time_s=60,
            ),
            imaging=[
                ImgConfig(
                    imager="wsclean",
                    pixels=1024,
                    fov_deg=1.0,
                    robust=-0.5,
                    wsclean_command="wsclean",
                    clean_iterations=500,
                )
            ],
        ),
    )
    manifest.add_milestone(
        "observation_configured",
        "completed",
        details={
            "min_frequency_mhz": 856.0,
            "max_frequency_mhz": 1712.0,
            "n_timesteps": 7,
            "start_time_utc": "2026-10-07 23:39:00",
            "phase_center_ra_deg": 150.0,
            "phase_center_dec_deg": 2.0,
        },
    )
    manifest.add_output("visibility", "visibilities.MS")
    manifest.add_output(
        "image_product",
        "setup-summary_wsclean-MFS-image.fits",
        image_product_id="setup-summary_wsclean",
        imager="wsclean",
        role="image",
    )

    html = render_weblog(manifest, tmp_path)

    assert "Observation & Telescope" in html
    assert "Observation Start" in html
    assert "2026-10-07 23:39:00 UTC" in html
    assert "Frequency Setup" not in html
    assert "Frequency Range" in html
    assert "856-1712 MHz" in html
    assert "Central Frequency" in html
    assert "1284 MHz" in html
    assert "Channels" in html
    assert "8 x 107 MHz" in html
    assert "107 MHz" in html
    assert "Total Bandwidth" in html
    assert "856 MHz" in html
    assert "Imaging Products" in html
    assert "1024 x 1024 pixels" in html
    assert "Pixels</th>" not in html
    assert "Resolved Geometry" in html
    assert "1°" in html
    assert "3.5156 arcsec" in html
    assert "Clean iterations" in html
    assert "500" in html
    assert "Major-cycle gain" in html
    assert "0.8" in html
    assert "Auto threshold" in html
    assert "Auto mask" in html
    assert "Channels out" in html
    assert "Output prefix" in html
    assert "setup-summary_wsclean" in html
    assert "Visibility input" in html
    assert "visibilities.MS" in html
    assert "Total FoV" not in html
    assert "Pixel Size" not in html
    assert "Image Size" not in html


def test_weblog_renders_antenna_count_in_telescope_section(tmp_path):
    """Telescope metadata exposes the station/antenna count with a clear label."""
    manifest = RunManifest(
        run_id="antennas",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(telescope="MeerKAT"),
    )
    manifest.add_milestone(
        "telescope_built",
        "completed",
        details={"n_stations": 64},
    )

    html = render_weblog(manifest, tmp_path)

    assert "Antennas" in html
    assert "64" in html


def test_weblog_renders_maximum_baseline_in_metres(tmp_path):
    """Resolved array geometry exposes the maximum baseline in metres."""
    manifest = RunManifest(
        run_id="baseline",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(telescope="MeerKAT"),
    )
    manifest.add_milestone(
        "image_geometry_resolved",
        "completed",
        details={"maximum_baseline_m": 7698.1234},
    )

    html = render_weblog(manifest, tmp_path)

    assert "Maximum Baseline" in html
    assert "7698.123 m" in html


def test_weblog_renders_uv_coverage_next_to_telescope_layout(tmp_path):
    """UV coverage is rendered in the telescope section, not duplicated in the gallery."""
    manifest = RunManifest(
        run_id="uv-layout",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (tmp_path / "telescope.png").write_bytes(png_bytes)
    (tmp_path / "uvcoverage.png").write_bytes(png_bytes)
    manifest.add_output("plot", "telescope.png", role="telescope")
    manifest.add_output("plot", "uvcoverage.png", role="uv_coverage")

    html = render_weblog(manifest, tmp_path)

    assert "Telescope Layout" in html
    assert "UV Coverage" in html
    assert html.index("Telescope Layout") < html.index("UV Coverage")
    assert "<h2>Supporting Plots</h2>" not in html


def test_weblog_uses_known_antenna_count_for_older_meerkat_manifests(tmp_path):
    """Older manifests without telescope-built counts still show known MeerKAT antennas."""
    manifest = RunManifest(
        run_id="old-meerkat",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(telescope="MeerKAT"),
    )
    manifest.add_milestone(
        "telescope_built",
        "completed",
        details={"name": "MeerKAT", "version": None},
    )

    html = render_weblog(manifest, tmp_path)

    assert "Antennas" in html
    assert "64" in html


def test_weblog_renders_software_versions(tmp_path):
    """Weblog includes a concise reproducibility software version table."""
    manifest = RunManifest(
        run_id="versions",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )

    html = render_weblog(manifest, tmp_path)

    assert "Software Versions" in html
    for name, _ in _software_versions():
        assert name in html


def test_sky_model_summary_typed_models(tmp_path):
    """_sky_model_summary describes typed model entries with clear labels."""
    from skasim.weblog import _sky_model_summary

    stokes_i = tmp_path / "stokes_i.fits"
    alpha_f = tmp_path / "alpha.fits"
    tt0 = tmp_path / "tt0.image"
    tt1 = tmp_path / "tt1.image"
    for p in (stokes_i, alpha_f, tt0, tt1):
        p.touch()

    config = SimConfig(
        models=[
            {"type": "component_sky_model", "catalog": "MIGHTEE"},
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha_f),
                "reference_frequency_hz": 1.4e9,
            },
        ]
    )
    manifest = RunManifest(
        run_id="test",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config=config,
    )
    summary = _sky_model_summary(manifest)

    assert len(summary) == 2
    assert summary[0]["label"] == "Catalog"
    assert summary[0]["source"] == "MIGHTEE"
    assert summary[1]["label"] == "Continuum I+α"
    assert str(stokes_i) in summary[1]["source"]

    # CASA ft entries are only valid on their own (multimodal runs reject them).
    taylor_config = SimConfig(
        models=[
            {
                "type": "casa_taylor_terms",
                "tt0": str(tt0),
                "tt1": str(tt1),
                "reference_frequency_hz": 1.5e9,
            },
        ]
    )
    taylor_summary = _sky_model_summary(
        RunManifest(
            run_id="test-taylor",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            config=taylor_config,
        )
    )
    assert taylor_summary[0]["label"] == "Taylor terms (nterms=2)"
    assert taylor_summary[0]["nterms"] == 2


def test_sky_model_summary_legacy_catalog():
    """_sky_model_summary falls back to legacy catalog info."""
    from skasim.weblog import _sky_model_summary

    config = SimConfig(catalog="GLEAM")
    manifest = RunManifest(
        run_id="test",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config=config,
    )
    manifest.add_milestone(
        "sky_model_loaded", "completed", details={"format": "GLEAM", "n_sources": 42}
    )
    summary = _sky_model_summary(manifest)

    assert len(summary) == 1
    assert summary[0]["label"] == "Built-in catalog"
    assert summary[0]["source"] == "GLEAM"
    assert summary[0]["n_sources"] == 42


def test_sky_model_summary_weblog_renders_typed_labels(tmp_path):
    """The weblog HTML contains rendered labels for each typed model entry."""
    tt0_path = tmp_path / "tt0"
    tt0_path.touch()

    catalog_config = SimConfig(
        models=[{"type": "component_sky_model", "catalog": "MIGHTEE"}]
    )
    # CASA ft entries are only valid on their own (multimodal runs reject them).
    taylor_config = SimConfig(
        models=[
            {
                "type": "casa_taylor_terms",
                "tt0": str(tt0_path),
                "reference_frequency_hz": 1.5e9,
            },
        ]
    )

    htmls = []
    for run_id, config in (("catalog", catalog_config), ("taylor", taylor_config)):
        manifest = RunManifest(
            run_id=run_id,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            config=config,
        )
        htmls.append(render_weblog(manifest, tmp_path))

    assert "Catalog" in htmls[0]
    assert "Taylor terms" in htmls[1]


def test_phase_centre_warning_flags_drift_beyond_tolerance(tmp_path):
    """A phase centre drifting >10\" from the sky model's own centre is flagged."""
    from skasim.weblog import _report_warnings

    manifest = RunManifest(
        run_id="phase-drift",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    manifest.add_milestone(
        "observation_configured",
        "completed",
        details={
            "phase_center_ra_deg": 150.0,
            "phase_center_dec_deg": 2.0,
            "sky_model_center_ra_deg": 150.0,
            # 60 arcsec of dec drift, well beyond the 10" tolerance
            "sky_model_center_dec_deg": 2.0 + 60.0 / 3600.0,
        },
    )

    warnings = _report_warnings(manifest)

    assert any("Phase centre" in w and "60.0" in w for w in warnings)
    html = render_weblog(manifest, tmp_path)
    assert "Phase centre is 60.0" in html


def test_phase_centre_warning_absent_within_tolerance(tmp_path):
    """A small (<10\") phase-centre drift is not flagged as a warning."""
    from skasim.weblog import _report_warnings

    manifest = RunManifest(
        run_id="phase-ok",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    manifest.add_milestone(
        "observation_configured",
        "completed",
        details={
            "phase_center_ra_deg": 150.0,
            "phase_center_dec_deg": 2.0,
            "sky_model_center_ra_deg": 150.0,
            "sky_model_center_dec_deg": 2.0 + 1.0 / 3600.0,
        },
    )

    assert not any("Phase centre" in w for w in _report_warnings(manifest))


def test_sky_model_summary_includes_flux_range_and_polarization():
    """Flux range and polarization presence surface from sky_model_loaded details."""
    from skasim.weblog import _sky_model_summary

    config = SimConfig(catalog="GLEAM")
    manifest = RunManifest(
        run_id="flux-summary",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        config=config,
    )
    manifest.add_milestone(
        "sky_model_loaded",
        "completed",
        details={
            "format": "GLEAM",
            "n_sources": 3,
            "flux_min_jy": 0.002,
            "flux_max_jy": 4.5,
            "has_polarization": True,
        },
    )

    summary = _sky_model_summary(manifest)

    assert summary[0]["flux_range"] == "2.000 mJy – 4.500 Jy"
    assert summary[0]["polarization_label"] == "polarized (Q/U/V)"


def test_imaging_products_render_clean_before_model_and_residual(tmp_path):
    """Product figures render in Clean, Model, Residual order."""
    manifest = RunManifest(
        run_id="product-order",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    prefix = "product-order_wsclean"
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    for role in ("model", "image", "residual"):
        stem = f"{prefix}-MFS-{role}"
        fits_path = tmp_path / f"{stem}.fits"
        png_path = tmp_path / f"{stem}.png"
        fits.writeto(fits_path, np.ones((2, 2)), overwrite=True)
        png_path.write_bytes(png_bytes)
        manifest.add_output(
            "image_product",
            fits_path.name,
            image_product_id=prefix,
            imager="wsclean",
            role=role,
        )
        manifest.add_output(
            "image_product",
            png_path.name,
            image_product_id=prefix,
            imager="wsclean",
            role=f"{role}_preview",
        )

    html = render_weblog(manifest, tmp_path)

    assert (
        html.index("<figcaption>Clean</figcaption>")
        < html.index("<figcaption>Model</figcaption>")
        < html.index("<figcaption>Residual</figcaption>")
    )


def test_summary_card_reports_status_class_imagers_and_no_issues(tmp_path):
    """The top summary card uses a real status badge class and lists imagers used."""
    manifest = RunManifest(
        run_id="summary-card",
        started_at=datetime(2026, 5, 22, 17, 30, 0, tzinfo=timezone.utc),
        config=SimConfig(imaging=[ImgConfig(imager="wsclean")]),
    )
    manifest.mark_completed()

    html = render_weblog(manifest, tmp_path)

    assert 'class="badge success"' in html
    assert "WSClean" in html
    assert "No issues" in html


# ---------------------------------------------------------------------------
# multimodal input: sky composition table
# ---------------------------------------------------------------------------


def _composition_manifest():
    manifest = RunManifest(
        run_id="composition",
        started_at=datetime(2026, 11, 3, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    manifest.add_milestone(
        "phase_centre_resolved",
        "completed",
        details={"rule": "catalogue", "ra_deg": 150.0, "dec_deg": 2.0},
    )
    manifest.add_milestone(
        "primary_beam_configured",
        "completed",
        details={
            "power_fwhm_deg_at_ref": 0.35,
            "ref_freq_hz": 1.4e9,
            "field_fwhm_deg": 0.5,
        },
    )
    manifest.add_milestone(
        "catalogue_beam_flux",
        "completed",
        details={
            "n_sources": 12,
            "intrinsic_flux_jy_at_ref": 0.0125,
            "attenuated_flux_jy_at_ref": 0.01,
            "max_offset_deg": 0.2,
        },
    )
    manifest.add_milestone(
        "image_model_footprints",
        "completed",
        details={
            "entries": [
                {
                    "model_entry_index": 0,
                    "source": "/data/target.fits",
                    "centre_offset_deg": 0.1,
                    "beyond_beam_half_power": True,
                }
            ]
        },
    )
    manifest.add_milestone(
        "image_model_injected",
        "completed",
        details={
            "model_entry_index": 0,
            "model_type": "continuum_i_alpha",
            "backend": "wsclean_predict",
            "wcs_placement": "fits_wcs",
            "resampled": True,
            "shift_applied": True,
            "offset_arcsec": 360.0,
            "intrinsic_flux_jy_at_ref": 0.004,
            "attenuated_flux_jy_at_ref": 0.002,
        },
    )
    manifest.add_milestone(
        "image_injection_completed",
        "completed",
        details={"n_merges": 1, "model_data_contents": "last_entry_only"},
    )
    return manifest


def test_sky_composition_rows_share_one_flux_unit():
    from skasim.weblog import _sky_composition

    composition = _sky_composition(_composition_manifest())

    assert composition["flux_unit"] == "mJy"
    catalogue, image = composition["rows"]
    assert catalogue["component"] == "Catalogue"
    assert catalogue["intrinsic_text"] == "12.50"
    assert catalogue["attenuation_pct"] == pytest.approx(20.0)
    assert image["component"] == "Image 0"
    assert image["source"] == "target.fits"
    assert image["placement"] == "regridded + shift"
    assert image["offset"] == "6.00′"
    assert image["attenuated_text"] == "2.00"
    assert image["beyond_half_power"] is True
    assert composition["phase_centre_rule"] == "catalogue centre"
    assert composition["beam_power_fwhm_arcmin"] == pytest.approx(21.0)
    assert composition["model_data_last_entry_only"] is True


def test_sky_composition_absent_without_image_models():
    from skasim.weblog import _sky_composition

    manifest = RunManifest(
        run_id="catalogue-only",
        started_at=datetime(2026, 11, 3, tzinfo=timezone.utc),
        config=SimConfig(),
    )

    assert _sky_composition(manifest) is None


def test_weblog_renders_sky_composition_section(tmp_path):
    manifest = _composition_manifest()
    manifest.config = SimConfig(
        models=[{"type": "component_sky_model", "catalog": "GLEAM"}]
    )

    html = render_weblog(manifest, tmp_path)

    assert "Sky composition" in html
    assert "regridded + shift" in html
    assert "Intrinsic (mJy)" in html
    assert "holds only the last image model" in html
