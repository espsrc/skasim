"""tests/test_pipeline.py — unit tests for pipeline functions that don't invoke Karabo simulators."""

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from skasim.config import (
    CasaTaylorTermsModelEntry,
    ImgConfig,
    ObsConfig,
    SimConfig,
    SpectralCubeModelEntry,
)
from skasim.loaders.image_models import CasaModelProduct
from skasim.manifest import create_run_context
from skasim.pipeline import (
    _load_sky_from_file,
    build_sky_model,
    compute_fov,
    parse_center,
    resolve_telescope_version,
    run_simulation,
    source_ref_get_best_observation_time,
)
from skasim.sky import SkyModel


def _write_minimal_fits_cube(path: Path, n_channels: int = 8, pixels: int = 64) -> Path:
    """Create a tiny valid FITS cube for spectral-cube model tests."""
    path = Path(path)
    data = np.zeros((n_channels, pixels, pixels), dtype=np.float32)
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = pixels
    header["NAXIS2"] = pixels
    header["NAXIS3"] = n_channels
    header["CRPIX1"] = pixels / 2.0
    header["CRPIX2"] = pixels / 2.0
    header["CRPIX3"] = n_channels / 2.0
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    center_hz = 700e6
    header["CRPIX3"] = 1.0
    # align the cube exactly with an 8-channel 12.5 MHz observation centred at 700 MHz
    header["CRVAL3"] = center_hz - (n_channels - 1) / 2.0 * 12.5e6
    header["CDELT1"] = -0.05
    header["CDELT2"] = 0.05
    header["CDELT3"] = 12.5e6
    header["CTYPE1"] = "GLON-CAR"
    header["CTYPE2"] = "GLAT-CAR"
    header["CTYPE3"] = "FREQ"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "Hz"
    header["BUNIT"] = "Jy/px"
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(path, overwrite=True)
    return path


def _build_fake_header() -> fits.Header:
    """Return a minimal WSClean-ready spectral header for pipeline mocking."""
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["NAXIS3"] = 8
    header["CRPIX1"] = 32.0
    header["CRPIX2"] = 32.0
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CDELT1"] = -0.05
    header["CDELT2"] = 0.05
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL3"] = 650e6
    header["CDELT3"] = 12.5e6
    header["CRPIX3"] = 1.0
    header["CTYPE3"] = "FREQ"
    header["CUNIT3"] = "Hz"
    header["BUNIT"] = "Jy/pixel"
    return header


class _FakeTelescope:
    def plot_telescope(self, file):
        Path(file).write_text("plot", encoding="utf-8")

    def max_baseline(self):
        return 10_000.0


class _FakeVersions(Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


def test_build_observation_centres_the_channel_grid_and_records_band_edges(
    tmp_path, monkeypatch
):
    """A 1284 MHz observation spans 856--1712 MHz without shifting channels."""
    import skasim.pipeline as pipeline

    captured = {}
    monkeypatch.setattr(
        pipeline,
        "require_karabo_module",
        lambda _: type(
            "ObservationModule",
            (),
            {"Observation": staticmethod(lambda **kwargs: captured.update(kwargs))},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "source_ref_get_best_observation_time",
        lambda center, telescope: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    ctx = create_run_context(
        SimConfig(
            output_dir=str(tmp_path / "lband"),
            observation=ObsConfig(
                frequency_mhz=1284.0,
                bandwidth_mhz=856.0,
                n_channels=8,
            ),
        )
    )

    _, center, _, n_channels, channel_width, start_frequency = (
        pipeline.build_observation(ctx, SkyCoord(0.0 * u.deg, 0.0 * u.deg), object())
    )
    milestone = next(
        item
        for item in ctx.manifest.milestones
        if item.name == "observation_configured"
    )

    assert center.to_value(u.MHz) == pytest.approx(1284.0)
    assert n_channels == 8
    assert channel_width.to_value(u.MHz) == pytest.approx(107.0)
    assert start_frequency.to_value(u.MHz) == pytest.approx(856.0)
    assert captured["start_frequency_hz"] == pytest.approx(856.0e6)
    assert milestone.details["min_frequency_mhz"] == pytest.approx(856.0)
    assert milestone.details["max_frequency_mhz"] == pytest.approx(1712.0)
    assert captured["start_date_and_time"] == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert milestone.details["start_time_utc"] == "2026-01-01 00:00:00"
    # no sky_center passed: it defaults to the effective centre, so no drift.
    assert milestone.details["sky_model_center_ra_deg"] == pytest.approx(0.0)


def test_build_observation_records_sky_model_centre_drift(tmp_path, monkeypatch):
    """A distinct sky_center is recorded separately from the effective centre."""
    import skasim.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "require_karabo_module",
        lambda _: type(
            "ObservationModule",
            (),
            {"Observation": staticmethod(lambda **kwargs: kwargs)},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "source_ref_get_best_observation_time",
        lambda center, telescope: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    ctx = create_run_context(SimConfig(output_dir=str(tmp_path / "drift")))

    pipeline.build_observation(
        ctx,
        SkyCoord(10.0 * u.deg, 5.0 * u.deg),
        object(),
        sky_center=SkyCoord(10.0 * u.deg, 5.1 * u.deg),
    )
    milestone = next(
        item
        for item in ctx.manifest.milestones
        if item.name == "observation_configured"
    )

    assert milestone.details["phase_center_dec_deg"] == pytest.approx(5.0)
    assert milestone.details["sky_model_center_dec_deg"] == pytest.approx(5.1)


def test_run_spectral_cube_pipeline_calls_wsclean_and_logs_summary(
    tmp_path, monkeypatch
):
    """A spectral-cube config triggers wsclean and logs a product summary."""
    import skasim.pipeline as pipeline

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        center = SkyCoord(10 * u.deg, 20 * u.deg)
        fake_sky = MagicMock()
        fake_sky.get_center.return_value = center

        monkeypatch.setattr(pipeline, "build_telescope", lambda ctx: _FakeTelescope())
        monkeypatch.setattr(
            pipeline, "compute_fov", lambda telescope, fov_deg, freq: 0.2 * u.deg
        )
        monkeypatch.setattr(
            pipeline, "build_sky_model", lambda ctx, fov: (fake_sky, center)
        )
        monkeypatch.setattr(
            pipeline,
            "build_observation",
            lambda ctx, center, telescope, **kwargs: (
                object(),
                700 * u.MHz,
                100 * u.MHz,
                8,
                12.5 * u.MHz,
                650 * u.MHz,
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "run_simulation",
            lambda ctx, telescope, observation, sky_model, fov_sim: ctx.visibility_path,
        )

        captured = []

        def fake_wsclean(ctx, vis_path, fov, img_config, sub_dir, n_channels=1):
            captured.append(("wsclean", img_config.tag, n_channels))
            ctx.manifest.add_output(
                "image_product",
                f"{img_config.tag}_wsclean-0000-image.fits",
                image_product_id=f"{img_config.tag}_wsclean",
                imager="wsclean",
                role="cube_image",
                metadata={"tag": img_config.tag},
            )
            ctx.manifest.add_output(
                "image_product",
                f"mom0_{img_config.tag}.fits",
                image_product_id=f"{img_config.tag}_wsclean_mom0",
                imager="casa-immoments",
                role="mom0",
                metadata={"tag": img_config.tag},
            )

        monkeypatch.setattr(pipeline, "run_wsclean_imaging", fake_wsclean)
        monkeypatch.setattr(
            pipeline,
            "write_image_model_previews",
            lambda ctx, center, fov: None,
        )
        monkeypatch.setattr(
            "skasim.loaders.image_models.prepare_spectral_cube_for_casa",
            lambda ctx, entry, index, report: CasaModelProduct(
                model_paths=[Path("/tmp/fake_cube.image")],
                nterms=1,
                reffreq=f"{ctx.config.observation.frequency_mhz * 1e6}Hz",
                intermediates=[],
                cube_data=np.zeros((8, 64, 64), dtype=np.float32),
                header=_build_fake_header(),
                freq_axis=3,
            ),
        )
        monkeypatch.setattr(
            "skasim.loaders.wsclean_predict.run_wsclean_predict",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            "skasim.loaders.image_models.run_casa_ft",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "skasim.loaders.image_models.merge_model_data_into_data",
            lambda visibility_path: None,
        )

        cube_path = tmp_path / "cube.fits"
        _write_minimal_fits_cube(cube_path, n_channels=8)
        config = SimConfig(
            output_dir="spectral_run",
            models=[
                SpectralCubeModelEntry(
                    type="spectral_cube",
                    cube=str(cube_path),
                    reference_frequency_hz=700e6,
                    channel_width_hz=12.5e6,
                    n_channels=8,
                )
            ],
            imaging=[ImgConfig(tag="line", imager="wsclean", pixels=64)],
            observation=ObsConfig(
                observation_time_s=1,
                frequency_mhz=700.0,
                bandwidth_mhz=100.0,
                n_channels=8,
                channel_width_mhz=12.5,
            ),
        )

        pipeline.run(config)

        assert captured == [("wsclean", "line", 8)]
        work_dir = tmp_path / "spectral_run"
        manifest = json.loads(
            (work_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        milestone_names = {m["name"] for m in manifest["milestones"]}
        assert "imaging_line_completed" in milestone_names
    finally:
        os.chdir(old_cwd)


def test_run_rejects_mixed_spectral_cube_and_component_models(tmp_path):
    """Spectral-cube model entries cannot coexist with other model types."""
    cube_path = tmp_path / "cube.fits"
    _write_minimal_fits_cube(cube_path, n_channels=8)
    tt0 = tmp_path / "tt0.fits"
    tt1 = tmp_path / "tt1.fits"
    _write_minimal_fits_cube(tt0, n_channels=1)
    _write_minimal_fits_cube(tt1, n_channels=1)
    with pytest.raises(ValueError, match="spectral_cube mode is exclusive"):
        SimConfig(
            models=[
                SpectralCubeModelEntry(type="spectral_cube", cube=str(cube_path)),
                CasaTaylorTermsModelEntry(
                    type="casa_taylor_terms",
                    tt0=str(tt0),
                    tt1=str(tt1),
                    reference_frequency_hz=700e6,
                ),
            ]
        )


# --------------------------------------------------------------------------- #
# telescope
# --------------------------------------------------------------------------- #


def test_resolve_telescope_version_accepts_enum_name():
    """CLI telescope version names are converted to Karabo enum members."""
    module = type(
        "Module",
        (),
        {"OSKAR_TELESCOPE_TO_VERSIONS": {"SKA-LOW-AAstar": _FakeVersions}},
    )

    version = resolve_telescope_version(
        module,
        "SKA-LOW-AAstar",
        "SKA_OST_ARRAY_CONFIG_2_3_1",
    )

    assert version is _FakeVersions.SKA_OST_ARRAY_CONFIG_2_3_1


def test_resolve_telescope_version_accepts_enum_value():
    """Karabo enum values are also accepted for telescope versions."""
    module = type(
        "Module",
        (),
        {"OSKAR_TELESCOPE_TO_VERSIONS": {"SKA-LOW-AAstar": _FakeVersions}},
    )

    version = resolve_telescope_version(
        module,
        "SKA-LOW-AAstar",
        "ska-ost-array-config-2.3.1",
    )

    assert version is _FakeVersions.SKA_OST_ARRAY_CONFIG_2_3_1


def test_resolve_telescope_version_rejects_unknown_version():
    """Bad telescope versions fail before Karabo raises opaque enum errors."""
    module = type(
        "Module",
        (),
        {"OSKAR_TELESCOPE_TO_VERSIONS": {"SKA-LOW-AAstar": _FakeVersions}},
    )

    with pytest.raises(ValueError, match="Accepted versions"):
        resolve_telescope_version(module, "SKA-LOW-AAstar", "bad-version")


# --------------------------------------------------------------------------- #
# parse_center
# --------------------------------------------------------------------------- #


def test_parse_center_none_returns_fallback():
    """None center string returns the fallback coordinate."""
    fallback = SkyCoord(10 * u.deg, 20 * u.deg)
    result = parse_center(None, fallback)
    assert result.ra.value == pytest.approx(10.0)
    assert result.dec.value == pytest.approx(20.0)


def test_parse_center_valid_hmsdms_string():
    """Space-separated HMS/DMS string parses correctly."""
    fallback = SkyCoord(0 * u.deg, 0 * u.deg)
    result = parse_center("10h01m35.1s 2d41m41s", fallback)
    assert result.ra.to(u.deg).value == pytest.approx(150.3962, abs=1e-3)
    assert result.dec.to(u.deg).value == pytest.approx(2.6947, abs=1e-4)


def test_parse_center_with_colons():
    """Colon-separated format also parses correctly."""
    fallback = SkyCoord(0 * u.deg, 0 * u.deg)
    result = parse_center("10:01:35.1 02:41:41", fallback)
    assert result.ra.to(u.deg).value == pytest.approx(150.3962, abs=1e-3)


def test_parse_center_invalid_returns_fallback():
    """Unparseable string falls back gracefully."""
    fallback = SkyCoord(5 * u.deg, -10 * u.deg)
    result = parse_center("not-a-coordinate", fallback)
    assert result.ra.value == pytest.approx(5.0)
    assert result.dec.value == pytest.approx(-10.0)


# --------------------------------------------------------------------------- #
# compute_fov
# --------------------------------------------------------------------------- #


def test_compute_fov_uses_explicit_fov_deg():
    """When fov_deg is set, return that value converted to radians."""
    freq = 700 * u.MHz
    fov = compute_fov("SKA1MID", 2.5, freq)
    assert fov.to(u.deg).value == pytest.approx(2.5, abs=1e-6)


@pytest.mark.parametrize(
    "telescope,expected_diameter_m",
    [
        ("SKA1MID", 15.0),
        ("SKA1LOW", 38.0),
    ],
)
def test_compute_fov_diffraction_limit(telescope, expected_diameter_m):
    """When fov_deg is None, compute diffraction-limited FoV."""
    freq = 700 * u.MHz
    fov = compute_fov(telescope, None, freq)
    wavelength = freq.to(u.m, equivalencies=u.spectral()).value
    expected_rad = 1.25 * wavelength / expected_diameter_m
    assert fov.to(u.rad).value == pytest.approx(expected_rad, rel=1e-6)


def test_compute_fov_positive():
    """Default config produces a positive FoV."""
    config = SimConfig(observation=ObsConfig(observation_time_s=1))
    freq = config.observation.frequency_mhz * u.MHz
    fov = compute_fov(config.telescope, config.imaging[0].fov_deg, freq)
    assert fov.to(u.deg).value > 0


# --------------------------------------------------------------------------- #
# create_run_context
# --------------------------------------------------------------------------- #


def test_create_run_context_creates_directory_and_log_file(tmp_path):
    """create_run_context creates working directory, returns RunContext with paths."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config = SimConfig(output_dir="test_run", telescope="SKA1MID")
        ctx = create_run_context(config)
        assert ctx.work_dir.is_absolute()
        assert ctx.work_dir.name == "test_run"
        assert ctx.work_dir.is_dir()
        assert ctx.log_path.exists()
        assert ctx.manifest_path.exists()
        assert ctx.manifest.run_id == "test_run"
        assert ctx.manifest.status == "running"
    finally:
        os.chdir(old_cwd)


def test_create_run_context_no_prefix_uses_timestamp(tmp_path):
    """When output_dir is None, directory name includes a timestamp-like string."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config = SimConfig(telescope="SKA1MID")
        ctx = create_run_context(config)
        assert len(ctx.work_dir.name) >= 8
        assert ctx.work_dir.is_dir()
        assert ctx.manifest_path.exists()
    finally:
        os.chdir(old_cwd)


# --------------------------------------------------------------------------- #
# helpers for JSON fixture
# --------------------------------------------------------------------------- #


def _make_ctx(tmp_path, config):
    """Build a RunContext inside tmp_path for tests that need one."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = create_run_context(config)
        return ctx
    finally:
        os.chdir(old_cwd)


def _single_source_json():
    return {
        "ra": 150.0,
        "dec": 2.5,
        "I": 1.0,
        "Q": 0.0,
        "U": 0.0,
        "V": 0.0,
        "ref_freq": 700e6,
        "spec_index": 0.0,
        "rot_meas": 0.0,
        "major_axis": 0.0,
        "minor_axis": 0.0,
        "pa": 0.0,
        "true_redshift": 0.0,
        "obs_redshift": 0.0,
    }


# --------------------------------------------------------------------------- #
# _load_sky_from_file — inline JSON via mock_open
# --------------------------------------------------------------------------- #


def test_load_sky_from_json_file():
    """JSON catalog loads successfully and returns a SkyModel with one source."""
    data = [_single_source_json()]
    mopen = mock_open(read_data=json.dumps(data))
    with patch("builtins.open", mopen):
        sky = _load_sky_from_file("catalog.json")
    assert sky.sources is not None
    assert len(sky.sources) == 1


def test_load_sky_from_json_preserves_shape_metadata():
    """JSON sky models retain source ellipse metadata for previews."""
    src = _single_source_json()
    src["major_axis"] = 120.0
    src["minor_axis"] = 30.0
    src["pa"] = 42.0
    src["spec_index"] = -0.63
    mopen = mock_open(read_data=json.dumps([src]))

    with patch("builtins.open", mopen):
        sky = _load_sky_from_file("catalog.json")

    rendered = sky.to_json()[0]
    assert rendered["major_axis"] == pytest.approx(120.0)
    assert rendered["minor_axis"] == pytest.approx(30.0)
    assert rendered["pa"] == pytest.approx(42.0)
    assert rendered["spec_index"] == pytest.approx(-0.63)


def test_load_sky_from_json_scales_intensity():
    """flux_scale != 1 multiplies source intensities."""
    data = [_single_source_json()]
    mopen = mock_open(read_data=json.dumps(data))
    with patch("builtins.open", mopen):
        sky_before = _load_sky_from_file("catalog.json", flux_scale=1.0)
    mopen2 = mock_open(read_data=json.dumps(data))
    with patch("builtins.open", mopen2):
        sky_after = _load_sky_from_file("catalog.json", flux_scale=3.0)
    assert len(sky_after.sources) == 1
    assert sky_after.sources[0, 2] == pytest.approx(sky_before.sources[0, 2] * 3.0)


def test_load_sky_from_json_assigns_ref_freq_when_zero():
    """If JSON source has ref_freq == 0, the loader assigns observing frequency."""
    src = _single_source_json()
    src["ref_freq"] = 0
    data = [src, {**src, "ra": 150.1}, {**src, "ra": 149.9}]
    mopen = mock_open(read_data=json.dumps(data))
    with patch("builtins.open", mopen):
        sky = _load_sky_from_file("catalog.json", frequency=1420 * u.MHz)
    assert sky.sources is not None
    assert len(sky.sources) == 3
    assert sky.to_json()[0]["ref_freq"] == pytest.approx(1420e6)


def test_load_sky_from_file_unknown_extension():
    """Passing an unsupported extension raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported sky-file extension"):
        _load_sky_from_file("foo.txt")


def test_load_sky_from_file_empty_json_raises():
    """An empty JSON array raises ValueError about no sources."""
    mopen = mock_open(read_data="[]")
    with (
        patch("builtins.open", mopen),
        pytest.raises(ValueError, match="No sources found in JSON"),
    ):
        _load_sky_from_file("empty.json")


def test_load_sky_from_jsonl_file():
    """JSONL (one source object per line) loads like a regular JSON array."""
    src = _single_source_json()
    src2 = {**src, "ra": 150.1}
    jsonl_content = json.dumps(src) + "\n" + json.dumps(src2) + "\n"
    mopen = mock_open(read_data=jsonl_content)
    with patch("builtins.open", mopen):
        sky = _load_sky_from_file("catalog.json")
    assert sky.sources is not None
    assert len(sky.sources) == 2


# --------------------------------------------------------------------------- #
# build_sky_model — random source generation
# --------------------------------------------------------------------------- #


def test_build_sky_model_file_records_sky_model_output(tmp_path):
    """File-backed sky model sources are recorded as sky_model outputs."""
    source_file = tmp_path / "sources.json"
    source_file.write_text(json.dumps([_single_source_json()]), encoding="utf-8")
    config = SimConfig(sky_file=str(source_file))
    ctx = _make_ctx(tmp_path, config)

    build_sky_model(ctx, fov=0.2 * u.deg)

    sky_outputs = [
        output for output in ctx.manifest.outputs if output.kind == "sky_model"
    ]
    assert sky_outputs[0].path == str(source_file.resolve())


def test_build_sky_model_random_source_count(tmp_path):
    """Random source generation produces one source per configured intensity."""
    config = SimConfig(source_flux_jy=[1.0, 5.0, 10.0, 20.0])
    ctx = _make_ctx(tmp_path, config)
    sky, center = build_sky_model(ctx, fov=0.2 * u.deg)
    assert len(sky.sources) == 4
    assert center is not None


def test_build_sky_model_random_single_source(tmp_path):
    """A single intensity produces one source."""
    config = SimConfig(source_flux_jy=[42.0])
    ctx = _make_ctx(tmp_path, config)
    sky, center = build_sky_model(ctx, fov=0.2 * u.deg)
    assert len(sky.sources) == 1


def test_build_sky_model_uses_source_flux_jy(tmp_path):
    """Generated source intensities create one generated source per value."""
    config = SimConfig(source_flux_jy=[1.0, 5.0, 10.0])
    ctx = _make_ctx(tmp_path, config)
    sky, center = build_sky_model(ctx, fov=0.2 * u.deg)

    assert len(sky.sources) == 3
    assert center is not None


def test_build_sky_model_uses_generated_source_polarization(tmp_path):
    """Generated source Stokes Q/U/V values are passed to the sky model."""
    config = SimConfig(
        source_flux_jy=[1.0, 5.0],
        stokes_q_jy=[0.1, 0.2],
        stokes_u_jy=[0.0, 0.3],
        stokes_v_jy=[0.0, -0.1],
    )
    ctx = _make_ctx(tmp_path, config)
    sky, _ = build_sky_model(ctx, fov=0.2 * u.deg)

    assert len(sky.sources) == 2
    assert sky.sources[0, 3] == pytest.approx(0.1)
    assert sky.sources[1, 3] == pytest.approx(0.2)
    assert sky.sources[1, 4] == pytest.approx(0.3)
    assert sky.sources[1, 5] == pytest.approx(-0.1)


def test_build_sky_model_records_flux_range_and_polarization(tmp_path):
    """sky_model_loaded records flux min/max and polarization presence."""
    config = SimConfig(
        source_flux_jy=[1.0, 5.0],
        stokes_q_jy=[0.1, 0.0],
        stokes_u_jy=[0.0, 0.0],
        stokes_v_jy=[0.0, 0.0],
    )
    ctx = _make_ctx(tmp_path, config)

    build_sky_model(ctx, fov=0.2 * u.deg)

    details = ctx.manifest.milestones[-1].details
    assert details["flux_min_jy"] == pytest.approx(1.0)
    assert details["flux_max_jy"] == pytest.approx(5.0)
    assert details["has_polarization"] is True


def test_build_sky_model_omits_polarization_flag_when_unpolarized(tmp_path):
    """Unpolarized random sources record flux range but no polarization flag."""
    config = SimConfig(source_flux_jy=[2.0, 3.0])
    ctx = _make_ctx(tmp_path, config)

    build_sky_model(ctx, fov=0.2 * u.deg)

    details = ctx.manifest.milestones[-1].details
    assert details["flux_min_jy"] == pytest.approx(2.0)
    assert details["flux_max_jy"] == pytest.approx(3.0)
    assert "has_polarization" not in details


@pytest.mark.parametrize(
    ("catalog", "loader_name"),
    [
        ("MIGHTEE", "get_MIGHTEE_Sky"),
        ("GLEAM", "get_GLEAM_Sky"),
    ],
)
def test_build_sky_model_named_catalog(tmp_path, monkeypatch, catalog, loader_name):
    """A named built-in catalog selects the matching catalog source."""
    fake_sky = SkyModel(np.array([[10.0, 20.0, 1.0]]))
    monkeypatch.setattr(
        SkyModel,
        loader_name,
        staticmethod(lambda: fake_sky),
        raising=False,
    )
    config = SimConfig(catalog=catalog)
    ctx = _make_ctx(tmp_path, config)

    sky, center = build_sky_model(ctx, fov=0.2 * u.deg)

    assert sky is fake_sky
    assert center.ra.value == pytest.approx(10.0)
    assert ctx.manifest.milestones[-1].details["format"] == catalog


# --------------------------------------------------------------------------- #
# build_sky_model — invalid configurations
# --------------------------------------------------------------------------- #


def test_build_sky_model_unsupported_catalog_raises(tmp_path):
    """Unsupported catalog names raise ValueError inside build_sky_model."""
    config = SimConfig(observation=ObsConfig(observation_time_s=1), catalog="MIGHTEE")
    # bypass pydantic validation; build_sky_model has its own ValueError guard
    object.__setattr__(config, "catalog", "UNKNOWN")
    ctx = _make_ctx(tmp_path, config)
    with pytest.raises(ValueError, match="Catalog UNKNOWN not available"):
        build_sky_model(ctx, fov=0.5 * u.deg)


# --------------------------------------------------------------------------- #
# source_ref_get_best_observation_time
# --------------------------------------------------------------------------- #


def test_source_ref_get_best_observation_time():
    """Returns an astropy Time around culmination for a mock telescope location."""
    center = SkyCoord(150 * u.deg, 2.5 * u.deg)
    telescope = MagicMock()
    telescope.centre_latitude = -30.0
    telescope.centre_longitude = 116.0
    telescope.centre_altitude = 300.0
    best_time = source_ref_get_best_observation_time(center, telescope)
    assert best_time is not None
    assert hasattr(best_time, "iso")  # astropy Time


def test_run_uses_resolved_wsclean_imager(tmp_path, monkeypatch):
    """run() selects imaging from config.imaging.imager and records it."""
    import skasim.pipeline as pipeline

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        center = SkyCoord(10 * u.deg, 20 * u.deg)
        fake_sky = MagicMock()
        fake_sky.get_center.return_value = center
        called = []
        simulation_fovs = []

        monkeypatch.setattr(pipeline, "build_telescope", lambda ctx: _FakeTelescope())
        monkeypatch.setattr(
            pipeline,
            "compute_fov",
            lambda telescope, fov_deg, freq: (
                0.2 * u.deg if fov_deg is None else fov_deg * u.deg
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "build_sky_model",
            lambda ctx, fov: simulation_fovs.append(fov) or (fake_sky, center),
        )
        monkeypatch.setattr(
            pipeline,
            "build_observation",
            lambda ctx, center, telescope, **kwargs: (
                object(),
                700 * u.MHz,
                100 * u.MHz,
                8,
                12.5 * u.MHz,
                650 * u.MHz,
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "run_simulation",
            lambda ctx, telescope, observation, sky_model, fov_sim: ctx.visibility_path,
        )
        monkeypatch.setattr(
            pipeline,
            "run_dirty_imaging",
            lambda *args, **kwargs: pytest.fail("dirty imaging should not run"),
        )
        monkeypatch.setattr(
            pipeline,
            "run_wsclean_imaging",
            lambda *args, **kwargs: called.append("wsclean"),
        )
        monkeypatch.setattr(
            pipeline,
            "run_shadems_command",
            lambda argv, work_dir: (
                called.append("uv")
                or (work_dir / "run_uvcoverage.png").write_bytes(b"\x89PNG")
                or type("R", (), {"stdout": "", "stderr": ""})()
            ),
        )

        config = SimConfig(
            output_dir="imager_run",
            imaging=[ImgConfig(imager="wsclean", fov_deg=0.05)],
            observation=ObsConfig(observation_time_s=1),
        )
        pipeline.run(config)

        manifest = json.loads(
            (tmp_path / "imager_run" / "run_manifest.json").read_text(encoding="utf-8")
        )
    finally:
        os.chdir(old_cwd)

    assert called == ["uv", "wsclean"]
    assert simulation_fovs == [0.2 * u.deg]
    assert manifest["config"]["imaging"][0]["imager"] == "wsclean"
    assert (tmp_path / "imager_run" / "weblog.html").exists()
    assert any(output["kind"] == "weblog" for output in manifest["outputs"])
    imaging_done = [
        item
        for item in manifest["milestones"]
        if item["name"] == "imaging_default_completed"
    ][0]
    assert imaging_done["details"]["imager"] == "wsclean"


def test_run_renders_weblog_on_failure(tmp_path, monkeypatch):
    """Failed runs save the manifest and render a weblog with the error."""
    import skasim.pipeline as pipeline

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monkeypatch.setattr(
            pipeline,
            "build_telescope",
            lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        config = SimConfig(
            output_dir="failed_run",
            imaging=[ImgConfig(imager="oskar-dirty")],
        )

        with pytest.raises(RuntimeError, match="boom"):
            pipeline.run(config)

        work_dir = tmp_path / "failed_run"
        manifest = json.loads((work_dir / "run_manifest.json").read_text())
        weblog = (work_dir / "weblog.html").read_text(encoding="utf-8")
    finally:
        os.chdir(old_cwd)

    assert manifest["status"] == "failed"
    assert "boom" in manifest["errors"][0]
    assert any(output["kind"] == "weblog" for output in manifest["outputs"])
    assert "failed" in weblog
    assert "boom" in weblog


def test_run_records_failure_milestone_details_as_dict(tmp_path, monkeypatch):
    """Failure milestones remain serializable when a phase raises."""
    import skasim.pipeline as pipeline

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        center = SkyCoord(10 * u.deg, 20 * u.deg)
        fake_sky = MagicMock()
        fake_sky.get_center.return_value = center
        monkeypatch.setattr(
            pipeline,
            "build_telescope",
            lambda ctx: _FakeTelescope(),
        )
        monkeypatch.setattr(
            pipeline, "compute_fov", lambda telescope, fov_deg, freq: 0.2 * u.deg
        )
        monkeypatch.setattr(
            pipeline,
            "build_sky_model",
            lambda ctx, fov: (fake_sky, center),
        )
        monkeypatch.setattr(
            pipeline,
            "build_observation",
            lambda ctx, center, telescope, **kwargs: (
                object(),
                700 * u.MHz,
                100 * u.MHz,
                8,
                12.5 * u.MHz,
                650 * u.MHz,
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "run_simulation",
            lambda ctx, telescope, observation, sky_model, fov_sim: (
                _ for _ in ()
            ).throw(RuntimeError("phase failed")),
        )
        config = SimConfig(
            output_dir="failed_milestone",
            imaging=[ImgConfig(imager="oskar-dirty")],
        )

        with pytest.raises(RuntimeError, match="phase failed"):
            pipeline.run(config)

        manifest = json.loads(
            (tmp_path / "failed_milestone" / "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    finally:
        os.chdir(old_cwd)

    failed = [
        item for item in manifest["milestones"] if item["name"] == "simulation_failed"
    ][0]
    assert failed["details"] == {"error": "phase failed"}


def test_run_simulation_does_not_rebuild_observation(tmp_path, monkeypatch):
    """Simulation uses the already-built observation instead of duplicating setup."""

    class FakeInterferometerSimulation:
        params = None
        run_args = None

        def __init__(self, **params):
            FakeInterferometerSimulation.params = params

        def run_simulation(self, **kwargs):
            FakeInterferometerSimulation.run_args = kwargs

    def fake_require(module_name):
        if module_name == "karabo.simulation.interferometer":
            return MagicMock(InterferometerSimulation=FakeInterferometerSimulation)
        if module_name == "karabo.simulator_backend":
            return MagicMock(SimulatorBackend=MagicMock(OSKAR="OSKAR"))
        raise AssertionError(module_name)

    monkeypatch.setattr("skasim.pipeline.require_karabo_module", fake_require)
    monkeypatch.setattr(
        "skasim.pipeline.build_observation",
        lambda *args, **kwargs: pytest.fail("build_observation should not run"),
    )
    config = SimConfig(output_dir=str(tmp_path / "sim"))
    ctx = create_run_context(config)
    sky_model = MagicMock()

    visibility_path = run_simulation(
        ctx, object(), object(), sky_model, fov_sim=0.2 * u.deg
    )

    assert visibility_path == ctx.visibility_path
    assert FakeInterferometerSimulation.params["channel_bandwidth_hz"] == pytest.approx(
        config.observation.channel_width_mhz * 1e6
    )
    assert FakeInterferometerSimulation.run_args["observation"] is not None
    # OSKAR's beam comes from the same helper used to attenuate image models.
    from skasim.pipeline import run_beam

    beam_params = run_beam(
        0.2 * u.deg, config.observation.frequency_mhz * u.MHz
    ).oskar_params()
    for key, value in beam_params.items():
        assert FakeInterferometerSimulation.params[key] == value
    assert FakeInterferometerSimulation.params["gauss_beam_fwhm_deg"] == 0.2


# --------------------------------------------------------------------------- #
# batch imaging
# --------------------------------------------------------------------------- #


def test_run_batch_imaging_creates_subdirs_and_calls_both_imagers(
    tmp_path, monkeypatch
):
    """Multiple imaging blocks each get their own sub-directory and imager call."""
    import skasim.pipeline as pipeline

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        center = SkyCoord(10 * u.deg, 20 * u.deg)
        fake_sky = MagicMock()
        fake_sky.get_center.return_value = center
        called = []
        simulation_fovs = []

        monkeypatch.setattr(pipeline, "build_telescope", lambda ctx: _FakeTelescope())
        monkeypatch.setattr(
            pipeline,
            "compute_fov",
            lambda telescope, fov_deg, freq: (
                0.2 * u.deg if fov_deg is None else fov_deg * u.deg
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "build_sky_model",
            lambda ctx, fov: simulation_fovs.append(fov) or (fake_sky, center),
        )
        monkeypatch.setattr(
            pipeline,
            "build_observation",
            lambda ctx, center, telescope, **kwargs: (
                object(),
                700 * u.MHz,
                100 * u.MHz,
                8,
                12.5 * u.MHz,
                650 * u.MHz,
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "run_simulation",
            lambda ctx, telescope, observation, sky_model, fov_sim: ctx.visibility_path,
        )
        monkeypatch.setattr(
            pipeline,
            "run_dirty_imaging",
            lambda ctx, vis_path, fov, center, img_config, sub_dir: called.append(
                ("dirty", img_config.tag, str(sub_dir))
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "run_wsclean_imaging",
            lambda ctx, vis_path, fov, img_config, sub_dir, n_channels=1: called.append(
                ("wsclean", img_config.tag, str(sub_dir))
            ),
        )

        config = SimConfig(
            output_dir="batch_run",
            imaging=[
                ImgConfig(tag="robust0", imager="wsclean", robust=0.0, fov_deg=0.05),
                ImgConfig(tag="robust2", imager="wsclean", robust=2.0, fov_deg=0.5),
            ],
            observation=ObsConfig(observation_time_s=1),
        )
        pipeline.run(config)

        # both imagers were invoked
        assert len(called) == 2
        tags = {c[1] for c in called}
        assert tags == {"robust0", "robust2"}
        assert simulation_fovs == [0.5 * u.deg]
        # subdirectories created under work_dir
        work_dir = tmp_path / "batch_run"
        assert (work_dir / "robust0").is_dir()
        assert (work_dir / "robust2").is_dir()

        # milestones per tag
        manifest = json.loads(
            (work_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        milestone_names = {m["name"] for m in manifest["milestones"]}
        assert "imaging_robust0_completed" in milestone_names
        assert "imaging_robust2_completed" in milestone_names
    finally:
        os.chdir(old_cwd)


def test_run_batch_imaging_fail_fast_on_first_error(tmp_path, monkeypatch):
    """If one imaging block fails, the batch aborts immediately."""
    import skasim.pipeline as pipeline

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        center = SkyCoord(10 * u.deg, 20 * u.deg)
        fake_sky = MagicMock()
        fake_sky.get_center.return_value = center

        monkeypatch.setattr(pipeline, "build_telescope", lambda ctx: _FakeTelescope())
        monkeypatch.setattr(
            pipeline, "compute_fov", lambda telescope, fov_deg, freq: 0.2 * u.deg
        )
        monkeypatch.setattr(
            pipeline, "build_sky_model", lambda ctx, fov: (fake_sky, center)
        )
        monkeypatch.setattr(
            pipeline,
            "build_observation",
            lambda ctx, center, telescope, **kwargs: (
                object(),
                700 * u.MHz,
                100 * u.MHz,
                8,
                12.5 * u.MHz,
                650 * u.MHz,
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "run_simulation",
            lambda ctx, telescope, observation, sky_model, fov_sim: ctx.visibility_path,
        )
        monkeypatch.setattr(
            pipeline,
            "run_dirty_imaging",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("imaging boom")),
        )

        config = SimConfig(
            output_dir="fail_fast_run",
            imaging=[
                ImgConfig(tag="a", imager="oskar-dirty"),
                ImgConfig(tag="b", imager="oskar-dirty"),
            ],
            observation=ObsConfig(observation_time_s=1),
        )
        with pytest.raises(RuntimeError, match="imaging boom"):
            pipeline.run(config)

        work_dir = tmp_path / "fail_fast_run"
        manifest = json.loads(
            (work_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        # only the first tag should have a failed milestone
        failed_names = {
            m["name"] for m in manifest["milestones"] if m["status"] == "failed"
        }
        assert "imaging_a_failed" in failed_names
        assert "imaging_b_started" not in {m["name"] for m in manifest["milestones"]}
    finally:
        os.chdir(old_cwd)


def test_weblog_has_tabs_for_multiple_tags(tmp_path, monkeypatch):
    """weblog template renders tab buttons when multiple tags exist."""
    import skasim.pipeline as pipeline

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        center = SkyCoord(10 * u.deg, 20 * u.deg)
        fake_sky = MagicMock()
        fake_sky.get_center.return_value = center

        monkeypatch.setattr(pipeline, "build_telescope", lambda ctx: _FakeTelescope())
        monkeypatch.setattr(
            pipeline, "compute_fov", lambda telescope, fov_deg, freq: 0.2 * u.deg
        )
        monkeypatch.setattr(
            pipeline, "build_sky_model", lambda ctx, fov: (fake_sky, center)
        )
        monkeypatch.setattr(
            pipeline,
            "build_observation",
            lambda ctx, center, telescope, **kwargs: (
                object(),
                700 * u.MHz,
                100 * u.MHz,
                8,
                12.5 * u.MHz,
                650 * u.MHz,
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "run_simulation",
            lambda ctx, telescope, observation, sky_model, fov_sim: ctx.visibility_path,
        )
        monkeypatch.setattr(
            pipeline,
            "run_wsclean_imaging",
            lambda ctx, vis_path, fov, img_config, sub_dir, n_channels=1: (
                ctx.manifest.add_output(
                    "image_product",
                    f"{img_config.tag}_wsclean-MFS-image.fits",
                    image_product_id=f"{img_config.tag}_wsclean",
                    imager="wsclean",
                    role="image",
                    metadata={"tag": img_config.tag},
                )
            ),
        )

        config = SimConfig(
            output_dir="weblog_tabs_run",
            imaging=[
                ImgConfig(tag="tagA", imager="wsclean"),
                ImgConfig(tag="tagB", imager="wsclean"),
            ],
            observation=ObsConfig(observation_time_s=1),
        )
        pipeline.run(config)

        html = (tmp_path / "weblog_tabs_run" / "weblog.html").read_text(
            encoding="utf-8"
        )
        assert 'class="tab-buttons"' in html
        assert "tagA" in html
        assert "tagB" in html
        assert "showTab('tagA')" in html
        assert "showTab('tagB')" in html
    finally:
        os.chdir(old_cwd)


# --------------------------------------------------------------------------- #
# multimodal input: phase-centre provenance and footprint warnings
# --------------------------------------------------------------------------- #


def _sin_model(tmp_path, name, centre_deg, n=32, cdelt_deg=1e-3):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.crval = list(centre_deg)
    wcs.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    wcs.wcs.cdelt = [-cdelt_deg, cdelt_deg]
    header = wcs.to_header()
    header["BUNIT"] = "Jy/pixel"
    path = tmp_path / f"{name}.fits"
    fits.writeto(path, np.ones((n, n), dtype=np.float32), header, overwrite=True)
    return path


def _continuum(path, alpha):
    return {
        "type": "continuum_i_alpha",
        "stokes_i": str(path),
        "alpha": str(alpha),
        "reference_frequency_hz": 1.4e9,
    }


@pytest.mark.parametrize(
    "case, expected",
    [
        ("explicit", "explicit"),
        ("catalogue_plus_image", "catalogue"),
        ("image_only", "image_model"),
        ("generated", "generated_sources"),
    ],
)
def test_phase_centre_rule_names_the_deciding_source(tmp_path, case, expected):
    from skasim.pipeline import phase_centre_rule

    image = _sin_model(tmp_path, "img", (150.0, 2.0))
    alpha = _sin_model(tmp_path, "alpha", (150.0, 2.0))
    configs = {
        "explicit": dict(
            center="10h00m00s +02d00m00s", models=[_continuum(image, alpha)]
        ),
        "catalogue_plus_image": dict(
            models=[
                {"type": "component_sky_model", "catalog": "GLEAM"},
                _continuum(image, alpha),
            ]
        ),
        "image_only": dict(models=[_continuum(image, alpha)]),
        "generated": dict(),
    }

    assert phase_centre_rule(SimConfig(**configs[case])) == expected


def test_record_sky_layout_records_rule_and_flags_far_footprint(tmp_path):
    from skasim.pipeline import _record_sky_layout
    from skasim.weblog import _report_warnings

    near = _sin_model(tmp_path, "near", (150.0, 2.0))
    # 0.2 deg north: beyond the half-power radius (0.177 deg), inside the
    # simulation field radius (0.25 deg)
    far = _sin_model(tmp_path, "far", (150.0, 2.2))
    alpha = _sin_model(tmp_path, "alpha", (150.0, 2.0))
    config = SimConfig(
        center="10h00m00s +02d00m00s",
        models=[_continuum(near, alpha), _continuum(far, alpha)],
        output_dir=str(tmp_path / "run"),
    )
    ctx = create_run_context(config)
    centre = SkyCoord(150.0, 2.0, unit="deg")

    _record_sky_layout(ctx, centre, fov_sim=0.5 * u.deg, freq=1400 * u.MHz)

    rule = [m for m in ctx.manifest.milestones if m.name == "phase_centre_resolved"]
    assert rule[0].details["rule"] == "explicit"
    assert rule[0].details["dec_deg"] == pytest.approx(2.0)
    footprints = [
        m for m in ctx.manifest.milestones if m.name == "image_model_footprints"
    ][0].details["entries"]
    assert [f["beyond_beam_half_power"] for f in footprints] == [False, True]
    assert [f["beyond_field"] for f in footprints] == [False, False]
    assert footprints[1]["beam_half_power_radius_deg"] == pytest.approx(
        0.5 / np.sqrt(2) / 2, rel=1e-3
    )
    warnings = _report_warnings(ctx.manifest)
    assert any("Image model 1" in w and "half-power" in w for w in warnings)
    assert not any("Image model 0" in w for w in warnings)


def test_catalogue_beam_flux_scales_spectrally_and_applies_beam():
    from skasim.pipeline import catalogue_beam_flux
    from skasim.primary_beam import GaussianStationBeam

    beam = GaussianStationBeam(field_fwhm_deg=1.0, ref_freq_hz=2.8e9)
    centre = SkyCoord(150.0, 0.0, unit="deg")
    ra = np.array([150.0, 150.3])
    dec = np.array([0.0, 0.0])
    flux = np.array([1.0, 2.0])
    ref = np.array([1.4e9, 0.0])  # second source: no reference -> flat
    alpha = np.array([-1.0, -0.7])

    record = catalogue_beam_flux((ra, dec, flux, ref, alpha), centre, beam)

    assert record["n_sources"] == 2
    assert record["intrinsic_flux_jy_at_ref"] == pytest.approx(0.5 + 2.0)
    expected = 0.5 + 2.0 * float(beam.power_response(0.3, 2.8e9))
    assert record["attenuated_flux_jy_at_ref"] == pytest.approx(expected, rel=1e-6)
    assert record["max_offset_deg"] == pytest.approx(0.3, rel=1e-6)


def test_record_sky_layout_writes_composition_panel_and_catalogue_flux(tmp_path):
    import matplotlib

    from skasim.pipeline import _record_sky_layout

    matplotlib.use("Agg", force=True)
    image = _sin_model(tmp_path, "img", (150.0, 2.0))
    alpha = _sin_model(tmp_path, "alpha", (150.0, 2.0))
    config = SimConfig(
        models=[
            {"type": "component_sky_model", "catalog": "GLEAM"},
            _continuum(image, alpha),
        ],
        output_dir=str(tmp_path / "run"),
    )
    ctx = create_run_context(config)
    sources = np.zeros((3, 12))
    sources[:, 0] = [150.0, 150.05, 149.9]
    sources[:, 1] = [2.0, 2.05, 1.95]
    sources[:, 2] = [0.1, 0.2, 0.3]

    class FakeSky:
        pass

    sky = FakeSky()
    sky.sources = sources

    _record_sky_layout(
        ctx, SkyCoord(150.0, 2.0, unit="deg"), 0.5 * u.deg, 1400 * u.MHz, sky
    )

    names = [m.name for m in ctx.manifest.milestones]
    assert "catalogue_beam_flux" in names
    panel = [o for o in ctx.manifest.outputs if o.role == "sky_composition"]
    assert len(panel) == 1
    assert (ctx.work_dir / panel[0].path).exists()
