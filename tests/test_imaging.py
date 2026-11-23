"""Image production behavior."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from skasim.config import (
    ImgConfig,
    ObsConfig,
    SimConfig,
    SpectralCubeModelEntry,
)
from skasim.imaging import (
    SKY_MODEL_CMAP,
    _compact_source_mask,
    _flux_marker_sizes,
    _padded_limits,
    _resolve_spectral_cube_wsclean_config,
    _sky_model_ellipses,
    _sky_model_position_angle,
    build_wsclean_argv,
    collect_wsclean_outputs,
    effective_wsclean_channels_out,
    resolve_wsclean_img_config,
    run_dirty_imaging,
    run_wsclean_command,
    run_wsclean_imaging,
    write_fits_preview,
    write_sky_model_previews,
    wsclean_output_prefix,
    wsclean_product_name,
)
from skasim.manifest import create_run_context


def test_resolve_spectral_cube_wsclean_config_forces_defaults():
    """Spectral-cube mode disables join_channels and multiscale."""
    config = SimConfig(
        imaging=[
            ImgConfig(
                imager="wsclean",
                pixels=64,
                join_channels=True,
                multiscale=True,
                multiscale_scales=[0, 4, 8],
            )
        ]
    )
    resolved = _resolve_spectral_cube_wsclean_config(config.imaging[0], n_channels=8)
    assert resolved.join_channels is False
    assert resolved.multiscale is False
    assert resolved.multiscale_scales is None
    assert resolved.channels_out == 8


def test_resolve_spectral_cube_wsclean_config_keeps_user_channels_out():
    """A user-supplied channels_out is preserved when present."""
    config = SimConfig(
        imaging=[
            ImgConfig(
                imager="wsclean",
                pixels=64,
                channels_out=4,
            )
        ]
    )
    resolved = _resolve_spectral_cube_wsclean_config(config.imaging[0], n_channels=16)
    assert resolved.channels_out == 4


def test_build_wsclean_argv_spectral_cube_has_no_join_channels():
    """Spectral-cube argv never contains -join-channels and matches channels."""
    config = SimConfig(
        observation=ObsConfig(bandwidth_mhz=10.0, n_channels=8),
        imaging=[ImgConfig(imager="wsclean", pixels=64)],
    )
    resolved = _resolve_spectral_cube_wsclean_config(
        config.imaging[0], n_channels=config.observation.n_channels
    )
    argv = build_wsclean_argv(
        resolved,
        visibility_path=Path("vis.MS"),
        fov=0.2 * u.deg,
        output_prefix="line",
        n_channels=config.observation.n_channels,
    )
    assert "-join-channels" not in argv
    assert "-multiscale" not in argv
    assert argv[argv.index("-channels-out") + 1] == "8"


def test_run_wsclean_imaging_triggers_moment8_for_spectral_cube(tmp_path, monkeypatch):
    """A spectral cube run calls moment8 after WSClean."""

    class FakeImage:
        def __init__(self, path):
            self.path = path

        def plot(self, filename, **kwargs):
            Path(filename).write_text("png", encoding="utf-8")

    def fake_require(module_name):
        if module_name == "karabo.imaging.image":
            return SimpleNamespace(Image=FakeImage)
        if module_name == "karabo.imaging.imager_wsclean":
            return SimpleNamespace(TMP_PREFIX_CUSTOM="tmp", TMP_PURPOSE_CUSTOM="test")
        if module_name == "karabo.util.file_handler":
            return SimpleNamespace(
                FileHandler=lambda: SimpleNamespace(
                    get_tmp_dir=lambda **kwargs: tmp_path
                )
            )
        raise AssertionError(module_name)

    captured = {}

    def fake_run_moment8(ctx, work_dir, output_prefix, tag):
        captured["moment8"] = (str(work_dir), output_prefix, tag)

    def fake_run(argv, work_dir):
        prefix = argv[argv.index("-name") + 1]
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"{prefix}-MFS-image.fits").write_text("fits", encoding="utf-8")

        class Result:
            stdout = "ok"

        return Result()

    monkeypatch.setattr("skasim.imaging.require_karabo_module", fake_require)
    monkeypatch.setattr("skasim.imaging.run_wsclean_command", fake_run)
    monkeypatch.setattr(
        "skasim.loaders.image_models.run_moment8_for_spectral_cube",
        fake_run_moment8,
    )
    monkeypatch.setattr(
        "skasim.imaging.write_fits_preview",
        lambda img_path, png_path, title: Path(png_path).write_text(
            "png", encoding="utf-8"
        ),
    )

    (tmp_path / "cube.fits").write_text("fits", encoding="utf-8")

    config = SimConfig(
        output_dir=str(tmp_path / "run"),
        models=[
            SpectralCubeModelEntry(
                type="spectral_cube", cube=str(tmp_path / "cube.fits")
            )
        ],
        imaging=[ImgConfig(imager="wsclean")],
    )
    ctx = create_run_context(config)

    run_wsclean_imaging(
        ctx,
        ctx.visibility_path,
        0.2 * u.deg,
        img_config=config.imaging[0],
        sub_dir=ctx.work_dir / config.imaging[0].tag,
    )

    assert captured["moment8"][1] == f"{config.imaging[0].tag}_wsclean"
    assert captured["moment8"][2] == config.imaging[0].tag


def test_run_wsclean_imaging_skips_moment8_without_spectral_cube(tmp_path, monkeypatch):
    """Non-cube runs do not trigger moment8."""

    class FakeImage:
        def __init__(self, path):
            self.path = path

        def plot(self, filename, **kwargs):
            Path(filename).write_text("png", encoding="utf-8")

    def fake_require(module_name):
        if module_name == "karabo.imaging.image":
            return SimpleNamespace(Image=FakeImage)
        if module_name == "karabo.imaging.imager_wsclean":
            return SimpleNamespace(TMP_PREFIX_CUSTOM="tmp", TMP_PURPOSE_CUSTOM="test")
        if module_name == "karabo.util.file_handler":
            return SimpleNamespace(
                FileHandler=lambda: SimpleNamespace(
                    get_tmp_dir=lambda **kwargs: tmp_path
                )
            )
        raise AssertionError(module_name)

    captured = {}

    def fake_run_moment8(ctx, work_dir, output_prefix, tag):
        captured["moment8"] = (str(work_dir), output_prefix, tag)

    def fake_run(argv, work_dir):
        prefix = argv[argv.index("-name") + 1]
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"{prefix}-MFS-image.fits").write_text("fits", encoding="utf-8")

        class Result:
            stdout = "ok"

        return Result()

    monkeypatch.setattr("skasim.imaging.require_karabo_module", fake_require)
    monkeypatch.setattr("skasim.imaging.run_wsclean_command", fake_run)
    monkeypatch.setattr(
        "skasim.loaders.image_models.run_moment8_for_spectral_cube",
        fake_run_moment8,
    )
    monkeypatch.setattr(
        "skasim.imaging.write_fits_preview",
        lambda img_path, png_path, title: Path(png_path).write_text(
            "png", encoding="utf-8"
        ),
    )

    config = SimConfig(
        output_dir=str(tmp_path / "run"),
        imaging=[ImgConfig(imager="wsclean")],
    )
    ctx = create_run_context(config)

    run_wsclean_imaging(
        ctx,
        ctx.visibility_path,
        0.2 * u.deg,
        img_config=config.imaging[0],
        sub_dir=ctx.work_dir / config.imaging[0].tag,
    )

    assert "moment8" not in captured


def test_resolve_wsclean_img_config_explicit_continuum_is_one_channel():
    """Explicit continuum mode images all channels into a single image."""
    img = ImgConfig(imager="wsclean", pixels=64, spectral_mode="continuum")

    resolved = resolve_wsclean_img_config(img, "continuum", n_channels=2048)

    assert resolved.channels_out == 1
    assert resolved.join_channels is False
    assert effective_wsclean_channels_out(resolved, 2048) == 1


def test_resolve_wsclean_img_config_continuum_keeps_user_channels_out():
    img = ImgConfig(
        imager="wsclean", pixels=64, spectral_mode="continuum", channels_out=4
    )

    resolved = resolve_wsclean_img_config(img, "continuum", n_channels=2048)

    assert resolved.channels_out == 4


def test_resolve_wsclean_img_config_auto_continuum_is_unchanged():
    """auto resolving to continuum keeps the historical settings."""
    img = ImgConfig(imager="wsclean", pixels=64)

    resolved = resolve_wsclean_img_config(img, "continuum", n_channels=16)

    assert resolved is img
    assert effective_wsclean_channels_out(resolved, 16) == 16


def test_resolve_wsclean_img_config_cube_mode_is_per_channel():
    img = ImgConfig(imager="wsclean", pixels=64, spectral_mode="cube")

    resolved = resolve_wsclean_img_config(img, "cube", n_channels=16)

    assert resolved.channels_out == 16
    assert resolved.join_channels is False


@pytest.mark.parametrize(
    "mode, expect_moment8, expect_channels_out",
    [("cube", True, "8"), ("continuum", False, "1")],
)
def test_run_wsclean_imaging_follows_block_spectral_mode_in_cube_run(
    tmp_path, monkeypatch, mode, expect_moment8, expect_channels_out
):
    """In a run with a cube, each block images as its spectral_mode says."""
    captured = {}

    def fake_require(module_name):
        if module_name == "karabo.imaging.imager_wsclean":
            return SimpleNamespace(TMP_PREFIX_CUSTOM="tmp", TMP_PURPOSE_CUSTOM="test")
        if module_name == "karabo.util.file_handler":
            return SimpleNamespace(
                FileHandler=lambda: SimpleNamespace(
                    get_tmp_dir=lambda **kwargs: tmp_path
                )
            )
        raise AssertionError(module_name)

    def fake_run(argv, work_dir):
        captured["argv"] = argv
        work_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("skasim.imaging.require_karabo_module", fake_require)
    monkeypatch.setattr("skasim.imaging.run_wsclean_command", fake_run)
    monkeypatch.setattr(
        "skasim.loaders.image_models.run_moment8_for_spectral_cube",
        lambda ctx, work_dir, prefix, tag: captured.setdefault("moment8", tag),
    )
    (tmp_path / "cube.fits").write_text("fits", encoding="utf-8")
    block = ImgConfig(imager="wsclean", pixels=64, tag=mode, spectral_mode=mode)
    config = SimConfig(
        output_dir=str(tmp_path / "run"),
        observation=ObsConfig(bandwidth_mhz=8.0, n_channels=8),
        models=[
            {"type": "component_sky_model", "catalog": "GLEAM"},
            SpectralCubeModelEntry(
                type="spectral_cube", cube=str(tmp_path / "cube.fits")
            ),
        ],
        imaging=[block],
    )
    ctx = create_run_context(config)

    run_wsclean_imaging(
        ctx,
        ctx.visibility_path,
        0.2 * u.deg,
        img_config=config.imaging[0],
        sub_dir=ctx.work_dir / mode,
        n_channels=8,
    )

    argv = captured["argv"]
    assert argv[argv.index("-channels-out") + 1] == expect_channels_out
    assert "-join-channels" not in argv
    assert ("moment8" in captured) is expect_moment8


def test_build_wsclean_argv_uses_default_command():
    """The default WSClean command builds an argv list starting with wsclean."""
    config = SimConfig(
        imaging=[ImgConfig(imager="wsclean", wsclean_command="wsclean", pixels=256)]
    )

    argv = build_wsclean_argv(
        img_config=config.imaging[0],
        visibility_path=Path("visibilities.MS"),
        fov=0.2 * u.deg,
        output_prefix="run-clean",
        n_channels=1,
    )
    assert argv[0] == "wsclean"
    assert "-name" in argv
    assert argv[argv.index("-name") + 1] == "run-clean"
    assert "visibilities.MS" in argv

    # previous hardcoded defaults still apply when fields are omitted
    assert "-multiscale" in argv
    assert "-local-rms" in argv
    assert "-join-channels" in argv
    assert argv[argv.index("-mgain") + 1] == "0.8"
    assert argv[argv.index("-auto-threshold") + 1] == "0.3"
    assert argv[argv.index("-auto-mask") + 1] == "3.0"


def test_build_wsclean_argv_respects_img_config_overrides():
    """ImgConfig wsclean fields override the skasim defaults."""
    config = SimConfig(
        imaging=[
            ImgConfig(
                imager="wsclean",
                pixels=256,
                clean_iterations=10000,
                mgain=0.95,
                multiscale=False,
                auto_threshold=1.0,
                auto_mask=5.0,
                local_rms=False,
                join_channels=False,
                channels_out=1,
                padding=1.2,
                threads=4,
            )
        ]
    )

    argv = build_wsclean_argv(
        img_config=config.imaging[0],
        visibility_path=Path("visibilities.MS"),
        fov=0.2 * u.deg,
        output_prefix="run-clean",
        n_channels=4,
    )

    assert argv[argv.index("-niter") + 1] == "10000"
    assert argv[argv.index("-mgain") + 1] == "0.95"
    assert argv[argv.index("-auto-threshold") + 1] == "1.0"
    assert argv[argv.index("-auto-mask") + 1] == "5.0"
    assert argv[argv.index("-channels-out") + 1] == "1"
    assert "-multiscale" not in argv
    assert "-local-rms" not in argv
    assert "-join-channels" not in argv
    assert argv[argv.index("-padding") + 1] == "1.2"
    assert argv[argv.index("-j") + 1] == "4"


def test_build_wsclean_argv_multiscale_scales():
    """multiscale_scales only appear when multiscale is active."""
    config = SimConfig(
        imaging=[
            ImgConfig(
                imager="wsclean",
                multiscale_scales=[0, 4, 8, 16],
            )
        ]
    )

    argv = build_wsclean_argv(
        img_config=config.imaging[0],
        visibility_path=Path("visibilities.MS"),
        fov=0.2 * u.deg,
        output_prefix="run-clean",
    )

    assert "-multiscale" in argv
    assert argv[argv.index("-multiscale-scales") + 1] == "0,4,8,16"

    config2 = SimConfig(
        imaging=[
            ImgConfig(
                imager="wsclean",
                multiscale=False,
                multiscale_scales=[0, 4, 8, 16],
            )
        ]
    )
    argv2 = build_wsclean_argv(
        img_config=config2.imaging[0],
        visibility_path=Path("visibilities.MS"),
        fov=0.2 * u.deg,
        output_prefix="run-clean",
    )
    assert "-multiscale-scales" not in argv2


def test_build_wsclean_argv_parses_singularity_command():
    """Containerized WSClean commands are parsed into argv without shell text."""
    command = "singularity exec /mnt/software/containers/wsclean-3.10-dysco.sif wsclean"
    config = SimConfig(
        imaging=[ImgConfig(imager="wsclean", wsclean_command=command, pixels=256)]
    )

    argv = build_wsclean_argv(
        img_config=config.imaging[0],
        visibility_path=Path("visibilities.MS"),
        fov=0.2 * u.deg,
        output_prefix="run-clean",
    )

    assert argv[:4] == [
        "singularity",
        "exec",
        "/mnt/software/containers/wsclean-3.10-dysco.sif",
        "wsclean",
    ]
    assert "-name" in argv


def test_build_wsclean_argv_caps_channels_out_to_available_channels():
    """Small smoke runs should not request more WSClean outputs than channels."""
    config = SimConfig(
        observation=ObsConfig(bandwidth_mhz=25.0, n_channels=2),
        imaging=[ImgConfig(imager="wsclean", pixels=256)],
    )

    argv = build_wsclean_argv(
        img_config=config.imaging[0],
        visibility_path=Path("visibilities.MS"),
        fov=0.2 * u.deg,
        output_prefix="run-clean",
        n_channels=config.observation.n_channels,
    )

    assert argv[argv.index("-channels-out") + 1] == "2"


def test_run_wsclean_command_uses_argv_and_working_directory(tmp_path, monkeypatch):
    """WSClean execution avoids shell execution and uses an explicit cwd."""
    calls = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            self.stdout = iter(["ok\n"])
            self.returncode = 0

        def wait(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("skasim.imaging.subprocess.Popen", FakePopen)

    result = run_wsclean_command(["wsclean", "-name", "run-clean"], tmp_path)

    assert result.stdout == "ok\n"
    assert calls[0][0] == ["wsclean", "-name", "run-clean"]
    assert calls[0][1]["cwd"] == str(tmp_path)
    assert calls[0][1]["shell"] is False


def test_run_dirty_imaging_passes_icrs_phase_centre(tmp_path, monkeypatch):
    """FITS-derived FK5 centres are transformed before calling Karabo dirty imaging."""
    captured = {}

    class FakeVisibility:
        def __init__(self, path):
            self.path = path

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["phase_centre"] = kwargs["imaging_phase_centre"]

    class FakeDirtyImage:
        def write_to_file(self, path, overwrite):
            Path(path).write_text("fits", encoding="utf-8")

    class FakeImager:
        def __init__(self, config):
            self.config = config

        def create_dirty_image(self, visibility):
            captured["visibility"] = visibility
            return FakeDirtyImage()

    def fake_require(module_name):
        if module_name == "karabo.imaging.imager_oskar":
            return SimpleNamespace(
                OskarDirtyImagerConfig=FakeConfig,
                OskarDirtyImager=FakeImager,
            )
        if module_name == "karabo.simulation.visibility":
            return SimpleNamespace(Visibility=FakeVisibility)
        raise AssertionError(module_name)

    monkeypatch.setattr("skasim.imaging.require_karabo_module", fake_require)
    monkeypatch.setattr(
        "skasim.imaging.write_fits_preview",
        lambda img_path, png_path, title: Path(png_path).write_text(
            "png", encoding="utf-8"
        ),
    )

    config = SimConfig(
        output_dir=str(tmp_path / "run"),
        imaging=[ImgConfig(pixels=64, tag="default")],
    )
    ctx = create_run_context(config)

    run_dirty_imaging(
        ctx,
        tmp_path / "visibilities.MS",
        0.1 * u.deg,
        SkyCoord(10.0 * u.deg, 2.0 * u.deg, frame="fk5"),
        ctx.config.imaging[0],
        ctx.work_dir,
    )

    assert captured["phase_centre"].frame.name == "icrs"
    assert captured["visibility"].path == str(tmp_path / "visibilities.MS")


def test_collect_wsclean_outputs_matches_only_configured_prefix(tmp_path):
    """WSClean output discovery ignores files from other runs."""
    expected = [
        tmp_path / "run-clean-MFS-dirty.fits",
        tmp_path / "run-clean-MFS-image.fits",
    ]
    unrelated = [
        tmp_path / "old-run-MFS-image.fits",
        tmp_path / "wsclean-0000-image.fits",
    ]
    for path in expected + unrelated:
        path.write_text("fits", encoding="utf-8")

    outputs = collect_wsclean_outputs(tmp_path, "run-clean")

    assert outputs == expected


def test_wsclean_output_prefix_is_run_scoped(tmp_path):
    """WSClean -name uses a stable prefix scoped to the current run."""
    config = SimConfig(output_dir=str(tmp_path / "example"))
    ctx = create_run_context(config)

    prefix = wsclean_output_prefix(ctx)

    assert prefix == "example_wsclean"


def test_write_fits_preview_uses_aplpy_cmasher_renderer(tmp_path):
    """WSClean FITS previews are rendered as WCS-aware APLpy PNGs."""
    pytest.importorskip("aplpy")
    pytest.importorskip("cmasher")

    fits_path = tmp_path / "image.fits"
    png_path = tmp_path / "image.png"
    data = np.array(
        [
            [0.0, 1.0e-6, 2.0e-6],
            [3.0e-6, 4.0e-6, 5.0e-6],
            [6.0e-6, 7.0e-6, 8.0e-6],
        ]
    )
    header = fits.Header()
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CRPIX1"] = 2.0
    header["CRPIX2"] = 2.0
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = 2.0 / 3600.0
    header["BMIN"] = 1.5 / 3600.0
    header["BPA"] = 35.0
    fits.writeto(fits_path, data, header, overwrite=True)

    write_fits_preview(fits_path, png_path, "Ignored title")

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_write_psf_profile_preview_writes_png_with_x_y_cuts(tmp_path):
    """The PSF preview renders a valid PNG (2D core + x/y profile panels)."""
    from skasim.imaging import write_psf_profile_preview

    psf_path = tmp_path / "psf.fits"
    png_path = tmp_path / "psf.png"
    size = 32
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    data = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * 3.0**2))
    header = fits.Header()
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["BMAJ"] = 5.0 / 3600.0
    header["BMIN"] = 4.0 / 3600.0
    fits.writeto(psf_path, data.astype(np.float32), header, overwrite=True)

    write_psf_profile_preview(psf_path, png_path, "Test PSF")

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_gaussian_fwhm_peak_and_half_max_points():
    """_gaussian_fwhm peaks at the given amplitude and reaches half-max at ±FWHM/2."""
    from skasim.imaging import _gaussian_fwhm

    fwhm = 6.0
    amplitude = 2.5
    x = np.array([0.0, fwhm / 2.0, -fwhm / 2.0, fwhm * 10.0])
    y = _gaussian_fwhm(x, fwhm, amplitude)

    assert y[0] == pytest.approx(amplitude)
    assert y[1] == pytest.approx(amplitude / 2.0)
    assert y[2] == pytest.approx(amplitude / 2.0)
    assert y[3] == pytest.approx(0.0, abs=1e-6)


def test_write_psf_profile_preview_handles_missing_bmin(tmp_path):
    """The PSF preview still renders when only BMAJ (no BMIN) is present."""
    from skasim.imaging import write_psf_profile_preview

    psf_path = tmp_path / "psf_bmaj_only.fits"
    png_path = tmp_path / "psf_bmaj_only.png"
    size = 32
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    data = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * 3.0**2))
    header = fits.Header()
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["BMAJ"] = 5.0 / 3600.0
    fits.writeto(psf_path, data.astype(np.float32), header, overwrite=True)

    write_psf_profile_preview(psf_path, png_path, "BMAJ-only PSF")

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_write_psf_profile_preview_falls_back_to_pixel_units_without_wcs(tmp_path):
    """Without a pixel scale in the header, the preview still renders in pixel units."""
    from skasim.imaging import write_psf_profile_preview

    psf_path = tmp_path / "psf_no_wcs.fits"
    png_path = tmp_path / "psf_no_wcs.png"
    size = 32
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    data = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * 3.0**2))
    fits.writeto(psf_path, data.astype(np.float32), overwrite=True)

    write_psf_profile_preview(psf_path, png_path, "No-WCS PSF")

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_write_sky_model_previews_writes_full_and_fov_pngs(tmp_path):
    """Sky model previews include full-catalog and FoV-matched views."""
    from astropy.coordinates import SkyCoord

    from skasim.sky import SkyModel

    sky_model = SkyModel(
        np.array(
            [
                [
                    150.0,
                    2.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    700e6,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    150.1,
                    2.1,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    700e6,
                    0.0,
                    0.0,
                    4.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )
    )

    outputs = write_sky_model_previews(
        sky_model,
        SkyCoord(150.0 * u.deg, 2.0 * u.deg),
        1.0 * u.deg,
        tmp_path,
        "example",
    )

    assert outputs == [
        ("example_sky_model.png", "sky_model"),
        ("example_sky_model_fov.png", "sky_model_fov"),
    ]
    for path, _ in outputs:
        assert (tmp_path / path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_sky_model_ellipses_preserve_source_shape_metadata():
    """Sky model source previews use major/minor axes and position angle."""
    ellipses = _sky_model_ellipses(
        np.array([150.0]),
        np.array([2.0]),
        np.array([18.0]),
        np.array([6.0]),
        np.array([35.0]),
    )

    assert len(ellipses) == 1
    ellipse = ellipses[0]
    assert ellipse.width == pytest.approx(18.0 / 3600.0)
    assert ellipse.height == pytest.approx(6.0 / 3600.0)
    assert ellipse.angle == pytest.approx(55.0)


def test_sky_model_position_angle_uses_astronomical_convention():
    """PA 0 is north-south and PA 90 is east-west in Matplotlib coordinates."""
    assert _sky_model_position_angle(0.0) == pytest.approx(90.0)
    assert _sky_model_position_angle(90.0) == pytest.approx(0.0)


def test_compact_source_mask_depends_on_plot_width():
    """Small sources become crosses when they are unresolved in the plotted FoV."""
    mask = _compact_source_mask(
        np.array([3.0, 100.0]),
        plot_width_deg=1.0,
    )

    assert mask.tolist() == [True, False]


def test_flux_marker_sizes_scale_with_flux_density():
    """Compact-source cross size encodes flux density."""
    sizes = _flux_marker_sizes(np.array([0.01, 1.0]))

    assert sizes[1] > sizes[0]


def test_reference_catalog_generator_writes_ds9_regions(tmp_path):
    """Reference JSON catalog generation has a matching DS9 region writer."""
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_gaussian_catalog.py"
    )
    spec = importlib.util.spec_from_file_location("generate_gaussian_catalog", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    region_path = tmp_path / "catalog.reg"
    module.write_ds9_regions(
        [
            {
                "ra": 150.0,
                "dec": 2.0,
                "I": 1.2,
                "spec_index": -0.5,
                "major_axis": 10.0,
                "minor_axis": 4.0,
                "pa": 35.0,
            }
        ],
        region_path,
    )

    text = region_path.read_text(encoding="utf-8")
    assert "fk5" in text
    assert 'ellipse(150.0000000000,2.0000000000,10.000000",4.000000",55.000000)' in text
    assert "point(150.0000000000,2.0000000000) # point=cross 8" in text


def test_reference_catalog_generator_uses_broad_demo_distributions(
    tmp_path, monkeypatch
):
    """Reference catalog fluxes and sizes exercise compact, faint, and extended sources."""
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_gaussian_catalog.py"
    )
    spec = importlib.util.spec_from_file_location("generate_gaussian_catalog", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)

    module.generate_catalog(seed=42, n_sources=100)

    import json

    sources = json.loads(
        (tmp_path / "demo_output" / "reference_gaussian_catalog.json").read_text()
    )
    fluxes = np.array([source["I"] for source in sources])
    major_axes = np.array([source["major_axis"] for source in sources])
    axis_ratios = np.array(
        [source["minor_axis"] / source["major_axis"] for source in sources]
    )
    position_angles = np.array([source["pa"] for source in sources])
    spectral_indices = np.array([source["spec_index"] for source in sources])

    assert len(sources) == 100
    assert np.count_nonzero(fluxes >= 0.06) == 2
    assert np.count_nonzero((fluxes >= 1.0e-5) & (fluxes <= 2.0e-5)) >= 10
    assert np.median(fluxes) == pytest.approx(1.0e-3, abs=5.0e-4)
    assert np.count_nonzero(major_axes < 1.0) >= 5
    assert np.count_nonzero((major_axes >= 1.0) & (major_axes < 15.0)) >= 40
    assert np.count_nonzero((major_axes >= 15.0) & (major_axes < 60.0)) >= 15
    assert np.count_nonzero(major_axes >= 60.0) >= 5
    assert axis_ratios.min() < 0.35
    assert axis_ratios.max() > 0.9
    assert position_angles.min() < 10.0
    assert position_angles.max() > 170.0
    assert spectral_indices.mean() == pytest.approx(-0.5, abs=0.1)
    assert (tmp_path / "demo_output" / "reference_gaussian_catalog.reg").exists()


def test_sky_model_previews_use_reversed_colormap():
    """Sky model previews render brighter sources darker than faint sources."""
    from matplotlib.collections import PatchCollection

    ellipses = _sky_model_ellipses(
        np.array([150.0]),
        np.array([2.0]),
        np.array([18.0]),
        np.array([6.0]),
        np.array([35.0]),
    )
    collection = PatchCollection(ellipses, cmap=SKY_MODEL_CMAP)

    assert collection.cmap.name == "viridis_r"


def test_padded_limits_use_source_coordinates():
    """Full sky-model previews set axes from source coordinates, not 0..1 defaults."""
    lower, upper = _padded_limits(np.array([149.5, 150.5]))

    assert lower < 149.5
    assert upper > 150.5


def test_run_wsclean_imaging_uses_run_prefix_and_stable_outputs(tmp_path, monkeypatch):
    """WSClean imaging records only stable outputs for the current prefix."""
    from skasim.imaging import run_wsclean_imaging

    class FakeImage:
        def __init__(self, path):
            self.path = path

        def plot(self, filename, **kwargs):
            Path(filename).write_text("png", encoding="utf-8")

    def fake_require(module_name):
        if module_name == "karabo.imaging.image":
            return SimpleNamespace(Image=FakeImage)
        if module_name == "karabo.imaging.imager_wsclean":
            return SimpleNamespace(TMP_PREFIX_CUSTOM="tmp", TMP_PURPOSE_CUSTOM="test")
        if module_name == "karabo.util.file_handler":
            return SimpleNamespace(
                FileHandler=lambda: SimpleNamespace(
                    get_tmp_dir=lambda **kwargs: tmp_path
                )
            )
        raise AssertionError(module_name)

    captured_argv = []

    def fake_run(argv, work_dir):
        captured_argv.append(argv)
        prefix = argv[argv.index("-name") + 1]
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"{prefix}-MFS-image.fits").write_text("fits", encoding="utf-8")
        (work_dir / "old-run-MFS-image.fits").write_text("old", encoding="utf-8")

        class Result:
            stdout = "ok"

        return Result()

    monkeypatch.setattr("skasim.imaging.require_karabo_module", fake_require)
    monkeypatch.setattr("skasim.imaging.run_wsclean_command", fake_run)
    monkeypatch.setattr(
        "skasim.imaging.write_fits_preview",
        lambda img_path, png_path, title: Path(png_path).write_text(
            "png", encoding="utf-8"
        ),
    )

    config = SimConfig(
        output_dir=str(tmp_path / "example"),
        imaging=[ImgConfig(imager="wsclean")],
    )
    ctx = create_run_context(config)

    run_wsclean_imaging(
        ctx,
        ctx.visibility_path,
        0.2 * u.deg,
        img_config=config.imaging[0],
        sub_dir=ctx.work_dir / config.imaging[0].tag,
    )

    prefix = f"{config.imaging[0].tag}_wsclean"
    output_paths = [output.path for output in ctx.manifest.outputs]
    assert captured_argv[0][captured_argv[0].index("-name") + 1] == prefix
    assert f"default/{prefix}-MFS-image.fits" in output_paths
    assert f"default/{prefix}-MFS-image.png" in output_paths
    assert all("old-run" not in output for output in output_paths)
    assert all("_bw" not in output for output in output_paths)


def test_run_wsclean_imaging_cleanup_keeps_run_scoped_outputs(tmp_path, monkeypatch):
    """Cleanup does not delete outputs when the run prefix starts like a temp file."""
    from skasim.imaging import run_wsclean_imaging

    class FakeImage:
        def __init__(self, path):
            self.path = path

        def plot(self, filename, **kwargs):
            Path(filename).write_text("png", encoding="utf-8")

    def fake_require(module_name):
        if module_name == "karabo.imaging.image":
            return SimpleNamespace(Image=FakeImage)
        if module_name == "karabo.imaging.imager_wsclean":
            return SimpleNamespace(TMP_PREFIX_CUSTOM="tmp", TMP_PURPOSE_CUSTOM="test")
        if module_name == "karabo.util.file_handler":
            return SimpleNamespace(
                FileHandler=lambda: SimpleNamespace(
                    get_tmp_dir=lambda **kwargs: tmp_path
                )
            )
        raise AssertionError(module_name)

    def fake_run(argv, work_dir):
        prefix = argv[argv.index("-name") + 1]
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"{prefix}-MFS-image.fits").write_text("fits", encoding="utf-8")
        (work_dir / "wsclean-0000-temp.fits").write_text("temp", encoding="utf-8")

        class Result:
            stdout = "ok"

        return Result()

    monkeypatch.setattr("skasim.imaging.require_karabo_module", fake_require)
    monkeypatch.setattr("skasim.imaging.run_wsclean_command", fake_run)
    monkeypatch.setattr(
        "skasim.imaging.write_fits_preview",
        lambda img_path, png_path, title: Path(png_path).write_text(
            "png", encoding="utf-8"
        ),
    )

    config = SimConfig(
        output_dir=str(tmp_path / "wsclean-00case"),
        imaging=[ImgConfig(imager="wsclean")],
    )
    ctx = create_run_context(config)

    run_wsclean_imaging(
        ctx,
        ctx.visibility_path,
        0.2 * u.deg,
        img_config=config.imaging[0],
        sub_dir=ctx.work_dir / config.imaging[0].tag,
    )

    prefix = f"{config.imaging[0].tag}_wsclean"
    output_paths = [output.path for output in ctx.manifest.outputs]
    assert f"default/{prefix}-MFS-image.fits" in output_paths
    assert not (ctx.work_dir / "default" / "wsclean-0000-temp.fits").exists()


@pytest.mark.parametrize(
    "filename, product, per_channel",
    [
        ("cont_wsclean-MFS-image.fits", "image", False),
        ("cont_wsclean-image.fits", "image", False),  # -channels-out 1
        ("cont_wsclean-psf.fits", "psf", False),
        ("cont_wsclean-0007-residual.fits", "residual", True),
        ("cont_wsclean-12345-model.fits", "model", True),
        ("sub/cont_wsclean-image-pb.fits", "image-pb", False),
        # a tag that looks like a product or holds digits is not misread
        ("model_test_wsclean-image.fits", "image", False),
    ],
)
def test_wsclean_product_name(filename, product, per_channel):
    prefix = Path(filename).name.split("-", 1)[0]
    assert wsclean_product_name(filename, prefix) == (product, per_channel)


def test_wsclean_product_name_without_prefix_spots_channel_index():
    assert wsclean_product_name("x-0001-image.fits", None) == (None, True)
    assert wsclean_product_name("x-MFS-image.fits", None) == (None, False)


def _wsclean_channel_image(path, freq_hz, width_hz, value):
    """A per-channel image as WSClean writes it: its own one-pixel FREQ axis."""
    header = fits.Header()
    header["CTYPE1"], header["CRPIX1"], header["CRVAL1"], header["CDELT1"] = (
        "RA---SIN",
        2.0,
        10.0,
        -1e-3,
    )
    header["CTYPE2"], header["CRPIX2"], header["CRVAL2"], header["CDELT2"] = (
        "DEC--SIN",
        2.0,
        2.0,
        1e-3,
    )
    header["CTYPE3"], header["CRPIX3"], header["CRVAL3"], header["CDELT3"] = (
        "FREQ",
        1.0,
        freq_hz,
        width_hz,
    )
    header["CUNIT3"] = "Hz"
    header["CTYPE4"], header["CRPIX4"], header["CRVAL4"], header["CDELT4"] = (
        "STOKES",
        1.0,
        1.0,
        1.0,
    )
    data = np.full((1, 1, 3, 3), value, dtype=np.float32)
    fits.writeto(path, data, header, overwrite=True)


def test_stack_channels_uses_each_channel_frequency(tmp_path):
    """The stacked FREQ axis starts at channel 0 and steps by the spacing.

    WSClean's CRVAL3 is the channel's own centre; reading it as a cube-wide
    reference used to double the spacing and shift the axis.
    """
    from skasim.imaging import stack_channels

    start, width = 1417.6e6, 50e3
    paths = []
    for ch in range(4):
        path = tmp_path / f"line_wsclean-{ch:04d}-image.fits"
        _wsclean_channel_image(path, start + ch * width, width, float(ch))
        paths.append(path)
    out = tmp_path / "line_wsclean-cube-image.fits"

    stack_channels(paths, out)

    header = fits.getheader(out)
    data = fits.getdata(out)
    assert data.shape == (4, 3, 3)
    assert [float(plane[0, 0]) for plane in data] == [0.0, 1.0, 2.0, 3.0]
    assert header["CRPIX3"] == 1.0
    assert header["CRVAL3"] == pytest.approx(start)
    assert header["CDELT3"] == pytest.approx(width)
    freqs = WCS(header).spectral.pixel_to_world(np.arange(4)).to_value(u.Hz)
    np.testing.assert_allclose(freqs, start + np.arange(4) * width)
