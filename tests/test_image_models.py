"""Image-model validation and reporting behavior."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from skasim.config import (
    ImgConfig,
    ObsConfig,
    SimConfig,
    SpectralCubeModelEntry,
    StaticStokesMapsModelEntry,
)
from skasim.loaders.image_models import (
    _resample_spectral_axis_to_ms_channels,
    _set_crval4_via_script,
    image_model_center,
    inject_image_models,
    merge_model_data_into_data,
    prepare_casa_taylor_terms,
    prepare_spectral_cube_for_casa,
    read_fits_cube_info,
    run_casa_ft,
    run_casa_importfits,
    validate_casa_taylor_terms,
    validate_continuum_i_alpha,
    validate_spectral_cube,
    validate_static_stokes_maps,
    write_image_model_previews,
)
from skasim.loaders.image_models.casa_interop import (
    _spectral_axis_number,
    run_casa_script,
    run_casa_set_spectral_coordinate,
)
from skasim.loaders.wsclean_predict import (
    inject_static_stokes_i_with_wsclean_predict,
)
from skasim.manifest import RunManifest, create_run_context
from skasim.weblog import render_weblog


def _write_model_pair(tmp_path, shape=(16, 16), alpha_shape=None):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 2.0]
    wcs.wcs.crpix = [shape[1] / 2.0, shape[0] / 2.0]
    wcs.wcs.cdelt = [-0.001, 0.001]

    stokes_header = wcs.to_header()
    stokes_header["BUNIT"] = "Jy/pixel"
    alpha_header = wcs.to_header()
    alpha_header["BUNIT"] = "1"

    stokes_i = tmp_path / "stokes_i.fits"
    alpha = tmp_path / "alpha.fits"
    fits.writeto(
        stokes_i, np.ones(shape, dtype=np.float32), stokes_header, overwrite=True
    )
    fits.writeto(
        alpha,
        np.zeros(alpha_shape or shape, dtype=np.float32),
        alpha_header,
        overwrite=True,
    )
    return stokes_i, alpha


def test_validate_continuum_i_alpha_accepts_matching_fits(tmp_path):
    """Continuum I+alpha validation reports model metadata."""
    stokes_i, alpha = _write_model_pair(tmp_path)
    cfg = SimConfig(
        models=[
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
            }
        ]
    )

    report = validate_continuum_i_alpha(cfg.models[0])

    assert report["spatial_shape"] == [16, 16]
    assert report["unit"] == "Jy/pixel"
    assert report["reference_frequency_hz"] == pytest.approx(1.4e9)


def test_validate_continuum_i_alpha_rejects_shape_mismatch(tmp_path):
    """Stokes I and alpha maps must share a spatial grid."""
    stokes_i, alpha = _write_model_pair(tmp_path, alpha_shape=(8, 8))
    cfg = SimConfig(
        models=[
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
            }
        ]
    )

    with pytest.raises(ValueError, match="matching spatial dimensions"):
        validate_continuum_i_alpha(cfg.models[0])


def test_validate_continuum_i_alpha_accepts_real_model_fixture(tmp_path):
    """Real WSClean model FITS headers are accepted as Jy/pixel image models."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "models_for_testing"
        / "CY4223_L_004_20180322_avg_target_3C320_vla_loop_01-0000-model.fits"
    )
    if not fixture.exists():
        pytest.skip("models_for_testing fixture directory is not present")
    alpha = tmp_path / "alpha.fits"
    with fits.open(fixture) as hdul:
        header = hdul[0].header.copy()
        header["BUNIT"] = "1"
        fits.writeto(alpha, np.zeros_like(hdul[0].data), header, overwrite=True)
    cfg = SimConfig(
        models=[
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(fixture),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
            }
        ]
    )

    report = validate_continuum_i_alpha(cfg.models[0])

    assert report["spatial_shape"] == [1600, 1600]
    assert report["unit"] == "JY/PIXEL"


def test_image_model_center_uses_first_model_wcs(tmp_path):
    """Image-only runs can derive the phase centre from FITS WCS."""
    stokes_i, alpha = _write_model_pair(tmp_path)
    cfg = SimConfig(
        models=[
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
            }
        ]
    )

    center = image_model_center(cfg.models)

    assert center is not None
    assert center.ra.deg == pytest.approx(10.0, abs=1e-3)
    assert center.dec.deg == pytest.approx(2.0, abs=1e-3)


def test_weblog_renders_fits_model_preview(tmp_path):
    """FITS model previews appear in the weblog Sky Model section."""
    pytest.importorskip("aplpy")
    pytest.importorskip("cmasher")
    stokes_i, alpha = _write_model_pair(tmp_path, shape=(32, 32))
    cfg = SimConfig(
        models=[
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
            }
        ],
        output_dir=str(tmp_path / "run"),
    )
    ctx = create_run_context(cfg)

    write_image_model_previews(
        ctx,
        0.05 * u.deg,
    )
    html = render_weblog(ctx.manifest, ctx.work_dir)

    assert "Footprint" in html
    assert "continuum_i_alpha" in html


def test_weblog_renders_existing_fits_model_output(tmp_path):
    """The weblog can render a pre-recorded FITS model preview output."""
    manifest = RunManifest(
        run_id="preview",
        started_at=datetime(2026, 5, 31, tzinfo=timezone.utc),
        config=SimConfig(),
    )
    png_path = tmp_path / "preview_fits_model.png"
    png_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    manifest.add_output(
        "plot",
        png_path.name,
        role="fits_model",
        metadata={"model_type": "continuum_i_alpha"},
    )

    html = render_weblog(manifest, tmp_path)

    assert "Footprint" in html
    assert "FITS model preview" in html


def test_weblog_renders_casa_taylor_term_preview(tmp_path, monkeypatch):
    """CASA Taylor-term previews are exported to FITS before rendering."""
    pytest.importorskip("aplpy")
    pytest.importorskip("cmasher")
    tt0 = tmp_path / "model.tt0"
    tt1 = tmp_path / "model.tt1"
    tt0.mkdir()
    tt1.mkdir()
    (tt0 / "table.dat").write_text("table", encoding="utf-8")
    (tt1 / "table.dat").write_text("table", encoding="utf-8")
    cfg = SimConfig(
        models=[
            {
                "type": "casa_taylor_terms",
                "tt0": str(tt0),
                "tt1": str(tt1),
                "reference_frequency_hz": 1.5e9,
            }
        ],
        output_dir=str(tmp_path / "run"),
    )
    ctx = create_run_context(cfg)
    calls = []

    def fake_export(work_dir, imagename, fitsimage):
        calls.append((work_dir, imagename, fitsimage))
        stokes_i, _ = _write_model_pair(tmp_path, shape=(32, 32))
        fitsimage.write_bytes(stokes_i.read_bytes())

    monkeypatch.setattr(
        "skasim.loaders.image_models.previews.run_casa_exportfits", fake_export
    )

    write_image_model_previews(
        ctx,
        0.05 * u.deg,
    )
    html = render_weblog(ctx.manifest, ctx.work_dir)

    assert calls[0][1] == tt0.resolve()
    assert "Footprint" in html
    assert "casa_taylor_terms" in html
    assert (ctx.work_dir / "run_fits_model.png").exists()


def test_inject_image_models_runs_continuum_entries_in_order(tmp_path, monkeypatch):
    """Image injection validates entries, predicts with CASA ft, and merges once."""
    stokes_i, alpha = _write_model_pair(tmp_path)
    cfg = SimConfig(
        models=[
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
                "injection_backend": "casa_ft",
            },
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
                "injection_backend": "casa_ft",
            },
        ],
        output_dir=str(tmp_path / "run"),
    )
    ctx = create_run_context(cfg)
    calls = []

    class Product:
        nterms = 2
        reffreq = "1400000000.0Hz"

        def __init__(self, index):
            self.model_paths = [ctx.work_dir / f"model-{index}.image"]
            self.intermediates = []

    def fake_prepare(ctx_arg, entry, index):
        assert ctx_arg is ctx
        return Product(index)

    def fake_run_casa_ft(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "skasim.loaders.image_models.prepare_continuum_i_alpha_for_casa", fake_prepare
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.prepare_continuum_i_alpha_for_casa",
        fake_prepare,
    )
    monkeypatch.setattr("skasim.loaders.image_models.run_casa_ft", fake_run_casa_ft)
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.run_casa_ft", fake_run_casa_ft
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": str(visibility_path)}),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": str(visibility_path)}),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": str(visibility_path)}),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casa_executable",
        lambda: Path("/opt/casa/bin/casa"),
    )

    inject_image_models(ctx, visibility_path=Path("fake.ms"))

    # two entries -> two ft calls + one merge
    assert len(calls) == 3
    ft_calls = [c for c in calls if "visibility_path" in c]
    assert ft_calls[0].get("incremental") is False
    assert ft_calls[1].get("incremental") is True
    assert any(c.get("merged") == "fake.ms" for c in calls)


def test_inject_image_models_runs_continuum_entries_with_wsclean_predict(
    tmp_path, monkeypatch
):
    """Two continuum entries run wsclean-predict in order and merge once."""
    stokes_i, alpha = _write_model_pair(tmp_path)
    cfg = SimConfig(
        models=[
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
                "injection_backend": "wsclean_predict",
            },
            {
                "type": "continuum_i_alpha",
                "stokes_i": str(stokes_i),
                "alpha": str(alpha),
                "reference_frequency_hz": 1.4e9,
                "injection_backend": "wsclean_predict",
            },
        ],
        output_dir=str(tmp_path / "run"),
    )
    ctx = create_run_context(cfg)
    calls = []

    class Product:
        nterms = 2
        reffreq = "1400000000.0Hz"

        def __init__(self, index):
            self.model_paths = [ctx.work_dir / f"model-{index}.fits"]
            self.intermediates = [stokes_i, alpha]

    def fake_prepare(ctx_arg, entry, index):
        assert ctx_arg is ctx
        return Product(index)

    def fake_inject(ctx_arg, entry, index, visibility_path, img_config, product):
        calls.append(
            {
                "backend": "wsclean_predict",
                "index": index,
                "visibility_path": str(visibility_path),
            }
        )
        return {"backend": "wsclean_predict"}

    monkeypatch.setattr(
        "skasim.loaders.image_models.prepare_continuum_i_alpha_for_casa", fake_prepare
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.prepare_continuum_i_alpha_for_casa",
        fake_prepare,
    )
    monkeypatch.setattr(
        "skasim.loaders.wsclean_predict.inject_continuum_i_alpha_with_wsclean_predict",
        fake_inject,
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": str(visibility_path)}),
    )

    inject_image_models(ctx, visibility_path=Path("fake.ms"))

    assert len(calls) == 3
    predict_calls = [c for c in calls if c.get("backend") == "wsclean_predict"]
    assert predict_calls[0]["index"] == 0
    assert predict_calls[1]["index"] == 1
    assert any(c.get("merged") == "fake.ms" for c in calls)


def test_merge_model_data_into_data(monkeypatch, tmp_path):
    """MODEL_DATA is accumulated into DATA in bounded row chunks."""
    columns = {
        "DATA": np.ones((5, 2, 2), dtype=np.complex64),
        "MODEL_DATA": np.ones((5, 2, 2), dtype=np.complex64) * 0.5,
    }
    calls = []

    class FakeTable:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def colnames(self):
            return ["DATA", "MODEL_DATA"]

        def nrows(self):
            return 5

        def getcell(self, name, row):
            return columns[name][row].copy()

        def getcol(self, name, startrow=0, nrow=-1):
            return columns[name][startrow : startrow + nrow].copy()

        def putcol(self, name, value, startrow=0, nrow=-1):
            calls.append((name, startrow, nrow))
            columns[name][startrow : startrow + nrow] = value

    # inject fake casacore.tables.table before the function imports it
    fake_tables = ModuleType("casacore.tables")
    fake_tables.table = FakeTable
    monkeypatch.setitem(sys.modules, "casacore.tables", fake_tables)

    # one row is 2*2*8 = 32 bytes, so 64 bytes -> chunks of 2 rows
    merge_model_data_into_data(tmp_path / "visibilities.MS", chunk_bytes=64)

    assert calls == [("DATA", 0, 2), ("DATA", 2, 2), ("DATA", 4, 1)]
    assert np.allclose(columns["DATA"], 1.5)
    assert np.allclose(columns["MODEL_DATA"], 0.5)


def test_casa_taylor_terms_validate_and_prepare_existing_images(tmp_path, monkeypatch):
    """Existing CASA Taylor-term image directories are copied and aligned."""
    tt0 = tmp_path / "model.tt0"
    tt1 = tmp_path / "model.tt1"
    tt0.mkdir()
    tt1.mkdir()
    (tt0 / "table.dat").write_text("table", encoding="utf-8")
    (tt1 / "table.dat").write_text("table", encoding="utf-8")
    cfg = SimConfig(
        models=[
            {
                "type": "casa_taylor_terms",
                "tt0": str(tt0),
                "tt1": str(tt1),
                "reference_frequency_hz": 1.5e9,
            }
        ],
        output_dir=str(tmp_path / "run"),
    )
    ctx = create_run_context(cfg)
    calls = []

    # fake casacore table to satisfy require_casacore + pixel reads
    fake_map_data = np.ones((1, 1, 4, 4), dtype=np.float32)

    class FakeCasacoreTable:
        def __init__(self, path, readonly=True, ack=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getcol(self, name):
            return fake_map_data

        def putcol(self, name, value):
            pass

        def getcolkeywords(self, name):
            return {"dimnames": ["frequency"]}

        def keywordnames(self):
            return ["coords"]

        def getkeyword(self, name):
            if name == "coords":
                return {"spectral0": {}, "pixelmap0": [3]}
            return {}

    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casacore",
        lambda: FakeCasacoreTable,
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.run_casa_set_spectral_coordinate",
        lambda work_dir, image_paths, frequency_hz: calls.append(
            (work_dir, image_paths, frequency_hz)
        ),
    )

    report = validate_casa_taylor_terms(cfg.models[0])
    product = prepare_casa_taylor_terms(ctx, cfg.models[0], 0)

    assert report["nterms"] == 2
    assert product.model_paths == [
        ctx.work_dir / "model_entry_01_casa_taylor.tt0.image",
        ctx.work_dir / "model_entry_01_casa_taylor.tt1.image",
    ]
    assert product.nterms == 2
    assert product.reffreq == "700000000.0Hz"
    assert (product.model_paths[0] / "table.dat").exists()
    # adjust_spectral_reference calls _set_crval4_via_script for each term
    assert len(calls) == 2
    assert calls[0] == (ctx.work_dir, [product.model_paths[0]], 700_000_000.0)
    assert calls[1] == (ctx.work_dir, [product.model_paths[1]], 700_000_000.0)


def test_inject_image_models_runs_casa_taylor_terms(tmp_path, monkeypatch):
    """CASA Taylor-term entries are predicted with ft and merged into DATA."""
    tt0 = tmp_path / "model.tt0"
    tt1 = tmp_path / "model.tt1"
    tt0.mkdir()
    tt1.mkdir()
    (tt0 / "table.dat").write_text("table", encoding="utf-8")
    (tt1 / "table.dat").write_text("table", encoding="utf-8")
    cfg = SimConfig(
        models=[
            {
                "type": "casa_taylor_terms",
                "tt0": str(tt0),
                "tt1": str(tt1),
                "reference_frequency_hz": 1.5e9,
            }
        ],
        output_dir=str(tmp_path / "run"),
    )
    ctx = create_run_context(cfg)
    calls = []

    # fake casacore table to satisfy require_casacore + pixel reads
    fake_map_data = np.ones((1, 1, 4, 4), dtype=np.float32)

    class FakeCasacoreTable:
        def __init__(self, path, readonly=True, ack=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getcol(self, name):
            return fake_map_data

        def putcol(self, name, value):
            pass

        def getcolkeywords(self, name):
            return {"dimnames": ["frequency"]}

        def keywordnames(self):
            return ["coords"]

        def getkeyword(self, name):
            if name == "coords":
                return {"spectral0": {}, "pixelmap0": [3]}
            return {}

    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casacore",
        lambda: FakeCasacoreTable,
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.run_casa_set_spectral_coordinate",
        lambda work_dir, image_paths, frequency_hz: calls.append(
            {"spectral_copy": image_paths, "frequency_hz": frequency_hz}
        ),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.run_casa_set_spectral_coordinate",
        lambda work_dir, image_paths, frequency_hz: calls.append(
            {"spectral_copy": image_paths, "frequency_hz": frequency_hz}
        ),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.run_casa_ft",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.run_casa_ft",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": visibility_path}),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": visibility_path}),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casa_executable",
        lambda: Path("/opt/casa/bin/casa"),
    )

    inject_image_models(ctx, tmp_path / "visibilities.MS")

    copied_paths = [
        ctx.work_dir / "model_entry_01_casa_taylor.tt0.image",
        ctx.work_dir / "model_entry_01_casa_taylor.tt1.image",
    ]
    # prepare_casa_taylor_terms now adjusts spectral reference inline:
    #   calls[0]: adjust_spectral_reference for tt0 (via run_casa_set_spectral_coordinate)
    #   calls[1]: adjust_spectral_reference for tt1 (via run_casa_set_spectral_coordinate)
    #   calls[2]: run_casa_ft
    #   calls[3]: merge_model_data_into_data
    assert calls[0]["frequency_hz"] == 700_000_000.0
    assert calls[1]["frequency_hz"] == 700_000_000.0
    assert calls[2]["model_paths"] == copied_paths
    assert calls[2]["nterms"] == 2
    assert calls[2]["reffreq"] == "700000000.0Hz"
    assert calls[3]["merged"] == tmp_path / "visibilities.MS"
    assert ctx.manifest.milestones[-1].name == "image_injection_completed"


def test_run_casa_importfits_uses_batch_fallback(tmp_path, monkeypatch):
    """CASA importfits can run through a local CASA executable when casatasks is absent."""
    calls = []

    class Result:
        returncode = 0
        stdout = "ok"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        # create the side-effects that the real CASA importfits would produce
        for _fitsimage, imagename in [
            (tmp_path / "model.tt0.fits", tmp_path / "model.tt0.image"),
        ]:
            table_dir = (
                Path(imagename) if not isinstance(imagename, Path) else imagename
            )
            table_dir.mkdir(parents=True, exist_ok=True)
            (table_dir / "table.dat").write_text("fake", encoding="utf-8")
        return Result()

    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casa_executable",
        lambda: Path("/opt/casa/bin/casa"),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.subprocess.run", fake_run
    )

    run_casa_importfits(
        tmp_path,
        [(tmp_path / "model.tt0.fits", tmp_path / "model.tt0.image")],
    )

    script = (tmp_path / "skasim_casa_importfits.py").read_text(encoding="utf-8")
    assert "from casatasks import importfits" in script
    assert "importfits(" in script
    assert "model.tt0.fits" in script
    assert calls[0][0][:5] == [
        "/opt/casa/bin/casa",
        "--nologger",
        "--nogui",
        "--log2term",
        "-c",
    ]
    assert calls[0][1]["cwd"] == str(tmp_path)


def test_inject_image_models_runs_spectral_cube_with_wsclean_predict(
    tmp_path, monkeypatch
):
    """Spectral-cube entries use wsclean -predict; ft is not called."""
    cube_path = _write_spectral_cube(tmp_path, n_channels=8, pixels=64, freq_mhz=700.0)
    cfg = SimConfig(
        output_dir=str(tmp_path / "run"),
        models=[
            {
                "type": "spectral_cube",
                "cube": str(cube_path),
            },
        ],
        observation=ObsConfig(
            frequency_mhz=700.0,
            bandwidth_mhz=100.0,
            n_channels=8,
            observation_time_s=1,
        ),
        imaging=[ImgConfig(pixels=64, imager="wsclean")],
    )
    ctx = create_run_context(cfg)
    calls = []

    monkeypatch.setattr(
        "skasim.loaders.wsclean_predict.run_wsclean_predict",
        lambda **kwargs: calls.append({"wsclean_predict": kwargs}),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": visibility_path}),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": visibility_path}),
    )

    effective_img = cfg.imaging[0].model_copy(
        update={"fov_deg": 0.2, "cell_size_arcsec": 11.25}
    )
    inject_image_models(
        ctx,
        tmp_path / "visibilities.MS",
        imaging_configs=[effective_img],
    )

    wsclean_calls = [c["wsclean_predict"] for c in calls if "wsclean_predict" in c]
    merge_calls = [c["merged"] for c in calls if "merged" in c]

    assert len(wsclean_calls) == 1
    assert wsclean_calls[0]["n_channels"] == 8
    assert wsclean_calls[0]["img_config"] is effective_img
    assert len(merge_calls) == 1
    assert ctx.manifest.milestones[-1].name == "image_injection_completed"


def test_run_casa_ft_uses_batch_fallback(tmp_path, monkeypatch):
    """CASA ft falls back to a batch CASA process when casatasks is absent."""
    import sys
    from types import ModuleType

    calls = []

    class Result:
        returncode = 0
        stdout = "ok"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    # mock casacore.tables.table for the post-batch verification
    class FakeTable:
        def __init__(self, path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def colnames(self):
            return ["DATA", "MODEL_DATA"]

    fake_tables = ModuleType("casacore.tables")
    fake_tables.table = FakeTable
    monkeypatch.setitem(sys.modules, "casacore.tables", fake_tables)

    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casa_executable",
        lambda: Path("/opt/casa/bin/casa"),
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.subprocess.run", fake_run
    )

    run_casa_ft(
        visibility_path=tmp_path / "visibilities.MS",
        model_paths=[tmp_path / "model.tt0.image", tmp_path / "model.tt1.image"],
        nterms=2,
        reffreq="1400000000.0Hz",
        incremental=True,
    )

    script = (tmp_path / "skasim_casa_ft.py").read_text(encoding="utf-8")
    assert "from casatasks import ft" in script
    assert "model.tt0.image" in script
    assert "model.tt1.image" in script
    assert "incremental=True" in script
    assert calls[0][0][-1] == str(tmp_path / "skasim_casa_ft.py")


def test_merge_model_data_into_data_adds_model_column(monkeypatch, tmp_path):
    """The final delivered DATA column includes image MODEL_DATA."""
    data = np.array([[[1 + 0j, 2 + 0j]]])
    model = np.array([[[3 + 0j, 4 + 0j]]])
    written = {}

    class FakeTable:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def colnames(self):
            return ["DATA", "MODEL_DATA"]

        def nrows(self):
            return 1

        def getcell(self, name, row):
            return {"DATA": data, "MODEL_DATA": model}[name][row]

        def getcol(self, name, startrow=0, nrow=-1):
            column = {"DATA": data, "MODEL_DATA": model}[name]
            return column[startrow : startrow + nrow].copy()

        def putcol(self, name, value, startrow=0, nrow=-1):
            written[name] = value

    casacore_module = ModuleType("casacore")
    tables_module = ModuleType("casacore.tables")
    tables_module.table = FakeTable
    monkeypatch.setitem(sys.modules, "casacore", casacore_module)
    monkeypatch.setitem(sys.modules, "casacore.tables", tables_module)

    merge_model_data_into_data(tmp_path / "visibilities.MS")

    assert np.array_equal(written["DATA"], data + model)


# ---------------------------------------------------------------------------
# static Stokes maps validation and injection
# ---------------------------------------------------------------------------


def _write_static_stokes_image(tmp_path, shape=(16, 16), bunit="Jy/pixel", ndim=2):
    """Create a minimal valid 2D FITS image for static_stokes_maps tests."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 2.0]
    wcs.wcs.crpix = [shape[1] / 2.0, shape[0] / 2.0]
    wcs.wcs.cdelt = [-0.001, 0.001]

    header = wcs.to_header()
    header["BUNIT"] = bunit

    data = np.ones(shape, dtype=np.float32)
    if ndim == 3:
        data = data[np.newaxis, :, :]

    path = tmp_path / "stokes_i.fits"
    fits.writeto(path, data, header, overwrite=True)
    return path


def test_validate_static_stokes_maps_accepts_2d_jy_per_pixel(tmp_path):
    """A 2D Jy/pixel image passes validation."""
    image_path = _write_static_stokes_image(tmp_path, shape=(64, 64))
    obs = ObsConfig(
        frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8, observation_time_s=1
    )
    img = ImgConfig(pixels=64, imager="wsclean")
    entry = StaticStokesMapsModelEntry(
        type="static_stokes_maps", stokes_i=str(image_path)
    )

    report = validate_static_stokes_maps(entry, obs, img)

    assert report["stokes_i"] == str(image_path)
    assert report["spatial_shape"] == [64, 64]
    assert report["unit"] == "Jy/pixel"
    assert report["n_channels"] == 8
    assert report["frequency_mhz"] == 700.0


def test_validate_static_stokes_maps_rejects_3d_image(tmp_path):
    """A 3D cube is rejected as stokes_i input."""
    image_path = _write_static_stokes_image(tmp_path, shape=(64, 64), ndim=3)
    obs = ObsConfig(
        frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8, observation_time_s=1
    )
    img = ImgConfig(pixels=64, imager="wsclean")
    entry = StaticStokesMapsModelEntry(
        type="static_stokes_maps", stokes_i=str(image_path)
    )

    with pytest.raises(ValueError, match="must be a 2D spatial image"):
        validate_static_stokes_maps(entry, obs, img)


def test_validate_static_stokes_maps_rejects_jy_per_beam(tmp_path):
    """Jy/beam units are rejected for stokes_i."""
    image_path = _write_static_stokes_image(tmp_path, shape=(64, 64), bunit="Jy/beam")
    obs = ObsConfig(
        frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8, observation_time_s=1
    )
    img = ImgConfig(pixels=64, imager="wsclean")
    entry = StaticStokesMapsModelEntry(
        type="static_stokes_maps", stokes_i=str(image_path)
    )

    with pytest.raises(ValueError, match="Jy/pixel-compatible"):
        validate_static_stokes_maps(entry, obs, img)


def test_inject_static_stokes_i_runs_wsclean_predict_and_merges(tmp_path, monkeypatch):
    """Static Stokes I injection writes per-channel models and runs predict."""
    image_path = _write_static_stokes_image(tmp_path, shape=(64, 64))
    cfg = SimConfig(
        output_dir=str(tmp_path / "run"),
        models=[
            StaticStokesMapsModelEntry(
                type="static_stokes_maps", stokes_i=str(image_path)
            )
        ],
        observation=ObsConfig(
            frequency_mhz=700.0,
            bandwidth_mhz=100.0,
            n_channels=4,
            observation_time_s=1,
        ),
        imaging=[ImgConfig(pixels=64, imager="wsclean")],
    )
    ctx = create_run_context(cfg)
    calls = []

    def fake_inject(ctx_arg, entry, index, visibility_path, img_config):
        calls.append(
            {
                "backend": "wsclean_predict",
                "index": index,
                "visibility_path": str(visibility_path),
            }
        )
        return {
            "model_paths": [
                ctx.work_dir
                / f"model_entry_{index + 1:02d}_static_stokes-0000-model.fits",
            ],
        }

    monkeypatch.setattr(
        "skasim.loaders.wsclean_predict.inject_static_stokes_i_with_wsclean_predict",
        fake_inject,
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.merge_model_data_into_data",
        lambda visibility_path: calls.append({"merged": str(visibility_path)}),
    )

    inject_image_models(ctx, visibility_path=Path("fake.ms"))

    assert len(calls) == 2
    assert calls[0]["backend"] == "wsclean_predict"
    assert calls[0]["index"] == 0
    assert any(c.get("merged") == "fake.ms" for c in calls)

    outputs = [o for o in ctx.manifest.outputs if o.kind == "sky_model"]
    assert len(outputs) == 1
    assert outputs[0].metadata["model_type"] == "static_stokes_maps"


def test_inject_static_stokes_i_generates_per_channel_fits(tmp_path, monkeypatch):
    """The real injection function writes one FITS per channel."""
    image_path = _write_static_stokes_image(tmp_path, shape=(64, 64))
    cfg = SimConfig(
        output_dir=str(tmp_path / "run"),
        models=[
            StaticStokesMapsModelEntry(
                type="static_stokes_maps", stokes_i=str(image_path)
            )
        ],
        observation=ObsConfig(
            frequency_mhz=700.0,
            bandwidth_mhz=100.0,
            n_channels=4,
            observation_time_s=1,
        ),
        imaging=[ImgConfig(pixels=64, imager="wsclean")],
    )
    ctx = create_run_context(cfg)

    run_calls = []
    monkeypatch.setattr(
        "skasim.loaders.wsclean_predict.run_wsclean_predict",
        lambda **kwargs: run_calls.append(kwargs),
    )

    report = inject_static_stokes_i_with_wsclean_predict(
        ctx, cfg.models[0], 0, Path("fake.ms"), cfg.imaging[0]
    )

    assert report["n_channels"] == 4
    assert len(report["model_paths"]) == 4
    for i, path_str in enumerate(report["model_paths"]):
        path = Path(path_str)
        assert path.exists()
        assert path.name == f"model_entry_01_static_stokes-{i:04d}-model.fits"
        with fits.open(path) as hdul:
            hdu = hdul[0]
            assert hdu.header["NAXIS"] == 3  # type: ignore[union-attr]
            assert hdu.header["NAXIS3"] == 1  # type: ignore[union-attr]
            assert hdu.header["CTYPE3"] == "FREQ"  # type: ignore[union-attr]
            expected_freq = (700.0 - 100.0 / 2.0 + (i + 0.5) * 100.0 / 4.0) * 1e6
            assert hdu.header["CRVAL3"] == pytest.approx(expected_freq)  # type: ignore[union-attr]
    assert len(run_calls) == 1
    assert run_calls[0]["n_channels"] == 4


def test_weblog_renders_static_stokes_map_preview(tmp_path, monkeypatch):
    """Static Stokes entries use the two-panel FITS preview generator."""
    image_path = _write_static_stokes_image(tmp_path, shape=(64, 64))
    cfg = SimConfig(
        output_dir=str(tmp_path / "run"),
        models=[
            StaticStokesMapsModelEntry(
                type="static_stokes_maps", stokes_i=str(image_path)
            )
        ],
        imaging=[ImgConfig(pixels=64, imager="wsclean")],
    )
    ctx = create_run_context(cfg)
    preview_calls = []

    def fake_preview(*args, **kwargs):
        preview_calls.append((args, kwargs))

    monkeypatch.setattr(
        "skasim.loaders.image_models.previews._write_image_model_two_panel_preview",
        fake_preview,
    )

    write_image_model_previews(ctx, 1.0 * u.deg)

    assert len(preview_calls) == 1
    assert preview_calls[0][0][0] == image_path
    assert preview_calls[0][0][2] == 1.0 * u.deg
    assert ctx.manifest.outputs[-1].role == "fits_model"


# ---------------------------------------------------------------------------
# spectral cube validation
# ---------------------------------------------------------------------------


def _write_spectral_cube(
    tmp_path, n_channels=8, pixels=64, freq_mhz=700.0, df_mhz=12.5
):
    """Create a minimal 3D FITS spectral cube with valid WCS."""
    cube_path = tmp_path / "cube.fits"
    data = np.zeros((n_channels, pixels, pixels), dtype=np.float32)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ"]
    wcs.wcs.crval = [10.0, 2.0, freq_mhz * 1e6]
    wcs.wcs.crpix = [pixels / 2.0 + 0.5, pixels / 2.0 + 0.5, n_channels / 2.0 + 0.5]
    wcs.wcs.cdelt = [-0.001, 0.001, df_mhz * 1e6]
    wcs.wcs.cunit = ["deg", "deg", "Hz"]
    header = wcs.to_header()
    header["BUNIT"] = "Jy/px"
    header["RESTFRQ"] = freq_mhz * 1e6
    fits.writeto(cube_path, data, header, overwrite=True)
    return cube_path


def test_read_fits_cube_info_returns_expected_shape(tmp_path):
    """read_fits_cube_info extracts spatial/spectral metadata from a 3D FITS cube."""
    cube_path = _write_spectral_cube(tmp_path, n_channels=16, pixels=32)
    info = read_fits_cube_info(cube_path)

    assert info.shape == (16, 32, 32)
    assert info.spatial_shape == (32, 32)
    assert info.n_channels == 16
    assert info.channel_width_hz == pytest.approx(12.5e6)
    assert info.unit == "jy/px"


def test_read_fits_cube_info_squeezes_4d_degenerate_stokes(tmp_path):
    """read_fits_cube_info accepts 4D cubes with a single degenerate Stokes axis."""
    cube_path = tmp_path / "cube_4d.fits"
    n_channels, pixels = 8, 64
    data = np.zeros((n_channels, 1, pixels, pixels), dtype=np.float32)
    # NAXIS order in the FITS file: NAXIS1=RA, NAXIS2=DEC, NAXIS3=STOKES, NAXIS4=FREQ
    header = fits.Header()
    header["NAXIS"] = 4
    header["NAXIS1"] = pixels
    header["NAXIS2"] = pixels
    header["NAXIS3"] = 1
    header["NAXIS4"] = n_channels
    header["CTYPE1"] = "RA---SIN"
    header["CRPIX1"] = pixels / 2.0 + 0.5
    header["CRVAL1"] = 10.0
    header["CDELT1"] = -0.001
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX2"] = pixels / 2.0 + 0.5
    header["CRVAL2"] = 2.0
    header["CDELT2"] = 0.001
    header["CUNIT2"] = "deg"
    header["CTYPE3"] = "STOKES"
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = 1.0
    header["CDELT3"] = 1.0
    header["CTYPE4"] = "FREQ"
    header["CRPIX4"] = n_channels / 2.0 + 0.5
    header["CRVAL4"] = 700e6
    header["CDELT4"] = 12.5e6
    header["CUNIT4"] = "Hz"
    header["BUNIT"] = "Jy/px"
    fits.writeto(cube_path, data, header, overwrite=True)

    info = read_fits_cube_info(cube_path)

    assert info.shape == (8, 64, 64)
    assert info.spatial_shape == (64, 64)
    assert info.n_channels == 8
    assert info.channel_width_hz == pytest.approx(12.5e6)
    assert info.unit == "jy/px"


def test_validate_spectral_cube_accepts_matching_config(tmp_path):
    """Matching cube/config passes validation and returns metadata."""
    cube_path = _write_spectral_cube(
        tmp_path, n_channels=8, pixels=64, freq_mhz=700.0, df_mhz=12.5
    )
    obs = ObsConfig(bandwidth_mhz=100.0, n_channels=8)
    img = ImgConfig(pixels=64)
    entry = SpectralCubeModelEntry(type="spectral_cube", cube=str(cube_path))

    report = validate_spectral_cube(entry, obs, img)

    assert report["cube"] == str(cube_path)
    assert report["n_channels"] == 8
    assert report["channel_width_hz"] == pytest.approx(12.5e6)
    assert report["unit"] == "jy/px"


def test_validate_spectral_cube_accepts_mismatched_grid_when_cube_is_inside_observation(
    tmp_path,
):
    """Cube spectral sampling may differ from observation; it only needs to lie within the observation band."""
    # cube: 4 channels x 12.5 MHz, centred at 700 MHz -> 668.75-731.25 MHz
    cube_path = _write_spectral_cube(
        tmp_path, n_channels=4, pixels=64, freq_mhz=700.0, df_mhz=12.5
    )
    # observation: 8 channels x 12.5 MHz, centred at 700 MHz -> 643.75-756.25 MHz
    obs = ObsConfig(frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8)
    img = ImgConfig(pixels=64)
    entry = SpectralCubeModelEntry(type="spectral_cube", cube=str(cube_path))

    report = validate_spectral_cube(entry, obs, img)

    assert report["cube"] == str(cube_path)
    assert report["n_channels"] == 4
    assert report["channel_width_hz"] == pytest.approx(12.5e6)
    assert report["unit"] == "jy/px"


def test_validate_spectral_cube_rejects_when_cube_exceeds_observation(tmp_path):
    """Cube frequency range must not extend beyond the observation band."""
    # cube: 16 channels x 12.5 MHz, centred at 900 MHz -> 806.25-993.75 MHz
    cube_path = _write_spectral_cube(
        tmp_path, n_channels=16, pixels=64, freq_mhz=900.0, df_mhz=12.5
    )
    # observation: 8 channels x 12.5 MHz, centred at 700 MHz -> 643.75-756.25 MHz
    obs = ObsConfig(frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8)
    img = ImgConfig(pixels=64)
    entry = SpectralCubeModelEntry(type="spectral_cube", cube=str(cube_path))

    with pytest.raises(ValueError, match="extends beyond the observation band"):
        validate_spectral_cube(entry, obs, img)


def test_validate_spectral_cube_accepts_narrow_cube_inside_wide_observation(tmp_path):
    """A narrow cube is valid inside a much wider observation band."""
    # cube: 8 channels x 0.925 kHz, centred at 1420.4 MHz (HI line cube)
    cube_path = _write_spectral_cube(
        tmp_path, n_channels=8, pixels=64, freq_mhz=1420.4, df_mhz=0.000925
    )
    # observation: 8 channels x 12.5 MHz, centred at 1420.4 MHz -> much wider
    obs = ObsConfig(frequency_mhz=1420.4, bandwidth_mhz=100.0, n_channels=8)
    img = ImgConfig(pixels=64)
    entry = SpectralCubeModelEntry(type="spectral_cube", cube=str(cube_path))

    report = validate_spectral_cube(entry, obs, img)

    assert report["cube"] == str(cube_path)
    assert report["n_channels"] == 8


def test_validate_spectral_cube_rejects_spatial_shape_mismatch(tmp_path):
    """Cube spatial dimensions must match imaging pixels."""
    cube_path = _write_spectral_cube(tmp_path, n_channels=8, pixels=128)
    obs = ObsConfig(frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8)
    img = ImgConfig(pixels=64)
    entry = SpectralCubeModelEntry(type="spectral_cube", cube=str(cube_path))

    with pytest.raises(ValueError, match="spatial dimensions"):
        validate_spectral_cube(entry, obs, img)


def test_validate_spectral_cube_rejects_bad_bunit(tmp_path):
    """Cube BUNIT must be Jy/pixel-compatible."""
    cube_path = _write_spectral_cube(tmp_path, n_channels=8, pixels=64)
    with fits.open(cube_path, mode="update") as hdul:
        hdul[0].header["BUNIT"] = "K"
    obs = ObsConfig(frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8)
    img = ImgConfig(pixels=64)
    entry = SpectralCubeModelEntry(type="spectral_cube", cube=str(cube_path))

    with pytest.raises(ValueError, match="Jy/pixel-compatible"):
        validate_spectral_cube(entry, obs, img)


def test_validate_spectral_cube_config_overrides_are_ignored(tmp_path):
    """Config spectral overrides are ignored; the cube header is authoritative."""
    cube_path = _write_spectral_cube(tmp_path, n_channels=8, pixels=64, freq_mhz=700.0)
    obs = ObsConfig(frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8)
    img = ImgConfig(pixels=64)
    entry = SpectralCubeModelEntry(
        type="spectral_cube",
        cube=str(cube_path),
        reference_frequency_hz=750e6,
        channel_width_hz=12.5e6,
        n_channels=8,
    )

    report = validate_spectral_cube(entry, obs, img)

    assert report["reference_frequency_hz"] == pytest.approx(700e6)
    assert report["channel_width_hz"] == pytest.approx(12.5e6)
    assert report["n_channels"] == 8


def test_prepare_spectral_cube_for_casa_writes_3d_fits_for_wsclean_predict(
    tmp_path, monkeypatch
):
    """The resampled spectral cube is now a 3D FITS ready for WSClean -predict."""
    cube_path = _write_spectral_cube(tmp_path, n_channels=8, pixels=64, freq_mhz=700.0)
    config = SimConfig(
        output_dir=str(tmp_path / "run"),
        models=[SpectralCubeModelEntry(type="spectral_cube", cube=str(cube_path))],
        observation=ObsConfig(
            frequency_mhz=700.0, bandwidth_mhz=100.0, n_channels=8, observation_time_s=1
        ),
        imaging=[ImgConfig(pixels=64, imager="wsclean")],
    )
    ctx = create_run_context(config)

    report = validate_spectral_cube(
        config.models[0], config.observation, config.imaging[0]
    )
    entry = config.models[0]
    assert entry.type == "spectral_cube"
    product = prepare_spectral_cube_for_casa(ctx, entry, 0, report)

    fits_out = product.model_paths[0]
    assert fits_out.exists()

    with fits.open(fits_out) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    assert data.ndim == 3
    assert data.shape == (8, 64, 64)
    assert header["NAXIS"] == 3
    assert header["NAXIS1"] == 64
    assert header["NAXIS2"] == 64
    assert header["NAXIS3"] == 8
    assert header["CTYPE1"] == "RA---SIN"
    assert header["CTYPE2"] == "DEC--SIN"
    assert header["CTYPE3"] == "FREQ"
    assert header["CUNIT3"] == "Hz"
    assert product.nterms == 1
    assert product.reffreq == "700000000.0Hz"
    assert np.array_equal(product.cube_data, data)
    assert product.freq_axis == 3


def test_resample_spectral_axis_conserves_line_flux():
    """A narrow line in a fine cube must survive resampling to coarse MS channels.

    Regression test for the ``np.interp`` dilution bug: when the input cube has
    many more (narrower) channels than the output MS, evaluating the input only at
    the output channel centres can drop the line almost entirely if the line does
    not coincide with a channel centre.  Band-limited integration must preserve
    the *integrated* flux.
    """
    # 128 fine channels of 0.1 MHz, centred at 700 MHz
    n_in = 128
    df_in = 0.1e6
    freq0 = 700.0e6 - (n_in - 1) * df_in / 2.0
    freqs_in = freq0 + np.arange(n_in) * df_in

    # a single 1 Jy/px line in the central pixel, Gaussian profile,
    # FWHM = 2.0 MHz, sampled on the fine grid
    data_in = np.zeros((n_in, 1, 1), dtype=np.float32)
    fwhm_hz = 2.0e6
    sigma_hz = fwhm_hz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    centre_idx = n_in // 2
    line_flux = 1.0  # Jy peak
    profile = np.exp(-0.5 * ((freqs_in - freqs_in[centre_idx]) / sigma_hz) ** 2)
    data_in[:, 0, 0] = profile * line_flux

    # 8 coarse output channels of 12.5 MHz, also centred at 700 MHz
    n_out = 8
    df_out = 12.5e6
    freq_out0 = 700.0e6 - (n_out - 1) * df_out / 2.0
    freqs_out = freq_out0 + np.arange(n_out) * df_out

    data_out = _resample_spectral_axis_to_ms_channels(
        freqs_in, data_in, freqs_out, df_out
    )

    # With np.interp, a 2 MHz line evaluated at the centre of a 12.5 MHz channel
    # gives ~1 Jy if centred, but ~0 Jy if it falls between channels.  Integration
    # should instead capture the total flux regardless of where the line sits.
    # Compare integrated flux using the same units: sum(data_i * channel_width_i).
    total_in = float(np.sum(data_in[:, 0, 0]) * df_in)
    total_out = float(np.sum(data_out[:, 0, 0]) * df_out)
    assert total_out == pytest.approx(total_in, rel=0.05), (
        f"integrated flux changed from {total_in:.3f} to {total_out:.3f}"
    )

    # All the flux should be contained in the output cube (the line is inside
    # the MS band).  We don't pin the peak to a single channel because a narrow
    # line centred between two coarse channels is split between them.
    assert data_out.max() > 0.0, "resampled cube is empty"


def test_resample_spectral_axis_beats_naive_interp_for_offcentre_line():
    """An off-centre narrow line is invisible to point sampling but not to integration."""
    n_in = 128
    df_in = 0.1e6
    freq0 = 700.0e6 - (n_in - 1) * df_in / 2.0
    freqs_in = freq0 + np.arange(n_in) * df_in

    # 8 coarse output channels of 12.5 MHz, centred at 700 MHz
    n_out = 8
    df_out = 12.5e6
    freq_out0 = 700.0e6 - (n_out - 1) * df_out / 2.0
    freqs_out = freq_out0 + np.arange(n_out) * df_out
    # place the line a quarter of the way into channel 3, away from any channel centre
    line_centre = freqs_out[3] + 0.25 * df_out

    data_in = np.zeros((n_in, 1, 1), dtype=np.float64)
    # line is much narrower than an output channel so it falls between the
    # coarse channel centres used by np.interp
    fwhm_hz = 2.0e6
    sigma_hz = fwhm_hz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    profile = np.exp(-0.5 * ((freqs_in - line_centre) / sigma_hz) ** 2)
    data_in[:, 0, 0] = 1.0 * profile

    data_out = _resample_spectral_axis_to_ms_channels(
        freqs_in, data_in.astype(np.float32), freqs_out, df_out
    )

    total_in = float(np.sum(data_in[:, 0, 0]) * df_in)
    total_out = float(np.sum(data_out[:, 0, 0]) * df_out)
    assert total_out == pytest.approx(total_in, rel=0.05)

    # np.interp at output channel centres would see almost nothing because the
    # line sits away from the centres; integration still captures it.
    naive_interp = np.interp(freqs_out, freqs_in, data_in[:, 0, 0])
    # if the naive interpolation is essentially zero, the resampler must be
    # strictly better; otherwise require it to capture much more flux.
    if float(np.max(np.abs(naive_interp))) < 1e-6:
        assert data_out.max() > 0.0, "resampler lost an off-centre line entirely"
    else:
        total_interp = float(np.sum(naive_interp) * df_out)
        assert total_out > 10 * total_interp, (
            f"resampler {total_out:.3f} not better than interp {total_interp:.3f}"
        )


# CASA pixel maps use coordinate-system order, not NumPy array order.
_COORDS_FREQ_THEN_STOKES = {
    "direction0": {},
    "pixelmap0": [0, 1],
    "stokes1": {"stokes": ["I"]},
    "pixelmap1": [3],
    "spectral2": {},
    "pixelmap2": [2],
}
_COORDS_STOKES_THEN_FREQ = {
    "direction0": {},
    "pixelmap0": [0, 1],
    "stokes1": {"stokes": ["I"]},
    "pixelmap1": [2],
    "spectral2": {},
    "pixelmap2": [3],
}


def _fake_casacore_with_coords(coords):
    class FakeTable:
        def __init__(self, path, readonly=True, ack=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keywordnames(self):
            return [] if coords is None else ["coords"]

        def getkeyword(self, name):
            assert name == "coords"
            return coords

    return FakeTable


@pytest.mark.parametrize(
    "coords, expected_axis",
    [(_COORDS_FREQ_THEN_STOKES, 3), (_COORDS_STOKES_THEN_FREQ, 4)],
)
@pytest.mark.parametrize("in_process", [False, True])
def test_set_spectral_coordinate_preserves_stokes_axis(
    monkeypatch, tmp_path, coords, expected_axis, in_process
):
    """Single-channel Taylor images must update frequency without changing Stokes."""
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casacore",
        lambda: _fake_casacore_with_coords(coords),
    )
    image_path = tmp_path / "model.image"
    if in_process:
        casatasks = ModuleType("casatasks")
        casatasks.imhead = MagicMock()
        monkeypatch.setitem(sys.modules, "casatasks", casatasks)

        _set_crval4_via_script(tmp_path, [image_path], 1.31e9)

        casatasks.imhead.assert_called_once_with(
            imagename=str(image_path),
            mode="put",
            hdkey=f"crval{expected_axis}",
            hdvalue="1310000000.0Hz",
        )
    else:
        monkeypatch.setattr(
            "skasim.loaders.image_models.casa_interop.require_casa_executable",
            lambda: Path("casa"),
        )
        scripts = []
        monkeypatch.setattr(
            "skasim.loaders.image_models.casa_interop.run_casa_script",
            lambda executable, script_path, lines: scripts.append(lines),
        )

        run_casa_set_spectral_coordinate(tmp_path, [image_path], 1.31e9)

        imhead_lines = [line for line in scripts[0] if line.startswith("imhead(")]
        assert imhead_lines == [
            f"imhead(imagename={str(image_path)!r}, mode='put', "
            f"hdkey='crval{expected_axis}', hdvalue='1310000000.0Hz')"
        ]


@pytest.mark.parametrize("returncode", [0, 1])
def test_run_casa_script_raises_on_severe_log_lines(monkeypatch, tmp_path, returncode):
    """CASA ft can report failure but exit zero with MODEL_DATA left at unity."""
    script_path = tmp_path / "skasim_casa_ft.py"

    def fake_run(command, **kwargs):
        kwargs["stdout"].write(
            "2026-08-19 09:42:43\tINFO\tft::casa\tTask ft complete.\n"
            "2026-08-19 09:42:43\tSEVERE\tft::imager::ft()\tException: Failed AlwaysAssert nStokesOut\n"
        )
        kwargs["stdout"].flush()
        return MagicMock(returncode=returncode)

    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.subprocess.run", fake_run
    )
    with pytest.raises(RuntimeError, match="nStokesOut") as error:
        run_casa_script(Path("casa"), script_path, ["ft()"])
    assert str(script_path) in str(error.value)
    assert "SEVERE" in str(error.value)


def test_run_casa_script_accepts_clean_log(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        kwargs["stdout"].write(
            "2026-06-24 15:23:10\tINFO\tft::casa\tTask ft complete.\n"
        )
        return MagicMock(returncode=0)

    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.subprocess.run", fake_run
    )
    run_casa_script(Path("casa"), tmp_path / "skasim_casa_ft.py", ["ft()"])


@pytest.mark.parametrize(
    "coords, expected",
    [
        (_COORDS_FREQ_THEN_STOKES, 3),
        (_COORDS_STOKES_THEN_FREQ, 4),
        ({"direction0": {}, "pixelmap0": [0, 1]}, None),
        (None, None),
        ({"spectral2": {}, "pixelmap2": [-1]}, None),
        ({"spectral2": {}}, None),
        ({"spectral2": {}, "pixelmap2": [2, 3]}, None),
    ],
)
def test_spectral_axis_number_reads_coordinate_mapping(
    monkeypatch, tmp_path, coords, expected
):
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casacore",
        lambda: _fake_casacore_with_coords(coords),
    )
    assert _spectral_axis_number(tmp_path / "model.image") == expected


def test_set_spectral_coordinate_skips_2d_image(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casacore",
        lambda: _fake_casacore_with_coords({"direction0": {}, "pixelmap0": [0, 1]}),
    )
    casatasks = ModuleType("casatasks")
    casatasks.imhead = MagicMock()
    monkeypatch.setitem(sys.modules, "casatasks", casatasks)
    batch = MagicMock()
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.run_casa_set_spectral_coordinate",
        batch,
    )

    _set_crval4_via_script(tmp_path, [tmp_path / "model.image"], 1.31e9)

    casatasks.imhead.assert_not_called()
    batch.assert_not_called()


def test_set_spectral_coordinate_fallback_handles_mixed_axes(monkeypatch, tmp_path):
    """Batch fallback selects an axis per image and skips spectrally flat inputs."""
    image_paths = [
        tmp_path / name for name in ("freq3.image", "freq4.image", "flat.image")
    ]
    coords_by_path = dict(
        zip(image_paths, [_COORDS_FREQ_THEN_STOKES, _COORDS_STOKES_THEN_FREQ, None])
    )
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casacore",
        lambda: (
            lambda path, **kwargs: _fake_casacore_with_coords(
                coords_by_path[Path(path)]
            )(path, **kwargs)
        ),
    )
    casatasks = ModuleType("casatasks")
    casatasks.imhead = MagicMock(
        side_effect=RuntimeError("in-process CASA unavailable")
    )
    monkeypatch.setitem(sys.modules, "casatasks", casatasks)
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.require_casa_executable",
        lambda: Path("casa"),
    )
    scripts = []
    monkeypatch.setattr(
        "skasim.loaders.image_models.casa_interop.run_casa_script",
        lambda executable, script_path, lines: scripts.append(lines),
    )

    _set_crval4_via_script(tmp_path, image_paths, 1.31e9)

    imhead_lines = [line for line in scripts[0] if line.startswith("imhead(")]
    assert len(imhead_lines) == 2
    for line, path, axis in zip(imhead_lines, image_paths, (3, 4)):
        assert f"imagename={str(path)!r}" in line
        assert f"hdkey='crval{axis}'" in line
        assert "hdvalue='1310000000.0Hz'" in line
