"""Unit tests for the WSClean -predict spectral-cube injection path."""

import numpy as np
import pytest
from astropy.io import fits

from skasim.config import ImgConfig
from skasim.loaders.wsclean_predict import (
    build_wsclean_predict_argv,
    merge_model_data_into_data,
    write_per_channel_model_fits,
)


def _build_spectral_header(n_channels: int = 4, pixels: int = 8) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = pixels
    header["NAXIS2"] = pixels
    header["NAXIS3"] = n_channels
    header["CTYPE1"] = "RA---SIN"
    header["CRPIX1"] = pixels / 2.0
    header["CRVAL1"] = 10.0
    header["CDELT1"] = -0.05
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX2"] = pixels / 2.0
    header["CRVAL2"] = 20.0
    header["CDELT2"] = 0.05
    header["CUNIT2"] = "deg"
    header["CTYPE3"] = "FREQ"
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = 650e6
    header["CDELT3"] = 12.5e6
    header["CUNIT3"] = "Hz"
    header["BUNIT"] = "Jy/pixel"
    return header


def test_write_per_channel_model_fits_creates_wsclean_named_planes(tmp_path):
    """Per-channel FITS follow the <prefix>-NNNN-model.fits convention."""
    n_channels, pixels = 4, 8
    cube_data = np.ones((n_channels, pixels, pixels), dtype=np.float32)
    header = _build_spectral_header(n_channels=n_channels, pixels=pixels)

    paths = write_per_channel_model_fits(
        cube_data, header, tmp_path, "my-line", freq_axis=3
    )

    assert len(paths) == n_channels
    for i, path in enumerate(paths):
        assert path.name == f"my-line-{i:04d}-model.fits"
        assert path.exists()
        with fits.open(path) as hdul:
            assert hdul[0].data.shape == (1, pixels, pixels)
            assert hdul[0].header["CTYPE3"] == "FREQ"
            assert hdul[0].header["CRVAL3"] == pytest.approx(650e6 + i * 12.5e6)
            assert hdul[0].header["NAXIS3"] == 1
            assert hdul[0].header["BUNIT"] == "Jy/pixel"


def test_write_per_channel_model_fits_rejects_missing_freq_keywords(tmp_path):
    """A header without CRVAL3/CDELT3 is rejected early."""
    cube_data = np.zeros((2, 4, 4), dtype=np.float32)
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["NAXIS3"] = 2
    header["CTYPE3"] = "FREQ"

    with pytest.raises(ValueError, match="CRVAL3"):
        write_per_channel_model_fits(cube_data, header, tmp_path, "bad", freq_axis=3)


def test_build_wsclean_predict_argv_includes_predict_and_channels_out():
    """Argv contains -predict, -channels-out and the MS path."""
    img_config = ImgConfig(
        pixels=64,
        fov_deg=0.2,
        wsclean_command="wsclean",
    )
    argv = build_wsclean_predict_argv(
        "wsclean",
        tmp_path := __import__("pathlib").Path("/tmp/vis.ms"),
        img_config,
        "line",
        16,
    )

    assert argv[0] == "wsclean"
    assert "-predict" in argv
    assert "-channels-out" in argv
    assert argv[argv.index("-channels-out") + 1] == "16"
    assert "-name" in argv
    assert argv[argv.index("-name") + 1] == "line"
    assert argv[-1] == str(tmp_path)


def test_merge_model_data_into_data_adds_model_to_data(monkeypatch, tmp_path):
    """DATA becomes DATA + MODEL_DATA."""
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

    casacore_module = __import__("types").ModuleType("casacore")
    tables_module = __import__("types").ModuleType("casacore.tables")
    tables_module.table = FakeTable
    monkeypatch.setitem(__import__("sys").modules, "casacore", casacore_module)
    monkeypatch.setitem(__import__("sys").modules, "casacore.tables", tables_module)

    merge_model_data_into_data(tmp_path / "visibilities.MS")

    assert np.array_equal(written["DATA"], data + model)
