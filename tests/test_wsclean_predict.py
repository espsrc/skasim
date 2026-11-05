"""Unit tests for the WSClean -predict spectral-cube injection path."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from skasim.config import ImgConfig
from skasim.loaders.wsclean_predict import (
    build_wsclean_predict_argv,
    format_wsclean_shift,
    merge_model_data_into_data,
    prepare_predict_grid,
    write_per_channel_model_fits,
    wsclean_predict_placement,
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


# ---------------------------------------------------------------------------
# multimodal input: WCS placement through -shift
# ---------------------------------------------------------------------------

PHASE_CENTRE = SkyCoord(232.855667, 35.561167, unit="deg")  # 15h31m25s +35d33m40s


def _sin_header(n, cdelt_deg, crval, crpix=None, projection="SIN", cdelt2=None):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    wcs.wcs.crval = list(crval)
    wcs.wcs.crpix = list(crpix) if crpix is not None else [n / 2 + 1, n / 2 + 1]
    wcs.wcs.cdelt = [-cdelt_deg, cdelt2 if cdelt2 is not None else cdelt_deg]
    return wcs.to_header()


def _wsclean_parse(text, hours):
    """Port of aocommon RaDecCoord::ParseRA/ParseDec ('h'/'d' formats)."""
    sign = text.lstrip().startswith("-")
    sep = "h" if hours else "d"
    head, rest = text.split(sep)
    mins, secs = rest.split("m")
    assert secs.endswith("s")
    whole, minutes, seconds = abs(int(head)), int(mins), float(secs[:-1])
    value = whole + minutes / 60 + seconds / 3600
    value = -value if sign else value
    return value * 15.0 if hours else value


def test_centred_image_tangent_at_phase_centre_needs_no_shift():
    header = _sin_header(64, 1e-3, (PHASE_CENTRE.ra.deg, PHASE_CENTRE.dec.deg))

    placement = wsclean_predict_placement(header, (64, 64), PHASE_CENTRE)

    assert placement["direct_placement_ok"] is True
    assert placement["shift_applied"] is False
    # geometric centre is half a pixel (diagonally) from WSClean's centre pixel
    assert placement["offset_arcsec"] == pytest.approx(0.5 * np.sqrt(2) * 3.6, 1e-3)
    assert placement["placement_error_pix"] == pytest.approx(0.0, abs=1e-6)


def test_off_centre_crpix_on_phase_centre_tangent_is_exact():
    """Same tangent plane, image centre elsewhere: -shift places it exactly."""
    header = _sin_header(
        64, 1e-3, (PHASE_CENTRE.ra.deg, PHASE_CENTRE.dec.deg), crpix=(-20.0, 90.0)
    )

    placement = wsclean_predict_placement(header, (64, 64), PHASE_CENTRE)

    centre = WCS(header).pixel_to_world(32, 32)
    assert placement["direct_placement_ok"] is True
    assert placement["shift_applied"] is True
    assert placement["shift_ra_deg"] == pytest.approx(centre.ra.deg, abs=1e-9)
    assert placement["shift_dec_deg"] == pytest.approx(centre.dec.deg, abs=1e-9)
    assert placement["placement_error_pix"] < 1e-6


def test_small_offset_with_own_tangent_point_is_placed_directly():
    target = PHASE_CENTRE.spherical_offsets_by(30 * u.arcsec, -20 * u.arcsec)
    header = _sin_header(64, 1e-3, (target.ra.deg, target.dec.deg))

    placement = wsclean_predict_placement(header, (64, 64), PHASE_CENTRE)

    assert placement["direct_placement_ok"] is True
    assert placement["shift_applied"] is True
    assert 0.0 < placement["placement_error_pix"] < 0.1


@pytest.mark.parametrize(
    "kwargs, shape, reason",
    [
        ({"projection": "TAN"}, (64, 64), "not RA/Dec SIN"),
        ({"cdelt2": 2e-3}, (64, 64), "pixels are not square"),
        ({}, (64, 32), "image is not square"),
    ],
)
def test_incompatible_headers_need_resampling(kwargs, shape, reason):
    header = _sin_header(
        64, 1e-3, (PHASE_CENTRE.ra.deg, PHASE_CENTRE.dec.deg), **kwargs
    )

    placement = wsclean_predict_placement(header, shape, PHASE_CENTRE)

    assert placement["direct_placement_ok"] is False
    assert reason in placement["reason"]


@pytest.mark.parametrize("mutation", ["mirrored", "far_own_tangent"])
def test_misplaced_grids_need_resampling(mutation):
    if mutation == "mirrored":
        header = _sin_header(64, 1e-3, (PHASE_CENTRE.ra.deg, PHASE_CENTRE.dec.deg))
        header["CDELT1"] = abs(header["CDELT1"])
        shape = (64, 64)
    else:
        target = PHASE_CENTRE.spherical_offsets_by(1.0 * u.deg, 0.5 * u.deg)
        header = _sin_header(1024, 2e-4, (target.ra.deg, target.dec.deg))
        shape = (1024, 1024)

    placement = wsclean_predict_placement(header, shape, PHASE_CENTRE)

    assert placement["direct_placement_ok"] is False
    assert "placement error" in placement["reason"]


def _point_source_image(header, shape, source):
    data = np.zeros(shape)
    x, y = WCS(header).celestial.world_to_pixel(source)
    data[int(round(float(y))), int(round(float(x)))] = 2.5
    snapped = WCS(header).celestial.pixel_to_world(
        int(round(float(x))), int(round(float(y)))
    )
    return data, snapped


def _flux_centroid(data, header):
    yy, xx = np.indices(data.shape)
    total = data.sum()
    xc = float((data * xx).sum() / total)
    yc = float((data * yy).sum() / total)
    return WCS(header).celestial.pixel_to_world(xc, yc)


@pytest.mark.parametrize(
    "projection, offset_deg, cdelt2",
    [
        ("TAN", (0.02, 0.01), None),
        ("SIN", (1.0, 0.5), None),
        ("SIN", (0.1, 0.0), 1.5e-3),
    ],
)
def test_prepare_predict_grid_regrids_onto_wsclean_grid(projection, offset_deg, cdelt2):
    """Regridded models conserve flux and land where their WCS says."""
    target = PHASE_CENTRE.spherical_offsets_by(
        offset_deg[0] * u.deg, offset_deg[1] * u.deg
    )
    header = _sin_header(
        96,
        1e-3,
        (target.ra.deg, target.dec.deg),
        projection=projection,
        cdelt2=cdelt2,
    )
    source = target.spherical_offsets_by(20 * u.arcsec, -15 * u.arcsec)
    data, snapped = _point_source_image(header, (96, 96), source)
    tt1 = 0.5 * data

    (out, out_tt1), out_header, placement = prepare_predict_grid(
        [data, tt1], header, PHASE_CENTRE
    )

    assert placement["resampled"] is True
    assert placement["resample_method"] == "reproject_exact"
    n = placement["n_pixels"]
    assert out.shape == (n, n) and n % 2 == 0
    assert out.sum() == pytest.approx(2.5, rel=1e-3)
    assert out_tt1.sum() == pytest.approx(1.25, rel=1e-3)
    # The output grid is exactly what WSClean assumes for this -shift.
    out_wcs = WCS(out_header).celestial
    assert list(out_wcs.wcs.ctype) == ["RA---SIN", "DEC--SIN"]
    assert out_wcs.wcs.crval[0] == pytest.approx(PHASE_CENTRE.ra.deg)
    assert out_wcs.wcs.crval[1] == pytest.approx(PHASE_CENTRE.dec.deg)
    shift = SkyCoord(placement["shift_ra_deg"], placement["shift_dec_deg"], unit="deg")
    assert out_wcs.pixel_to_world(n / 2, n / 2).separation(shift).arcsec < 1e-6
    redone = wsclean_predict_placement(out_header, out.shape, PHASE_CENTRE)
    assert redone["direct_placement_ok"] is True
    # Flux stays at the source's sky position (within 0.1 output pixel).
    pixel_arcsec = placement["pixel_size_arcsec"]
    moved = _flux_centroid(out, out_header).separation(snapped).arcsec
    assert moved < 0.1 * pixel_arcsec


def test_prepare_predict_grid_conserves_flux_of_filled_image():
    """Edge pixels partly covered by the input must not inflate the flux."""
    target = PHASE_CENTRE.spherical_offsets_by(10 * u.arcmin, 5 * u.arcmin)
    header = _sin_header(128, 1 / 3600, (target.ra.deg, target.dec.deg))
    data = np.random.default_rng(3).random((128, 128))

    (out,), _, placement = prepare_predict_grid([data], header, PHASE_CENTRE)

    assert placement["resampled"] is True
    # reproject_exact treats pixel edges as great-circle arcs; the residual
    # against the SIN plane geometry is ~3e-5 here (edge clipping was 4e-3).
    assert out.sum() == pytest.approx(data.sum(), rel=1e-4)


def test_prepare_predict_grid_leaves_compatible_planes_untouched():
    header = _sin_header(64, 1e-3, (PHASE_CENTRE.ra.deg, PHASE_CENTRE.dec.deg))
    data = np.random.default_rng(0).random((64, 64))

    (out,), out_header, placement = prepare_predict_grid([data], header, PHASE_CENTRE)

    assert placement["resampled"] is False
    assert out is data
    assert out_header is header


@pytest.mark.parametrize(
    "ra, dec", [(232.855667, 35.561167), (0.001, -0.5), (359.9999, -89.5), (12.5, 0.0)]
)
def test_shift_strings_round_trip_through_wsclean_parser(ra, dec):
    ra_text, dec_text = format_wsclean_shift(SkyCoord(ra, dec, unit="deg"))

    assert _wsclean_parse(ra_text, hours=True) == pytest.approx(ra, abs=1e-8)
    assert _wsclean_parse(dec_text, hours=False) == pytest.approx(dec, abs=1e-8)


def test_build_predict_argv_includes_shift_before_ms():
    img = ImgConfig(pixels=64, fov_deg=0.064, imager="wsclean")

    argv = build_wsclean_predict_argv(
        "wsclean",
        Path("vis.ms"),
        img,
        "prefix",
        4,
        pixel_size_arcsec=3.6,
        n_pixels=64,
        shift=("15h31m25.36s", "35d33m40.2s"),
    )

    i = argv.index("-shift")
    assert argv[i + 1 : i + 3] == ["15h31m25.36s", "35d33m40.2s"]
    assert argv[-1] == "vis.ms"
