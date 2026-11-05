"""tests/test_primary_beam.py"""

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from skasim.primary_beam import (
    GaussianStationBeam,
    angular_separation_deg,
    simulation_beam,
)


def _oskar_field_pattern(theta_deg, field_fwhm_deg, ref_hz, freq_hz):
    """Independent transcription of oskar_station_beam_gaussian."""
    fwhm_rad = math.radians(field_fwhm_deg * ref_hz / freq_hz)
    std = math.sin(fwhm_rad) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    l2m2 = math.sin(math.radians(theta_deg)) ** 2
    return math.exp(-l2m2 / (2.0 * std**2))


def test_simulation_beam_oskar_params_use_fov_as_field_fwhm():
    beam = simulation_beam(fov_sim_deg=1.2, ref_freq_hz=1.4e9)

    assert beam.oskar_params() == {
        "station_type": "Gaussian beam",
        "gauss_beam_fwhm_deg": 1.2,
        "gauss_ref_freq_hz": 1.4e9,
    }


@pytest.mark.parametrize("freq_hz", [0.9e9, 1.4e9, 2.1e9])
@pytest.mark.parametrize("theta_deg", [0.0, 0.1, 0.4, 0.9, 2.0])
def test_power_response_is_square_of_oskar_field_pattern(theta_deg, freq_hz):
    """Visibilities see E_p * conj(E_q) = E^2 for identical stations."""
    beam = GaussianStationBeam(field_fwhm_deg=1.5, ref_freq_hz=1.4e9)

    expected = _oskar_field_pattern(theta_deg, 1.5, 1.4e9, freq_hz) ** 2

    assert beam.power_response(theta_deg, freq_hz) == pytest.approx(expected)


def test_power_response_is_one_at_phase_centre():
    beam = GaussianStationBeam(field_fwhm_deg=0.5, ref_freq_hz=1e9)

    assert beam.power_response(0.0, 1.3e9) == pytest.approx(1.0)


def test_power_fwhm_is_field_fwhm_over_root_two():
    beam = GaussianStationBeam(field_fwhm_deg=1.0, ref_freq_hz=1.4e9)

    fwhm = beam.power_fwhm_deg(1.4e9)

    assert fwhm == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-4)
    assert beam.power_response(fwhm / 2.0, 1.4e9) == pytest.approx(0.5)


def test_power_fwhm_scales_inversely_with_frequency():
    beam = GaussianStationBeam(field_fwhm_deg=0.4, ref_freq_hz=1.0e9)

    ratio = beam.power_fwhm_deg(1.0e9) / beam.power_fwhm_deg(2.0e9)

    assert ratio == pytest.approx(2.0, rel=1e-4)


def test_power_response_vectorises_over_arrays():
    beam = GaussianStationBeam(field_fwhm_deg=1.0, ref_freq_hz=1e9)
    theta = np.linspace(0.0, 1.0, 11).reshape(11, 1)

    response = beam.power_response(theta, 1e9)

    assert response.shape == (11, 1)
    assert np.all(np.diff(response[:, 0]) < 0)


def test_beam_provenance_record():
    record = GaussianStationBeam(field_fwhm_deg=1.0, ref_freq_hz=1e9).as_dict()

    assert record["model"] == "oskar_gaussian_station_beam"
    assert record["field_fwhm_deg"] == 1.0
    assert record["power_fwhm_deg_at_ref"] == pytest.approx(1.0 / math.sqrt(2.0), 1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"field_fwhm_deg": 0.0, "ref_freq_hz": 1e9},
        {"field_fwhm_deg": 1.0, "ref_freq_hz": -1.0},
    ],
)
def test_beam_rejects_non_positive_parameters(kwargs):
    with pytest.raises(ValueError):
        GaussianStationBeam(**kwargs)


def test_angular_separation_matches_astropy():
    rng = np.random.default_rng(1)
    ra = rng.uniform(0, 360, 50)
    dec = rng.uniform(-89, 89, 50)
    centre = SkyCoord(232.85 * u.deg, 35.56 * u.deg)

    sep = angular_separation_deg(ra, dec, centre.ra.deg, centre.dec.deg)
    expected = SkyCoord(ra * u.deg, dec * u.deg).separation(centre).deg

    np.testing.assert_allclose(sep, expected, atol=1e-10)
