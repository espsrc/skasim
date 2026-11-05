"""Shared Gaussian primary-beam model used for every sky component.

OSKAR attenuates catalogue sources with its ``Gaussian beam`` station type.
Image models are predicted by WSClean, which applies no beam, so skasim
multiplies them by the same response before prediction. Both paths obtain the
beam from :class:`GaussianStationBeam` so they cannot drift apart.

OSKAR semantics (``oskar_station_beam_gaussian``, verified against the OSKAR
source):

* the FWHM parameter describes the station **field (voltage)** pattern
  ``E(l, m) = exp(-(l^2 + m^2) / (2 sigma^2))`` with
  ``sigma = sin(FWHM) / (2 sqrt(2 ln 2))``;
* the FWHM scales as ``ref_freq / freq``;
* the beam peaks at 1 in the beam direction (the phase centre here).

Visibilities see the product of two identical station beams, so a source at
angular distance ``theta`` from the phase centre is attenuated by the power
pattern ``E^2 = exp(-sin^2(theta) / sigma^2)``. Its FWHM is close to
``FWHM / sqrt(2)``.

Karabo passes ``gauss_beam_fwhm_deg`` through unchanged as the field FWHM
unless ``enable_power_pattern=True``; skasim does not set that flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class GaussianStationBeam:
    """OSKAR-equivalent Gaussian station beam centred on the phase centre."""

    field_fwhm_deg: float
    ref_freq_hz: float

    def __post_init__(self) -> None:
        if not self.field_fwhm_deg > 0:
            raise ValueError(f"field_fwhm_deg must be > 0, got {self.field_fwhm_deg}")
        if not self.ref_freq_hz > 0:
            raise ValueError(f"ref_freq_hz must be > 0, got {self.ref_freq_hz}")

    def oskar_params(self) -> dict:
        """Return the Karabo ``InterferometerSimulation`` beam parameters."""
        return {
            "station_type": "Gaussian beam",
            "gauss_beam_fwhm_deg": self.field_fwhm_deg,
            "gauss_ref_freq_hz": self.ref_freq_hz,
        }

    def _sigma(self, freq_hz: float) -> float:
        field_fwhm_rad = math.radians(self.field_fwhm_deg * self.ref_freq_hz / freq_hz)
        return math.sin(field_fwhm_rad) * _FWHM_TO_SIGMA

    def power_response(self, separation_deg: ArrayLike, freq_hz: float) -> np.ndarray:
        """Return the visibility (power) attenuation at ``separation_deg``."""
        if not freq_hz > 0:
            raise ValueError(f"freq_hz must be > 0, got {freq_hz}")
        sin_theta = np.sin(np.radians(np.asarray(separation_deg, dtype=float)))
        return np.exp(-(sin_theta**2) / self._sigma(freq_hz) ** 2)

    def power_fwhm_deg(self, freq_hz: float) -> float:
        """Return the power-pattern FWHM at ``freq_hz``."""
        # E^2 = 1/2  ->  sin(theta_half) = sigma * sqrt(ln 2)
        half = math.asin(self._sigma(freq_hz) * math.sqrt(math.log(2.0)))
        return 2.0 * math.degrees(half)

    def as_dict(self) -> dict:
        """Return a JSON-serializable provenance record."""
        return {
            "model": "oskar_gaussian_station_beam",
            "field_fwhm_deg": self.field_fwhm_deg,
            "ref_freq_hz": self.ref_freq_hz,
            "power_fwhm_deg_at_ref": self.power_fwhm_deg(self.ref_freq_hz),
            "centre": "phase_centre",
        }


def simulation_beam(fov_sim_deg: float, ref_freq_hz: float) -> GaussianStationBeam:
    """Return the run beam: field FWHM equal to the simulation field of view."""
    return GaussianStationBeam(field_fwhm_deg=fov_sim_deg, ref_freq_hz=ref_freq_hz)


def angular_separation_deg(
    ra_deg: ArrayLike, dec_deg: ArrayLike, ra0_deg: ArrayLike, dec0_deg: ArrayLike
) -> np.ndarray:
    """Great-circle separation (Vincenty formula) in degrees; broadcasts."""
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    ra0 = np.radians(np.asarray(ra0_deg, dtype=float))
    dec0 = np.radians(np.asarray(dec0_deg, dtype=float))
    dra = ra - ra0
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    sin_dec0, cos_dec0 = np.sin(dec0), np.cos(dec0)
    num = np.hypot(
        cos_dec * np.sin(dra),
        cos_dec0 * sin_dec - sin_dec0 * cos_dec * np.cos(dra),
    )
    den = sin_dec0 * sin_dec + cos_dec0 * cos_dec * np.cos(dra)
    return np.degrees(np.arctan2(num, den))


__all__ = [
    "GaussianStationBeam",
    "angular_separation_deg",
    "simulation_beam",
]
