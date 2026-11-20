"""Check a multimodal run: compact components land where and as bright as expected.

For every compact component of a finished run (each image model's brightest
pixel, and catalogue sources with major axis <= ``--max-size-arcsec``), the
expected continuum-image peak is computed independently from the *input* files:

    S_expected = I(nu_ref) * (nu0 / nu_ref)^alpha * P(theta, nu0)

with ``P`` the shared Gaussian power beam recorded in the manifest
(``primary_beam_configured``) and ``theta`` the distance from the phase centre
(``phase_centre_resolved``). The script then finds the image peak near the
expected position and reports flux ratio and position error. It exits non-zero
when any component misses the tolerances.

Peaks are measured on the restored image of a WSClean block imaged in
``continuum`` mode when the run has one, otherwise on the dirty image. Dirty
images are not reliable for this: a source a few synthesized beams from a
brighter one picks up its dirty-beam sidelobes (in the VLA-C demos, ~5 mJy on
an 8 mJy source 28 arcsec from a 20 mJy one). ``--dirty`` forces the dirty
image.

For every ``spectral_cube`` entry, the line is checked on the run's line cube
(the WSClean ``cube_image``): the spectrum at the cube's brightest moment-0
position is continuum-subtracted with a straight-line fit to the line-free
channels (outside the ``line_channels`` recorded in the manifest, with a
two-channel guard), then integrated over the line channels. It is compared
with the expected beam-attenuated integrated line flux, computed per channel
from the input cube, and the flux-weighted line frequency is compared with the
expected one. Continuum peaks include the band-averaged line of any cube at the
same position.

Designed for ``configs/fits_model/vla_c_catalog_plus_two_images_quick.json``
and ``configs/fits_model/vla_c_catalog_image_hi_cube_quick.json`` (compact
sources well separated relative to the synthesized beam). Extended or blended
components are not meaningful peak-flux tests.

Usage:
    python scripts/check_multimodal_run.py demo_output/<run_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from skasim.primary_beam import GaussianStationBeam


def _milestone(manifest: dict, name: str) -> dict:
    for item in manifest.get("milestones", []):
        if item.get("name") == name:
            return item.get("details") or {}
    raise SystemExit(f"manifest has no '{name}' milestone; is this a multimodal run?")


def _continuum_image(
    run_dir: Path, manifest: dict, prefer_dirty: bool = False
) -> tuple:
    """Return ``(path, label)`` of the continuum image to measure peaks on.

    The restored image of a WSClean ``continuum`` block when present (CLEAN
    removes the dirty-beam sidelobes), else the OSKAR dirty image, else any
    dirty image. Per-channel WSClean outputs (``<prefix>-NNNN-*``) are never
    used.
    """
    outputs = [
        output
        for output in manifest.get("outputs", [])
        if str(output.get("path", "")).endswith(".fits")
        and not re.search(r"-\d{4,}-[^/]*$", str(output.get("path", "")))
    ]
    if not prefer_dirty:
        modes = {}
        for item in manifest.get("milestones", []):
            if item.get("name") == "imaging_spectral_modes":
                modes = (item.get("details") or {}).get("modes") or {}
        for output in outputs:
            tag = (output.get("metadata") or {}).get("tag")
            if (
                output.get("imager") == "wsclean"
                and output.get("role") == "image"
                and modes.get(tag) == "continuum"
            ):
                return run_dir / output["path"], f"WSClean restored, block '{tag}'"
    dirty = [output for output in outputs if output.get("role") == "dirty"]
    dirty.sort(key=lambda output: output.get("imager") != "oskar-dirty")
    if dirty:
        chosen = dirty[0]
        tag = (chosen.get("metadata") or {}).get("tag")
        return run_dir / chosen["path"], f"dirty {chosen.get('imager')}, block '{tag}'"
    raise SystemExit("no continuum FITS image recorded in the manifest")


def _line_cube_image(run_dir: Path, manifest: dict) -> Optional[Path]:
    for output in manifest.get("outputs", []):
        if output.get("role") == "cube_image" and str(output.get("path", "")).endswith(
            ".fits"
        ):
            return run_dir / output["path"]
    return None


def _cube_lines(
    models: List[dict], manifest: dict, beam: GaussianStationBeam, centre: SkyCoord
) -> List[dict]:
    """Expected beam-attenuated line of each spectral_cube entry (input cube)."""
    injected = {
        item["details"].get("model_entry_index"): item["details"]
        for item in manifest.get("milestones", [])
        if item.get("name") == "image_model_injected" and item.get("details")
    }
    lines = []
    image_index = -1
    for model in models:
        # model_entry_index counts every non-catalogue entry in models order
        if model.get("type") != "component_sky_model":
            image_index += 1
        if model.get("type") != "spectral_cube":
            continue
        path = Path(model["cube"])
        header = fits.getheader(path)
        cube = np.asarray(fits.getdata(path), dtype=float).squeeze()
        wcs = WCS(header)
        n_chan = cube.shape[0]
        freqs = wcs.spectral.pixel_to_world(np.arange(n_chan)).to_value(u.Hz)
        mom0 = cube.sum(axis=0)
        y, x = np.unravel_index(int(np.argmax(mom0)), mom0.shape)
        position = wcs.celestial.pixel_to_world(x, y)
        theta = float(position.separation(centre).deg)
        spectrum = cube.sum(axis=(1, 2))
        attenuated = np.array(
            [s * float(beam.power_response(theta, f)) for s, f in zip(spectrum, freqs)]
        )
        width_hz = abs(float(freqs[1] - freqs[0]))
        restfrq = header.get("RESTFRQ", header.get("RESTFREQ"))
        if restfrq:
            unit, dx = "Jy km/s", 299792.458 * width_hz / float(restfrq)
        else:
            unit, dx = "Jy MHz", width_hz / 1e6
        details = injected.get(image_index, {})
        lines.append(
            {
                "name": f"cube {image_index} ({path.name})",
                "position": position,
                "expected": float(attenuated.sum()) * dx,
                "expected_centre_hz": float(
                    (attenuated * freqs).sum() / attenuated.sum()
                ),
                "unit": unit,
                "dx": dx,
                "line_channels": details.get("line_channels"),
                "band_mean_jy": float(attenuated.sum()),
            }
        )
    return lines


def _measure_line(
    image: np.ndarray, wcs: WCS, freqs: np.ndarray, line: dict, guard: int = 2
) -> Optional[tuple]:
    """Continuum-subtracted integrated line at the source pixel."""
    x, y = (
        int(round(float(v))) for v in wcs.celestial.world_to_pixel(line["position"])
    )
    n_chan, ny, nx = image.shape
    if not (0 <= x < nx and 0 <= y < ny) or not line["line_channels"]:
        return None
    first, last = line["line_channels"]
    spectrum = image[:, y, x]
    channels = np.arange(n_chan)
    free = (channels < first - guard) | (channels > last + guard)
    slope, intercept = np.polyfit(channels[free], spectrum[free], 1)
    residual = spectrum - (slope * channels + intercept)
    window = slice(first, last + 1)
    integrated = float(residual[window].sum()) * line["dx"]
    weights = residual[window]
    centre_hz = float((weights * freqs[window]).sum() / weights.sum())
    return integrated, centre_hz


def _image_components(models: List[dict], nu0: float) -> List[dict]:
    components = []
    image_index = 0
    for model in models:
        if model.get("type") not in ("continuum_i_alpha", "static_stokes_maps"):
            continue
        path = Path(model["stokes_i"])
        data = np.asarray(fits.getdata(path), dtype=float).squeeze()
        header = fits.getheader(path)
        wcs = WCS(header).celestial
        y, x = np.unravel_index(int(np.nanargmax(data)), data.shape)
        flux = float(data[y, x])
        alpha = 0.0
        ref = nu0
        if model.get("type") == "continuum_i_alpha":
            alpha_map = np.asarray(fits.getdata(model["alpha"]), dtype=float).squeeze()
            alpha = float(alpha_map[y, x])
            ref = float(model["reference_frequency_hz"])
        components.append(
            {
                "name": f"image {image_index} ({path.name})",
                "position": wcs.pixel_to_world(x, y),
                "flux_nu0": flux * (nu0 / ref) ** alpha,
            }
        )
        image_index += 1
    return components


def _catalogue_components(
    models: List[dict], nu0: float, max_size_arcsec: float
) -> List[dict]:
    components = []
    for model in models:
        if model.get("type") != "component_sky_model" or not model.get("path"):
            continue
        path = Path(model["path"])
        if path.suffix.lower() != ".json":
            print(f"skipping catalogue {path}: only JSON catalogues are checked")
            continue
        for i, source in enumerate(json.loads(path.read_text())):
            if float(source.get("major_axis", 0.0)) > max_size_arcsec:
                continue
            ref = float(source.get("ref_freq") or 0.0)
            alpha = float(source.get("spec_index") or 0.0)
            scale = (nu0 / ref) ** alpha if ref > 0 else 1.0
            components.append(
                {
                    "name": f"catalogue #{i}",
                    "position": SkyCoord(source["ra"] * u.deg, source["dec"] * u.deg),
                    "flux_nu0": float(source["I"]) * scale,
                }
            )
    return components


def _measure(
    image: np.ndarray, wcs: WCS, position: SkyCoord, box: int
) -> Optional[tuple]:
    x, y = (float(v) for v in wcs.world_to_pixel(position))
    xi, yi = int(round(x)), int(round(y))
    ny, nx = image.shape
    x0, x1 = max(0, xi - box), min(nx, xi + box + 1)
    y0, y1 = max(0, yi - box), min(ny, yi + box + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    cut = image[y0:y1, x0:x1]
    py, px = np.unravel_index(int(np.nanargmax(cut)), cut.shape)
    peak = float(cut[py, px])
    offset_pix = float(np.hypot(px + x0 - x, py + y0 - y))
    return peak, offset_pix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--flux-tol", type=float, default=0.10)
    parser.add_argument("--pos-tol-pix", type=float, default=1.0)
    parser.add_argument("--max-size-arcsec", type=float, default=2.0)
    parser.add_argument("--search-box-pix", type=int, default=3)
    parser.add_argument(
        "--dirty",
        action="store_true",
        help="measure on the dirty image even when a cleaned one exists",
    )
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    beam_details = _milestone(manifest, "primary_beam_configured")
    beam = GaussianStationBeam(
        field_fwhm_deg=float(beam_details["field_fwhm_deg"]),
        ref_freq_hz=float(beam_details["ref_freq_hz"]),
    )
    nu0 = beam.ref_freq_hz
    centre_details = _milestone(manifest, "phase_centre_resolved")
    centre = SkyCoord(
        centre_details["ra_deg"] * u.deg, centre_details["dec_deg"] * u.deg
    )

    image_path, image_label = _continuum_image(args.run_dir, manifest, args.dirty)
    image = np.asarray(fits.getdata(image_path), dtype=float).squeeze()
    wcs = WCS(fits.getheader(image_path)).celestial

    models = manifest["config"]["models"]
    components = _catalogue_components(
        models, nu0, args.max_size_arcsec
    ) + _image_components(models, nu0)
    lines = _cube_lines(models, manifest, beam, centre)
    n_ms_channels = int(manifest["config"]["observation"].get("n_channels") or 1)
    if not components and not lines:
        print("no compact components to check")
        return 1

    print(f"image       : {image_path} ({image_label})")
    print(
        f"beam        : power FWHM {beam.power_fwhm_deg(nu0) * 60:.2f}' at "
        f"{nu0 / 1e6:.2f} MHz; phase centre {centre.to_string('hmsdms')} "
        f"({centre_details.get('rule')})"
    )
    header = (
        f"{'component':<34}{'offset':>9}{'beam':>8}{'expected':>12}"
        f"{'measured':>12}{'ratio':>8}{'dpos(px)':>10}  status"
    )
    print(header)
    print("-" * len(header))
    failures = 0
    for comp in components:
        sep = float(comp["position"].separation(centre).deg)
        attenuation = float(beam.power_response(sep, nu0))
        expected = comp["flux_nu0"] * attenuation
        # A line on the same position adds its band average to the continuum.
        pixel_deg = float(np.abs(wcs.proj_plane_pixel_scales()[0].to_value(u.deg)))
        for line in lines:
            if line["position"].separation(comp["position"]).deg <= pixel_deg:
                expected += line["band_mean_jy"] / n_ms_channels
        measured = _measure(image, wcs, comp["position"], args.search_box_pix)
        if measured is None:
            print(f"{comp['name']:<34} outside the image  FAIL")
            failures += 1
            continue
        peak, dpos = measured
        ratio = peak / expected if expected else float("nan")
        ok = abs(ratio - 1.0) <= args.flux_tol and dpos <= args.pos_tol_pix
        failures += 0 if ok else 1
        print(
            f"{comp['name']:<34}{sep * 60:>8.2f}'{attenuation:>8.3f}"
            f"{expected * 1e3:>9.2f} mJy{peak * 1e3:>9.2f} mJy{ratio:>8.3f}"
            f"{dpos:>10.2f}  {'ok' if ok else 'FAIL'}"
        )

    if lines:
        cube_path = _line_cube_image(args.run_dir, manifest)
        if cube_path is None:
            print("FAIL: the run has spectral cubes but no line cube image")
            failures += 1
        else:
            line_image = np.asarray(fits.getdata(cube_path), dtype=float).squeeze()
            line_wcs = WCS(fits.getheader(cube_path))
            freqs = line_wcs.spectral.pixel_to_world(
                np.arange(line_image.shape[0])
            ).to_value(u.Hz)
            channel_hz = abs(float(freqs[1] - freqs[0]))
            print(f"\nline cube  : {cube_path}")
            print(
                f"{'line':<34}{'expected':>16}{'measured':>16}{'ratio':>8}"
                f"{'dnu(ch)':>9}  status"
            )
            for line in lines:
                measured = _measure_line(line_image, line_wcs, freqs, line)
                if measured is None:
                    print(f"{line['name']:<34} not measurable  FAIL")
                    failures += 1
                    continue
                flux, centre_hz = measured
                ratio = flux / line["expected"] if line["expected"] else float("nan")
                dnu = abs(centre_hz - line["expected_centre_hz"]) / channel_hz
                ok = abs(ratio - 1.0) <= args.flux_tol and dnu <= 1.0
                failures += 0 if ok else 1
                print(
                    f"{line['name']:<34}{line['expected']:>9.4f} {line['unit']:<6}"
                    f"{flux:>9.4f} {line['unit']:<6}{ratio:>8.3f}{dnu:>9.2f}  "
                    f"{'ok' if ok else 'FAIL'}"
                )

    reported = [
        item.get("details") or {}
        for item in manifest.get("milestones", [])
        if item.get("name") == "image_model_injected"
    ]
    merges = [
        item
        for item in manifest.get("milestones", [])
        if item.get("name") == "image_model_merged"
    ]
    print(f"\nimage models injected: {len(reported)}; merged into DATA: {len(merges)}")
    if len(merges) != len(reported):
        print("FAIL: every injected image model must be merged once")
        failures += 1
    print("PASS" if failures == 0 else f"{failures} check(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
