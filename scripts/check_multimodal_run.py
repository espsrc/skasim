"""Check a multimodal run: compact components land where and as bright as expected.

For every compact component of a finished run (each image model's brightest
pixel, and catalogue sources with major axis <= ``--max-size-arcsec``), the
expected dirty-image peak is computed independently from the *input* files:

    S_expected = I(nu_ref) * (nu0 / nu_ref)^alpha * P(theta, nu0)

with ``P`` the shared Gaussian power beam recorded in the manifest
(``primary_beam_configured``) and ``theta`` the distance from the phase centre
(``phase_centre_resolved``). The script then finds the dirty-image peak near
the expected position and reports flux ratio and position error. It exits
non-zero when any component misses the tolerances.

Designed for ``configs/fits_model/vla_c_catalog_plus_two_images_quick.json``
(compact sources well separated relative to the synthesized beam). Extended or
blended components are not meaningful peak-flux tests.

Usage:
    python scripts/check_multimodal_run.py demo_output/<run_dir>
"""

from __future__ import annotations

import argparse
import json
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


def _dirty_image(run_dir: Path, manifest: dict) -> Path:
    for output in manifest.get("outputs", []):
        if output.get("role") == "dirty" and str(output.get("path", "")).endswith(
            ".fits"
        ):
            return run_dir / output["path"]
    raise SystemExit("no dirty FITS image recorded in the manifest")


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

    dirty_path = _dirty_image(args.run_dir, manifest)
    image = np.asarray(fits.getdata(dirty_path), dtype=float).squeeze()
    wcs = WCS(fits.getheader(dirty_path)).celestial

    models = manifest["config"]["models"]
    components = _catalogue_components(
        models, nu0, args.max_size_arcsec
    ) + _image_components(models, nu0)
    if not components:
        print("no compact components to check")
        return 1

    print(f"dirty image : {dirty_path}")
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
        measured = _measure(image, wcs, comp["position"], args.search_box_pix)
        if measured is None:
            print(f"{comp['name']:<34} outside the dirty image  FAIL")
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
