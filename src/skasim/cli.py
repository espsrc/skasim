"""cli.py — command-line entry point (adapted from synthsim.py)"""

# The environment and warning filters must be configured before importing the
# application modules, which import plotting and xarray dependencies.
# ruff: noqa: E402

# suppress xarray accessor collision between rascil and ska-sdp-datamodels
import os
import warnings

warnings.filterwarnings("ignore", message="registration of accessor")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import ImgConfig, ObsConfig, SimConfig
from .runtime import require_karabo_module

# options that may coexist with --config
_CONFIG_COMPATIBLE = frozenset({"config", "output_dir", "overwrite", "show_telescopes"})


def _has_non_default_content_args(parser, args, argv):
    """Return True if any content argument was explicitly set (not default).

    For actions with ``argparse.SUPPRESS`` defaults, we inspect ``argv`` to
    decide whether the option was supplied, because argparse cannot distinguish
    an explicit value that happens to equal a legacy default from an omitted
    option once the default is removed.
    """
    # Build a set of option strings that appeared literally in argv, including
    # --key=value variants.
    explicit_options: set[str] = set()
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            if "=" in token:
                explicit_options.add(token.split("=", 1)[0])
            else:
                explicit_options.add(token)
        i += 1

    for action in parser._actions:
        dest = action.dest
        if dest in _CONFIG_COMPATIBLE or dest == "help":
            continue
        # deprecated hidden actions already error before reaching here
        if action.help is argparse.SUPPRESS:
            continue
        if action.default is argparse.SUPPRESS:
            if not any(opt in explicit_options for opt in action.option_strings):
                continue
            # --pixels 512 is the Pydantic default; treat it as not supplying
            # a content argument so that legacy invocations still work.
            if dest == "pixels" and getattr(args, dest, None) == 512:
                continue
            return True
        value = getattr(args, dest, action.default)
        if value != action.default:
            return True
    return False


def _deprecated_action(message: str):
    class DeprecatedAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            parser.error(message)

    return DeprecatedAction


class _DeprecatedIAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            "--I was removed in skasim 0.2; use --flux-density for generated "
            "source flux densities in Jy."
        )


class _DeprecatedCleaningAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error("--cleaning was removed in skasim 0.2; use --imager wsclean.")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="SKA simulator: sky model -> visibilities -> image products",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_conf = p.add_argument_group("run from config file")
    run_conf.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (exclusive with content arguments; --output-dir and --overwrite allowed)",
    )

    sky = p.add_argument_group("sky model source")
    sky.add_argument(
        "--model",
        type=str,
        default=None,
        help="Sky model file: FITS, JSON, pickle, or Karabo model",
    )
    sky.add_argument(
        "--fits-image",
        dest="fits_image",
        type=str,
        default=None,
        help=(
            "FITS image file (2D continuum image). Deprecated: use a "
            "static_stokes_maps model entry in a config file."
        ),
    )
    sky.add_argument(
        "--catalog",
        dest="catalog",
        type=str,
        default=None,
        help="Built-in catalog name: MIGHTEE, GLEAM, or SKAMid",
    )
    sky.add_argument(
        "--flux-density",
        dest="source_flux_jy",
        metavar="FLUX_JY",
        type=float,
        nargs="+",
        default=None,
        help="Generated source flux density values in Jy",
    )
    sky.add_argument(
        "--stokes-q",
        dest="stokes_q_jy",
        metavar="Q_JY",
        type=float,
        nargs="+",
        default=None,
        help="Generated source Stokes Q values in Jy; must match --flux-density length",
    )
    sky.add_argument(
        "--stokes-u",
        dest="stokes_u_jy",
        metavar="U_JY",
        type=float,
        nargs="+",
        default=None,
        help="Generated source Stokes U values in Jy; must match --flux-density length",
    )
    sky.add_argument(
        "--stokes-v",
        dest="stokes_v_jy",
        metavar="V_JY",
        type=float,
        nargs="+",
        default=None,
        help="Generated source Stokes V values in Jy; must match --flux-density length",
    )
    sky.add_argument(
        "--continuum-stokes-i",
        type=str,
        default=None,
        help="Continuum image-model Stokes I FITS image in Jy/pixel",
    )
    sky.add_argument(
        "--continuum-alpha",
        type=str,
        default=None,
        help="Continuum image-model spectral-index FITS image",
    )
    sky.add_argument(
        "--reference-frequency-hz",
        type=float,
        default=None,
        help="Reference frequency for --continuum-stokes-i in Hz",
    )

    pointing = p.add_argument_group("observation")
    pointing.add_argument(
        "--telescope", type=str, default="SKA1MID", help="Telescope name"
    )
    pointing.add_argument(
        "--telescope-version", type=str, default=None, help="Explicit telescope version"
    )
    pointing.add_argument(
        "--center",
        type=str,
        default=None,
        help="Field centre, e.g. '10h01m35.1s 2d41m41s'",
    )
    pointing.add_argument(
        "--frequency-mhz",
        type=float,
        default=argparse.SUPPRESS,
        help="Central observing frequency in MHz",
    )
    pointing.add_argument(
        "--bandwidth-mhz",
        type=float,
        default=argparse.SUPPRESS,
        help="Bandwidth in MHz",
    )
    pointing.add_argument(
        "--n-channels", type=int, default=argparse.SUPPRESS, help="Number of channels"
    )
    pointing.add_argument(
        "--channel-width-mhz",
        type=float,
        default=argparse.SUPPRESS,
        help="Channel width in MHz",
    )
    pointing.add_argument(
        "--observation-time",
        dest="observation_time_s",
        metavar="SECONDS",
        type=int,
        default=argparse.SUPPRESS,
        help="Observation duration in seconds (default: 600)",
    )

    imaging = p.add_argument_group("image product")
    imaging.add_argument(
        "--imager",
        choices=["oskar-dirty", "wsclean"],
        default="oskar-dirty",
        help="Image product imager",
    )
    imaging.add_argument(
        "--wsclean-command",
        default="wsclean",
        help="Command used for WSClean imaging",
    )
    imaging.add_argument(
        "--pixels",
        type=int,
        default=argparse.SUPPRESS,
        help="Image size in pixels (legacy default: 512)",
    )
    imaging.add_argument(
        "--fov", type=str, default=None, help="Field of view, e.g. '1deg'"
    )
    imaging.add_argument("--robust", type=float, default=0.0, help="Briggs robustness")
    imaging.add_argument(
        "--clean-iterations", type=int, default=5000, help="WSClean CLEAN iterations"
    )

    coverage = p.add_argument_group("uv coverage")
    coverage.add_argument(
        "--no-uv-coverage",
        dest="uv_coverage",
        action="store_false",
        default=True,
        help="Skip shadeMS UV-coverage plot generation",
    )
    coverage.add_argument(
        "--shadems-command",
        default="shadems",
        help="Command used for shadeMS UV-coverage plotting (default: shadems; set to a full path or conda run wrapper if shadeMS is installed in a separate environment)",
    )
    coverage.add_argument(
        "--uv-coverage-canvas-size",
        type=int,
        default=600,
        help="Square shadeMS UV-coverage canvas size in pixels",
    )

    outputs = p.add_argument_group("outputs")
    outputs.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Exact output directory; defaults to YYYYMMDD_HHMMSS_<telescope>",
    )
    outputs.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )

    advanced = p.add_argument_group("advanced")
    advanced.add_argument(
        "--rms", action="store_true", default=False, help="Enable noise"
    )
    advanced.add_argument(
        "--rms-value", type=float, default=0.0, help="RMS noise level in Jy"
    )
    advanced.add_argument(
        "--rms-sigma", type=float, default=3.0, help="RMS sigma multiplier"
    )
    advanced.add_argument(
        "--noise-rms-start",
        type=float,
        default=None,
        help="OSKAR station noise RMS start (Jy) — overrides telescope model",
    )
    advanced.add_argument(
        "--noise-rms-end",
        type=float,
        default=None,
        help="OSKAR station noise RMS end (Jy) — overrides telescope model",
    )
    advanced.add_argument(
        "--flux-scale",
        type=float,
        default=1.0,
        help="Scale file-backed Stokes I flux density",
    )
    advanced.add_argument(
        "--column-mapping",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12",
        help="FITS column indices: id,ra,dec,I,Q,U,V,spec_index,ref_freq,rm,major,minor,pa",
    )
    advanced.add_argument(
        "--show-telescopes",
        action="store_true",
        help="List available telescopes and exit",
    )

    # Hidden migration paths for removed or renamed 0.1/early-0.2 options.
    p.add_argument(
        "--I",
        type=float,
        nargs="+",
        default=None,
        action=_DeprecatedIAction,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--stokes-i",
        dest="removed_stokes_i",
        type=float,
        nargs="+",
        default=None,
        action=_deprecated_action(
            "--stokes-i was removed; use --flux-density for generated source "
            "flux densities in Jy."
        ),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--cleaning",
        nargs=0,
        default=None,
        action=_DeprecatedCleaningAction,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--fits",
        action=_deprecated_action(
            "--fits was removed; use --model for FITS, JSON, pickle, or Karabo model files."
        ),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--json",
        action=_deprecated_action(
            "--json was removed; use --model for FITS, JSON, pickle, or Karabo model files."
        ),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--json-fg",
        action=_deprecated_action(
            "--json-fg was removed; provide one sky model source with --model, --catalog, or --flux-density."
        ),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--Q",
        action=_deprecated_action("--Q was renamed to --stokes-q."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--U",
        action=_deprecated_action("--U was renamed to --stokes-u."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--V",
        action=_deprecated_action("--V was renamed to --stokes-v."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ref-freq",
        nargs="+",
        action=_deprecated_action(
            "--ref-freq was removed from the CLI; observing frequency is set with --frequency-mhz."
        ),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--freq",
        action=_deprecated_action("--freq was renamed to --frequency-mhz."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--bandwidth",
        action=_deprecated_action("--bandwidth was renamed to --bandwidth-mhz."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--delta-freq",
        action=_deprecated_action("--delta-freq was renamed to --channel-width-mhz."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--seconds",
        action=_deprecated_action("--seconds was renamed to --observation-time."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--prefix",
        action=_deprecated_action(
            "--prefix was renamed to --output-dir and now names the exact output directory."
        ),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--niter",
        action=_deprecated_action("--niter was renamed to --clean-iterations."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--scale-I",
        action=_deprecated_action("--scale-I was renamed to --flux-scale."),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--imaging-niter",
        action=_deprecated_action(
            "--imaging-niter was removed; use --clean-iterations for WSClean imaging."
        ),
        help=argparse.SUPPRESS,
    )

    args = p.parse_args(argv)

    # ------------------------------------------------------------------- #
    # --config branch: load SimConfig from JSON
    # ------------------------------------------------------------------- #
    if args.config is not None:
        if _has_non_default_content_args(p, args, argv or sys.argv[1:]):
            p.error("--config is exclusive with content arguments")
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            p.error(f"Config file not found: {config_path}")
        config = SimConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        # allow --overwrite / --output-dir to override JSON values
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        if args.overwrite:
            config.overwrite = True

        from .pipeline import run

        run(config)
        return

    if args.show_telescopes:
        from typing import get_args

        telescope_module = require_karabo_module("karabo.simulation.telescope")
        OSKARTelescopesWithoutVersionType = (
            telescope_module.OSKARTelescopesWithoutVersionType
        )
        OSKARTelescopesWithVersionType = telescope_module.OSKARTelescopesWithVersionType
        OSKAR_TELESCOPE_TO_VERSIONS = telescope_module.OSKAR_TELESCOPE_TO_VERSIONS

        print("=== Telescopes (No version required) ===")
        for t in sorted(get_args(OSKARTelescopesWithoutVersionType)):
            print(f"  {t}")

        print(
            "\n=== Telescopes requiring a version (Supply via --telescope-version) ==="
        )
        for t in sorted(get_args(OSKARTelescopesWithVersionType)):
            versions = [v.name for v in OSKAR_TELESCOPE_TO_VERSIONS.get(t, [])]
            print(f"  {t:<18} | Accepted versions: {', '.join(versions)}")
        sys.exit(0)

    # map CLI args to SimConfig
    # pass only the flags given, so unset values take the ObsConfig defaults
    # and the CLI and config-file entry points agree
    obs_fields = (
        "frequency_mhz",
        "bandwidth_mhz",
        "n_channels",
        "channel_width_mhz",
        "observation_time_s",
    )
    obs = ObsConfig(**{f: getattr(args, f) for f in obs_fields if hasattr(args, f)})

    fov_deg = None
    if args.fov:
        try:
            import astropy.units as u

            fov_deg = u.Quantity(args.fov).to(u.deg).value
        except Exception:
            fov_deg = float(args.fov)

    img_kwargs = {
        "tag": "default",
        "fov_deg": fov_deg,
        "pixels": getattr(args, "pixels", None),
        "robust": args.robust,
        "imager": args.imager,
        "wsclean_command": args.wsclean_command,
        "clean_iterations": args.clean_iterations,
    }
    img_kwargs = {k: v for k, v in img_kwargs.items() if v is not None}
    img = ImgConfig(**img_kwargs)

    # Build typed model entries from legacy CLI flags
    model_entries = []
    if args.model is not None:
        model_entries.append(
            {
                "type": "component_sky_model",
                "path": args.model,
                "sky_format": "auto",
                "column_mapping": args.column_mapping,
                "flux_scale": args.flux_scale,
            }
        )
    if args.catalog is not None:
        model_entries.append(
            {
                "type": "component_sky_model",
                "catalog": args.catalog,
            }
        )
    continuum_fields = (
        args.continuum_stokes_i,
        args.continuum_alpha,
        args.reference_frequency_hz,
    )
    if any(value is not None for value in continuum_fields):
        if not all(value is not None for value in continuum_fields):
            p.error(
                "--continuum-stokes-i, --continuum-alpha, and "
                "--reference-frequency-hz must be provided together."
            )
        model_entries.append(
            {
                "type": "continuum_i_alpha",
                "stokes_i": args.continuum_stokes_i,
                "alpha": args.continuum_alpha,
                "reference_frequency_hz": args.reference_frequency_hz,
            }
        )

    config_kwargs = {
        "telescope": args.telescope,
        "telescope_version": args.telescope_version,
        "sky_format": "auto",
        "column_mapping": args.column_mapping,
        "flux_scale": args.flux_scale,
        "center": args.center,
        "rms": args.rms,
        "rms_value": args.rms_value,
        "rms_sigma": args.rms_sigma,
        "noise_rms_start": args.noise_rms_start,
        "noise_rms_end": args.noise_rms_end,
        "observation": obs,
        "imaging": [img],
        "output_dir": args.output_dir,
        "overwrite": args.overwrite,
        "uv_coverage": args.uv_coverage,
        "shadems_command": args.shadems_command,
        "uv_coverage_canvas_size": args.uv_coverage_canvas_size,
    }
    if model_entries and args.fits_image is not None:
        # fits_image is a legacy field and cannot sit beside typed models
        p.error(
            "--fits-image cannot be combined with --model, --catalog, or "
            "--continuum-* options; provide one sky model source per run."
        )
    if model_entries:
        config_kwargs["models"] = model_entries
    else:
        config_kwargs["sky_file"] = args.model
        config_kwargs["catalog"] = args.catalog
        config_kwargs["fits_image"] = args.fits_image
    if args.source_flux_jy is not None:
        config_kwargs["source_flux_jy"] = args.source_flux_jy
    if args.stokes_q_jy is not None:
        config_kwargs["stokes_q_jy"] = args.stokes_q_jy
    if args.stokes_u_jy is not None:
        config_kwargs["stokes_u_jy"] = args.stokes_u_jy
    if args.stokes_v_jy is not None:
        config_kwargs["stokes_v_jy"] = args.stokes_v_jy

    config = SimConfig(**config_kwargs)

    from .pipeline import run

    run(config)


if __name__ == "__main__":
    main()
