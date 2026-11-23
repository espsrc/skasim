"""CLI configuration behavior."""

import json
import warnings

import pytest
from pydantic import ValidationError

import skasim.pipeline
from skasim import cli
from skasim.config import ImgConfig, ObsConfig, SimConfig


def _make_config_json(tmp_path, flux=None):
    """Write a minimal SimConfig JSON to a temp file."""
    config = SimConfig(
        source_flux_jy=[1.0],
        observation=ObsConfig(frequency_mhz=800.0),
        imaging=[ImgConfig(imager="oskar-dirty")],
    )
    if flux is not None:
        config.source_flux_jy = flux
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")
    return str(path)


def test_config_file_loads_simconfig(monkeypatch, tmp_path):
    """--config <json> deserialises the SimConfig and runs the pipeline."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    config_path = _make_config_json(tmp_path, flux=[2.5, 5.0])
    cli.main(["--config", config_path])

    assert captured[0].source_flux_jy == [2.5, 5.0]
    assert captured[0].observation.frequency_mhz == 800.0
    assert captured[0].imaging[0].imager == "oskar-dirty"


def test_config_file_allows_explicit_default_pixel_value(monkeypatch, tmp_path):
    """--config plus --pixels 512 must not be treated as a content conflict."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    config_path = _make_config_json(tmp_path, flux=[2.5, 5.0])
    cli.main(["--config", config_path, "--pixels", "512"])

    assert captured[0].imaging[0]._geometry_fields_set() == {"pixels"}


def test_config_file_overrides_output_dir_and_overwrite(monkeypatch, tmp_path):
    """--output-dir and --overwrite are permitted alongside --config."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    config_path = _make_config_json(tmp_path)
    cli.main(
        [
            "--config",
            config_path,
            "--output-dir",
            str(tmp_path / "custom"),
            "--overwrite",
        ]
    )

    assert captured[0].output_dir == str(tmp_path / "custom")
    assert captured[0].overwrite is True


def test_config_file_blocks_content_flags(monkeypatch, tmp_path, capsys):
    """--config rejects content arguments such as --frequency-mhz."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))
    config_path = _make_config_json(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["--config", config_path, "--frequency-mhz", "900"])

    captured = capsys.readouterr()
    assert "--config is exclusive with content arguments" in captured.err


def test_config_file_blocks_content_flags_equals_syntax(monkeypatch, tmp_path, capsys):
    """--config catches content arguments even with --flag=value syntax."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))
    config_path = _make_config_json(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["--config", config_path, "--frequency-mhz=900"])

    captured = capsys.readouterr()
    assert "--config is exclusive with content arguments" in captured.err


def test_config_file_missing_file(monkeypatch, tmp_path, capsys):
    """--config with a nonexistent path errors cleanly."""
    with pytest.raises(SystemExit):
        cli.main(["--config", str(tmp_path / "nonexistent.json")])

    captured = capsys.readouterr()
    assert "Config file not found" in captured.err


# --------------------------------------------------------------------------- #
# existing tests below
# --------------------------------------------------------------------------- #


def test_catalog_sets_named_catalog(monkeypatch):
    """--catalog is the canonical CLI spelling for built-in catalogs."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(["--catalog", "GLEAM"])

    assert captured[0].catalog is None
    assert captured[0].models[0].catalog == "GLEAM"


def test_numeric_catalog_cli_value_has_migration_message():
    """Numeric catalog CLI values fail with the 0.2 migration message."""
    with pytest.raises(ValidationError, match="Numeric catalog IDs were removed"):
        cli.main(["--catalog", "1"])


def test_flux_density_sets_generated_source_flux_jy(monkeypatch):
    """--flux-density defines generated source intensities."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(["--flux-density", "1.0", "5.0", "10.0"])

    assert captured[0].source_flux_jy == [1.0, 5.0, 10.0]


def test_cli_observation_defaults_match_config_defaults(monkeypatch):
    """Without observation flags the CLI uses the ObsConfig defaults.

    The CLI used to default to a 10 s observation while a config file
    defaulted to 600 s, so the same run differed by entry point.
    """
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(["--flux-density", "1.0"])

    assert captured[0].observation == ObsConfig()
    assert captured[0].observation.observation_time_s == 600


def test_cli_observation_flags_are_passed_through(monkeypatch):
    """Explicit observation flags still reach the config."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(
        [
            "--flux-density",
            "1.0",
            "--observation-time",
            "10",
            "--frequency-mhz",
            "1400",
            "--bandwidth-mhz",
            "64",
            "--n-channels",
            "4",
        ]
    )

    obs = captured[0].observation
    assert obs.observation_time_s == 10
    assert obs.frequency_mhz == 1400.0
    assert obs.bandwidth_mhz == 64.0
    assert obs.n_channels == 4
    assert obs.channel_width_mhz == 16.0


def test_generated_source_stokes_flags(monkeypatch):
    """--stokes-q/u/v define generated source polarization values."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(
        [
            "--flux-density",
            "2.5",
            "3.5",
            "--stokes-q",
            "0.1",
            "0.2",
            "--stokes-u",
            "0.0",
            "0.3",
            "--stokes-v",
            "0.0",
            "-0.1",
        ]
    )

    assert captured[0].source_flux_jy == [2.5, 3.5]
    assert captured[0].stokes_q_jy == [0.1, 0.2]
    assert captured[0].stokes_u_jy == [0.0, 0.3]
    assert captured[0].stokes_v_jy == [0.0, -0.1]


def test_stokes_i_has_migration_message(capsys):
    """--stokes-i fails with a migration message for --flux-density."""
    with pytest.raises(SystemExit):
        cli.main(["--stokes-i", "2.5", "3.5"])

    captured = capsys.readouterr()
    assert "--flux-density" in captured.err


def test_legacy_i_flag_has_migration_message(capsys):
    """--I fails with a migration message for the clearer source intensity flags."""
    with pytest.raises(SystemExit):
        cli.main(["--I", "1.0"])

    captured = capsys.readouterr()
    assert "--flux-density" in captured.err


def test_default_imager_is_oskar_dirty(monkeypatch):
    """The CLI defaults to the OSKAR dirty imager."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main([])

    assert captured[0].imaging[0].imager == "oskar-dirty"


def test_default_cli_geometry_uses_legacy_path(monkeypatch):
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cli.main([])

    assert captured[0].imaging[0]._geometry_fields_set() == set()
    assert any(item.category is DeprecationWarning for item in caught)


def test_explicit_cli_default_pixel_count_remains_explicit(monkeypatch):
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(["--pixels", "512"])

    assert captured[0].imaging[0]._geometry_fields_set() == {"pixels"}


def test_wsclean_imager_is_selected_explicitly(monkeypatch):
    """--imager wsclean selects WSClean image production."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(["--imager", "wsclean"])

    assert captured[0].imaging[0].imager == "wsclean"


def test_cleaning_flag_has_migration_message(capsys):
    """--cleaning fails with a migration message for --imager wsclean."""
    with pytest.raises(SystemExit):
        cli.main(["--cleaning"])

    captured = capsys.readouterr()
    assert "--imager wsclean" in captured.err


def test_wsclean_command_cli_override(monkeypatch):
    """--wsclean-command stores the configured WSClean command."""
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))
    command = "singularity exec /mnt/software/containers/wsclean-3.10-dysco.sif wsclean"

    cli.main(["--imager", "wsclean", "--wsclean-command", command])

    assert captured[0].imaging[0].wsclean_command == command


def test_cli_rejects_source_flux_jy_with_catalog():
    """CLI users cannot combine generated-source intensities with a catalog."""
    with pytest.raises(ValidationError, match="typed models"):
        cli.main(["--catalog", "MIGHTEE", "--flux-density", "1.0"])


def test_cli_accepts_catalog_plus_continuum_image_model(tmp_path, monkeypatch):
    """A catalog contribution can be combined with a continuum image model."""
    stokes_i = tmp_path / "stokes_i.fits"
    alpha = tmp_path / "alpha.fits"
    stokes_i.write_text("placeholder", encoding="utf-8")
    alpha.write_text("placeholder", encoding="utf-8")
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(
        [
            "--catalog",
            "MIGHTEE",
            "--continuum-stokes-i",
            str(stokes_i),
            "--continuum-alpha",
            str(alpha),
            "--reference-frequency-hz",
            "1400000000",
        ]
    )

    assert [entry.type for entry in captured[0].models] == [
        "component_sky_model",
        "continuum_i_alpha",
    ]
    assert captured[0].models[0].catalog == "MIGHTEE"


def test_cli_runs_json_config_file(tmp_path, monkeypatch):
    """--config loads a JSON SimConfig and runs it directly."""
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "telescope": "VLA",
                "telescope_version": "C",
                "catalog": "GLEAM",
                "observation": {
                    "frequency_mhz": 1400.0,
                    "bandwidth_mhz": 8.0,
                    "n_channels": 2,
                },
                "output_dir": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )
    captured = []
    monkeypatch.setattr(skasim.pipeline, "run", lambda config: captured.append(config))

    cli.main(["--config", str(config_path)])

    assert captured[0].telescope == "VLA"
    assert captured[0].telescope_version == "C"
    assert captured[0].catalog == "GLEAM"
    assert captured[0].observation.n_channels == 2


def test_cli_help_has_single_canonical_surface(capsys):
    """Public help exposes canonical flags and hides removed aliases."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    captured = capsys.readouterr()
    help_text = captured.out
    assert "--frequency-mhz" in help_text
    assert "--observation-time" in help_text
    assert "--output-dir" in help_text
    assert "--clean-iterations" in help_text
    assert "--flux-density" in help_text
    assert "--stokes-q" in help_text
    assert "--stokes-u" in help_text
    assert "--stokes-v" in help_text
    assert "--catalog" in help_text
    assert "--continuum-stokes-i" in help_text
    assert "--continuum-alpha" in help_text
    assert "--reference-frequency-hz" in help_text

    for removed in (
        "--stokes-i",
        "--json",
        "--json-fg",
        "--freq ",
        "--seconds",
        "--prefix",
        "--niter",
        "--Q",
        "--U",
        "--V",
    ):
        assert removed not in help_text
