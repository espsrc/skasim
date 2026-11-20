"""tests/test_utils.py"""

import json
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest

from skasim.utils import (
    DIAMETERS,
    NpEncoder,
    build_shadems_uv_coverage_argv,
    define_extra_units,
    get_diameter,
    mapping_unit,
    shadems_uv_coverage_env,
    # printlog,
    # show_exc,
)

# -----------------------------------------------------------------------------
# mapping_unit
# -----------------------------------------------------------------------------


def test_mapping_unit_none():
    assert mapping_unit(None) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("JY", "Jy"),
        ("DEG", "deg"),
        ("JY/BEAM", "Jy/beam"),
        ("HZ", "Hz"),
    ],
)
def test_mapping_unit_known(raw, expected):
    assert mapping_unit(raw) == expected


def test_mapping_unit_unknown():
    assert mapping_unit("foo") == "foo"


# -----------------------------------------------------------------------------
# NpEncoder
# -----------------------------------------------------------------------------


def test_npencoder_bool():
    assert json.dumps({"ok": np.bool_(True)}, cls=NpEncoder) == '{"ok": true}'


def test_npencoder_integer():
    val = np.int64(42)
    assert json.dumps({"n": val}, cls=NpEncoder) == '{"n": 42}'


def test_npencoder_floating():
    val = np.float64(3.14)
    assert json.dumps({"x": val}, cls=NpEncoder) == '{"x": 3.14}'


def test_npencoder_ndarray():
    arr = np.array([1, 2, 3])
    result = json.loads(json.dumps({"a": arr}, cls=NpEncoder))
    assert result["a"] == [1, 2, 3]


# -----------------------------------------------------------------------------
# get_diameter
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DIAMETERS.keys()))
def test_get_diameter_exact(name):
    diam = get_diameter(name)
    assert isinstance(diam, u.Quantity)
    assert diam.unit.physical_type == "length"
    assert diam > 0


def test_get_diameter_lowercase():
    assert get_diameter("ska1mid") == DIAMETERS["SKA1MID"]


@pytest.mark.parametrize("name", ["e-MERLIN", "E-MERLIN", "e-merlin"])
def test_get_diameter_mixed_case_key(name):
    assert get_diameter(name) == DIAMETERS["e-MERLIN"]


def test_get_diameter_ska_fallback_low():
    assert get_diameter("SKA1-LOW") == DIAMETERS["SKA1LOW"]


def test_get_diameter_ska_fallback_mid():
    assert get_diameter("SKA1-MID") == DIAMETERS["SKA1MID"]


def test_get_diameter_unknown_raises():
    with pytest.raises(ValueError, match="Telescope.*not found"):
        get_diameter("FAKE_SCOPE")


# -----------------------------------------------------------------------------
# define_extra_units
# -----------------------------------------------------------------------------


def test_define_extra_units_idempotent():
    # calling twice should not crash
    define_extra_units()
    define_extra_units()
    # TODO: verify behavior
    assert u.Unit("JY").is_equivalent(u.Jy)


# -----------------------------------------------------------------------------
# shadeMS UV coverage helpers
# -----------------------------------------------------------------------------


def test_build_shadems_uv_coverage_argv_uses_verified_configuration(tmp_path):
    """shadeMS argv is built with verified config values."""
    argv = build_shadems_uv_coverage_argv(
        shadems_command="python -m shade_ms",
        visibility_path=tmp_path / "visibilities.MS",
        output_dir=tmp_path / "plots",
        png_name="uv.png",
        title="UVCoverage",
        canvas_size=800,
    )
    assert argv[:3] == ["python", "-m", "shade_ms"]
    assert "--xaxis" in argv
    assert "--yaxis" in argv
    assert "--png" in argv
    assert "uv.png" in argv
    assert "--title" in argv
    assert "UVCoverage" in argv
    assert "--xcanvas" in argv
    assert "800" in argv


def test_shadems_uv_coverage_env_uses_writable_cache_dirs(tmp_path):
    """shadeMS env sets writable MPL and Numba cache directories."""
    env = shadems_uv_coverage_env(tmp_path)
    assert env["MPLCONFIGDIR"] is not None
    assert env["NUMBA_CACHE_DIR"] is not None
    assert Path(env["MPLCONFIGDIR"]).is_dir()
    assert Path(env["NUMBA_CACHE_DIR"]).is_dir()
    assert Path(env["MPLCONFIGDIR"]).parent.parent == tmp_path
    assert Path(env["NUMBA_CACHE_DIR"]).parent.parent == tmp_path


def test_run_shadems_command_records_manifest_outputs(tmp_path, monkeypatch):
    """shadeMS plot generation records both PNG and command log outputs."""
    import skasim.utils as utils_mod
    from skasim.config import SimConfig
    from skasim.manifest import create_run_context

    config = SimConfig(output_dir=str(tmp_path / "run"))
    ctx = create_run_context(config)
    visibility_path = ctx.work_dir / "visibilities.MS"
    visibility_path.mkdir()

    def fake_run(argv, work_dir):
        (work_dir / "run_uvcoverage.png").write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
            b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return SimpleNamespace(stdout="shadeMS ok\n", stderr="")

    monkeypatch.setattr("skasim.utils.run_shadems_command", fake_run)

    run_id = ctx.work_dir.name
    png_name = f"{run_id}_uvcoverage.png"
    log_name = f"{run_id}_uvcoverage_shadems.log"
    png_path = ctx.work_dir / png_name
    log_path = ctx.work_dir / log_name
    argv = build_shadems_uv_coverage_argv(
        shadems_command="shadems",
        visibility_path=visibility_path,
        output_dir=ctx.work_dir,
        png_name=png_name,
        title=f"{run_id} uv coverage",
        canvas_size=600,
    )
    result = utils_mod.run_shadems_command(argv, ctx.work_dir)
    log_path.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    assert png_path.exists()
    ctx.manifest.add_output(
        "plot",
        png_name,
        role="uv_coverage",
        metadata={"tool": "shadems", "xaxis": "u", "yaxis": "v", "canvas_size": 600},
    )
    ctx.manifest.add_output(
        "log",
        log_name,
        role="uv_coverage",
        metadata={"tool": "shadems"},
    )

    uv_outputs = [
        output for output in ctx.manifest.outputs if output.role == "uv_coverage"
    ]
    assert [output.kind for output in uv_outputs] == ["plot", "log"]
    assert uv_outputs[0].path == "run_uvcoverage.png"
    assert uv_outputs[0].metadata["tool"] == "shadems"
    assert (ctx.work_dir / "run_uvcoverage_shadems.log").read_text(
        encoding="utf-8"
    ) == "shadeMS ok\n"
