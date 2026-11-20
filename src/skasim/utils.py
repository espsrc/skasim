"""helpers extracted from legacy scripts/utils.py."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import astropy.units as u
import numpy as np
from loguru import logger


def init_logger(log_file: Optional[str] = None) -> None:
    """Cconfigure loguru: stderr + optional log file."""
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        level="INFO",
    )
    if log_file:
        logger.add(
            log_file,
            colorize=False,
            level="INFO",
            enqueue=True,
        )


def define_extra_units() -> None:
    extra_units = [
        u.def_unit("JY", 1 * u.Jy),
        u.def_unit("DEG", 1 * u.deg),
        u.def_unit("JY/BEAM", 1 * u.Jy / u.sr),
        u.def_unit("HZ", 1 * u.Hz),
    ]
    u.add_enabled_units(extra_units)


def mapping_unit(unit_str: Optional[str]) -> Optional[str]:
    if unit_str is None:
        return None
    return {
        "JY": "Jy",
        "DEG": "deg",
        "JY/BEAM": "Jy/beam",
        "HZ": "Hz",
    }.get(unit_str, unit_str)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# TODO: define elsewhere or check if karabo provides
DIAMETERS = {
    "ALMA": 25 * u.m,
    "APEX": 12 * u.m,
    "ATCA": 22 * u.m,
    "CARMA": 10.4 * u.m,
    "e-MERLIN": 25 * u.m,
    "GBT": 100 * u.m,
    "GMRT": 45 * u.m,
    "IRAM30M": 30 * u.m,
    "JCMT": 15 * u.m,
    "LOFAR": 25 * u.m,
    "MEERKAT": 13.5 * u.m,
    "MRT": 30 * u.m,
    "NRAO12M": 12 * u.m,
    "NRAO20M": 20 * u.m,
    "NRAO40M": 40 * u.m,
    "NRAO45M": 45 * u.m,
    "NRAO90M": 90 * u.m,
    "PARKES": 64 * u.m,
    "SMA": 6.5 * u.m,
    "SKA1LOW": 38 * u.m,
    "SKA1MID": 15 * u.m,
    "VLA": 25 * u.m,
}


def get_diameter(telescope_name: str):
    name = telescope_name.upper()
    # Keys are not all upper case (e.g. "e-MERLIN"), so compare case-insensitively.
    diameters = {key.upper(): value for key, value in DIAMETERS.items()}
    if name in diameters:
        return diameters[name]
    if "SKA" in name or "SKA1" in name:
        if "LOW" in name:
            return DIAMETERS["SKA1LOW"]
        if "MID" in name:
            return DIAMETERS["SKA1MID"]
        raise ValueError(
            f"Telescope {telescope_name} not found. "
            f"Available: {', '.join(DIAMETERS.keys())}"
        )
    raise ValueError(
        f"Telescope {telescope_name} not found. "
        f"Available: {', '.join(DIAMETERS.keys())}"
    )


# --------------------------------------------------------------------------- #
# shadeMS UV coverage helpers                                                 #
# --------------------------------------------------------------------------- #


def build_shadems_uv_coverage_argv(
    shadems_command: str,
    visibility_path: Path,
    output_dir: Path,
    png_name: str,
    title: str,
    canvas_size: int = 600,
) -> list[str]:
    """Build a shell-free shadeMS argv list for a U/V coverage plot."""
    return shlex.split(shadems_command) + [
        str(visibility_path),
        "--xaxis",
        "u",
        "--yaxis",
        "v",
        "--dir",
        str(output_dir),
        "--png",
        png_name,
        "--title",
        title,
        "--xlabel",
        "u",
        "--ylabel",
        "v",
        "--xcanvas",
        str(canvas_size),
        "--ycanvas",
        str(canvas_size),
        "--spread-pix",
        "2",
        "--no-lim-save",
    ]


def shadems_uv_coverage_env(work_dir: Path) -> dict[str, str]:
    """Return an environment with writable cache directories for shadeMS imports."""
    env = os.environ.copy()
    cache_dir = work_dir / ".cache"
    mpl_dir = cache_dir / "matplotlib"
    numba_dir = cache_dir / "numba"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    numba_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["NUMBA_CACHE_DIR"] = str(numba_dir)
    return env


def run_shadems_command(argv: list[str], work_dir: Path):
    """Run shadeMS with argv, an explicit cwd, and writable cache directories."""
    return subprocess.run(
        argv,
        shell=False,
        cwd=str(work_dir),
        env=shadems_uv_coverage_env(work_dir),
        capture_output=True,
        text=True,
        check=True,
    )
