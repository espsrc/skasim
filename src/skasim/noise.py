"""Realistic thermal-noise injection into measurement-set visibilities.

Implements the radiometer equation from Radcliffe et al. (2024), MNRAS 527, 942:
    sigma_pq = (1/eta_c) * sqrt(SEFD_p * SEFD_q / (2 * t_int * delta_nu))

Noise is circularly complex Gaussian with zero mean and variance sigma_pq^2
per visibility (real and imaginary components each have variance sigma_pq^2/2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from .runtime import require_casacore


def load_sefd_file(path: str | Path) -> list[dict]:
    """Load an SEFD JSON file and return a list of antenna entries.

    Each entry contains at least 'name' and 'sefd_jy'. The list order is the
    station order used by the telescope model and must match MS ANTENNA indices.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"SEFD file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"SEFD file must be a JSON object, got {type(data).__name__}")

    antennas = data.get("antennas")
    if antennas is None:
        raise ValueError("SEFD file missing required 'antennas' field")
    if not isinstance(antennas, list):
        raise ValueError("SEFD 'antennas' field must be a list")
    if not antennas:
        raise ValueError("SEFD 'antennas' list is empty")

    result = []
    for idx, entry in enumerate(antennas):
        if not isinstance(entry, dict):
            raise ValueError(f"SEFD antenna entry {idx} must be an object")
        name = entry.get("name")
        sefd = entry.get("sefd_jy")
        if name is None or sefd is None:
            raise ValueError(f"SEFD antenna entry {idx} missing 'name' or 'sefd_jy'")
        if not isinstance(sefd, (int, float)) or sefd <= 0:
            raise ValueError(f"SEFD for {name} must be a positive number, got {sefd!r}")
        result.append({"name": str(name), "sefd_jy": float(sefd)})

    return result


def sefd_values(antennas: list[dict]) -> np.ndarray:
    """Return ordered SEFD array (Jy) from loaded antenna metadata."""
    return np.array([a["sefd_jy"] for a in antennas], dtype=np.float64)


def compute_baseline_sigma(
    sefd: np.ndarray,
    eta_c: float,
    t_int_s: float,
    delta_nu_hz: float,
) -> dict[tuple[int, int], float]:
    """Compute per-baseline noise standard deviation.

    Parameters
    ----------
    sefd : np.ndarray
        SEFD values in Jy, one per antenna, indexed by ANTENNA index.
    eta_c : float
        System efficiency factor.
    t_int_s : float
        Integration time in seconds.
    delta_nu_hz : float
        Channel bandwidth in Hz.

    Returns
    -------
    dict mapping (ant1, ant2) -> sigma_pq (Jy).
    """
    if eta_c <= 0:
        raise ValueError("eta_c must be positive")
    if t_int_s <= 0:
        raise ValueError("t_int must be positive")
    if delta_nu_hz <= 0:
        raise ValueError("delta_nu must be positive")

    n_ant = len(sefd)
    sigma = {}
    for i in range(n_ant):
        for j in range(i, n_ant):
            sigma_ij = (1.0 / eta_c) * np.sqrt(
                (sefd[i] * sefd[j]) / (2.0 * t_int_s * delta_nu_hz)
            )
            sigma[(i, j)] = float(sigma_ij)
            if i != j:
                sigma[(j, i)] = float(sigma_ij)
    return sigma


def _read_ms_metadata(ms_path: str | Path):
    """Read antenna indices, interval, and DATA shape from the MS."""
    table = require_casacore()
    ms_path = Path(ms_path).expanduser()
    with table(str(ms_path), readonly=True, ack=False) as tbl:
        ant1 = tbl.getcol("ANTENNA1").astype(np.int64)
        ant2 = tbl.getcol("ANTENNA2").astype(np.int64)
        interval = tbl.getcol("INTERVAL").astype(np.float64)
        # inspect first row for shape
        first = tbl.getcell("DATA", 0)
        if not isinstance(first, np.ndarray) or first.ndim != 2:
            raise ValueError(f"DATA column first row has unexpected shape/type: {type(first)}")
        n_channels, n_correlations = first.shape
        n_rows = tbl.nrows()
    return ant1, ant2, interval, n_rows, n_channels, n_correlations


def _per_row_sigma(
    ant1: np.ndarray,
    ant2: np.ndarray,
    interval: np.ndarray,
    sigma_map: dict[tuple[int, int], float],
) -> np.ndarray:
    """Build a per-row sigma vector using the baseline map and row interval."""
    sigma = np.empty(len(ant1), dtype=np.float64)
    for idx, (a1, a2, tint) in enumerate(zip(ant1, ant2, interval)):
        base_sigma = sigma_map[(int(a1), int(a2))]
        # sigma scales as 1/sqrt(t_int)
        sigma[idx] = base_sigma / np.sqrt(tint)
    return sigma


def inject_noise(
    ms_path: str | Path,
    sefd_antennas: list[dict],
    delta_nu_hz: float,
    eta_c: float = 0.88,
    chunk_rows: int = 5000,
    random_seed: Optional[int] = None,
) -> dict:
    """Inject realistic thermal noise into the DATA column of a measurement set.

    Parameters
    ----------
    ms_path : str | Path
        Path to the MS to modify (read-write).
    sefd_antennas : list[dict]
        Loaded SEFD antenna metadata from load_sefd_file().
    eta_c : float, optional
        System efficiency factor. Default 0.88.
    delta_nu_hz : float
        Channel bandwidth in Hz.
    chunk_rows : int, optional
        Number of rows to process per I/O chunk. Default 5000.
    random_seed : int, optional
        If provided, seed the RNG for reproducible noise.

    Returns
    -------
    dict with summary statistics (min_sigma, max_sigma, mean_sigma, n_rows).
    """
    table = require_casacore()
    ms_path = Path(ms_path).expanduser()

    ant1, ant2, interval, n_rows, n_channels, n_correlations = _read_ms_metadata(ms_path)
    sefd = sefd_values(sefd_antennas)

    # validate antenna indices
    max_ant = max(int(ant1.max()), int(ant2.max()))
    if max_ant >= len(sefd):
        raise ValueError(
            f"MS references antenna index {max_ant}, but SEFD file only has "
            f"{len(sefd)} antennas"
        )

    # baseline sigma for unit integration time (t_int = 1 s)
    # actual per-row sigma scales with interval^{-1/2}
    sigma_map = compute_baseline_sigma(sefd, eta_c, t_int_s=1.0, delta_nu_hz=delta_nu_hz)
    per_row_sigma = _per_row_sigma(ant1, ant2, interval, sigma_map)

    rng = np.random.default_rng(random_seed)
    table = require_casacore()

    with table(str(ms_path), readonly=False, ack=False) as tbl:
        has_weight_spectrum = "WEIGHT_SPECTRUM" in tbl.colnames()

        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            nrow = end - start
            # read current DATA chunk
            data_chunk = tbl.getcol("DATA", startrow=start, nrow=nrow)
            # generate circularly complex Gaussian noise:
            # variance per component (real/imag) = sigma^2
            sigmas = per_row_sigma[start:end]
            real = rng.standard_normal((nrow, n_channels, n_correlations))
            imag = rng.standard_normal((nrow, n_channels, n_correlations))
            noise = (real + 1j * imag) * sigmas[:, None, None]
            # add to existing data and write back
            tbl.putcol("DATA", data_chunk + noise, startrow=start, nrow=nrow)

            # Update WEIGHT and SIGMA so imagers (wsclean, tclean) correctly
            # weight baselines by their actual noise level.
            # WEIGHT = 1/σ², SIGMA = σ  — both shaped (nrow, n_correlations).
            row_weight = 1.0 / sigmas**2
            weight_vals = np.repeat(row_weight[:, np.newaxis], n_correlations, axis=1)
            sigma_vals = np.repeat(sigmas[:, np.newaxis], n_correlations, axis=1)
            tbl.putcol("WEIGHT", weight_vals, startrow=start, nrow=nrow)
            tbl.putcol("SIGMA", sigma_vals, startrow=start, nrow=nrow)

            # wsclean prefers WEIGHT_SPECTRUM over WEIGHT when present.
            if has_weight_spectrum:
                weight_spec = np.repeat(
                    row_weight[:, np.newaxis, np.newaxis],
                    n_channels, axis=1,
                )
                weight_spec = np.repeat(weight_spec, n_correlations, axis=2)
                tbl.putcol(
                    "WEIGHT_SPECTRUM", weight_spec, startrow=start, nrow=nrow,
                )

    summary = {
        "n_rows": int(n_rows),
        "n_channels": int(n_channels),
        "n_correlations": int(n_correlations),
        "min_sigma_jy": float(per_row_sigma.min()),
        "max_sigma_jy": float(per_row_sigma.max()),
        "mean_sigma_jy": float(per_row_sigma.mean()),
        "weight_spectrum_updated": has_weight_spectrum,
    }
    logger.info(
        f"Injected noise into {n_rows} rows × {n_channels} channels × "
        f"{n_correlations} correlations; sigma range "
        f"{summary['min_sigma_jy']:.6e} – {summary['max_sigma_jy']:.6e} Jy; "
        f"WEIGHT/SIGMA updated"
        f"{', WEIGHT_SPECTRUM updated' if has_weight_spectrum else ''}"
    )
    return summary
