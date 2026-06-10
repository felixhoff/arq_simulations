#!/usr/bin/env python3
"""
Generate I2 B-X transition positions and improved relative strengths.

This script keeps Luc (1980) Tables IV/V line positions, but replaces the
single-Morse Franck-Condon model with a spectroscopic-potential workflow:

  1. Default: use expanded-Morse-oscillator (EMO) potentials fitted to Luc's
     observed vibrational term values and B_v constants.
  2. Optional: refit the EMO potentials on your machine with --fcf-source refit-emo.
  3. Optional: supply external high-quality pointwise potentials or literature FCFs.

The default EMO presets are much more constrained than a simple Morse model:
X uses v''=0..9 energies and B_v values; B uses v'=1..62 band origins and B_v
values. The transition positions still come directly from Tables IV/V.

Limitations: this is still a relative-intensity model. Absolute cross sections
require an electronic transition moment function, line broadening, instrumental
response, isotopic abundance choices, pressure/temperature conditions, and
possibly perturbation/predissociation corrections.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import least_squares

# Exact physical constants, SI unless otherwise noted.
H = 6.62607015e-34
HBAR = H / (2.0 * math.pi)
C_M_S = 2.99792458e8
C_CM_S = 2.99792458e10
AMU_KG = 1.66053906660e-27
K_B = 1.380649e-23
K_B_OVER_HC_CM = 0.695034800  # k_B/(h c) in cm^-1 K^-1
CM_TO_THz = C_CM_S / 1.0e12

# 127I atomic mass. Natural iodine is essentially monoisotopic 127I.
I127_MASS_U = 126.9044719
MU_I2_KG = (I127_MASS_U * AMU_KG) / 2.0


# Ground X state constants from Luc Table IV.
# E is vibrational term energy relative to X(v''=0,J''=0), in cm^-1.
# D values are cm^-1; Table IV reports 10^9 * D.
X_STATE: Dict[int, Dict[str, float]] = {
    0: {"E": 0.0000,    "B": 0.037311571, "D": 4.5773338e-9},
    1: {"E": 213.3006,  "B": 0.037197081, "D": 4.6028129e-9},
    2: {"E": 425.3733,  "B": 0.037082021, "D": 4.6311452e-9},
    3: {"E": 636.2102,  "B": 0.036966025, "D": 4.6445633e-9},
    4: {"E": 845.8034,  "B": 0.036849704, "D": 4.6747097e-9},
    5: {"E": 1054.1457, "B": 0.036732571, "D": 4.7058148e-9},
    6: {"E": 1261.2279, "B": 0.036614776, "D": 4.7363730e-9},
    7: {"E": 1467.0414, "B": 0.036496162, "D": 4.7662557e-9},
    8: {"E": 1671.5786, "B": 0.036376769, "D": 4.7970460e-9},
    9: {"E": 1874.8283, "B": 0.036256529, "D": 4.8294091e-9},
}


# Excited B state constants from Luc Table V.
# sigma0 is the J=0 band origin from X(v''=0) to B(v'), in cm^-1.
# B: Table V reports 10^2 * B, so values here are multiplied by 1e-2.
# D: Table V reports 10^8 * D, so values here are multiplied by 1e-8.
# H: Table V reports 10^14 * H for v'=1..30 and 10^13 * H for v'=31..62;
#    values here are actual H in cm^-1.
B_STATE_RAW = [
    # v, sigma0,        1e2*B,     1e8*D,     scaled_H,  H_scale
    (1,  15848.7476, 2.8773913, 0.6335021,  -0.2884,   1e-14),
    (2,  15971.3668, 2.8619756, 0.6479861,  -0.2446,   1e-14),
    (3,  16092.4368, 2.8462991, 0.6614539,  -0.2482,   1e-14),
    (4,  16211.9334, 2.8303558, 0.6743953,  -0.2952,   1e-14),
    (5,  16329.8365, 2.8141348, 0.6883674,  -0.3201,   1e-14),
    (6,  16446.1265, 2.7976259, 0.7025975,  -0.3658,   1e-14),
    (7,  16560.7843, 2.7808307, 0.7174997,  -0.4337,   1e-14),
    (8,  16673.7866, 2.7637777, 0.7399957,  -0.2693,   1e-14),
    (9,  16785.1140, 2.7463287, 0.7531339,  -0.4547,   1e-14),
    (10, 16894.7443, 2.7285885, 0.7729565,  -0.4491,   1e-14),
    (11, 17002.6565, 2.7104914, 0.7895742,  -0.6369,   1e-14),
    (12, 17108.8275, 2.6920279, 0.8062299,  -0.8579,   1e-14),
    (13, 17213.2364, 2.6733027, 0.8386592,  -0.5422,   1e-14),
    (14, 17315.8600, 2.6540804, 0.8563920,  -0.8006,   1e-14),
    (15, 17416.6768, 2.6345073, 0.8821992,  -0.8345,   1e-14),
    (16, 17515.6645, 2.6145169, 0.9091475,  -0.8477,   1e-14),
    (17, 17612.7997, 2.5940917, 0.9340232,  -1.0517,   1e-14),
    (18, 17708.0619, 2.5732357, 0.9623762,  -1.1821,   1e-14),
    (19, 17801.4283, 2.5519456, 0.9940724,  -1.2885,   1e-14),
    (20, 17892.8777, 2.5301959, 1.0298997,  -1.2951,   1e-14),
    (21, 17982.3892, 2.5079402, 1.0637983,  -1.4757,   1e-14),
    (22, 18069.9423, 2.4851807, 1.0997442,  -1.6424,   1e-14),
    (23, 18155.5174, 2.4619286, 1.1389726,  -1.8106,   1e-14),
    (24, 18239.0953, 2.4381466, 1.1803433,  -2.0038,   1e-14),
    (25, 18320.6579, 2.4138215, 1.2243490,  -2.2031,   1e-14),
    (26, 18400.1887, 2.3889522, 1.2713989,  -2.4057,   1e-14),
    (27, 18477.6718, 2.3634965, 1.3192903,  -2.6854,   1e-14),
    (28, 18553.0929, 2.3374820, 1.3711216,  -2.9696,   1e-14),
    (29, 18626.4388, 2.3108798, 1.4262925,  -3.2887,   1e-14),
    (30, 18697.6996, 2.2836890, 1.4865103,  -3.5557,   1e-14),
    (31, 18766.8672, 2.2557445, 1.5355186,  -0.42412,  1e-13),
    (32, 18833.9309, 2.2274924, 1.6174051,  -0.41267,  1e-13),
    (33, 18898.8899, 2.1984085, 1.6774918,  -0.49287,  1e-13),
    (34, 18961.7422, 2.1687831, 1.7535683,  -0.52878,  1e-13),
    (35, 19022.4879, 2.1385023, 1.8283588,  -0.58823,  1e-13),
    (36, 19081.1306, 2.1075769, 1.9028476,  -0.67257,  1e-13),
    (37, 19137.6777, 2.0761056, 1.9921597,  -0.72310,  1e-13),
    (38, 19192.1382, 2.0439748, 2.0780748,  -0.80951,  1e-13),
    (39, 19244.5247, 2.0112771, 2.1751283,  -0.87680,  1e-13),
    (40, 19294.8524, 1.9779293, 2.2681896,  -0.99053,  1e-13),
    (41, 19343.1405, 1.9441079, 2.3835023,  -1.04402,  1e-13),
    (42, 19389.4099, 1.9096043, 2.4857190,  -1.18789,  1e-13),
    (43, 19433.6851, 1.8741319, 2.5538711,  -1.46122,  1e-13),
    (44, 19475.9912, 1.8387999, 2.6907424,  -1.58054,  1e-13),
    (45, 19516.3611, 1.8002895, 2.8468907,  -1.62780,  1e-13),
    (46, 19554.8256, 1.7663019, 2.9869330,  -1.78339,  1e-13),
    (47, 19591.4197, 1.7291363, 3.1239311,  -2.00202,  1e-13),
    (48, 19626.1797, 1.6915023, 3.2686776,  -2.28222,  1e-13),
    (49, 19659.1449, 1.6534103, 3.4333592,  -2.50632,  1e-13),
    (50, 19690.3546, 1.6148747, 3.6115706,  -2.78861,  1e-13),
    (51, 19719.8506, 1.5756474, 3.7506198,  -3.36943,  1e-13),
    (52, 19747.6750, 1.5362813, 3.9934358,  -3.43854,  1e-13),
    (53, 19773.8703, 1.4962899, 4.1878601,  -4.04905,  1e-13),
    (54, 19798.4824, 1.4557470, 4.3919239,  -4.65555,  1e-13),
    (55, 19821.5542, 1.4149379, 4.6611350,  -5.05647,  1e-13),
    (56, 19843.1311, 1.3734652, 4.9008455,  -5.89927,  1e-13),
    (57, 19863.2565, 1.3317148, 5.2082764,  -6.60287,  1e-13),
    (58, 19881.9834, 1.2892046, 5.4377336,  -8.19568,  1e-13),
    (59, 19899.3523, 1.2464889, 5.8114065,  -8.87653,  1e-13),
    (60, 19915.4127, 1.2031186, 6.1303268, -10.72266,  1e-13),
    (61, 19930.2139, 1.1595435, 6.5619420, -12.10565,  1e-13),
    (62, 19943.8081, 1.1151630, 6.9177555, -14.81389,  1e-13),
]

B_STATE: Dict[int, Dict[str, float]] = {
    v: {"sigma0": sigma0, "B": b_scaled * 1e-2, "D": d_scaled * 1e-8, "H": h_scaled * h_scale}
    for v, sigma0, b_scaled, d_scaled, h_scaled, h_scale in B_STATE_RAW
}

# Table III analyzed bands, excluding the (13,0) blend noted in Table II.
ANALYZED_BANDS: Dict[int, List[int]] = {
    0: [v for v in range(10, 63) if v != 13],
    1: list(range(9, 31)),
    2: list(range(7, 22)),
    3: list(range(4, 17)),
    4: list(range(4, 13)),
    5: list(range(3, 10)),
    6: list(range(1, 8)),
    7: list(range(1, 7)),
    8: list(range(1, 6)),
    9: list(range(2, 5)),
}



# -----------------------------------------------------------------------------
# Preset EMO potentials
# -----------------------------------------------------------------------------
# Parameterization:
#   V(R) = D_e * [1 - exp(- beta(R) * (R - R_e))]^2
#   beta(R) = exp(sum_i c_i * y_p(R)^i), y_p=(R^p-R_e^p)/(R^p+R_e^p)
# The parameters below are fitted to Luc Tables IV/V term values and B_v values,
# using the same reduced mass as the line-strength calculation.

PRESET_EMO_X = {
    "name": "X_1Sigma_g+",
    "p": 3,
    "params": [
        9.84904477, 2.66638474, 0.41344173, 0.04253079,
        0.45109480, -2.61384153, 3.35170949,
    ],
    "r_min_A": 1.8,
    "r_max_A": 7.0,
    "n_levels": 10,
    "energy_anchor_v": 0,
    "source": "preset EMO fit to Luc Table IV E_v and B_v",
}

PRESET_EMO_B = {
    "name": "B_3Pi_0u+",
    "p": 3,
    "params": [
        8.37541086, 3.02147949, 0.61278977, 0.18588792,
        0.91215472, -3.41470894, 1.84856316, 1.17935261,
        -1.31136790, 0.51668116, 0.18613713,
    ],
    "r_min_A": 1.9,
    "r_max_A": 9.5,
    "n_levels": 63,
    "energy_anchor_v": 1,
    "source": "preset EMO fit to Luc Table V sigma0(v) and B_v",
}


def rotational_energy(B: float, D: float, J: int, Hc: float = 0.0) -> float:
    """Rotational term value F_v(J), in cm^-1."""
    jj = J * (J + 1)
    return B * jj - D * jj**2 + Hc * jj**3


def transition_wavenumber(v_upper: int, v_lower: int, branch: str, J_lower: int) -> Tuple[float, int]:
    """Return transition wavenumber in cm^-1 and J_upper."""
    if branch == "R":
        J_upper = J_lower + 1
    elif branch == "P":
        if J_lower < 1:
            raise ValueError("P branch requires J_lower >= 1")
        J_upper = J_lower - 1
    else:
        raise ValueError("branch must be 'P' or 'R'")

    upper = B_STATE[v_upper]
    lower = X_STATE[v_lower]
    band_origin = upper["sigma0"] - lower["E"]
    F_upper = rotational_energy(upper["B"], upper["D"], J_upper, upper["H"])
    F_lower = rotational_energy(lower["B"], lower["D"], J_lower, 0.0)
    return band_origin + F_upper - F_lower, J_upper


def lower_state_energy(v_lower: int, J_lower: int) -> float:
    lower = X_STATE[v_lower]
    return lower["E"] + rotational_energy(lower["B"], lower["D"], J_lower, 0.0)


def iter_band_pairs(mode: str) -> Iterator[Tuple[int, int]]:
    if mode == "all":
        for v_lower in sorted(X_STATE):
            for v_upper in sorted(B_STATE):
                yield v_upper, v_lower
    elif mode == "analyzed":
        for v_lower, uppers in sorted(ANALYZED_BANDS.items()):
            for v_upper in uppers:
                yield v_upper, v_lower
    else:
        raise ValueError("mode must be 'all' or 'analyzed'")


def nuclear_spin_weight_I127_X(J_lower: int) -> int:
    """Nuclear-spin statistical weight for 127I2 X 1Sigma_g+ lower levels."""
    return 15 if (J_lower % 2 == 0) else 21


def honl_london_parallel_omega0(branch: str, J_lower: int) -> Tuple[float, float]:
    """Return normalized H-L factor and population*H-L rotational factor.

    For a parallel Omega=0 <- Omega=0 band:
      P: H_L = J''/(2J''+1), product with (2J''+1) = J''
      R: H_L = (J''+1)/(2J''+1), product with (2J''+1) = J''+1
    """
    denom = 2.0 * J_lower + 1.0
    if branch == "P":
        hl = J_lower / denom
        rot = float(J_lower)
    elif branch == "R":
        hl = (J_lower + 1.0) / denom
        rot = float(J_lower + 1)
    else:
        raise ValueError(branch)
    return hl, rot


def kinetic_prefactor_cm_A2(mu_kg: float = MU_I2_KG) -> float:
    """Finite-difference kinetic prefactor K for R in Angstrom and energy in cm^-1."""
    return (HBAR**2 / (2.0 * mu_kg)) / (H * C_M_S * 100.0) * 1.0e20


def rotational_prefactor_cm_A2(mu_kg: float = MU_I2_KG) -> float:
    """B_v = const * <1/R^2>, with R in Angstrom."""
    return H / (8.0 * math.pi**2 * C_CM_S * mu_kg) * 1.0e20


def emo_potential_cm(r_A: np.ndarray, params: List[float], p: int = 3) -> np.ndarray:
    log_De, r_e, *coeffs = params
    De = math.exp(log_De)
    y = (r_A**p - r_e**p) / (r_A**p + r_e**p)
    poly = np.zeros_like(r_A, dtype=float)
    for i, c in enumerate(coeffs):
        poly += c * y**i
    beta = np.exp(np.clip(poly, -8.0, 4.0))
    expo = np.clip(-beta * (r_A - r_e), -70.0, 45.0)
    return De * (1.0 - np.exp(expo)) ** 2


def solve_potential_on_grid(
    r_A: np.ndarray,
    V_cm: np.ndarray,
    n_levels: int,
    mu_kg: float = MU_I2_KG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 1D vibrational Schrodinger equation on a uniform grid.

    Returns eigenvalues, eigenvectors, and calculated B_v values. Eigenvectors are
    columns and are Euclidean-normalized; dot products between eigenvectors on the
    same grid approximate continuous overlap integrals.
    """
    if len(r_A) < 10:
        raise ValueError("Need at least 10 grid points")
    dr = float(r_A[1] - r_A[0])
    if not np.allclose(np.diff(r_A), dr, rtol=1e-6, atol=1e-10):
        raise ValueError("Grid must be uniform")
    K = kinetic_prefactor_cm_A2(mu_kg)
    diag = V_cm + 2.0 * K / dr**2
    offdiag = np.full(len(r_A) - 1, -K / dr**2)
    vals, vecs = eigh_tridiagonal(
        diag,
        offdiag,
        select="i",
        select_range=(0, n_levels - 1),
        check_finite=False,
    )
    Bconst = rotational_prefactor_cm_A2(mu_kg)
    Bv = Bconst * np.sum((vecs**2) / (r_A[:, None] ** 2), axis=0)
    # deterministic phase convention for reproducible overlap signs
    for j in range(vecs.shape[1]):
        idx = int(np.argmax(np.abs(vecs[:, j])))
        if vecs[idx, j] < 0:
            vecs[:, j] *= -1.0
    return vals, vecs, Bv


def solve_emo_state(config: Dict, r_A: np.ndarray, n_levels: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    levels = int(n_levels if n_levels is not None else config["n_levels"])
    V = emo_potential_cm(r_A, config["params"], int(config.get("p", 3)))
    vals, vecs, Bv = solve_potential_on_grid(r_A, V, levels)
    return V, vals, vecs, Bv


def read_pointwise_potential(path: Path, r_A: np.ndarray) -> np.ndarray:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"r_A", "V_cm"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must have columns r_A,V_cm")
        for row in reader:
            rows.append((float(row["r_A"]), float(row["V_cm"])))
    rows.sort()
    rp = np.array([x[0] for x in rows])
    vp = np.array([x[1] for x in rows])
    if r_A[0] < rp[0] or r_A[-1] > rp[-1]:
        raise ValueError(
            f"Grid [{r_A[0]:.3f}, {r_A[-1]:.3f}] A extends outside potential {path} "
            f"range [{rp[0]:.3f}, {rp[-1]:.3f}] A"
        )
    return np.interp(r_A, rp, vp)


def observed_data_for_state(state: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if state == "X":
        v = np.array(sorted(X_STATE), dtype=int)
        E = np.array([X_STATE[int(i)]["E"] for i in v], dtype=float)
        Bv = np.array([X_STATE[int(i)]["B"] for i in v], dtype=float)
        anchor = 0
    elif state == "B":
        v = np.array(sorted(B_STATE), dtype=int)
        sigma = np.array([B_STATE[int(i)]["sigma0"] for i in v], dtype=float)
        E = sigma - sigma[0]
        Bv = np.array([B_STATE[int(i)]["B"] for i in v], dtype=float)
        anchor = 1
    else:
        raise ValueError(state)
    return v, E, Bv, anchor


def fit_emo_state(
    state: str,
    start_config: Dict,
    n_grid_fit: int,
    max_nfev: int,
    sigma_E_cm: float,
    sigma_B_cm: float,
    verbose: bool,
) -> Dict:
    """Refit one EMO state to observed term values and B_v constants."""
    obs_v, obs_E_rel, obs_B, anchor = observed_data_for_state(state)
    n_levels = int(max(obs_v) + 1)
    r_min = float(start_config["r_min_A"])
    r_max = float(start_config["r_max_A"])
    r_A = np.linspace(r_min, r_max, n_grid_fit)
    p = int(start_config.get("p", 3))
    x0 = np.array(start_config["params"], dtype=float)
    order = len(x0) - 3

    def residual(theta: np.ndarray) -> np.ndarray:
        V = emo_potential_cm(r_A, theta.tolist(), p)
        vals, _vecs, Bcalc = solve_potential_on_grid(r_A, V, n_levels)
        Ecalc_rel = vals[obs_v] - vals[anchor]
        return np.concatenate([
            (Ecalc_rel - obs_E_rel) / sigma_E_cm,
            (Bcalc[obs_v] - obs_B) / sigma_B_cm,
        ])

    # Broad but physical bounds: De, Re, beta polynomial coefficients.
    if state == "X":
        lo = [math.log(5000.0), 2.35, math.log(0.2)] + [-10.0] * order
        hi = [math.log(35000.0), 3.05, math.log(8.0)] + [10.0] * order
    else:
        lo = [math.log(1000.0), 2.35, math.log(0.08)] + [-14.0] * order
        hi = [math.log(20000.0), 3.90, math.log(10.0)] + [14.0] * order

    result = least_squares(
        residual,
        x0,
        bounds=(np.array(lo), np.array(hi)),
        max_nfev=max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        verbose=2 if verbose else 0,
    )
    out = dict(start_config)
    out["params"] = [float(x) for x in result.x]
    out["source"] = f"runtime EMO refit to Luc Table {'IV' if state == 'X' else 'V'}"
    out["fit_cost"] = float(result.cost)
    out["fit_nfev"] = int(result.nfev)
    return out


def potential_fit_report_rows(config_X: Dict, config_B: Dict, r_A: np.ndarray) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for label, config, obs_source in [("X", config_X, X_STATE), ("B", config_B, B_STATE)]:
        n_levels = int(config["n_levels"])
        _, vals, _, Bcalc = solve_emo_state(config, r_A, n_levels)
        obs_v, obs_E_rel, obs_B, anchor = observed_data_for_state(label)
        for k, v in enumerate(obs_v):
            ecalc_rel = vals[v] - vals[anchor]
            rows.append({
                "state": label,
                "v": str(int(v)),
                "E_obs_rel_cm-1": f"{obs_E_rel[k]:.8f}",
                "E_calc_rel_cm-1": f"{ecalc_rel:.8f}",
                "E_resid_cm-1": f"{ecalc_rel - obs_E_rel[k]:.8f}",
                "B_obs_cm-1": f"{obs_B[k]:.10f}",
                "B_calc_cm-1": f"{Bcalc[v]:.10f}",
                "B_resid_cm-1": f"{Bcalc[v] - obs_B[k]:.10e}",
            })
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float_list(text: str | None) -> List[float] | None:
    if text is None or text.strip() == "":
        return None
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def transition_moment_values(r_A: np.ndarray, coeffs: List[float] | None, r_ref_A: float) -> np.ndarray:
    if coeffs is None:
        return np.ones_like(r_A)
    x = r_A - r_ref_A
    mu = np.zeros_like(r_A)
    power = np.ones_like(r_A)
    for c in coeffs:
        mu += c * power
        power *= x
    return mu


def compute_fcf_from_wavefunctions(
    r_A: np.ndarray,
    vecs_X: np.ndarray,
    vecs_B: np.ndarray,
    mu_R: np.ndarray,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    fcf: Dict[Tuple[int, int], Dict[str, float]] = {}
    for v_upper in sorted(B_STATE):
        psi_u = vecs_B[:, v_upper]
        for v_lower in sorted(X_STATE):
            psi_l = vecs_X[:, v_lower]
            overlap = float(np.dot(psi_u, psi_l))
            fcf_val = overlap * overlap
            mu_int = float(np.dot(psi_u * mu_R, psi_l))
            vibronic_strength = mu_int * mu_int
            if abs(overlap) > 1.0e-12:
                r_centroid = float(np.dot(psi_u * r_A, psi_l) / overlap)
            else:
                r_centroid = float("nan")
            fcf[(v_upper, v_lower)] = {
                "overlap": overlap,
                "FCF": fcf_val,
                "transition_moment_integral": mu_int,
                "vibronic_strength": vibronic_strength,
                "r_centroid_A": r_centroid,
            }
    return fcf


def read_external_fcf(path: Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"v_upper", "v_lower"}.issubset(fields):
            raise ValueError("FCF CSV must include v_upper,v_lower")
        strength_col = "vibronic_strength" if "vibronic_strength" in fields else "FCF"
        if strength_col not in fields:
            raise ValueError("FCF CSV must include FCF or vibronic_strength")
        for row in reader:
            vu, vl = int(row["v_upper"]), int(row["v_lower"])
            fcf_val = float(row.get("FCF", row[strength_col]) or row[strength_col])
            strength = float(row[strength_col])
            out[(vu, vl)] = {
                "overlap": float(row.get("overlap", "nan") or "nan"),
                "FCF": fcf_val,
                "transition_moment_integral": float(row.get("transition_moment_integral", "nan") or "nan"),
                "vibronic_strength": strength,
                "r_centroid_A": float(row.get("r_centroid_A", "nan") or "nan"),
            }
    return out


def build_fcf_table(args: argparse.Namespace) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], List[Dict[str, str]]]:
    if args.fcf_source == "external-fcf":
        if args.fcf_input is None:
            raise SystemExit("--fcf-source external-fcf requires --fcf-input")
        return read_external_fcf(Path(args.fcf_input)), []

    r_A = np.linspace(args.r_min, args.r_max, args.grid)
    fit_rows: List[Dict[str, str]] = []

    if args.fcf_source == "external-potential":
        if args.x_potential_csv is None or args.b_potential_csv is None:
            raise SystemExit("--fcf-source external-potential requires --x-potential-csv and --b-potential-csv")
        Vx = read_pointwise_potential(Path(args.x_potential_csv), r_A)
        Vb = read_pointwise_potential(Path(args.b_potential_csv), r_A)
        vals_X, vecs_X, Bv_X = solve_potential_on_grid(r_A, Vx, 10)
        vals_B, vecs_B, Bv_B = solve_potential_on_grid(r_A, Vb, 63)
    else:
        config_X = dict(PRESET_EMO_X)
        config_B = dict(PRESET_EMO_B)
        if args.fcf_source == "refit-emo":
            config_X = fit_emo_state("X", config_X, args.fit_grid, args.fit_max_nfev, args.fit_sigma_E, args.fit_sigma_B, args.verbose_fit)
            # The B-state has more data; use slightly looser default energy scale if not overridden.
            config_B = fit_emo_state("B", config_B, args.fit_grid, args.fit_max_nfev, args.fit_sigma_E_B, args.fit_sigma_B_B, args.verbose_fit)
        Vx, vals_X, vecs_X, Bv_X = solve_emo_state(config_X, r_A, 10)
        Vb, vals_B, vecs_B, Bv_B = solve_emo_state(config_B, r_A, 63)
        if args.potential_output_prefix:
            prefix = Path(args.potential_output_prefix)
            write_csv(prefix.with_name(prefix.name + "_X_potential.csv"), ["r_A", "V_cm"], ({"r_A": f"{r:.8f}", "V_cm": f"{v:.10f}"} for r, v in zip(r_A, Vx)))
            write_csv(prefix.with_name(prefix.name + "_B_potential.csv"), ["r_A", "V_cm"], ({"r_A": f"{r:.8f}", "V_cm": f"{v:.10f}"} for r, v in zip(r_A, Vb)))
        fit_rows = potential_fit_report_rows(config_X, config_B, r_A)

    coeffs = parse_float_list(args.tdm_coeffs)
    r_ref = args.tdm_r_ref if args.tdm_r_ref is not None else 0.5 * (PRESET_EMO_X["params"][1] + PRESET_EMO_B["params"][1])
    mu_R = transition_moment_values(r_A, coeffs, r_ref)
    return compute_fcf_from_wavefunctions(r_A, vecs_X, vecs_B, mu_R), fit_rows


def iter_transition_rows(args: argparse.Namespace, fcf: Dict[Tuple[int, int], Dict[str, float]]) -> List[Dict[str, object]]:
    branch_list = {"both": ["P", "R"], "P": ["P"], "R": ["R"]}[args.branches]
    rows: List[Dict[str, object]] = []
    kT_cm = K_B_OVER_HC_CM * args.temperature
    for v_upper, v_lower in iter_band_pairs(args.mode):
        vib = fcf.get((v_upper, v_lower))
        if vib is None:
            continue
        for branch in branch_list:
            j_start = 1 if branch == "P" else 0
            for J_lower in range(j_start, args.max_j + 1):
                wn, J_upper = transition_wavenumber(v_upper, v_lower, branch, J_lower)
                if args.cap_upper_j and J_upper > args.max_j:
                    continue
                if wn <= 0:
                    continue
                E_low = lower_state_energy(v_lower, J_lower)
                boltz = math.exp(-E_low / kT_cm) if args.temperature > 0 else 0.0
                g_ns = nuclear_spin_weight_I127_X(J_lower)
                degeneracy = 2 * J_lower + 1
                hl_norm, rot_factor = honl_london_parallel_omega0(branch, J_lower)
                freq_factor = wn ** args.frequency_power
                strength = freq_factor * vib["vibronic_strength"] * g_ns * rot_factor * boltz
                rows.append({
                    "v_upper": v_upper,
                    "v_lower": v_lower,
                    "branch": branch,
                    "J_lower": J_lower,
                    "J_upper": J_upper,
                    "wavenumber_cm-1": wn,
                    "frequency_THz": wn * CM_TO_THz,
                    "wavelength_vac_nm": 1.0e7 / wn,
                    "E_lower_cm-1": E_low,
                    "FCF": vib["FCF"],
                    "overlap": vib["overlap"],
                    "r_centroid_A": vib["r_centroid_A"],
                    "transition_moment_integral": vib["transition_moment_integral"],
                    "vibronic_strength": vib["vibronic_strength"],
                    "boltzmann": boltz,
                    "nuclear_spin_weight": g_ns,
                    "lower_rot_degeneracy": degeneracy,
                    "honl_london_normalized": hl_norm,
                    "rotational_line_factor": rot_factor,
                    "relative_strength": strength,
                })
    max_strength = max((float(r["relative_strength"]) for r in rows), default=0.0)
    for r in rows:
        r["relative_strength_norm"] = float(r["relative_strength"]) / max_strength if max_strength > 0 else 0.0
    return rows


def format_transition_row(row: Dict[str, object]) -> Dict[str, object]:
    float_formats = {
        "wavenumber_cm-1": ".6f",
        "frequency_THz": ".9f",
        "wavelength_vac_nm": ".9f",
        "E_lower_cm-1": ".6f",
        "FCF": ".12e",
        "overlap": ".12e",
        "r_centroid_A": ".8f",
        "transition_moment_integral": ".12e",
        "vibronic_strength": ".12e",
        "boltzmann": ".12e",
        "honl_london_normalized": ".12e",
        "rotational_line_factor": ".8f",
        "relative_strength": ".12e",
        "relative_strength_norm": ".12e",
    }
    out = {}
    for k, v in row.items():
        if k in float_formats:
            try:
                out[k] = format(float(v), float_formats[k])
            except ValueError:
                out[k] = "nan"
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate I2 B-X line positions and improved relative strengths from spectroscopic potentials."
    )
    parser.add_argument("--max-j", type=int, default=150, help="maximum lower-state J'' to include")
    parser.add_argument("--temperature", type=float, default=300.0, help="Boltzmann temperature in K")
    parser.add_argument("--mode", choices=["all", "analyzed"], default="all")
    parser.add_argument("--branches", choices=["both", "P", "R"], default="both")
    parser.add_argument("--cap-upper-j", action="store_true", help="also require J' <= --max-j")
    parser.add_argument("--output", default="iodine_bx_accurate_strengths_J150_T300.csv")
    parser.add_argument("--fcf-output", default="iodine_bx_accurate_fcf.csv")
    parser.add_argument("--fit-report-output", default="iodine_bx_potential_fit_report.csv")
    parser.add_argument(
        "--fcf-source",
        choices=["preset-emo", "refit-emo", "external-potential", "external-fcf"],
        default="preset-emo",
        help="source of vibrational factors; preset-emo is the fast default",
    )
    parser.add_argument("--fcf-input", help="CSV with v_upper,v_lower,FCF for --fcf-source external-fcf")
    parser.add_argument("--x-potential-csv", help="CSV with r_A,V_cm for X state")
    parser.add_argument("--b-potential-csv", help="CSV with r_A,V_cm for B state")
    parser.add_argument("--potential-output-prefix", help="write the generated EMO potentials to <prefix>_X/B_potential.csv")
    parser.add_argument("--grid", type=int, default=1000, help="uniform grid points for final FCF calculation")
    parser.add_argument("--r-min", type=float, default=2.0, help="minimum R in Angstrom for final FCF grid")
    parser.add_argument("--r-max", type=float, default=9.0, help="maximum R in Angstrom for final FCF grid")
    parser.add_argument("--fit-grid", type=int, default=1200, help="grid points for runtime EMO refits")
    parser.add_argument("--fit-max-nfev", type=int, default=250, help="max optimizer evaluations per state for runtime refits")
    parser.add_argument("--fit-sigma-E", type=float, default=0.02, help="X-state energy residual scale in cm^-1")
    parser.add_argument("--fit-sigma-B", type=float, default=2.0e-6, help="X-state B_v residual scale in cm^-1")
    parser.add_argument("--fit-sigma-E-B", type=float, default=0.10, help="B-state energy residual scale in cm^-1")
    parser.add_argument("--fit-sigma-B-B", type=float, default=1.0e-5, help="B-state B_v residual scale in cm^-1")
    parser.add_argument("--verbose-fit", action="store_true", help="print scipy optimizer progress")
    parser.add_argument(
        "--tdm-coeffs",
        help="comma-separated polynomial coefficients for mu(R)=sum c_n*(R-r_ref)^n; default constant mu=1",
    )
    parser.add_argument("--tdm-r-ref", type=float, help="reference R in Angstrom for --tdm-coeffs")
    parser.add_argument(
        "--frequency-power",
        type=float,
        default=1.0,
        help="multiply relative strengths by wavenumber^power; use 0 to disable frequency factor",
    )
    args = parser.parse_args()

    if args.max_j < 0:
        raise SystemExit("--max-j must be non-negative")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if args.grid < 200:
        raise SystemExit("--grid should be at least 200")

    fcf, fit_rows = build_fcf_table(args)

    fcf_fields = [
        "v_upper", "v_lower", "FCF", "overlap", "r_centroid_A",
        "transition_moment_integral", "vibronic_strength",
    ]
    fcf_rows = []
    for (vu, vl), d in sorted(fcf.items()):
        fcf_rows.append({
            "v_upper": vu,
            "v_lower": vl,
            "FCF": f"{d['FCF']:.12e}",
            "overlap": f"{d['overlap']:.12e}",
            "r_centroid_A": f"{d['r_centroid_A']:.8f}",
            "transition_moment_integral": f"{d['transition_moment_integral']:.12e}",
            "vibronic_strength": f"{d['vibronic_strength']:.12e}",
        })
    write_csv(Path(args.fcf_output), fcf_fields, fcf_rows)

    if fit_rows and args.fit_report_output:
        write_csv(
            Path(args.fit_report_output),
            ["state", "v", "E_obs_rel_cm-1", "E_calc_rel_cm-1", "E_resid_cm-1", "B_obs_cm-1", "B_calc_cm-1", "B_resid_cm-1"],
            fit_rows,
        )

    rows = iter_transition_rows(args, fcf)
    fields = [
        "v_upper", "v_lower", "branch", "J_lower", "J_upper",
        "wavenumber_cm-1", "frequency_THz", "wavelength_vac_nm",
        "E_lower_cm-1", "FCF", "overlap", "r_centroid_A",
        "transition_moment_integral", "vibronic_strength", "boltzmann",
        "nuclear_spin_weight", "lower_rot_degeneracy", "honl_london_normalized",
        "rotational_line_factor", "relative_strength", "relative_strength_norm",
    ]
    write_csv(Path(args.output), fields, (format_transition_row(r) for r in rows))

    print(f"Wrote {len(rows):,} transitions to {args.output}")
    print(f"Wrote {len(fcf_rows):,} vibrational factors to {args.fcf_output}")
    if fit_rows and args.fit_report_output:
        print(f"Wrote potential fit diagnostics to {args.fit_report_output}")
    print("Model: line positions from Luc Tables IV/V; vibrational factors from", args.fcf_source)
    print("Strength convention: nu^frequency_power * vibronic_strength * g_ns * rotational_line_factor * Boltzmann")


if __name__ == "__main__":
    main()
