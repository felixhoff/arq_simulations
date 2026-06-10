#!/usr/bin/env python3
"""
Generate I2 B-X transition positions and approximate relative strengths using
Morse-potential Franck-Condon factors.

Input physics model
-------------------
Line positions use Luc (1980) Tables IV/V constants, as in iodine_bx_line_list.py.
Franck-Condon factors are computed numerically from two Morse potentials:

    V(r) = D_e [1 - exp(-a (r - r_e))]^2

where D_e = omega_e^2 / (4 omega_e x_e), and a is determined from omega_e,
D_e, and the reduced mass. Equilibrium separations r_e are inferred from B_e.

Default Morse parameters:
  X 1Sigma_g+: omega_e = 214.5186 cm^-1, omega_e x_e = 0.6072284 cm^-1,
               B_e = 0.037368670 cm^-1  (Luc Table VI)
  B 3Pi_0u+:   omega_e = 125.6724 cm^-1, omega_e x_e = 0.752677 cm^-1,
               B_e = 0.02999694599 cm^-1 (low-order Luc Table VII values)

These Morse FCFs are approximate. The Luc line-position constants are much more
accurate than the Morse intensity model. This script assumes a constant electronic
transition moment; no r-centroid correction is included.

Example:
    python iodine_bx_morse_strengths.py --max-j 150 --temperature 300 \
        --output iodine_bx_strengths_J150_T300.csv \
        --fcf-output iodine_bx_morse_fcf.csv
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


@dataclass(frozen=True)
class MorseSpec:
    name: str
    omega_e_cm: float
    omega_exe_cm: float
    B_e_cm: float
    mu_kg: float = MU_I2_KG

    @property
    def D_e_cm(self) -> float:
        return self.omega_e_cm**2 / (4.0 * self.omega_exe_cm)

    @property
    def r_e_A(self) -> float:
        # B_e = h/(8 pi^2 c I), I = mu r_e^2. c is in cm/s for B_e in cm^-1.
        r_m = math.sqrt(H / (8.0 * math.pi**2 * C_CM_S * self.mu_kg * self.B_e_cm))
        return r_m * 1.0e10

    @property
    def a_Ainv(self) -> float:
        # omega = 2 pi c * omega_e, with omega_e in cm^-1.
        angular_freq = 2.0 * math.pi * C_M_S * 100.0 * self.omega_e_cm
        D_e_J = H * C_M_S * 100.0 * self.D_e_cm
        a_minv = math.sqrt(self.mu_kg * angular_freq**2 / (2.0 * D_e_J))
        return a_minv * 1.0e-10

    @property
    def vmax_morse(self) -> float:
        return self.omega_e_cm / (2.0 * self.omega_exe_cm) - 0.5


def morse_potential_cm(r_A: np.ndarray, spec: MorseSpec) -> np.ndarray:
    y = np.exp(-spec.a_Ainv * (r_A - spec.r_e_A))
    return spec.D_e_cm * (1.0 - y) ** 2


def kinetic_prefactor_cm_A2(mu_kg: float = MU_I2_KG) -> float:
    # H_kin = -K d^2/dR^2. R is in Angstrom here.
    return (HBAR**2 / (2.0 * mu_kg)) / (H * C_M_S * 100.0) * 1.0e20


def solve_vibrations(
    spec: MorseSpec,
    n_levels: int,
    r_min_A: float,
    r_max_A: float,
    n_grid: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grid, eigenvalues, and eigenvectors for the Morse Hamiltonian.

    Eigenvectors are column-wise and are normalized so sum(vec**2) = 1. For two
    wavefunctions on the same grid, dot(vec_i, vec_j) is the overlap integral.
    """
    if n_grid < 100:
        raise ValueError("n_grid is too small; use at least 100")
    if r_min_A >= r_max_A:
        raise ValueError("r_min_A must be < r_max_A")

    r = np.linspace(r_min_A, r_max_A, n_grid)
    dr = r[1] - r[0]
    K = kinetic_prefactor_cm_A2(spec.mu_kg)
    V = morse_potential_cm(r, spec)
    diag = V + 2.0 * K / dr**2
    off = np.full(n_grid - 1, -K / dr**2)

    # Request exactly the low-lying states needed.
    evals, evecs = eigh_tridiagonal(
        diag,
        off,
        select="i",
        select_range=(0, n_levels - 1),
        check_finite=False,
    )
    return r, evals, evecs


def compute_fcf_matrix(
    x_spec: MorseSpec,
    b_spec: MorseSpec,
    max_v_lower: int,
    max_v_upper: int,
    r_min_A: float,
    r_max_A: float,
    n_grid: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_x = max_v_lower + 1
    n_b = max_v_upper + 1
    r_x, e_x, psi_x = solve_vibrations(x_spec, n_x, r_min_A, r_max_A, n_grid)
    r_b, e_b, psi_b = solve_vibrations(b_spec, n_b, r_min_A, r_max_A, n_grid)
    if not np.allclose(r_x, r_b):
        raise RuntimeError("Internal error: X and B grids differ")
    # psi_b[:, v_upper].T @ psi_x[:, v_lower]
    overlaps = psi_b.T @ psi_x
    fcf = overlaps**2
    return fcf, e_x, e_b, r_x


def rotational_energy(B: float, D: float, J: int, Hrot: float = 0.0) -> float:
    jj = J * (J + 1)
    return B * jj - D * jj**2 + Hrot * jj**3


def transition_wavenumber(v_upper: int, v_lower: int, branch: str, J_lower: int) -> Tuple[float, int]:
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


def lower_state_energy_cm(v_lower: int, J_lower: int) -> float:
    lower = X_STATE[v_lower]
    return lower["E"] + rotational_energy(lower["B"], lower["D"], J_lower, 0.0)


def nuclear_spin_weight_i2_x_state(J_lower: int) -> int:
    # 127I has I = 5/2. For X 1Sigma_g+ of homonuclear I2:
    # even J'' -> antisymmetric nuclear spin manifold = 15 states
    # odd J''  -> symmetric nuclear spin manifold = 21 states
    return 15 if (J_lower % 2 == 0) else 21


def honl_london_0_to_0(branch: str, J_lower: int) -> float:
    # Normalized Hönl-London factor for a parallel Omega=0 <- Omega=0 band.
    # Multiplying by lower rotational degeneracy (2J+1) gives J for P and J+1 for R.
    denom = 2.0 * J_lower + 1.0
    if branch == "P":
        return J_lower / denom
    if branch == "R":
        return (J_lower + 1.0) / denom
    raise ValueError("branch must be 'P' or 'R'")


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


def iter_transition_rows(
    *,
    max_j: int,
    mode: str,
    branches: str,
    cap_upper_j: bool,
    temperature_K: float,
    fcf: np.ndarray,
    include_nuclear_spin: bool,
    normalize_to: float | None,
) -> Iterator[Dict[str, object]]:
    branch_list = {"both": ["P", "R"], "P": ["P"], "R": ["R"]}[branches]
    kT_cm = K_B_OVER_HC_CM * temperature_K

    for v_upper, v_lower in iter_band_pairs(mode):
        q_vv = float(fcf[v_upper, v_lower])
        for branch in branch_list:
            j_start = 1 if branch == "P" else 0
            for J_lower in range(j_start, max_j + 1):
                wn, J_upper = transition_wavenumber(v_upper, v_lower, branch, J_lower)
                if cap_upper_j and J_upper > max_j:
                    continue
                if wn <= 0.0:
                    continue

                E_lower = lower_state_energy_cm(v_lower, J_lower)
                boltz = math.exp(-E_lower / kT_cm)
                g_ns = nuclear_spin_weight_i2_x_state(J_lower) if include_nuclear_spin else 1
                lower_rot_degeneracy = 2 * J_lower + 1
                HL = honl_london_0_to_0(branch, J_lower)
                rot_line_factor = lower_rot_degeneracy * HL

                # Relative integrated absorption strength under constant electronic transition moment.
                # The wavenumber factor is appropriate for oscillator-strength-like absorption scaling.
                strength = wn * q_vv * g_ns * rot_line_factor * boltz
                strength_norm = strength / normalize_to if normalize_to else strength

                yield {
                    "v_upper": v_upper,
                    "v_lower": v_lower,
                    "branch": branch,
                    "J_lower": J_lower,
                    "J_upper": J_upper,
                    "wavenumber_cm-1": f"{wn:.6f}",
                    "frequency_THz": f"{wn * CM_TO_THz:.9f}",
                    "wavelength_vac_nm": f"{1e7 / wn:.9f}",
                    "E_lower_cm-1": f"{E_lower:.6f}",
                    "FCF_Morse": f"{q_vv:.12e}",
                    "boltzmann": f"{boltz:.12e}",
                    "nuclear_spin_weight": g_ns,
                    "lower_rot_degeneracy": lower_rot_degeneracy,
                    "honl_london_factor": f"{HL:.12e}",
                    "rot_line_factor": f"{rot_line_factor:.12e}",
                    "relative_strength": f"{strength:.12e}",
                    "relative_strength_norm": f"{strength_norm:.12e}",
                }


def write_fcf_csv(path: Path, fcf: np.ndarray, e_x: np.ndarray, e_b: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "v_upper",
                "v_lower",
                "FCF_Morse",
                "E_B_morse_cm-1",
                "E_X_morse_cm-1",
            ],
        )
        writer.writeheader()
        for v_upper in range(fcf.shape[0]):
            for v_lower in range(fcf.shape[1]):
                writer.writerow(
                    {
                        "v_upper": v_upper,
                        "v_lower": v_lower,
                        "FCF_Morse": f"{fcf[v_upper, v_lower]:.12e}",
                        "E_B_morse_cm-1": f"{e_b[v_upper]:.6f}",
                        "E_X_morse_cm-1": f"{e_x[v_lower]:.6f}",
                    }
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate I2 B-X line positions and approximate Morse-FCF relative strengths."
    )
    p.add_argument("--max-j", type=int, default=150, help="maximum lower-state J'' to include")
    p.add_argument("--mode", choices=["all", "analyzed"], default="all")
    p.add_argument("--branches", choices=["both", "P", "R"], default="both")
    p.add_argument("--cap-upper-j", action="store_true", help="also require J' <= --max-j")
    p.add_argument("--temperature", type=float, default=300.0, help="rotational/vibrational temperature in K")
    p.add_argument("--output", default="iodine_bx_morse_strengths_J150_T300.csv")
    p.add_argument("--fcf-output", default="", help="optional CSV path for the Morse FCF matrix")
    p.add_argument("--ngrid", type=int, default=3500, help="number of radial grid points")
    p.add_argument("--r-min", type=float, default=1.5, help="minimum internuclear distance in Angstrom")
    p.add_argument("--r-max", type=float, default=8.5, help="maximum internuclear distance in Angstrom")
    p.add_argument("--no-nuclear-spin", action="store_true", help="omit iodine nuclear-spin statistical weights")
    p.add_argument("--no-normalize", action="store_true", help="do not normalize strengths to max=1")

    # Morse-parameter overrides for sensitivity testing.
    p.add_argument("--x-omega-e", type=float, default=214.5186)
    p.add_argument("--x-omega-exe", type=float, default=0.6072284)
    p.add_argument("--x-Be", type=float, default=0.037368670)
    p.add_argument("--b-omega-e", type=float, default=125.6724)
    p.add_argument("--b-omega-exe", type=float, default=0.752677)
    p.add_argument("--b-Be", type=float, default=0.02999694599)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_j < 0:
        raise SystemExit("--max-j must be non-negative")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive")

    x_spec = MorseSpec("X", args.x_omega_e, args.x_omega_exe, args.x_Be)
    b_spec = MorseSpec("B", args.b_omega_e, args.b_omega_exe, args.b_Be)

    max_v_lower = max(X_STATE)
    max_v_upper = max(B_STATE)
    if b_spec.vmax_morse < max_v_upper:
        raise SystemExit(
            f"B-state Morse potential supports only about v <= {b_spec.vmax_morse:.1f}, "
            f"but v'={max_v_upper} is requested. Decrease --b-omega-exe or increase --b-omega-e."
        )
    if x_spec.vmax_morse < max_v_lower:
        raise SystemExit(
            f"X-state Morse potential supports only about v <= {x_spec.vmax_morse:.1f}, "
            f"but v''={max_v_lower} is requested."
        )

    print("Morse parameters used:")
    for spec in (x_spec, b_spec):
        print(
            f"  {spec.name}: omega_e={spec.omega_e_cm:.6f} cm^-1, "
            f"omega_e*x_e={spec.omega_exe_cm:.6f} cm^-1, "
            f"D_e={spec.D_e_cm:.3f} cm^-1, r_e={spec.r_e_A:.5f} A, "
            f"a={spec.a_Ainv:.5f} A^-1, vmax~{spec.vmax_morse:.1f}"
        )

    print("Solving Morse vibrational wavefunctions...")
    fcf, e_x, e_b, _ = compute_fcf_matrix(
        x_spec=x_spec,
        b_spec=b_spec,
        max_v_lower=max_v_lower,
        max_v_upper=max_v_upper,
        r_min_A=args.r_min,
        r_max_A=args.r_max,
        n_grid=args.ngrid,
    )

    if args.fcf_output:
        write_fcf_csv(Path(args.fcf_output), fcf, e_x, e_b)
        print(f"Wrote FCF matrix to {args.fcf_output}")

    # First pass to get max strength for normalized output. For 186k rows this is fine.
    rows = list(
        iter_transition_rows(
            max_j=args.max_j,
            mode=args.mode,
            branches=args.branches,
            cap_upper_j=args.cap_upper_j,
            temperature_K=args.temperature,
            fcf=fcf,
            include_nuclear_spin=not args.no_nuclear_spin,
            normalize_to=None,
        )
    )
    strengths = [float(r["relative_strength"]) for r in rows]
    max_strength = max(strengths) if strengths else None
    normalize_to = None if args.no_normalize else max_strength

    if normalize_to:
        # Recompute rows with normalized strengths to avoid formatting/rounding from first pass.
        rows = list(
            iter_transition_rows(
                max_j=args.max_j,
                mode=args.mode,
                branches=args.branches,
                cap_upper_j=args.cap_upper_j,
                temperature_K=args.temperature,
                fcf=fcf,
                include_nuclear_spin=not args.no_nuclear_spin,
                normalize_to=normalize_to,
            )
        )

    fieldnames = [
        "v_upper",
        "v_lower",
        "branch",
        "J_lower",
        "J_upper",
        "wavenumber_cm-1",
        "frequency_THz",
        "wavelength_vac_nm",
        "E_lower_cm-1",
        "FCF_Morse",
        "boltzmann",
        "nuclear_spin_weight",
        "lower_rot_degeneracy",
        "honl_london_factor",
        "rot_line_factor",
        "relative_strength",
        "relative_strength_norm",
    ]

    out = Path(args.output)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} transitions to {out}")
    print("Strength model: wavenumber * FCF * nuclear-spin weight * rotational line factor * Boltzmann population.")
    if normalize_to:
        print(f"Normalized relative_strength_norm to max=1; max unnormalized strength = {normalize_to:.12e}")
    print("Wavelengths are vacuum wavelengths. J_lower is J'' in the X state.")


if __name__ == "__main__":
    main()
