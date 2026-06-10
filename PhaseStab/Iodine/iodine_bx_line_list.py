#!/usr/bin/env python3
"""
Generate I2 B-X transition line positions from Luc (1980), Tables IV/V.

Units:
  - wavenumber_cm-1: cm^-1
  - frequency_THz: THz = c * wavenumber
  - wavelength_vac_nm: vacuum wavelength in nm = 1e7 / wavenumber_cm-1

Branch convention:
  J_lower is J'' in the X state, matching the paper's R/P branch formula.
  R(J''): Delta J = +1, so J_upper = J'' + 1.
  P(J''): Delta J = -1, so J_upper = J'' - 1 and J'' >= 1.

Default output:
  all v' = 1..62 and v'' = 0..9 combinations, both P and R branches,
  with J'' through --max-j, default 150.

Use --mode analyzed to restrict bands to the 139 analyzed bands in Table III
(excluding the (13,0) blend noted in Table II).
"""

import argparse
import csv
from pathlib import Path

C_CM_PER_S = 2.99792458e10
CM_TO_THz = C_CM_PER_S / 1e12  # 0.0299792458 THz per cm^-1


# Ground X state constants from Table IV.
# E_X is relative to X(v''=0,J''=0), in cm^-1.
# D_X values are stored in cm^-1; Table IV reports 10^9 * D.
X_STATE = {
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


# Excited B state constants from Table V.
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

B_STATE = {
    v: {"sigma0": sigma0, "B": b_scaled * 1e-2, "D": d_scaled * 1e-8, "H": h_scaled * h_scale}
    for v, sigma0, b_scaled, d_scaled, h_scaled, h_scale in B_STATE_RAW
}


# Table III analyzed bands, with the (13,0) blend from Table II excluded.
ANALYZED_BANDS = {
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


def rotational_energy(B: float, D: float, J: int, H: float = 0.0) -> float:
    """Rotational term value F_v(J), in cm^-1."""
    jj = J * (J + 1)
    return B * jj - D * jj**2 + H * jj**3


def transition_wavenumber(v_upper: int, v_lower: int, branch: str, J_lower: int) -> tuple[float, int]:
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


def iter_band_pairs(mode: str):
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


def iter_transitions(max_j: int, mode: str, branches: str, cap_upper_j: bool):
    branch_list = {"both": ["P", "R"], "P": ["P"], "R": ["R"]}[branches]
    for v_upper, v_lower in iter_band_pairs(mode):
        for branch in branch_list:
            j_start = 1 if branch == "P" else 0
            for J_lower in range(j_start, max_j + 1):
                wn, J_upper = transition_wavenumber(v_upper, v_lower, branch, J_lower)
                if cap_upper_j and J_upper > max_j:
                    continue
                if wn <= 0:
                    continue
                frequency_THz = wn * CM_TO_THz
                wavelength_nm = 1e7 / wn
                yield {
                    "v_upper": v_upper,
                    "v_lower": v_lower,
                    "branch": branch,
                    "J_lower": J_lower,
                    "J_upper": J_upper,
                    "wavenumber_cm-1": f"{wn:.6f}",
                    "frequency_THz": f"{frequency_THz:.9f}",
                    "wavelength_vac_nm": f"{wavelength_nm:.9f}",
                }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate I2 B-X P/R branch transition line list from Luc (1980), Tables IV/V."
    )
    parser.add_argument("--max-j", type=int, default=150, help="maximum lower-state J'' to include")
    parser.add_argument(
        "--mode",
        choices=["all", "analyzed"],
        default="all",
        help="'all' gives all v'=1..62 and v''=0..9 combinations; "
             "'analyzed' restricts to Table III analyzed bands",
    )
    parser.add_argument("--branches", choices=["both", "P", "R"], default="both")
    parser.add_argument(
        "--cap-upper-j",
        action="store_true",
        help="also require J' <= --max-j; otherwise R(max_j) has J'=max_j+1",
    )
    parser.add_argument(
        "--output",
        default="i2_bx_transitions_J150.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    if args.max_j < 0:
        raise SystemExit("--max-j must be non-negative")

    rows = list(iter_transitions(args.max_j, args.mode, args.branches, args.cap_upper_j))
    fieldnames = [
        "v_upper",
        "v_lower",
        "branch",
        "J_lower",
        "J_upper",
        "wavenumber_cm-1",
        "frequency_THz",
        "wavelength_vac_nm",
    ]

    output = Path(args.output)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} transitions to {output}")
    print("Wavelengths are vacuum wavelengths. J_lower is J'' in the X state.")


if __name__ == "__main__":
    main()
