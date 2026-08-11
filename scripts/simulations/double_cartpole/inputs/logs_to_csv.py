"""Convert a DOUBLE cart-pole rig [LOOP] log dump into a tidy per-step CSV for the replay test.

Log format (two lines per control step):
    [LOOP] [FUNCTION] dt_s = .., cart_pos_m = .., cart_vel_mps = .., inner_cos = .., inner_sin = ..,
                      inner_angular_vel_radps = .., outer_cos = .., outer_sin = ..,
                      outer_angular_vel_radps = ..
    [LOOP] [NN] force_n = .., torque_nm = ..

Frames (verified against double_pendulum.cpp:108,129 and :371):
    the [FUNCTION] line prints double_pendulum_policy's PARAMETERS, and the firmware calls it as
    `-double_pendulum_policy(-pos_m, -cart_v_mps, ...) * 40`, so cart_pos_m / cart_vel_mps are
    already NEGATED (policy frame) while force_n is sensor frame. replay.py flips force_n to put
    everything in one frame -- same as the single. Angles: 0 = upright; outer is RELATIVE to inner.

force_n on a record is computed FROM that record's state and applied over the NEXT interval:
    state[i] + force_n[i] --(dt_s[i])--> state[i+1]

Usage:
    python logs_to_csv.py [src.txt] [dst.csv]   # defaults: logs.txt / logs.csv next to this script
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys

DT_S = 0.01  # 100 Hz control loop

_PAIR = re.compile(r"([A-Za-z_]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse(path: str) -> list[dict]:
    rows: list[dict] = []
    pending: dict | None = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if "[FUNCTION]" in line:
                pending = {k: float(v) for k, v in _PAIR.findall(line)}
            elif "[NN]" in line and pending is not None:
                pending.update({k: float(v) for k, v in _PAIR.findall(line)})
                rows.append(pending)
                pending = None
    return rows


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "logs.txt")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "logs.csv")

    records = parse(src)
    if not records:
        raise SystemExit(f"no [FUNCTION]/[NN] pairs found in {src}")

    cols = [
        "step",
        "t_s",
        "dt_s",
        "cart_pos_m",
        "cart_vel_mps",
        "inner_cos",
        "inner_sin",
        "inner_angle_rad",  # atan2(sin, cos): 0 = upright, +/-pi = hanging
        "inner_vel_radps",
        "outer_cos",
        "outer_sin",
        "outer_angle_rad",  # RELATIVE to the inner link
        "outer_vel_radps",
        "force_n",  # applied over [step, step+1]
        "torque_nm",
    ]

    has_dt = any("dt_s" in r for r in records)
    t_cum = 0.0
    dts = []
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(records):
            dt = float(r.get("dt_s", DT_S))
            dts.append(dt)
            t_cum += dt
            w.writerow(
                [
                    i,
                    round(t_cum if has_dt else i * DT_S, 6),
                    round(dt, 6),
                    r["cart_pos_m"],
                    r["cart_vel_mps"],
                    r["inner_cos"],
                    r["inner_sin"],
                    round(math.atan2(r["inner_sin"], r["inner_cos"]), 6),
                    r["inner_angular_vel_radps"],
                    r["outer_cos"],
                    r["outer_sin"],
                    round(math.atan2(r["outer_sin"], r["outer_cos"]), 6),
                    r["outer_angular_vel_radps"],
                    r["force_n"],
                    r["torque_nm"],
                ]
            )

    # ---- sanity summary ----
    n = len(records)
    col = lambda k: [r[k] for r in records]  # noqa: E731
    ia = [math.atan2(r["inner_sin"], r["inner_cos"]) for r in records]
    oa = [math.atan2(r["outer_sin"], r["outer_cos"]) for r in records]
    w1, w2 = col("inner_angular_vel_radps"), col("outer_angular_vel_radps")
    dt_mean = sum(dts) / len(dts)

    print(f"parsed {n} steps ({n - 1} transitions) -> {dst}  ({t_cum:.3f} s of run)")
    for name, v, unit in (
        ("cart_pos_m", col("cart_pos_m"), "m"),
        ("cart_vel_mps", col("cart_vel_mps"), "m/s"),
        ("inner_angle_rad", ia, "rad"),
        ("inner_vel_radps", w1, "rad/s"),
        ("outer_angle_rad", oa, "rad"),
        ("outer_vel_radps", w2, "rad/s"),
        ("force_n", col("force_n"), "N"),
    ):
        print(f"  {name:<16} [{min(v):+9.4f}, {max(v):+9.4f}] {unit}")

    if has_dt:
        print(
            f"  dt_s (REAL)      mean {dt_mean * 1000:.3f} ms  min {min(dts) * 1000:.3f}  max {max(dts) * 1000:.3f}"
            f"  -> {1 / dt_mean:.1f} Hz, spread {(max(dts) / min(dts) - 1) * 100:.1f}%"
        )
        print("  replay.py steps the sim at the MEAN dt (per-step dt is not applied).")
    else:
        print(f"  dt_s not in log -> assumed nominal {DT_S * 1000:.0f} ms per step.")

    # startup artifacts: the firmware's finite-diff velocity needs one prior call, and a stale
    # encoder read between two calls yields an exact 0.0
    skip = 0
    if abs(w1[0]) > 2.0 or abs(w2[0]) > 2.0:
        skip = 1
    if n > 1 and (w1[1] == 0.0 or w2[1] == 0.0):
        skip = 2
    if skip:
        print(
            f"  NOTE: rows 0..{skip - 1} look like finite-diff startup artifacts "
            f"(w1[0]={w1[0]:+.3f}, w1[1]={w1[1]:+.3f}) -> replay.py SKIP_FIRST={skip}"
        )

    alias = sum(1 for v in w2 if abs(v) > 20.0)
    if alias:
        step_deg = math.degrees(max(abs(v) for v in w2) * dt_mean)
        print(f"  NOTE: {alias}/{n} rows have |outer_vel| > 20 rad/s (peak moves {step_deg:.1f} deg/sample).")
        print("        The firmware's finite-diff estimator aliases there -- treat that regime as unreliable.")


if __name__ == "__main__":
    main()
