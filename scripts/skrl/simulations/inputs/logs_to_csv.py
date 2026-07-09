"""Convert a rig [LOOP] log dump into a tidy per-step CSV for the sim2real replay test.

Log format (two lines per control step, blank line between records):
    [LOOP] [FUNCTION] cart_pos_m = .., cart_vel_mps = .., cos = .., sin = .., pole_vel_radps = ..
    [LOOP] [NN] force_n = .., torque_nm = ..

Timing: the loop targets 100 Hz but the ACTUAL per-step dt varies slightly (the firmware divides
by the true micros() elapsed). Newer logs print `dt_s = ...` in the [FUNCTION] line -- when present
we record that real dt per row and make `t_s` its cumulative sum (accurate timestamps); otherwise
we fall back to a nominal 0.01 s. force_n on a record is the action COMPUTED FROM that record's
state and applied over the NEXT interval, so the causal transition for the replay test is:
    state[i]  +  force_n[i]  --(step dt_s[i])-->  state[i+1]
i.e. use row i's force_n (and row i's dt_s) to drive from row i to row i+1.

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

# grab every `key = number` pair on a line (tolerant of tabs / spaces)
_PAIR = re.compile(r"([A-Za-z_]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse(path: str) -> list[dict]:
    rows: list[dict] = []
    pending: dict | None = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if "[FUNCTION]" in line:
                vals = {k: float(v) for k, v in _PAIR.findall(line)}
                pending = vals
            elif "[NN]" in line and pending is not None:
                vals = {k: float(v) for k, v in _PAIR.findall(line)}
                pending.update(vals)
                rows.append(pending)
                pending = None
    return rows


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "logs.txt")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "logs.csv")

    records = parse(src)

    cols = [
        "step",
        "t_s",  # cumulative time (sum of dt_s if present, else step*0.01)
        "dt_s",  # ACTUAL per-step loop period (real dt from the rig, or 0.01 nominal)
        "cart_pos_m",
        "cart_vel_mps",
        "cos",
        "sin",
        "pole_angle_rad",  # atan2(sin, cos): 0 = upright, +/-pi = hanging down
        "pole_vel_radps",
        "force_n",  # action applied over [step, step+1]
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
            t_s = t_cum if has_dt else i * DT_S
            angle = math.atan2(r["sin"], r["cos"])
            w.writerow(
                [
                    i,
                    round(t_s, 6),
                    round(dt, 6),
                    r["cart_pos_m"],
                    r["cart_vel_mps"],
                    r["cos"],
                    r["sin"],
                    round(angle, 6),
                    r["pole_vel_radps"],
                    r["force_n"],
                    r["torque_nm"],
                ]
            )

    # quick sanity summary
    n = len(records)
    pv = [r["pole_vel_radps"] for r in records]
    cp = [r["cart_pos_m"] for r in records]
    dt_mean = sum(dts) / len(dts)
    print(f"parsed {n} steps -> {dst}  ({t_cum:.2f} s of run)")
    print(f"  cart_pos_m range : [{min(cp):+.4f}, {max(cp):+.4f}]")
    print(f"  pole_vel_radps   : [{min(pv):+.2f}, {max(pv):+.2f}]")
    if has_dt:
        print(
            f"  dt_s (REAL)      : mean {dt_mean * 1000:.2f} ms  min {min(dts) * 1000:.2f}  max "
            f"{max(dts) * 1000:.2f}  -> {1 / dt_mean:.1f} Hz avg"
        )
        print("  play_replay will read the mean dt_s and step the sim by it (matches the rig).")
    else:
        print(f"  dt_s not in log -> assumed nominal {DT_S * 1000:.0f} ms per step.")
    if abs(pv[0]) > 25:
        print(
            f"  NOTE: step 0 pole_vel = {pv[0]:+.2f} rad/s with cart at rest "
            f"-> startup finite-diff artifact; skip transition 0 in replay."
        )


if __name__ == "__main__":
    main()
