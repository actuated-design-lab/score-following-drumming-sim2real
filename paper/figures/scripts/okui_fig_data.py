"""okui_fig_data.py — data for Fig. 6 (reproduction of a human performance, 2026-11-11)

Reads the two scored load-cell runs of okui2026 (MLP policy GEN2B/E, second run of each part) from a jetson_project
checkout and writes, with the same scoring as core_score (contact_event + 40 ms chain rule):
  figures/data/okui_sss_strikes.csv : part, onset, target_s, strike_s, timing_err_ms, peak_force_N, success
  figures/data/okui_sss_force.csv   : part, time_s, force_N (load cell), wrist_deg and grip_deg (joint angles, encoders, deg),
                                      pres_G_cmd (PAM-G pressure command sent by the policy, MPa; held for 20 ms),
                                      pres_G_meas (PAM-G measured pressure, MPa);
                                      5 ms steps, time = rate-corrected deploy time (same samples for force and angle)
Time is the robot's MIDI time (the 2.0 s lead of the task file included); make_figures.py shifts it so that the first
onset of the original performance (part A, onset 0) is at 0 s.

  python3 paper/figures/scripts/okui_fig_data.py --jetson ~/jetson_project
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

RUNS = {"A": "data/user0/okui2026/gen2b_20261111/deploy_okui_handA_bpm200_GEN2B-E_trial01_1794365787.csv",   # 02:56
        "B": "data/user0/okui2026/gen2b_20261111/deploy_okui_handB_bpm200_GEN2B-E_trial01_1794365998.csv"}   # 03:00
OUT = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetson", required=True, help="jetson_project checkout")
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.jetson))
    sys.path.insert(0, os.path.join(root, "tools"))
    sys.path.insert(0, os.path.join(root, "analysis"))
    import core_score  # noqa: E402
    from strike_metrics import apply_time_source  # noqa: E402
    S, F = [], []
    for part, rel in RUNS.items():
        f = os.path.join(root, rel)
        st, _ = core_score.score(f, core_score.classify(f))
        st = st.reset_index(drop=True)
        hit = st.peak_force >= core_score.THR_N
        S.append(pd.DataFrame({"part": part, "onset": st.strike_idx, "target_s": st.target_time,
                               "strike_s": st.peak_time.where(hit), "timing_err_ms": st.timing_err_ms.where(hit),
                               "peak_force_N": st.peak_force.where(hit), "success": st.success.astype(int)}))
        df = apply_time_source(pd.read_csv(f), "rate_corrected")
        F.append(pd.DataFrame({"part": part, "time_s": df.time.round(4), "force_N": df.force_N.round(3),
                               "wrist_deg": df.wrist_angle_deg.round(3),
                               "grip_deg": df.grip_angle_deg.round(3),
                               "pres_G_cmd": df.cmd_G.round(4),
                               "pres_G_meas": df.meas_pres_G.round(4)}))
        print(part, len(st), "onsets, success", round(st.success.mean(), 3))
    pd.concat(S).to_csv(OUT / "okui_sss_strikes.csv", index=False)
    pd.concat(F).to_csv(OUT / "okui_sss_force.csv", index=False)


if __name__ == "__main__":
    main()
