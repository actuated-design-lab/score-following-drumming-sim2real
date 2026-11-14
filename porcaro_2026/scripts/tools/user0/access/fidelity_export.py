"""fidelity_export.py — 忠実度の図（論文 Fig. fidelity）の元データ：同じ指令の実機と sim（Gen2、DR なし）の時系列をそろえて書き出す（2026-11-05）

入力：data/user0/jfps2026/replay/gen2_check/sim_<動作>__gen2_cfg.csv（Gen2 の圧力モデル＋通信の遅れ 8 ms、指令 → 圧力 → 角度の通し）と
      sim_<動作>__gen2_measured.csv（実測の圧力をそのまま入れる。生の実測、平滑化なし）。どちらも replay_open_loop.py の出力（check_gen2.sh・run_1105_pkg.sh fidelity）
      実機：jetson_project の playback_20261005 のログを、エコー（flag）で指令の時間軸にそろえる（real_log.align_real_log、compare_hys_modes.py と同じ）
出力：data/user0/eval/access_1104/pkg/fidelity_ts_<動作>.csv（5 ms 刻み）
  time_s, cmd_DF/F/G（指令 [MPa]）, real_P_DF/F/G（実測の圧力 [MPa]、筋の名前どおり）, real_wrist_deg, real_grip_deg,
  simcfg_P_DF/F/G（sim の圧力モデルの出口圧）, simcfg_wrist_deg, simcfg_grip_deg, simmeas_wrist_deg, simmeas_grip_deg
角度は実機の向き。sim のない列は空

  python scripts/tools/user0/access/fidelity_export.py [--motions tm_A_quasistatic tm_B_steps tm_C_sine tm_E_dbl160_seed2 tm_E_gmd138_seed2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/jfps2026"))
import real_log as RL  # noqa: E402

DT = 0.005
ap = argparse.ArgumentParser()
ap.add_argument("--jetson", default=str(ROOT.parent / "jetson_project"))
ap.add_argument("--dir", default="data/user0/jfps2026/replay/gen2_check")
ap.add_argument("--motions", nargs="+", default=["tm_A_quasistatic", "tm_B_steps", "tm_C_sine", "tm_E_dbl160_seed2", "tm_E_gmd138_seed2"])
ap.add_argument("--out", default="data/user0/eval/access_1104/pkg")
a = ap.parse_args()
out = ROOT / a.out
out.mkdir(parents=True, exist_ok=True)
for m in a.motions:
    sims = {}
    for tag in ("gen2_cfg", "gen2_measured"):
        f = ROOT / a.dir / f"sim_{m}__{tag}.csv"
        if f.exists():
            sims[tag] = pd.read_csv(f)
    if not sims:
        print(f"[skip] {m}：sim の csv がない（{a.dir}）")
        continue
    meta = json.load(open(ROOT / a.dir / f"sim_{m}__{next(iter(sims))}.json", encoding="utf-8"))
    cmd = pd.read_csv(RL.jetson_signal_path(a.jetson, m))
    c3 = cmd[["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].values
    rl = meta.get("real_log")
    path = rl if rl and os.path.exists(rl) else RL.find_real_log(a.jetson, m, os.path.basename(rl or "") or None)
    A, lag = RL.align_real_log(pd.read_csv(path), c3, DT, columns=RL.PRES_COLS + ("wrist_angle_deg", "grip_angle_deg"))
    n = min([len(A)] + [len(s) for s in sims.values()])
    t = np.arange(n) * DT
    D = pd.DataFrame({"time_s": t})
    ci = np.clip((t / 0.02).astype(int), 0, len(c3) - 1)
    for j, c in enumerate(("DF", "F", "G")):
        D[f"cmd_{c}"] = c3[ci, j]
    for j, c in enumerate(("DF", "F", "G")):
        D[f"real_P_{c}"] = A[:n, j]
    D["real_wrist_deg"], D["real_grip_deg"] = A[:n, 3], A[:n, 4]
    if "gen2_cfg" in sims:
        s = sims["gen2_cfg"]
        for c in ("DF", "F", "G"):
            D[f"simcfg_P_{c}"] = s[f"P_out_{c}"].values[:n]
        D["simcfg_wrist_deg"], D["simcfg_grip_deg"] = s.wrist_angle_deg.values[:n], s.grip_angle_deg.values[:n]
    if "gen2_measured" in sims:
        s = sims["gen2_measured"]
        D["simmeas_wrist_deg"], D["simmeas_grip_deg"] = s.wrist_angle_deg.values[:n], s.grip_angle_deg.values[:n]
    D.to_csv(out / f"fidelity_ts_{m}.csv", index=False, float_format="%.5g")
    print(f"[ok] {m}：{n} 行、実機ログ {os.path.basename(path)}（エコーのずれ {lag * 5} ms）、sim {sorted(sims)}")
