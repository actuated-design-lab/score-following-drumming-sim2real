"""sim_dr_summary.py — DR あり・新しい判定の sim 評価（論文 §6.2）：F0・E・B × 実機コアの 5 タスク × DR 32 回の集計（2026-11-05）

入力：data/user0/eval/access_1104/sim/dr2_<方法>_<タスク>/strikes.csv（sim_trace.py --n_dr 32 --dr_seed 7000、run_1105_pkg.sh dr）
  採点は ff_score.score_sim（新しい判定 = contact_event ＋ 押し付けの連なり 40 ms）。env = DR の引き（eval_b と同じ引き方）
  F0 = opt2 r0 favorite（ff_lib.synth で MIDI から指令を作る）、E・B = b_eval/val/selection.json の ckpt
集計：引き（env）ごとの成功率 → 32 個の平均・SD・95 % CI（1.96·SD/√32）。5 タスクの平均は、引きごとに 5 タスクを平均してから同じように
出力：data/user0/eval/access_1104/pkg/sim_dr.csv（方法 × タスク）、sim_dr.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SIM = ROOT / "data/user0/eval/access_1104/sim"
OUT = ROOT / "data/user0/eval/access_1104/pkg"
TASKS = ["b_holdout_00_bpm90", "b_holdout_01_bpm96", "double_lead2_bpm160", "gmd_legacy_gmd_03_high_recut_bpm138",
         "gmd_legacy_gmd_04_extreme_recut_bpm170"]
rows, per_env = [], []
for M in ("F0", "E", "B"):
    for t in TASKS:
        f = SIM / f"dr2_{M}_{t}" / "strikes.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        e = s.groupby("env").success.mean()
        per_env.append(e.rename("success").reset_index().assign(method=M, task=t))
        rows.append(dict(method=M, task=t, n_draws=len(e), onsets_per_draw=int(len(s) / len(e)), success_mean=e.mean(), success_sd=e.std(),
                         ci95=1.96 * e.std() / np.sqrt(len(e))))
if not rows:
    raise SystemExit("dr2_* がない（run_1105_pkg.sh dr を先に）")
T = pd.DataFrame(rows)
P = pd.concat(per_env)
for M, x in P.groupby("method"):
    if x.task.nunique() == len(TASKS):
        m5 = x.groupby("env").success.mean()
        T = pd.concat([T, pd.DataFrame([dict(method=M, task="mean of 5 tasks", n_draws=len(m5), onsets_per_draw=np.nan, success_mean=m5.mean(),
                                             success_sd=m5.std(), ci95=1.96 * m5.std() / np.sqrt(len(m5)))])])
OUT.mkdir(parents=True, exist_ok=True)
T.to_csv(OUT / "sim_dr.csv", index=False, float_format="%.4f")
L = ["# DR あり・新しい判定の sim（sim_dr_summary.py）", "", "DR 32 回（dr_seed 7000）。成功率 = 引きごとの成功率の平均 ± 95 % CI", "",
     "| 方法 | タスク | 引き | 成功率 | SD |", "|---|---|---|---|---|"]
for r in T.itertuples():
    L.append(f"| {r.method} | {r.task} | {r.n_draws} | {r.success_mean:.3f} ± {r.ci95:.3f} | {r.success_sd:.3f} |")
open(OUT / "sim_dr.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
