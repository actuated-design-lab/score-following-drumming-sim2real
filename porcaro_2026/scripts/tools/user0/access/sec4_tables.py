"""sec4_tables.py — IEEE Access 原稿の §4 以降に入れる確定データの表（2026-11-05）

入力（どれも git に入っている）：
  - jetson_project data/user0/access2026v3/core2026_score/{runs.csv, ff_weak_targets.csv}
      本表（RL = 11/09 の取り直し（圧力の枠を直したあと）、FF = 11/05 のやり直し、各方法 15 本 = 5 タスク × 3 反復）。採点は新しい判定（contact_event ＋ 押し付けの連なり 40 ms）
      失敗の内訳は ff_weak の kind（成功 / 早い / 遅い = 当たったが ±30 ms の外、押し付け = 当たりがなく、目標の ±1.5·T16 の窓が
      採点で外した接触の連なりと重なる、未打 = それ以外の当たりなし）
  - data/user0/ff/opt2/diag/diag.csv：sim（DR なし、公称、n_env 1）。B・E = 11/04 の sim を新しい判定で採点し直したもの、F0 = opt2 r0 favorite
  - data/user0/eval/access_1104/sim/nom_F0_<task>/trace.npz の onsets：前打間隔の区分を FF が指令を作るときと同じ方法（ff_lib.ioi_bins）で決める
  - data/user0/ff/opt2/r0/{favorite.json, summary.md}、data/user0/eval/b_eval2/val/selection.json

  python scripts/tools/user0/access/sec4_tables.py
出力：data/user0/eval/access_1104/sec4/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
import ff_lib as FL  # noqa: E402

J = ROOT.parent / "jetson_project"
SC = J / "data/user0/access2026v3/core2026_score"
OUT = ROOT / "data/user0/eval/access_1104/sec4"
M = ["F0", "F3", "E", "B"]
TS = {"b_holdout_00_bpm90": "holdout0（90 bpm）", "b_holdout_01_bpm96": "holdout1（96 bpm）", "double_lead2_bpm160": "double160（160 bpm）",
      "gmd_legacy_gmd_03_high_recut_bpm138": "gmd138（138 bpm）", "gmd_legacy_gmd_04_extreme_recut_bpm170": "gmd170（170 bpm）"}
BIN = ["< 120", "120–200", "200–320", "≥ 320（最初を含む）"]
KINDS = ["未打", "早い", "遅い", "押し付け"]

R = pd.read_csv(SC / "runs.csv")
R = R[~R.bridge.astype(bool)].copy()
R["rep"] = R.groupby(["method", "task"]).cumcount() + 1
T = pd.read_csv(SC / "ff_weak_targets.csv")
on = {t: np.load(ROOT / f"data/user0/eval/access_1104/sim/nom_F0_{t}/trace.npz")["onsets"] for t in TS}
T["ffbin"] = [FL.ioi_bins(on[t])[i] for t, i in zip(T.task, T.strike_idx)]
T["ioi_grid"] = [np.r_[np.inf, np.diff(on[t]) * 1000][i] for t, i in zip(T.task, T.strike_idx)]
D = pd.read_csv(ROOT / "data/user0/ff/opt2/diag/diag.csv")
sim = {"B": D[(D.source == "11/04 の sim") & (D.cand == "B")], "E": D[(D.source == "11/04 の sim") & (D.cand == "E")],
       "F0": D[(D.source == "opt2") & (D.cand == "r0_favorite")]}

L = ["# §4 以降の確定データ（sec4_tables.py）", ""]

# --------------------------------------------------------------------------- 本表
L += ["## 本表：方法 × タスクの成功率（実機、新しい判定）", "",
      "成功率 = 反復ごとの成功率の平均 ± SD（n = 3）。旧採点 = 押し付けの連なりなし。失敗の内訳は全打点（3 反復をまとめる）に占める割合", "",
      "| 方法 | タスク | 打点（3 反復） | 成功率 | 旧採点 | " + " | ".join(KINDS) + " |", "|---|---|---|---|---|" + "---|" * len(KINDS)]
K = T.groupby(["method", "task"]).kind.value_counts(normalize=True).unstack(fill_value=0)
for m in M:
    for t in TS:
        x = R[(R.method == m) & (R.task == t)]
        k = K.loc[(m, t)]
        n = len(T[(T.method == m) & (T.task == t)])
        L.append(f"| {m} | {TS[t]} | {n} | {x.success.mean():.3f} ± {x.success.std():.3f} | {x.success_old.mean():.3f} | "
                 + " | ".join(f"{k.get(c, 0):.3f}" for c in KINDS) + " |")
L += ["", "5 タスクの平均（タスクの重みを等しく。SD は反復ごとの 5 タスク平均の SD、反復 = タスクごとの実行順）", "",
      "| 方法 | 成功率 | 旧採点 | " + " | ".join(KINDS) + " |", "|---|---|---|" + "---|" * len(KINDS)]
for m in M:
    x = R[R.method == m]
    rm = x.groupby("rep").success.mean()
    k = K.loc[m].mean()
    L.append(f"| {m} | {x.groupby('task').success.mean().mean():.3f} ± {rm.std():.3f} | {x.groupby('task').success_old.mean().mean():.3f} | "
             + " | ".join(f"{k.get(c, 0):.3f}" for c in KINDS) + " |")

# --------------------------------------------------------------------------- sim
L += ["", "## sim の対応値（同じ 5 タスク）", "",
      "条件：DR なし（公称値）、env 1 本、通信の遅れ 8 ms（公称）、新しい判定（連なり 40 ms）。B・E は 11/04 の sim の時系列（run_1104.sh の nom_*）を"
      "diag_opt2.py で採点し直したもの、F0 は opt2 r0 favorite（diag_opt2.sh）。F3（Δ は実機の較正で決める）は sim で回していない", "",
      "| タスク | F0 sim | F0 実機 | E sim | E 実機 | B sim | B 実機 |", "|---|---|---|---|---|---|---|"]
rt = R.groupby(["method", "task"]).success.mean()
for t in TS:
    c = [f"{float(sim[m][sim[m].task == t].success.iloc[0]):.3f} | {rt[(m, t)]:.3f}" for m in ["F0", "E", "B"]]
    L.append(f"| {TS[t]} | " + " | ".join(c) + " |")
L.append("| 5 タスクの平均 | " + " | ".join(f"{sim[m].success.mean():.3f} | {rt[m].mean():.3f}" for m in ["F0", "E", "B"]) + " |")
sel = json.load(open(ROOT / "data/user0/eval/b_eval2/val/selection.json"))
L += ["", f"検証用の列（16 本 × DR 32 回、dr_seed 7000、新しい判定）：F0_r0 の成功率は `data/user0/eval/b_eval2/val/selection.md` の表（選択：{sel.get('F0', {}).get('run', sel.get('F0'))}）。"
      "RL（B・E）の検証用の列は新しい判定でまだ回し直していない（FF段2後の todo §4）"]

# --------------------------------------------------------------------------- 前打間隔
L += ["", "## 前打間隔（直前の打撃との間隔）別の成功率（実機）", "",
      "区分は FF が指令を作るときと同じ（ff_lib.ioi_bins：20 ms 格子の目標時刻の差、最初の打撃は一番長い区分）。成功率（打点数、15 本をまとめる）", "",
      "| 前打間隔 [ms] | " + " | ".join(M) + " |", "|---|" + "---|" * len(M)]
for b in range(4):
    x = T[T.ffbin == b]
    L.append(f"| {BIN[b]} | " + " | ".join(f"{x[x.method == m].success.mean():.3f}（{(x.method == m).sum()}）" for m in M) + " |")
L += ["", "休符明けと連続の中（休符明け = 直前の間隔 ≥ 400 ms、または曲の最初の打撃）", "",
      "| | " + " | ".join(M) + " |", "|---|" + "---|" * len(M)]
T["ctx"] = np.where(T.ioi_grid >= 400, "休符明け", "連続の中")
for c in ["休符明け", "連続の中"]:
    x = T[T.ctx == c]
    L.append(f"| {c} | " + " | ".join(f"{x[x.method == m].success.mean():.3f}（{(x.method == m).sum()}）" for m in M) + " |")
x = T[T.task != "double_lead2_bpm160"]
L += ["", "double160 を除く", "", "| | " + " | ".join(M) + " |", "|---|" + "---|" * len(M)]
for c in ["休符明け", "連続の中"]:
    y = x[x.ctx == c]
    L.append(f"| {c} | " + " | ".join(f"{y[y.method == m].success.mean():.3f}（{(y.method == m).sum()}）" for m in M) + " |")

OUT.mkdir(parents=True, exist_ok=True)
open(OUT / "tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
