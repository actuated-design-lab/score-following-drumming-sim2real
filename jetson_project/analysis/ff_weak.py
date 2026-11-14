"""ff_weak.py — 実機コアの本表（access2026v3：RL = 11/09 の取り直し、FF = 11/05 のやり直し。環境変数 FF_WEAK_SCORE で別の採点フォルダ）で、FF が弱くなる条件を探す（CPU）

採点は core_score と同じ（strikes.csv をそのまま使う。新しい判定 = contact_event ＋ 押し付けの連なり 40 ms）。
  1. 直前の打撃との間隔（IOI）別の成功率。区分は FF のプリミティブと同じ（120 / 200 / 320 ms、直前の IOI で決める。最初の打撃は一番長い区分）
     ＋ 細かい区分（80 / 100 / 120 / 160 / 200 / 280 / 320 / 480 ms）
  2. double160 の失敗の内訳：未打 / 押し付け / 早い / 遅い、1 打目（直前の IOI が長い）と 2 打目（直前の IOI が 16 分 = 94 ms）
     押し付け = 目標の ±1.5·T16 の窓が、採点で外した接触の連なり（離れ 40 ms 未満でつながって 100 ms を超える）と重なる
  3. 休符明け（直前の IOI ≥ 400 ms、または曲の最初の打撃）と、連続の中（< 400 ms）の成功率
  4. run ごとに前半・後半（目標時刻の中央で分ける）の成功率と、その差

  python3 analysis/ff_weak.py
出力：data/user0/access2026v3/core2026_score/ff_weak.md、ff_weak_targets.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import core_score as cs  # noqa: E402

SC = os.path.join(ROOT, os.environ.get("FF_WEAK_SCORE", "data/user0/access2026v3/core2026_score"))
METHODS = ["E", "F0", "F3", "B"]
BPM = {"b_holdout_00_bpm90": 90, "b_holdout_01_bpm96": 96, "double_lead2_bpm160": 160,
       "gmd_legacy_gmd_03_high_recut_bpm138": 138, "gmd_legacy_gmd_04_extreme_recut_bpm170": 170}
SHORT = {"b_holdout_00_bpm90": "holdout0", "b_holdout_01_bpm96": "holdout1", "double_lead2_bpm160": "double160",
         "gmd_legacy_gmd_03_high_recut_bpm138": "gmd138", "gmd_legacy_gmd_04_extreme_recut_bpm170": "gmd170"}


def long_trains(t, f, on=1.0, gap=cs.TRAIN_GAP_S, maxlen=0.100):
    above = f >= on
    segs, i, n = [], 0, len(f)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            segs.append([i, j]); i = j
        else:
            i += 1
    tr = []
    for s, e in segs:
        if tr and t[s] - t[tr[-1][1] - 1] < gap:
            tr[-1][1] = e
        else:
            tr.append([s, e])
    return [(t[s], t[e - 1]) for s, e in tr if t[e - 1] - t[s] > maxlen]


def table(g, by):
    T = g.groupby(by + ["method"]).success.agg(["mean", "size"]).unstack("method")
    out = pd.DataFrame(index=T.index)
    for m in METHODS:
        if ("mean", m) in T:
            out[m] = [f"{a:.3f}（{int(b)}）" if np.isfinite(a) else "" for a, b in zip(T[("mean", m)], T[("size", m)])]
    return out


def md(df):
    df = df.reset_index()
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main():
    R = pd.read_csv(os.path.join(SC, "runs.csv"))
    S = pd.read_csv(os.path.join(SC, "strikes.csv"))
    S = S[~S.bridge.astype(bool)] if "bridge" in S else S
    rows = []
    for f, g in S.groupby("file"):
        g = g.sort_values("target_time").copy()
        r = R[R.file == f].iloc[0]
        df = cs.apply_time_source(pd.read_csv(os.path.join(ROOT, f)), "rate_corrected") if r.kind_log == "deploy" \
            else cs.playback_time(pd.read_csv(os.path.join(ROOT, f)))
        tr = long_trains(df.time.to_numpy(float), df.force_N.to_numpy(float))
        t16 = 15.0 / BPM[r.task]
        tt = g.target_time.to_numpy(float)
        # 目標は 20 ms の格子の上にある。ログの時刻のゆらぎ（数 ms）で区分の境目（120・200・320 ms）をまたがないよう、20 ms に丸める
        g["ioi_prev_ms"] = np.r_[np.inf, np.round(np.diff(tt) / 0.02) * 20.0]
        g["ioi_next_ms"] = np.r_[np.round(np.diff(tt) / 0.02) * 20.0, np.inf]
        hit = g.peak_force.fillna(0).to_numpy() >= cs.THR_N
        press = np.array([any(a <= t + 1.5 * t16 and b >= t - 1.5 * t16 for a, b in tr) for t in tt])
        e = g.timing_err_ms.to_numpy(float)
        g["kind"] = np.where(g.success, "成功", np.where(hit & (e < 0), "早い", np.where(hit, "遅い", np.where(press, "押し付け", "未打"))))
        g["half"] = np.where(tt <= np.median(tt), "前半", "後半")
        g["rep"] = r.rep
        rows.append(g)
    D = pd.concat(rows, ignore_index=True)
    D["task_s"] = D.task.map(SHORT)
    D.to_csv(os.path.join(SC, "ff_weak_targets.csv"), index=False)
    L = ["# FF が弱くなる条件（ff_weak.py）", "",
         f"出典：{os.path.relpath(SC, ROOT)} の strikes.csv（本表、各方法 15 本）。成功率（打数）。採点は新しい判定", ""]

    # 1. IOI 別
    D["ioi_bin"] = pd.cut(D.ioi_prev_ms, [0, 120, 200, 320, np.inf], right=False, labels=["< 120", "120–200", "200–320", "≥ 320（最初を含む）"])
    D["ioi_fine"] = pd.cut(D.ioi_prev_ms, [0, 100, 120, 160, 200, 280, 320, 480, np.inf], right=False)
    L += ["## 1. 直前の IOI 別の成功率", "", "FF のプリミティブの区分（直前の IOI で決める）", "", md(table(D, ["ioi_bin"])), "",
          "細かい区分 [ms]", "", md(table(D, ["ioi_fine"])), "",
          "タスク × FF の区分", "", md(table(D, ["task_s", "ioi_bin"])), ""]

    # 2. double160 の失敗の内訳
    d = D[D.task_s == "double160"].copy()
    d["pos"] = np.where(d.ioi_prev_ms < 150, "2 打目（直前 94 ms）", "1 打目（直前 ≥ 281 ms か最初）")
    K = d.groupby(["method", "pos"]).kind.value_counts(normalize=True).unstack().fillna(0)
    K["打数"] = d.groupby(["method", "pos"]).size()
    cols = [c for c in ["成功", "未打", "押し付け", "早い", "遅い"] if c in K] + ["打数"]
    K = K[cols].loc[[m for m in METHODS if m in K.index.get_level_values(0)]]
    L += ["## 2. double160 の内訳（割合）", "", md(K.round(3)), ""]

    # 3. 休符明け
    D["ctx"] = np.where(D.ioi_prev_ms >= 400, "休符明け（直前 ≥ 400 ms か最初）", "連続の中（直前 < 400 ms）")
    L += ["## 3. 休符明けと連続の中", "", md(table(D, ["ctx"])), "", "double160 を除く", "", md(table(D[D.task_s != "double160"], ["ctx"])), ""]

    # 4. 前半・後半
    H = D.groupby(["method", "file", "half"]).success.mean().unstack("half")
    H["差（後半 − 前半）"] = H["後半"] - H["前半"]
    G = H.groupby("method").agg(前半=("前半", "mean"), 後半=("後半", "mean"), 差の平均=("差（後半 − 前半）", "mean"),
                                差の_SD=("差（後半 − 前半）", "std"), 下がった_run=("差（後半 − 前半）", lambda x: int((x < 0).sum())),
                                run=("差（後半 − 前半）", "size"))
    G = G.loc[[m for m in METHODS if m in G.index]]
    L += ["## 4. run の前半と後半（目標時刻の中央で分ける）", "", md(G.round(3)), ""]
    open(os.path.join(SC, "ff_weak.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
