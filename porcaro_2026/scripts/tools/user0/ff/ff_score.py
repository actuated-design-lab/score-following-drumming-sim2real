"""ff_score.py — FF と RL を同じ式で採点する（Isaac も GPU も使わない）

打点の判定は analysis/harness/strike_extract.py の extract_strikes（contact_event、
力 1 N 以上かつ |誤差| ≤ 30 ms、1 対 1 の対応付け）。打撃は「0.5 N 未満に離れてから 1 N 以上に当たり、また 0.5 N 未満に
離れる」接触で、接触が 100 ms を超えるもの（押し付け）は数えない（2026-11-03。peak_match は押し付けたまま力を上下させる
FF の解を打撃として数えてしまったため）。旧い数字の再現は METHOD = "peak_match"。実機ログは jetson_project/analysis/strike_metrics.py の
extract_strikes_real（同じ判定、実機の時刻の扱いだけ違う）を使う。

summarize() は成功率に加えて、20 N に対する力の誤差（R1 の要求）と失敗の内訳を返す:
  success        成功（±30 ms かつ 1 N 以上）の割合
  no_strike      対応する打撃が無い（1 N 未満）割合
  late / early   打ったが ±30 ms の外（遅い／早い）の割合。success + no_strike + late + early = 1
  err_ms_*       打った打点の符号つきタイミング誤差（正 = 遅い）
  F_med / F_iqr  打った打点のピーク力 [N]
  dF_abs_med     打った打点の |ピーク力 − 20 N| の中央値 [N]
  dF_rmse_all    全目標の力の誤差の RMSE（打っていない目標は 0 N として数える）[N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from analysis.harness.strike_extract import extract_strikes  # noqa: E402

TOL_MS, MIN_FRAC, TARGET = 30.0, 0.05, 20.0
METHOD, RELEASE_N, MAX_CONTACT_S = "contact_event", 0.5, 0.100
# 押し付けの判定（2026-11-04 本人の決定）：離れが TRAIN_GAP_S 未満でつながった接触の連なりが 100 ms を超えたら、中の接触をまとめて
# 打撃に数えない（strike_extract.mask_long_trains）。採点は 40 ms。FF の目的関数の押し付けの罰は厳しめの 60 ms（ff_opt.py）。
# TRAIN_GAP_S = 0 で 11/03 の採点（contact_event だけ）を再現する
TRAIN_GAP_S = 0.040
PRESS_TRAIN_GAP_S = 0.060


def excess_contact_frac(t, f, on_n: float = 1.0, release_n: float = RELEASE_N, max_contact_s: float = MAX_CONTACT_S,
                        train_gap_s: float = 0.0) -> float:
    """押し付けの量：接触（on_n 以上で始まり release_n 未満で終わる）のうち、max_contact_s を超えた時間の合計 ÷ 全体の時間。
    RL の報酬の contact_limit（接触が長すぎる分への罰）に当たる。FF の目的関数の罰に使う（2026-11-03）。
    train_gap_s > 0 なら、離れが train_gap_s 未満でつながった接触（on_n 以上の区間）の連なりを 1 つの接触として数える（2026-11-04。
    sim の跳ねの連続は 1 回ずつは 5 ms なので、0 のままだと罰が効かない）"""
    t = np.asarray(t, float); f = np.asarray(f, float)
    if len(t) < 2:
        return 0.0
    dt = float(np.median(np.diff(t)))
    if train_gap_s > 0:
        above = f >= on_n
        segs, i, n = [], 0, len(f)
        while i < n:
            if above[i]:
                j = i
                while j < n and above[j]:
                    j += 1
                segs.append([i, j]); i = j
            else:
                i += 1
        if not segs:
            return 0.0
        tr = [segs[0][:]]
        for s0, e0 in segs[1:]:
            if t[s0] - t[tr[-1][1] - 1] < train_gap_s:
                tr[-1][1] = e0
            else:
                tr.append([s0, e0])
        excess = sum(max(0.0, (e0 - s0) * dt - max_contact_s) for s0, e0 in tr)
        return excess / (len(t) * dt)
    inside, start, excess = False, 0, 0.0
    for i in range(len(f)):
        if not inside and f[i] >= on_n:
            inside, start = True, i
        elif inside and f[i] < release_n:
            inside = False
            excess += max(0.0, (i - start) * dt - max_contact_s)
    if inside:
        excess += max(0.0, (len(f) - start) * dt - max_contact_s)
    return excess / (len(t) * dt)


def score_sim(ts: pd.DataFrame, bpm: float) -> pd.DataFrame:
    """ts: time_s, target_force, force_z（simulation_log.csv と同じ列）"""
    return extract_strikes(ts, bpm=bpm, target_ref=TARGET, tol_ms=TOL_MS, min_strike_frac=MIN_FRAC,
                           method=METHOD, release_n=RELEASE_N, max_contact_s=MAX_CONTACT_S, train_gap_s=TRAIN_GAP_S)


def summarize(st: pd.DataFrame) -> dict:
    n = len(st)
    if n == 0:
        return dict(n=0)
    hit = st.peak_force >= MIN_FRAC * TARGET
    e = st.timing_err_ms.where(hit)
    F = st.peak_force.where(hit)
    dF_all = np.where(hit, st.peak_force - TARGET, -TARGET)
    q1, q3 = (F.quantile(0.25), F.quantile(0.75)) if hit.any() else (np.nan, np.nan)
    return dict(
        n=n,
        success=float(st.success.mean()),
        no_strike=float((~hit).mean()),
        late=float((hit & (e > TOL_MS)).mean()),
        early=float((hit & (e < -TOL_MS)).mean()),
        err_ms_mean=float(e.mean()), err_ms_sd=float(e.std()), err_ms_med=float(e.median()),
        F_med=float(F.median()), F_iqr=float(q3 - q1),
        dF_abs_med=float((F - TARGET).abs().median()),
        dF_rmse_all=float(np.sqrt(np.mean(dF_all ** 2))),
    )


def fmt_row(d: dict) -> dict:
    """summary.md 用に丸める"""
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 3) if k in ("success", "no_strike", "late", "early") else round(v, 1)
        else:
            out[k] = v
    return out
