"""core_score.py — 実機コア（core2026）のログを、元のタスクの MIDI の打撃時刻で採点する（sim の ff_score と同じ判定：±30 ms・1 N）

deploy_policy（B・E・RL+Δ）のログと run_signal_playback（F0・F3）のログを見分け、方法・タスク・反復を付けて採点する。
  - 目標は元のタスクの MIDI（songs/core2026/ か songs/core2026_calib/）から作り直す。RL+Δ はずらした MIDI で叩いたが、採点は元の時刻
  - 時刻：deploy は strike_metrics の rate_corrected、playback はログの time（開始の瞬間が t = 0）
  - 打点の判定：analysis/strike_metrics.extract_strikes_real（contact_event：0.5 N 未満に離れてから 1 N 以上に当たる接触、
    接触 100 ms 以内。押し付けたまま力が上下する山は数えない。2026-11-03 に peak_match から変更）
core_runner.py の記録（core2026_session.csv）で excluded = 1 の run は採点しない
--calib：較正の列のログだけを集計し、B と F0 の符号つきタイミング誤差の平均から Δ（20 ms に丸める）を出す

  python3 tools/core_score.py data/user0/<venue>/                 # 評価
  python3 tools/core_score.py data/user0/<venue>/ --calib         # 較正
出力：<フォルダ>/core2026_score/{runs.csv, strikes.csv, summary.md}（--calib は calib.md）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from midi_rhythm_generator import MidiRhythmGenerator  # noqa: E402
from strike_metrics import apply_time_source, extract_strikes_real  # noqa: E402

THR_N, TOL_MS, TARGET = 1.0, 30.0, 20.0
BRIDGE_TRIAL = 4      # 日の差の橋渡しの試行番号（porcaro_2026 real_core.py の BRIDGE_TRIAL と同じ）
# 打点の判定（2026-11-03）：contact_event = 0.5 N 未満に離れてから当たる接触だけ、接触 100 ms 以内（押し付けを数えない）。
# 環境変数 CORE_SCORE_METHOD=peak_match で旧い判定（11/03 夕方までの数字）を再現できる
METHOD = os.environ.get("CORE_SCORE_METHOD", "contact_event")
# 押し付けの判定（2026-11-04 本人の決定）：離れが 40 ms 未満でつながった接触の連なりが 100 ms を超えたら打撃に数えない（strike_metrics.mask_long_trains）。
# 本表はこの新しい判定を主にし、旧い採点（連なりなし = 11/03〜11/04 の数字）を success_old として併記する。CORE_SCORE_TRAIN_GAP_MS=0 で旧い採点だけ
TRAIN_GAP_S = float(os.environ.get("CORE_SCORE_TRAIN_GAP_MS", "40")) / 1000.0
SONGS, CALIB = os.path.join(ROOT, "songs/core2026"), os.path.join(ROOT, "songs/core2026_calib")
SSS = os.path.join(ROOT, "songs/sss2026")       # Sing Sing Sing（アレンジ実証の別枠、2026-11-04）。ログは venue sss2026 に分ける


def orig_midi(stem):
    """ずらした MIDI・指令の名前から、元のタスク（か較正）の MIDI を探す"""
    base = re.sub(r"_rld\d+", "", stem)
    for d, kind in ((SONGS, "task"), (CALIB, "calib"), (SSS, "task")):
        p = os.path.join(d, base + ".mid")
        if os.path.exists(p):
            return p, kind
    return None, None


def classify(csv):
    name = os.path.splitext(os.path.basename(csv))[0]
    js = os.path.splitext(csv)[0] + ".json"
    if name.startswith("deploy_") and os.path.exists(js):
        meta = json.load(open(js, encoding="utf-8"))
        if meta.get("mock") or meta.get("verify"):
            return None
        mk = meta.get("model_key", "")
        midi = os.path.basename(meta.get("midi", ""))
        if not mk.startswith("GEN2B/"):
            return None
        stem = os.path.splitext(midi)[0]
        method = "RL+Δ" if "_rld" in stem else mk.split("/")[1]
        return dict(kind_log="deploy", method=method, stem=stem, rep=meta.get("trial", ""), t_unix=int(name.rsplit("_", 1)[-1]))
    m = re.match(r"data_(F0|F3)_(.+)_(\d+)$", name)
    if m:
        return dict(kind_log="playback", method=m.group(1), stem=m.group(2), rep="", t_unix=int(m.group(3)))
    return None


def playback_time(df):
    """FF（run_signal_playback）のログの時刻を、RL（deploy_policy）と同じ定義にする（2026-11-03）。

    deploy_policy：受信時刻（Jetson の時計）の下側包絡に直線を当て、シリアルの転送時間を引いた「発生時刻」。原点は制御ループの開始。
    run_signal_playback の time 列は「同期のあとの最初のサンプルからの番号 × 5 ms」で、原点が数 ms ずれる（11/03 の F0 6 本で −3.4〜+1.9 ms）。
    ここでは timestamp_pc（受信時刻）と cmd_pc（指令を送った時刻）から、deploy と同じ式で発生時刻を作り、原点を最初の指令を送った時刻にする。
    列がないとき、または原点が 50 ms 以上ずれる（受信が途中で切れたログなど）ときは、元の time 列のまま。
    """
    if not {"timestamp_pc", "cmd_pc", "time"} <= set(df.columns) or len(df) < 10:
        return df
    k = np.arange(len(df))
    tr = df["timestamp_pc"].to_numpy(float)
    c = np.polyfit(k, tr, 1)
    env = np.percentile(tr - np.polyval(c, k), 2)
    xfer = 58 * 10 / 230400
    t0 = float(np.nanmin(df["cmd_pc"].to_numpy(float)))
    t = np.polyval(c, k) + env - xfer - t0
    if abs(np.median(t - df["time"].to_numpy(float))) > 0.050:
        print(f"[core_score] 時刻の原点が 50 ms 以上ずれるので、ログの time 列のまま（{np.median(t - df['time'].to_numpy(float)) * 1000:+.0f} ms）")
        return df
    df = df.copy()
    df["time"] = t
    return df


def score(csv, info):
    p, kind = orig_midi(info["stem"])
    if p is None:
        return None, None
    g = MidiRhythmGenerator(p, device=torch.device("cpu"), dt=0.02, target_force=TARGET, lookahead_steps=1)
    buf = g.trajectory_buffer.cpu().numpy() if hasattr(g.trajectory_buffer, "cpu") else np.asarray(g.trajectory_buffer)
    if buf.max() <= 1.5:
        buf = buf * TARGET
    df = pd.read_csv(csv)
    if info["kind_log"] == "deploy":
        df = apply_time_source(df, "rate_corrected")
    else:
        df = playback_time(df)
    t = df["time"].to_numpy(float)
    k = np.clip((t / 0.02).astype(int), 0, len(buf) - 1)
    df["target_force"] = np.where(t / 0.02 < len(buf), buf[k], 0.0)
    st = extract_strikes_real(df, bpm=float(g.bpm), target_ref=TARGET, force_threshold=THR_N, method=METHOD, train_gap_s=TRAIN_GAP_S)
    old = extract_strikes_real(df, bpm=float(g.bpm), target_ref=TARGET, force_threshold=THR_N, method=METHOD, train_gap_s=0.0)
    st["success_old"] = old.success.to_numpy() if len(old) == len(st) else np.nan
    return st, kind


def summ(st):
    hit = st.peak_force >= THR_N
    e = st.timing_err_ms.where(hit)
    F = st.peak_force.where(hit)
    dF_all = np.where(hit, st.peak_force - TARGET, -TARGET)
    return dict(n=len(st), success=st.success.mean(), no_strike=(~hit).mean(), late=(hit & (e > TOL_MS)).mean(),
                early=(hit & (e < -TOL_MS)).mean(), err_ms_mean=e.mean(), err_ms_sd=e.std(), F_med=F.median(),
                F_iqr=F.quantile(0.75) - F.quantile(0.25), dF_abs_med=(F - TARGET).abs().median(),
                dF_rmse_all=float(np.sqrt(np.mean(dF_all ** 2))))     # 四分位幅と全目標の RMSE は sim の ff_score.summarize と同じ定義


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="採点から外すログ（パスの一部でよい。例：access2026/playback_20261104 で 11/04 の旧い FF を外す）")
    a = ap.parse_args()
    files = sorted({f for d in a.dirs for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)})
    # core_runner.py の記録で除外した run（QC HARD、直後のチェックの FAIL、オペレータの取り直し）は採点しない
    excl = set()
    sp = os.path.join(ROOT, "core2026_session.csv")
    if os.path.exists(sp):
        ss = pd.read_csv(sp, dtype=str).fillna("")
        excl = {os.path.normpath(os.path.join(ROOT, x)) for x in ss[ss.excluded == "1"].log if x}
    runs, strikes = [], []
    n_ex = 0
    for f in files:
        if os.path.normpath(os.path.abspath(f)) in excl or any(x in os.path.relpath(f, ROOT) for x in a.exclude):
            n_ex += 1
            continue
        info = classify(f)
        if info is None:
            continue
        st, kind = score(f, info)
        if st is None or st.empty:
            continue
        if (kind == "calib") != a.calib:
            continue
        info.update(file=os.path.relpath(f, ROOT), task=info["stem"], **summ(st), success_old=float(st.success_old.mean()))
        # 日の差の橋渡し（11/04、FF フェーズに差し込んだ B・E。試行番号 4）：本表には入れず、別の表にする
        info["bridge"] = bool(info["kind_log"] == "deploy" and str(info.get("rep", "")).strip() == str(BRIDGE_TRIAL))
        runs.append(info)
        st = st.assign(method=info["method"], task=info["stem"], file=info["file"], bridge=info["bridge"])
        strikes.append(st)
    if not runs:
        raise SystemExit("対象のログがない")
    print(f"[core_score] 除外した run：{n_ex} 本（core2026_session.csv）")
    R = pd.DataFrame(runs).sort_values("t_unix")
    S = pd.concat(strikes, ignore_index=True)
    Rb, R = R[R.bridge], R[~R.bridge].copy()
    S = S[~S.bridge]
    R["rep"] = R.groupby(["method", "task"]).cumcount() + 1          # playback は試行番号がないので、時刻の順で数える
    out = os.path.join(a.dirs[0], "core2026_score")
    os.makedirs(out, exist_ok=True)
    R.to_csv(os.path.join(out, "calib_runs.csv" if a.calib else "runs.csv"), index=False)
    S.to_csv(os.path.join(out, "calib_strikes.csv" if a.calib else "strikes.csv"), index=False)
    L = []
    if a.calib:
        L += ["# 較正（core_score.py --calib）", "", "| 方法 | 本数 | 打った打点 | 誤差の平均 [ms] | SD | Δ（20 ms に丸め）[ms] |", "|---|---|---|---|---|---|"]
        for m, x in S.groupby("method"):
            e = x.timing_err_ms[x.peak_force >= THR_N]
            d = 20 * round(e.mean() / 20) if len(e) else np.nan
            L.append(f"| {m} | {R[R.method == m].shape[0]} | {len(e)} | {e.mean():+.1f} | {e.std():.1f} | {d:+.0f} |")
        L += ["", "Δ = 打った打点の符号つき誤差（正 = 遅い）の平均。F3 は F0 の発火を Δ_FF だけ前に、RL+Δ は B の目標を Δ_RL だけ前にずらす：",
              "porcaro_2026 で `python scripts/tools/user0/eval/real_core.py shift --ff <F0 の best.json> --delta_ff_ms <F0 の Δ> --delta_rl_ms <B の Δ>`"]
        md = "calib.md"
    else:
        L += ["# 実機コアの採点（core_score.py）", "",
              f"判定：contact_event ＋ 押し付けの連なり（離れ {TRAIN_GAP_S * 1000:.0f} ms 未満でつながった接触が 100 ms を超えたら数えない、2026-11-04）。"
              "旧採点 = 連なりなし（11/03〜11/04 の数字）。反復ごとの成功率の平均 ± SD（反復の間）と、全打点をまとめた値", "",
              "| 方法 | タスク | 反復 | 成功率（反復の平均 ± SD） | 旧採点 | 未打 | 遅い | 早い | 誤差の平均 [ms] | F 中央値 [N] | |ΔF| 中央値 [N] |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (m, t), x in R.groupby(["method", "task"]):
            s = summ(S[(S.method == m) & (S.task == t)])
            L.append(f"| {m} | {t} | {len(x)} | {x.success.mean():.3f} ± {x.success.std():.3f} | {x.success_old.mean():.3f} | {s['no_strike']:.3f} | {s['late']:.3f} | "
                     f"{s['early']:.3f} | {s['err_ms_mean']:+.1f} | {s['F_med']:.1f} | {s['dF_abs_med']:.1f} |")
        L += ["", "## 5 タスクの平均（タスクごとの反復の平均を、5 タスクで平均）", "", "| 方法 | 成功率 | 旧採点 | タスク数 |", "|---|---|---|---|"]
        for m, x in R.groupby("method"):
            g = x.groupby("task")[["success", "success_old"]].mean()
            L.append(f"| {m} | {g.success.mean():.3f} | {g.success_old.mean():.3f} | {len(g)} |")
        if len(Rb):
            L += ["", "## 日の差の橋渡し（11/04 に FF の間に差し込んだ B・E、各タスク 1 本。本表には入れていない）", "",
                  "| 方法 | タスク | 11/03 の成功率（反復の平均、本数） | 11/04 の成功率 | 差（11/04 − 11/03） |", "|---|---|---|---|---|"]
            dd = []
            for (m, t), x in Rb.groupby(["method", "task"]):
                y = R[(R.method == m) & (R.task == t)]
                d = x.success.mean() - y.success.mean() if len(y) else np.nan
                dd.append((m, d))
                L.append(f"| {m} | {t} | {y.success.mean():.3f}（{len(y)} 本） | {x.success.mean():.3f} | {d:+.3f} |")
            for m in sorted({a for a, _ in dd}):
                v = np.array([d for a, d in dd if a == m and np.isfinite(d)])
                if len(v):
                    L.append(f"| {m} | 5 タスクの平均 | | | {v.mean():+.3f}（{len(v)} タスク） |")
        md = "summary.md"
    open(os.path.join(out, md), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
