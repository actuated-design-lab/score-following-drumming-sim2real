"""pkg_1105.py — IEEE Access の図表の残り（11/05 夜の依頼 1〜7）の元データを、論文リポジトリの paper/figures/data にそのまま入れる CSV にする（CPU）

入力（どれも git に入っているもの）
  jetson_project
    data/user0/access2026v3/core2026_score/{runs.csv, strikes.csv}：本表（RL = 11/09、FF = 11/05、各方法 15 本、新しい判定）
    data/user0/sss2026/sss_strikes.csv と、その 1 本目のログ（SSS 初試奏、11/04 17:59 JST）
    analysis/itv/access_table2.csv：圧力段の検証 NRMSE（access_table2.py）
  porcaro_2026
    data/user0/jfps2026/replay/gen2_check/metrics.csv：Gen2（DR なし）の再生の手首角 RMSE（compare_hys_modes.py。check_gen2.sh・run_1105_pkg.sh fidelity）
    data/user0/eval/access_1104/sim/nom_F0_<task>/trace.npz の onsets：前打間隔の区分（ff_lib.ioi_bins）
    data/user0/eval/access_1104/pkg/{sim_dr.csv, fidelity_ts_*.csv}、c6/strikes.csv：WS の出力（run_1105_pkg.sh。無ければ飛ばす）
出力：data/user0/eval/access_1104/pkg/paper/*.csv（列の意味は各関数の docstring と、論文側の package md）

  python scripts/tools/user0/access/pkg_1105.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
J = ROOT.parent / "jetson_project"
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
sys.path.insert(0, str(J / "analysis"))
import ff_lib as FL  # noqa: E402
from strike_metrics import apply_time_source  # noqa: E402

SC = J / "data/user0/access2026v3/core2026_score"
PKG = ROOT / "data/user0/eval/access_1104/pkg"
OUT = PKG / "paper"
OUT.mkdir(parents=True, exist_ok=True)
NAME = {"F0": "FF", "F3": "FF-cal", "E": "MLP", "B": "LSTM"}
TASK = {"b_holdout_00_bpm90": "Held-out 1", "b_holdout_01_bpm96": "Held-out 2", "double_lead2_bpm160": "Double 160",
        "gmd_legacy_gmd_03_high_recut_bpm138": "GMD 138", "gmd_legacy_gmd_04_extreme_recut_bpm170": "GMD 170"}
BIN = ["<120", "120-200", "200-320", ">=320"]
THR, TARGET = 1.0, 20.0
written = []


def save(df, name):
    df.to_csv(OUT / name, index=False, float_format="%.4g")
    written.append(name)
    print(f"[ok] {name}（{len(df)} 行）")


# --------------------------------------------------------------------------- 1. 圧力段の NRMSE と手首角 RMSE
def item1():
    """pkg2_pressure_nrmse.csv：data, recording, channel, n_runs, nrmse_pct（中央値）, nrmse_iqr_pct, delay_ms
       pkg2_wrist_rmse.csv：motion, input（measured = 実測圧力を入力 / model = Gen2 の圧力モデル＋通信の遅れ）, independent（同定に使っていない）,
       rmse_deg, offset_deg（sim − 実機）, rmse_rest_deg, rmse_moving_deg, lag_ms（正 = sim が遅い）, amp_ratio（sim ÷ 実機の SD）,
       grip_rmse_deg, grip_amp_ratio"""
    P = pd.read_csv(J / "analysis/itv/access_table2.csv")
    save(P.rename(columns={"n": "n_runs", "nrmse_pct": "nrmse_pct", "iqr_pct": "nrmse_iqr_pct"}), "pkg2_pressure_nrmse.csv")
    M = pd.read_csv(ROOT / "data/user0/jfps2026/replay/gen2_check/metrics.csv")
    M = M[M.tag.isin(["gen2_measured", "gen2_cfg"])].copy()
    M["input"] = M.tag.map({"gen2_measured": "measured", "gen2_cfg": "model"})
    lab = {"tm_A_quasistatic": "Quasi-static", "tm_B_steps": "Steps", "tm_C_sine": "Sinusoid", "tm_D_antagonist": "Antagonistic",
           "tm_E_dbl160_seed2": "Replay double strokes 1", "tm_E_dbl160_seed3": "Replay double strokes 2",
           "tm_E_gmd138_seed2": "Replay GMD 1", "tm_E_gmd138_seed3": "Replay GMD 2"}
    indep = {"tm_C_sine", "tm_E_dbl160_seed2", "tm_E_dbl160_seed3", "tm_E_gmd138_seed2", "tm_E_gmd138_seed3"}
    W = pd.DataFrame({"motion": M.motion.map(lab).fillna(M.motion), "motion_id": M.motion, "input": M["input"],
                      "independent": M.motion.isin(indep), "rmse_deg": M.rmse, "offset_deg": M.offset, "rmse_rest_deg": M.rmse_rest,
                      "rmse_moving_deg": M.rmse_moving, "lag_ms": M.lag_ms, "amp_ratio": M.amp, "grip_rmse_deg": M.grip_rmse,
                      "grip_amp_ratio": M.grip_amp, "real_log": M.real_log})
    save(W.sort_values(["motion_id", "input"]), "pkg2_wrist_rmse.csv")


# --------------------------------------------------------------------------- 本表の打点
R = pd.read_csv(SC / "runs.csv")
R = R[~R.bridge.astype(bool)].copy()
S = pd.read_csv(SC / "strikes.csv")
S = S[~S.bridge.astype(bool)].copy()
ON = {t: np.load(ROOT / f"data/user0/eval/access_1104/sim/nom_F0_{t}/trace.npz")["onsets"] for t in TASK}


def item2():
    """pkg2_timing_force.csv：method, task（all = 5 タスクをまとめる）, onsets, struck（1 N 以上の当たりがある打点）,
       err_median_ms, err_q1_ms, err_q3_ms（当たった打点の符号つきタイミング誤差、正 = 遅い）, F_median_N, F_q1_N, F_q3_N（当たった打点のピーク力）,
       dF_abs_median_N（当たった打点の |F − 20 N| の中央値）。3 反復をまとめる"""
    rows = []
    for m in ["B", "E", "F0", "F3"]:
        for t in list(TASK) + ["all"]:
            x = S[(S.method == m) & ((S.task == t) | (t == "all"))]
            h = x[x.peak_force >= THR]
            e, F = h.timing_err_ms, h.peak_force
            rows.append(dict(method=NAME[m], task=TASK.get(t, "all"), onsets=len(x), struck=len(h), err_median_ms=e.median(),
                             err_q1_ms=e.quantile(.25), err_q3_ms=e.quantile(.75), F_median_N=F.median(), F_q1_N=F.quantile(.25),
                             F_q3_N=F.quantile(.75), dF_abs_median_N=(F - TARGET).abs().median()))
    save(pd.DataFrame(rows), "pkg2_timing_force.csv")


def item3():
    """pkg2_ioi_counts.csv：task, bin（FF の区分、ff_lib.ioi_bins。最初の打撃は >=320）, onsets_per_run, onsets_3runs, share"""
    rows = []
    for t, on in ON.items():
        b = FL.ioi_bins(on)
        for k, lab in enumerate(BIN):
            rows.append(dict(task=TASK[t], bin=lab, onsets_per_run=int((b == k).sum()), onsets_3runs=int(3 * (b == k).sum()),
                             share=float((b == k).mean())))
    D = pd.DataFrame(rows)
    tot = D.groupby("bin", sort=False)[["onsets_per_run", "onsets_3runs"]].sum().reset_index()
    tot["task"] = "all"
    tot["share"] = tot.onsets_per_run / tot.onsets_per_run.sum()
    save(pd.concat([D, tot[D.columns]]), "pkg2_ioi_counts.csv")


def item4a():
    """pkg2_runs.csv：method, task, rep（タスクごとの実行順 1〜3）, date（JST）, onsets, success（新しい判定）, success_old, log（jetson_project の相対パス）"""
    x = R.sort_values("t_unix").copy()
    x["rep"] = x.groupby(["method", "task"]).cumcount() + 1
    x["date"] = pd.to_datetime(x.t_unix, unit="s", utc=True).dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d %H:%M")
    save(pd.DataFrame({"method": x.method.map(NAME), "task": x.task.map(TASK), "rep": x.rep, "date": x.date, "onsets": x.n,
                       "success": x.success, "success_old": x.success_old, "log": x.file}).sort_values(["method", "task", "rep"]), "pkg2_runs.csv")


def item4b():
    """pkg2_sss_first_strikes.csv：onset（0 始まり）, velocity（楽譜のベロシティ、105 = アクセント、12 = 弱打。方策には渡していない）, target_s,
       strike_s（当たりの時刻、当たりなしは空）, timing_err_ms, peak_force_N（当たりなしは 0）, success, success_old
       pkg2_sss_first_force.csv：time_s（採点と同じ時刻）, force_N（ロードセル、5 ms 刻み）"""
    s = pd.read_csv(J / "data/user0/sss2026/sss_strikes.csv")
    first = s[s["take"] == "E_sss"]["time"].min()
    x = s[(s["take"] == "E_sss") & (s["time"] == first)].sort_values("strike_idx")
    hit = x.peak_force >= THR
    save(pd.DataFrame({"onset": x.strike_idx, "velocity": x.velocity, "target_s": x.target_time,
                       "strike_s": x.peak_time.where(hit), "timing_err_ms": x.timing_err_ms.where(hit),
                       "peak_force_N": x.peak_force.where(hit, 0.0), "success": x.success, "success_old": x.success_old}),
         "pkg2_sss_first_strikes.csv")
    log = x.log.iloc[0]
    d = apply_time_source(pd.read_csv(J / log), "rate_corrected")
    save(pd.DataFrame({"time_s": d.time.values, "force_N": d.force_N.values}), "pkg2_sss_first_force.csv")
    print(f"     SSS 1 本目：{first}（UTC）、{log}")


def item5():
    """pkg2_sim_dr.csv：method, task, n_draws, onsets_per_draw, success_mean, success_sd, ci95（sim_dr_summary.py）"""
    f = PKG / "sim_dr.csv"
    if not f.exists():
        print("[--] sim_dr.csv がない（run_1105_pkg.sh dr の後）")
        return
    T = pd.read_csv(f)
    T["method"] = T.method.map(NAME)
    T["task"] = T.task.map(TASK).fillna(T.task)
    save(T, "pkg2_sim_dr.csv")


def item6():
    """pkg2_swing_simreal.csv：src（real / sim）, bin, prev_bin（all = 分けない）, onsets, omega_deg_s, A_deg, theta_top_deg（中央値）, PF_MPa, success（実機のみ）
       c6_swing_bins.py の strikes.csv（sim 側を含む版）から"""
    f = ROOT / "data/user0/eval/access_1104/c6/strikes.csv"
    if not f.exists():
        print("[--] c6/strikes.csv がない（run_1105_pkg.sh c6 の後）")
        return
    D = pd.read_csv(f)
    # 実機側は c6_swing_bins.py --real_only の strikes_real.csv（package_sec4 §5 と同じ値）を使う。WS の実行の実機側は 46 / 918 打点で
    # 振れ幅が小さく違った（振れ幅 3° 未満の打点。原因は未特定、時刻の基準の版の違いと見ている）ので、§5 とそろえるため
    fr = ROOT / "data/user0/eval/access_1104/c6/strikes_real.csv"
    if fr.exists():
        D = pd.concat([D[D.src == "sim"], pd.read_csv(fr).assign(src="real")], ignore_index=True)
    D["prev_bin"] = D.sort_values(["src", "file", "strike_idx"]).groupby(["src", "file"]).bin.shift(1)
    rows = []
    for (src, b), x in D.groupby(["src", "bin"]):
        rows.append(dict(src=src, bin=BIN[b], prev_bin="all", onsets=len(x), omega_deg_s=x.omega_dn.median(), A_deg=x.A.median(),
                         theta_top_deg=x.th_top.median(), PF_MPa=x.P_F.median(), success=x.success.mean() if src == "real" else np.nan))
        for pb, y in x.dropna(subset=["prev_bin"]).groupby("prev_bin"):
            rows.append(dict(src=src, bin=BIN[b], prev_bin=BIN[int(pb)], onsets=len(y), omega_deg_s=y.omega_dn.median(), A_deg=y.A.median(),
                             theta_top_deg=y.th_top.median(), PF_MPa=y.P_F.median(), success=y.success.mean() if src == "real" else np.nan))
    save(pd.DataFrame(rows), "pkg2_swing_simreal.csv")


def item7():
    """fidelity_ts_<動作>.csv：fidelity_export.py の出力をそのまま"""
    fs = sorted(PKG.glob("fidelity_ts_*.csv"))
    if not fs:
        print("[--] fidelity_ts_*.csv がない（run_1105_pkg.sh fidelity の後）")
    for f in fs:
        shutil.copy(f, OUT / f.name)
        written.append(f.name)
        print(f"[ok] {f.name}")


for fn in (item1, item2, item3, item4a, item4b, item5, item6, item7):
    fn()
print("→", OUT, written)
