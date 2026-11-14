"""c3_sim_real.py — sim と実機の一致（CPU）：打点ごとの成否・タイミング誤差・時系列（手首角・圧力）

入力
  実機：本表の run（core2026_score/runs.csv・strikes.csv、橋渡しを除く）。B・E・F0・F3 × 5 タスク × 3 反復
  sim（WS、run_1104.sh trace の出力、data/user0/eval/access_1104/sim/）
    dr_<方法>_<タスク>/   DR 32 本：strikes.csv（env ごと）、trace.npz（env をまたいだ四分位）
    nom_<方法>_<タスク>/  DR なし 1 本：strikes.csv、trace.npz（生の時系列）
  RL は閉ループ（sim と実機で指令そのものが違ってよい）、FF は開ループ（指令は同じ。違いは軌道だけ）

1. 打点ごとの成否：sim の成功確率（DR 32 本の平均）と実機の成功率（3 反復の平均）。Spearman、AUC（sim の確率で実機の当たり／外れを当てる）、
   2×2（sim ≥ 0.5、実機 ≥ 0.5）
2. タイミング誤差：打点ごとの誤差の平均（当たった打点だけ、±30 ms の内外を問わず検出された打点）の、sim と実機の相関・差
3. 時系列：DR なしの sim と実機の各 run、曲の窓で、手首角と圧力（sim は ITV 出口、実機は実測）の相関（ずれ 0）と、
   相互相関が最大になるずれ（±200 ms、正 = 実機が遅い）。比べる基準として、実機の反復どうしの相関も出す
4. FF の分かれ方：目標時刻に合わせた |手首角の差|（実機 − sim、DR なし）の中央値の時間変化

  python scripts/tools/user0/access/c3_sim_real.py
出力：data/user0/eval/access_1104/c3/{c3_targets.csv, c3_concord.csv, c3_timing.csv, c3_ts.csv, c3_summary.md, *.png}
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, pearsonr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realcore as R  # noqa: E402

SIM = R.ROOT / "data/user0/eval/access_1104/sim"
OUT = R.ROOT / "data/user0/eval/access_1104/c3"
METHODS = ["E", "B", "F0", "F3"]
COLOR = {"E": "#2a78d6", "B": "#eb6834", "F0": "#1baf7a", "F3": "#eda100"}
MAXLAG = 0.200


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    s1, s0 = score[y], score[~y]
    return float(np.mean((s1[:, None] > s0[None]) + 0.5 * (s1[:, None] == s0[None])))


def match_targets(real_t, sim_t, tol=0.03):
    """実機の目標時刻ごとに、いちばん近い sim の目標の添字（tol を超えたら -1）"""
    j = np.abs(real_t[:, None] - sim_t[None]).argmin(1)
    return np.where(np.abs(real_t - sim_t[j]) <= tol, j, -1)


def lagcorr(t_ref, y_ref, t, y, lo, hi, maxlag=MAXLAG, dt=R.DT):
    """y_ref（基準、sim）と y（実機）。ずれ 0 の相関、最大の相関とそのずれ（正 = y が遅い）、RMSE"""
    g = np.arange(lo, hi, dt)
    a = np.interp(g, t_ref, y_ref)
    best = (-np.inf, 0.0)
    r0 = np.nan
    for L in np.arange(-maxlag, maxlag + dt / 2, dt):
        b = np.interp(g + L, t, y)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
            continue
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if abs(L) < dt / 2:
            r0 = r
            rmse = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
        if r > best[0]:
            best = (r, float(L))
    return dict(r0=r0, r_max=best[0], lag_ms=best[1] * 1e3, rmse=rmse if np.isfinite(r0) else np.nan)


def nom_trace(m, task):
    p = SIM / f"nom_{m}_{task}/trace.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return pd.DataFrame({"time": z["time"], "meas_pres_DF": z["P_out"][:, 0], "meas_pres_F": z["P_out"][:, 1],
                         "meas_pres_G": z["P_out"][:, 2], "cmd_DF": z["P_cmd"][:, 0], "cmd_F": z["P_cmd"][:, 1],
                         "cmd_G": z["P_cmd"][:, 2], "wrist_angle_deg": z["q"][:, 0], "grip_angle_deg": z["q"][:, 1],
                         "force_N": z["force"]})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rr, S = R.runs(), R.strikes()
    S = S[~S.bridge.astype(bool)] if "bridge" in S else S
    trow, tsrow, missing = [], [], []
    lockdiff = {m: [] for m in METHODS}
    for (m, task), grp in rr.groupby(["method", "task"]):
        dr_p, nom_p = SIM / f"dr_{m}_{task}/strikes.csv", SIM / f"nom_{m}_{task}/strikes.csv"
        if not dr_p.exists():
            missing.append(dr_p.parent.name)
            continue
        sd = pd.read_csv(dr_p)
        sim_t = np.sort(sd.target_time.unique())
        sp = sd.groupby("target_time").success.mean().reindex(sim_t).to_numpy()
        se = sd[sd.peak_time.notna()].groupby("target_time").timing_err_ms.mean().reindex(sim_t).to_numpy()
        sn = pd.read_csv(nom_p).set_index("target_time").reindex(sim_t) if nom_p.exists() else None
        # 実機：反復ごとの打点
        Rs = S[(S.method == m) & (S.task == task)]
        files = list(grp.file)
        ref_t = np.sort(Rs[Rs.file == files[0]].target_time.to_numpy())
        j = match_targets(ref_t, sim_t)
        for k, t_k in enumerate(ref_t):
            if j[k] < 0:
                continue
            rows = Rs[np.abs(Rs.target_time - t_k) < 0.03]
            trow.append(dict(method=m, task=task, k=k, target_time=t_k, sim_p=sp[j[k]], sim_err=se[j[k]],
                             sim_nom=float(sn.success.iloc[j[k]]) if sn is not None else np.nan,
                             real_p=rows.success.mean(), n_rep=len(rows),
                             real_hits=[bool(x) for x in rows.success],
                             real_err=rows[rows.peak_time.notna()].timing_err_ms.mean()))
        # 時系列：DR なしの sim と各 run、実機の反復どうし
        nt = nom_trace(m, task)
        if nt is None:
            missing.append(f"nom_{m}_{task}")
            continue
        lo, hi = ref_t[0] - 0.5, ref_t[-1] + 0.5
        reals = {f: R.load(f, k) for f, k in zip(grp.file, grp.kind_log)}
        sig = ["wrist_angle_deg", "meas_pres_DF", "meas_pres_F", "meas_pres_G"]
        for f, df in reals.items():
            for c in sig:
                tsrow.append(dict(method=m, task=task, pair="sim_vs_real", file=f, signal=c,
                                  **lagcorr(nt.time.to_numpy(), nt[c].to_numpy(), df.time.to_numpy(), df[c].to_numpy(), lo, hi)))
            g, A = R.lock(df, ref_t, ["wrist_angle_deg"], lo=-0.30, hi=0.40)
            g, B_ = R.lock(nt, ref_t, ["wrist_angle_deg"], lo=-0.30, hi=0.40)
            lockdiff[m].append(np.abs(A["wrist_angle_deg"] - B_["wrist_angle_deg"]))
        for (f1, d1), (f2, d2) in itertools.combinations(reals.items(), 2):
            for c in sig:
                tsrow.append(dict(method=m, task=task, pair="real_vs_real", file=f"{f1}|{f2}", signal=c,
                                  **lagcorr(d1.time.to_numpy(), d1[c].to_numpy(), d2.time.to_numpy(), d2[c].to_numpy(), lo, hi)))
    if missing:
        print(f"[c3] sim の出力がない（run_1104.sh trace の前？）：{len(missing)}、例 {missing[:4]}")
    T = pd.DataFrame(trow)
    if T.empty:
        return
    T.to_csv(OUT / "c3_targets.csv", index=False)

    # 1. 成否の一致
    crow = []
    for m in METHODS + ["all"]:
        d = T if m == "all" else T[T.method == m]
        if d.empty:
            continue
        y = np.concatenate([np.asarray(h, bool) for h in d.real_hits])
        s = np.repeat(d.sim_p.to_numpy(), d.n_rep.to_numpy())
        ss, rs = d.sim_p >= 0.5, d.real_p >= 0.5
        crow.append(dict(method=m, n_targets=len(d), sim_p_mean=d.sim_p.mean(), real_p_mean=d.real_p.mean(),
                         spearman=spearmanr(d.sim_p, d.real_p).statistic if d.sim_p.nunique() > 1 and d.real_p.nunique() > 1 else np.nan,
                         auc=auc(s, y), both_hit=int((ss & rs).sum()), sim_only=int((ss & ~rs).sum()),
                         real_only=int((~ss & rs).sum()), both_miss=int((~ss & ~rs).sum()),
                         agree=float((ss == rs).mean())))
    Cc = pd.DataFrame(crow)
    Cc.to_csv(OUT / "c3_concord.csv", index=False)

    # 2. タイミング誤差
    trows = []
    for m in METHODS:
        d = T[(T.method == m)].dropna(subset=["sim_err", "real_err"])
        if len(d) < 5:
            continue
        r = pearsonr(d.sim_err, d.real_err)
        trows.append(dict(method=m, n=len(d), pearson=r.statistic, p=r.pvalue, bias_ms=float((d.real_err - d.sim_err).mean()),
                          rms_ms=float(np.sqrt(((d.real_err - d.sim_err) ** 2).mean())), sim_err_mean=d.sim_err.mean(), real_err_mean=d.real_err.mean()))
    Tt = pd.DataFrame(trows)
    Tt.to_csv(OUT / "c3_timing.csv", index=False)

    # 3. 時系列
    TS = pd.DataFrame(tsrow)
    TS.to_csv(OUT / "c3_ts.csv", index=False)
    tsum = TS.groupby(["pair", "method", "signal"])[["r0", "r_max", "lag_ms", "rmse"]].median().reset_index() if not TS.empty else TS

    # 図 1：成否の一致（目標ごと）
    fig, ax = plt.subplots(1, len(METHODS), figsize=(12, 3.2), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for a, m in zip(ax, METHODS):
        d = T[T.method == m]
        a.scatter(d.sim_p, d.real_p + rng.uniform(-0.03, 0.03, len(d)), s=10, color=COLOR[m], alpha=0.6, lw=0)
        c = Cc[Cc.method == m]
        if len(c):
            a.set_title(f"{m}: Spearman {c.spearman.iloc[0]:.2f}, AUC {c.auc.iloc[0]:.2f}", fontsize=9)
        a.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
        a.grid(alpha=0.25, lw=0.5)
        a.set_xlabel("sim success prob. (DR 32)")
    ax[0].set_ylabel("real success rate (3 reps)")
    fig.tight_layout(); fig.savefig(OUT / "c3_concord.png", dpi=200); fig.savefig(OUT / "c3_concord.pdf"); plt.close(fig)

    # 図 2：FF と RL の分かれ方（目標時刻に合わせた |手首角の差|）
    fig, a = plt.subplots(figsize=(5.5, 3.5))
    g = np.round(np.arange(-0.30, 0.40 + R.DT / 2, R.DT), 6)
    for m in METHODS:
        if lockdiff[m]:
            x = np.concatenate(lockdiff[m])
            a.plot(g * 1e3, np.nanmedian(x, 0), color=COLOR[m], lw=2, label=m)
    a.axvline(0, color="#999", lw=0.8); a.grid(alpha=0.25, lw=0.5); a.legend(frameon=False, fontsize=8)
    a.set_xlabel("time from target onset [ms]"); a.set_ylabel("|wrist real − sim (nominal)| [deg]")
    fig.tight_layout(); fig.savefig(OUT / "c3_divergence.png", dpi=200); fig.savefig(OUT / "c3_divergence.pdf"); plt.close(fig)

    # 図 3：時系列の重ね描き（E と F0、double160 と holdout0、最初の 3 s）
    pairs = [(m, t) for m in ["E", "F0"] for t in ["double_lead2_bpm160", "b_holdout_00_bpm90"]]
    fig, ax = plt.subplots(3, len(pairs), figsize=(13, 7), sharex=True, sharey="row")
    for j, (m, task) in enumerate(pairs):
        nt = nom_trace(m, task)
        drz = SIM / f"dr_{m}_{task}/trace.npz"
        grp = rr[(rr.method == m) & (rr.task == task)]
        if nt is None or grp.empty:
            continue
        t0 = nt.time.iloc[0]
        for f, k in zip(grp.file, grp.kind_log):
            df = R.load(f, k)
            for i, c in enumerate(["wrist_angle_deg", "meas_pres_G", "force_N"]):
                ax[i, j].plot(df.time, df[c], color=COLOR[m], lw=0.8, alpha=0.7)
        if drz.exists():
            z = np.load(drz)
            for i, (key, col) in enumerate([("q", 0), ("P_out", 2)]):
                ax[i, j].fill_between(z["time"], z[key][0, :, col], z[key][2, :, col], color="#52514e", alpha=0.18, lw=0)
        for i, c in enumerate(["wrist_angle_deg", "meas_pres_G", "force_N"]):
            ax[i, j].plot(nt.time, nt[c], color="#52514e", lw=1.4)
        ax[0, j].set_title(f"{m} / {task[:22]}", fontsize=9)
        ax[2, j].set_xlim(1.8, 4.8)
        ax[2, j].set_xlabel("time [s]")
    for i, yl in enumerate(["wrist [deg]", "G pressure [MPa]", "force [N]"]):
        ax[i, 0].set_ylabel(yl)
    for a in ax.flat:
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("colored = real (3 reps), dark = sim nominal, band = sim DR IQR", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "c3_timeseries.png", dpi=200); fig.savefig(OUT / "c3_timeseries.pdf"); plt.close(fig)

    with open(OUT / "c3_summary.md", "w") as fp:
        fp.write("# C3 sim と実機の一致\n\n出典：scripts/tools/user0/access/c3_sim_real.py。実機 = 本表 60 本、sim = run_1104.sh trace（sim_trace.py、Gen2_B、DR なし と DR 32 本）\n\n")
        fp.write("## 1. 打点ごとの成否\n\n" + Cc.round(3).to_markdown(index=False) + "\n\n")
        fp.write("## 2. タイミング誤差（打点ごとの平均、ms）\n\n" + Tt.round(3).to_markdown(index=False) + "\n\n")
        fp.write("## 3. 時系列（曲の窓、run の中央値。lag 正 = 実機が遅い）\n\n" + tsum.round(3).to_markdown(index=False) + "\n")
        if missing:
            fp.write(f"\n（sim の出力がないもの {len(missing)}）\n")
    print(Cc.round(3).to_string()); print(Tt.round(3).to_string()); print(tsum.round(3).to_string())


if __name__ == "__main__":
    main()
