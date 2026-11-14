"""c1_ff_press.py — 実機の F0 が「押し付け」になる原因：同じ指令の実機と sim を比べる（CPU）

入力
  実機：F0 の本表の run 15 本（core2026_score/runs.csv、採点と同じ時刻）
  sim（WS、run_1104.sh replay の出力、data/user0/eval/access_1104/replay/）
    F0_<task>_cfg.csv.gz        : sim の圧力モデル（Gen2、DR なし）＝ 段2 の選択と同じ sim
    F0_<task>_meas_<unix>.csv.gz: 実機で測った圧力をそのまま入れた sim（replay_open_loop --pmodel measured、50 ms の移動平均＋±10 kPa の遊び）
切り分け（打点ごと、目標時刻のあとの窓での手首角）
  real − sim_cfg = (sim_meas − sim_cfg) [圧力の側] + (real − sim_meas) [機構の側]
  「sim は低い圧力での DF の効きを大きく見ている」なら、機構の側が大きく、sim_meas は実機より手首が上がる

  python scripts/tools/user0/access/c1_ff_press.py
出力：data/user0/eval/access_1104/c1/{c1_contact.csv, c1_decomp.csv, c1_summary.md, c1_locked.png, c1_wrist_vs_dp.png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ff"))
import realcore as R  # noqa: E402

REP = R.ROOT / "data/user0/eval/access_1104/replay"
OUT = R.ROOT / "data/user0/eval/access_1104/c1"
SONGS = R.ROOT / "data/user0/eval/real_core/songs"
SRC = {"real": "#2a78d6", "sim_cfg": "#eb6834", "sim_meas": "#1baf7a"}
LAB = {"real": "real", "sim_cfg": "sim (model pressure)", "sim_meas": "sim (measured pressure in)"}
WIN_POST = (0.08, 0.20)          # 目標時刻のあと、振り上げの窓 [s]（手首角の平均）


def onsets(task):
    import ff_lib
    tgt, bpm = ff_lib.midi_target(SONGS / f"{task}.mid")
    return ff_lib.onsets_from_target(tgt, bpm), bpm


def sim_df(path):
    d = pd.read_csv(path)
    return pd.DataFrame({"time": d.time + d.time.diff().median(),     # 物理 1 ステップのあとの状態
                         "cmd_DF": d.P_cmd_DF, "cmd_F": d.P_cmd_F, "cmd_G": d.P_cmd_G,
                         "meas_pres_DF": d.P_out_DF, "meas_pres_F": d.P_out_F, "meas_pres_G": d.P_out_G,
                         "wrist_angle_deg": d.wrist_angle_deg, "grip_angle_deg": d.grip_angle_deg, "force_N": d.force_N})


def contact_stats(df, t_lo, t_hi):
    """F ≥ 1 N の時間の割合、接触（1 N で始まり 0.5 N 未満で終わる）の長さ、100 ms 以内の接触イベントの数"""
    d = df[(df.time >= t_lo) & (df.time <= t_hi)]
    t, f = d.time.to_numpy(float), d.force_N.to_numpy(float)
    frac = float(np.mean(f >= 1.0))
    segs, inside, s0 = [], False, 0
    for i in range(len(f)):
        if not inside and f[i] >= 1.0:
            inside, s0 = True, i
        elif inside and f[i] < 0.5:
            inside = False
            segs.append(t[i - 1] - t[s0] + R.DT)
    if inside:
        segs.append(t[-1] - t[s0] + R.DT)
    segs = np.asarray(segs)
    return dict(frac_F_gt1=frac, n_contacts=len(segs), n_short=int(np.sum(segs <= 0.100)), n_press=int(np.sum(segs > 0.100)),
                contact_ms_med=float(np.median(segs) * 1e3) if len(segs) else np.nan,
                press_time_frac=float(segs[segs > 0.100].sum() / (t_hi - t_lo)) if len(segs) else 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rr = R.runs()
    rr = rr[rr.method == "F0"]
    cols = ["cmd_DF", "cmd_F", "cmd_G", "meas_pres_DF", "meas_pres_F", "meas_pres_G", "wrist_angle_deg", "force_N"]
    crow, drow = [], []
    L = {s: {c: [] for c in cols} for s in SRC}
    dp = {s: [] for s in SRC}
    missing = []
    for task, grp in rr.groupby("task"):
        ons, bpm = onsets(task)
        lo, hi = ons[0] - 0.5, ons[-1] + 0.5
        cfgp = REP / f"F0_{task}_cfg.csv.gz"
        S_cfg = sim_df(cfgp) if cfgp.exists() else None
        if S_cfg is None:
            missing.append(cfgp.name)
        for _, r in grp.iterrows():
            u = Path(r.file).stem.split("_")[-1]
            real = R.load(r.file, r.kind_log)
            mp = REP / f"F0_{task}_meas_{u}.csv.gz"
            S_meas = sim_df(mp) if mp.exists() else None
            if S_meas is None:
                missing.append(mp.name)
            srcs = {"real": real, "sim_cfg": S_cfg, "sim_meas": S_meas}
            for s, df in srcs.items():
                if df is None:
                    continue
                crow.append(dict(task=task, run=u, source=s, **contact_stats(df, lo, hi)))
                g, A = R.lock(df, ons, cols, lo=-0.40, hi=0.30)
                if not (s == "sim_cfg" and r.rep != grp.rep.min()):     # sim_cfg は決定的：図の束には 1 回分だけ
                    for c in cols:
                        L[s][c].append(A[c])
                w = (g >= WIN_POST[0]) & (g <= WIN_POST[1])
                for k, t_k in enumerate(ons):
                    drow.append(dict(task=task, run=u, source=s, k=k, t_k=t_k,
                                     wrist_post=float(np.nanmean(A["wrist_angle_deg"][k, w])),
                                     DF_post=float(np.nanmean(A["meas_pres_DF"][k, w])),
                                     F_post=float(np.nanmean(A["meas_pres_F"][k, w]))))
                sel = (df.time >= lo) & (df.time <= hi)
                dp[s].append(df.loc[sel, ["meas_pres_DF", "meas_pres_F", "wrist_angle_deg"]].to_numpy(float))
    if missing:
        print(f"[c1] sim の出力がない（run_1104.sh replay の前？）：{len(missing)} 本、例 {missing[:3]}")
    C = pd.DataFrame(crow)
    C.to_csv(OUT / "c1_contact.csv", index=False)
    D = pd.DataFrame(drow)
    D.to_csv(OUT / "c1_targets.csv", index=False)

    # 切り分け：打点ごとに real・sim_meas（同じ run）・sim_cfg（同じタスク）をそろえる
    P = D.pivot_table(index=["task", "run", "k"], columns="source", values="wrist_post").dropna()
    dec = None
    if {"real", "sim_cfg", "sim_meas"} <= set(P.columns):
        P["pressure_side"] = P.sim_meas - P.sim_cfg
        P["mechanism_side"] = P.real - P.sim_meas
        P["total"] = P.real - P.sim_cfg
        dec = P.groupby("task")[["real", "sim_cfg", "sim_meas", "total", "pressure_side", "mechanism_side"]].median()
        dec.loc["all"] = P[["real", "sim_cfg", "sim_meas", "total", "pressure_side", "mechanism_side"]].median()
        dec.to_csv(OUT / "c1_decomp.csv")
    cs = C.groupby(["task", "source"])[["frac_F_gt1", "press_time_frac", "n_short", "n_press", "contact_ms_med"]].median()
    ca = C.groupby("source")[["frac_F_gt1", "press_time_frac", "n_short", "n_press", "contact_ms_med"]].median()

    # 図 1：目標時刻に合わせた DF・F の圧力（実機の実測 / sim の ITV 出口 / sim に入れた実測）、手首角、力
    fig, ax = plt.subplots(4, 1, figsize=(6.5, 9), sharex=True)
    for s in SRC:
        if not L[s]["wrist_angle_deg"]:
            continue
        g = np.round(np.arange(-0.40, 0.30 + R.DT / 2, R.DT), 6)
        for a, c, ls in [(ax[0], "meas_pres_DF", "-"), (ax[1], "meas_pres_F", "-"), (ax[2], "wrist_angle_deg", "-"), (ax[3], "force_N", "-")]:
            x = np.concatenate(L[s][c])
            a.fill_between(g * 1e3, np.nanpercentile(x, 25, 0), np.nanpercentile(x, 75, 0), color=SRC[s], alpha=0.15, lw=0)
            a.plot(g * 1e3, np.nanmedian(x, 0), color=SRC[s], lw=2, ls=ls, label=LAB[s])
    if L["real"]["cmd_DF"]:
        x = np.concatenate(L["real"]["cmd_DF"]); ax[0].plot(g * 1e3, np.nanmedian(x, 0), color="#52514e", lw=1, ls="--", label="command")
        x = np.concatenate(L["real"]["cmd_F"]); ax[1].plot(g * 1e3, np.nanmedian(x, 0), color="#52514e", lw=1, ls="--", label="command")
    for a, yl in zip(ax, ["DF pressure [MPa]", "F pressure [MPa]", "wrist angle [deg]", "force [N]"]):
        a.set_ylabel(yl); a.axvline(0, color="#999", lw=0.8); a.grid(alpha=0.25, lw=0.5)
        a.axvspan(WIN_POST[0] * 1e3, WIN_POST[1] * 1e3, color="#999", alpha=0.08, lw=0)
    ax[0].legend(frameon=False, fontsize=8)
    ax[3].set_xlabel("time from target onset [ms]")
    fig.suptitle("F0 (same command): real vs sim, median and IQR over all targets", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "c1_locked.png", dpi=200); fig.savefig(OUT / "c1_locked.pdf"); plt.close(fig)

    # 図 2：手首角と DF − F の圧力差（同じ圧力のときの手首の位置。機構の側の差を直接見る）
    fig, a = plt.subplots(figsize=(5.5, 4))
    bins = np.arange(-0.5, 0.55, 0.05)
    for s in ["real", "sim_meas", "sim_cfg"]:
        if not dp[s]:
            continue
        x = np.concatenate(dp[s])
        d = x[:, 0] - x[:, 1]
        idx = np.digitize(d, bins)
        cen, med, q1, q3 = [], [], [], []
        for b in range(1, len(bins)):
            y = x[idx == b, 2]
            if len(y) >= 30:
                cen.append((bins[b - 1] + bins[b]) / 2); med.append(np.median(y)); q1.append(np.percentile(y, 25)); q3.append(np.percentile(y, 75))
        a.fill_between(cen, q1, q3, color=SRC[s], alpha=0.15, lw=0)
        a.plot(cen, med, color=SRC[s], lw=2, marker="o", ms=4, label=LAB[s])
    a.set_xlabel("P_DF − P_F (pressure acting on the joint) [MPa]"); a.set_ylabel("wrist angle [deg]")
    a.grid(alpha=0.25, lw=0.5); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "c1_wrist_vs_dp.png", dpi=200); fig.savefig(OUT / "c1_wrist_vs_dp.pdf"); plt.close(fig)

    with open(OUT / "c1_summary.md", "w") as fp:
        fp.write("# C1 FF（F0）の押し付けの原因：同じ指令の実機と sim\n\n出典：scripts/tools/user0/access/c1_ff_press.py。"
                 "実機 = core2026_score/runs.csv の F0 15 本、sim = run_1104.sh replay（replay_open_loop.py、Gen2・DR なし）\n\n")
        fp.write("## 接触（曲の窓：最初の目標 − 0.5 s 〜 最後の目標 + 0.5 s、run の中央値）\n\n")
        fp.write("frac_F_gt1 = F ≥ 1 N の時間の割合、press_time_frac = 100 ms を超える接触の時間の割合、n_short / n_press = 100 ms 以内 / 超えの接触の数\n\n")
        fp.write(ca.round(3).to_markdown() + "\n\n" + cs.round(3).to_markdown() + "\n\n")
        if dec is not None:
            fp.write(f"## 手首角の切り分け（目標時刻 + {WIN_POST[0]*1e3:.0f}〜{WIN_POST[1]*1e3:.0f} ms の平均、打点ごとの差の中央値、deg）\n\n")
            fp.write("total = real − sim_cfg、pressure_side = sim_meas − sim_cfg、mechanism_side = real − sim_meas\n\n")
            fp.write(dec.round(2).to_markdown() + "\n")
        if missing:
            fp.write(f"\n（sim の出力が {len(missing)} 本ない）\n")
    print(ca.round(3).to_string())
    if dec is not None:
        print(dec.round(2).to_string())


if __name__ == "__main__":
    main()
