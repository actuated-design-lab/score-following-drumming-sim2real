"""c2_strike_motion.py — 実機の RL（E、比較に B）の打撃動作：接触の立ち上がりに合わせた指令・実測圧・関節角・力（CPU）

対象：実機コアの本表の run（core2026_score/runs.csv、橋渡しを除く）
イベント：採点と同じ接触イベント（realcore.contact_events）。t = 0 は力が初めて 1 N 以上になった時刻
「当たり」：採点の strikes.csv で success = True の打点（peak_time が一致、±3 ms）

  python scripts/tools/user0/access/c2_strike_motion.py
出力：data/user0/eval/access_1104/c2/{c2_strike_motion.png, c2_summary.csv, c2_summary.md, c2_traces.npz}
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
import realcore as R  # noqa: E402

OUT = R.ROOT / "data/user0/eval/access_1104/c2"
COLOR = {"E": "#2a78d6", "B": "#eb6834"}
MUS = {"G": "#2a78d6", "DF": "#eb6834", "F": "#1baf7a"}


def collect(method):
    rr = R.runs()
    rr = rr[rr.method == method]
    S = R.strikes()
    cols = R.COLS + ["obs_contact"]
    acc = {c: [] for c in cols}
    meta = []
    for _, r in rr.iterrows():
        df = R.load(r.file, r.kind_log)
        t, f = df["time"].to_numpy(float), df["force_N"].to_numpy(float)
        pk, on = R.contact_events(t, f)
        hit_pk = S[(S.file == r.file) & S.success].peak_time.to_numpy(float)
        hit = np.array([np.any(np.abs(hit_pk - t[k]) < 0.003) for k in pk], bool)
        g, L = R.lock(df, t[on], [c for c in cols if c in df])
        for c in cols:
            acc[c].append(L[c] if c in L else np.full((len(on), len(g)), np.nan))
        meta.append(pd.DataFrame({"file": r.file, "task": r.task, "t_on": t[on], "t_pk": t[pk], "F_pk": f[pk], "hit": hit}))
    return g, {c: np.concatenate(v) for c, v in acc.items()}, pd.concat(meta, ignore_index=True)


def at(g, x, tt):
    return np.nanmedian(x[:, np.argmin(np.abs(g - tt))])


def summarize(method, g, A, M):
    rows = []
    for sel_name, sel in [("hit", M.hit.to_numpy()), ("all", np.ones(len(M), bool))]:
        cg = A["cmd_G"][sel]
        pre = (g >= -0.25) & (g <= 0.0)
        i_max = np.nanargmax(np.where(pre, cg, -np.inf), axis=1)
        t_max = g[i_max]
        # G の指令が最大値と t=0 以降の最小値の中点を下回る最初の時刻（最大のあと）
        t_drop = []
        for e in range(len(cg)):
            post = g >= g[i_max[e]]
            lo_ = np.nanmin(cg[e][(g >= 0) & (g <= 0.15)]) if np.any(np.isfinite(cg[e][(g >= 0) & (g <= 0.15)])) else np.nan
            mid = (cg[e, i_max[e]] + lo_) / 2
            idx = np.where(post & (cg[e] < mid))[0]
            t_drop.append(g[idx[0]] if len(idx) and cg[e, i_max[e]] - lo_ > 0.05 else np.nan)
        t_drop = np.asarray(t_drop)
        d = dict(method=method, events=sel_name, n=int(sel.sum()),
                 n_runs=int(M[sel].file.nunique()),
                 G_cmd_m100=at(g, cg, -0.10), G_cmd_m40=at(g, cg, -0.04), G_cmd_0=at(g, cg, 0.0), G_cmd_p40=at(g, cg, 0.04),
                 G_cmd_max_pre=float(np.nanmedian(np.nanmax(np.where(pre, cg, np.nan), axis=1))),
                 t_G_cmd_max_ms=float(np.nanmedian(t_max) * 1e3),
                 t_G_cmd_drop_ms=float(np.nanmedian(t_drop) * 1e3), frac_G_drop=float(np.mean(np.isfinite(t_drop))),
                 G_meas_m40=at(g, A["meas_pres_G"][sel], -0.04), G_meas_0=at(g, A["meas_pres_G"][sel], 0.0),
                 G_meas_p40=at(g, A["meas_pres_G"][sel], 0.04),
                 DF_cmd_m100=at(g, A["cmd_DF"][sel], -0.10), DF_cmd_0=at(g, A["cmd_DF"][sel], 0.0),
                 F_cmd_m100=at(g, A["cmd_F"][sel], -0.10), F_cmd_0=at(g, A["cmd_F"][sel], 0.0),
                 wrist_m100=at(g, A["wrist_angle_deg"][sel], -0.10), wrist_0=at(g, A["wrist_angle_deg"][sel], 0.0),
                 grip_m100=at(g, A["grip_angle_deg"][sel], -0.10), grip_0=at(g, A["grip_angle_deg"][sel], 0.0),
                 grip_p40=at(g, A["grip_angle_deg"][sel], 0.04),
                 F_peak_med=float(M[sel].F_pk.median()),
                 contact_ms_med=float(((M[sel].t_pk - M[sel].t_on) * 1e3).median()))
        rows.append(d)
    return rows


def band(ax, g, x, color, label, ls="-"):
    m = np.nanmedian(x, 0)
    q1, q3 = np.nanpercentile(x, 25, 0), np.nanpercentile(x, 75, 0)
    ax.fill_between(g * 1e3, q1, q3, color=color, alpha=0.18, lw=0)
    ax.plot(g * 1e3, m, color=color, lw=2, ls=ls, label=label)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, rows = {}, []
    for m in ["E", "B"]:
        g, A, M = collect(m)
        res[m] = (g, A, M)
        rows += summarize(m, g, A, M)
        print(f"[c2] {m}: events {len(M)}, hits {int(M.hit.sum())}")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "c2_summary.csv", index=False)
    np.savez_compressed(OUT / "c2_traces.npz", **{f"{m}_{c}": res[m][1][c] for m in res for c in res[m][1]},
                        g=res["E"][0], **{f"{m}_hit": res[m][2].hit.to_numpy() for m in res})

    # 図：列 = E / B（当たりのイベントだけ）、行 = 指令圧・実測圧・関節角・力
    fig, ax = plt.subplots(4, 2, figsize=(9, 9), sharex=True, sharey="row")
    for j, m in enumerate(["E", "B"]):
        g, A, M = res[m]
        h = M.hit.to_numpy()
        for mu in ["G", "DF", "F"]:
            band(ax[0, j], g, A[f"cmd_{mu}"][h], MUS[mu], mu)
            band(ax[1, j], g, A[f"meas_pres_{mu}"][h], MUS[mu], mu)
        band(ax[2, j], g, A["wrist_angle_deg"][h], "#52514e", "wrist")
        band(ax[2, j], g, A["grip_angle_deg"][h], "#6250d6", "grip", ls="--")
        band(ax[3, j], g, A["force_N"][h], "#52514e", "force")
        ax[0, j].set_title(f"{m}: {int(h.sum())} hit strikes / {M.file.nunique()} runs")
        for a in ax[:, j]:
            a.axvline(0, color="#999", lw=0.8)
            a.grid(alpha=0.25, lw=0.5)
        ax[3, j].set_xlabel("time from contact onset (F ≥ 1 N) [ms]")
    ax[0, 0].set_ylabel("command [MPa]")
    ax[1, 0].set_ylabel("measured pressure [MPa]")
    ax[2, 0].set_ylabel("joint angle [deg]")
    ax[3, 0].set_ylabel("force [N]")
    ax[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    ax[2, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Real robot, core tasks: median and IQR around contact onset", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "c2_strike_motion.png", dpi=200)
    fig.savefig(OUT / "c2_strike_motion.pdf")

    with open(OUT / "c2_summary.md", "w") as fp:
        fp.write("# C2 実機の打撃動作（接触の立ち上がりに合わせる、中央値）\n\n出典：scripts/tools/user0/access/c2_strike_motion.py、"
                 "実機コア本表の run（core2026_score/runs.csv）。圧力 MPa、角度 deg、時刻は t = 0（初めて 1 N 以上）からの ms\n\n")
        fp.write(tab.round(3).to_markdown(index=False))
        fp.write("\n")
    print(tab.round(3).T.to_string())


if __name__ == "__main__":
    main()
