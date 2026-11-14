"""c1b_lift.py — 打点の前に手首が持ち上がっていたか（振り上げのない「打撃」＝ 押し付けたままの接触の跳ね返り）を、sim と実機で数える（CPU）

接触イベント（採点と同じ contact_event：1 N で始まり 0.5 N 未満で終わる、100 ms 以内）ごとに、
  lift = 前のイベントのピークから、このイベントの立ち上がりまでの手首角の最大 − 立ち上がりの手首角
lift < LIFT_DEG を「持ち上げなし」とする。採点で当たり（success）になった打点のうち、持ち上げなしの割合も出す
sim：run_1104.sh trace の nom_<方法>_<タスク>（DR なし、力は k_t × z、5 ms）。実機：本表の run

  python scripts/tools/user0/access/c1b_lift.py
出力：data/user0/eval/access_1104/c1/{c1b_lift.csv, c1b_lift.md, c1b_example.png}
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
from c3_sim_real import nom_trace, SIM  # noqa: E402

OUT = R.ROOT / "data/user0/eval/access_1104/c1"
LIFT_DEG = 2.0


def events(df, hit_pk):
    t, f, w = df.time.to_numpy(float), df.force_N.to_numpy(float), df.wrist_angle_deg.to_numpy(float)
    pk, on = R.contact_events(t, f)
    rows = []
    for i, (k, s) in enumerate(zip(pk, on)):
        a = pk[i - 1] if i else max(0, s - int(0.3 / R.DT))
        lift = float(np.max(w[a:s + 1]) - w[s]) if s > a else np.nan
        rows.append(dict(t_pk=t[k], F_pk=f[k], lift=lift, hit=bool(np.any(np.abs(hit_pk - t[k]) < 0.003))))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rr, S = R.runs(), R.strikes()
    rows = []
    for (m, task), grp in rr.groupby(["method", "task"]):
        for f, kind in zip(grp.file, grp.kind_log):
            df = R.load(f, kind)
            ev = events(df, S[(S.file == f) & S.success].peak_time.to_numpy(float))
            rows.append(ev.assign(source="real", method=m, task=task, run=Path(f).stem))
        nt = nom_trace(m, task)
        sp = SIM / f"nom_{m}_{task}/strikes.csv"
        if nt is not None and sp.exists():
            st = pd.read_csv(sp)
            ev = events(nt, st[st.success].peak_time.to_numpy(float))
            rows.append(ev.assign(source="sim_nom", method=m, task=task, run="nom"))
    E = pd.concat(rows, ignore_index=True)
    E["no_lift"] = E.lift < LIFT_DEG
    E.to_csv(OUT / "c1b_lift.csv", index=False)
    g = E.groupby(["method", "source"])
    tab = pd.DataFrame({
        "events": g.size(),
        "no_lift_frac": g.no_lift.mean(),
        "hits": g.hit.sum(),
        "hits_no_lift_frac": E[E.hit].groupby(["method", "source"]).no_lift.mean(),
        "lift_med_deg": g.lift.median(),
        "F_pk_med": g.F_pk.median(),
    }).reset_index()
    with open(OUT / "c1b_lift.md", "w") as fp:
        fp.write(f"# 打点の前の手首の持ち上げ（持ち上げなし = {LIFT_DEG}° 未満）\n\n出典：scripts/tools/user0/access/c1b_lift.py。"
                 "sim = DR なし（run_1104.sh trace）、実機 = 本表の run。events は接触イベントの総数（sim は 1 本、実機は 3 本の合計）\n\n")
        fp.write(tab.round(3).to_markdown(index=False) + "\n")
    print(tab.round(3).to_string(index=False))

    # 例：F0・double160 の同じ区間（sim と実機 1 本）
    task = "double_lead2_bpm160"
    nt = nom_trace("F0", task)
    f = rr[(rr.method == "F0") & (rr.task == task)].iloc[0]
    df = R.load(f.file, f.kind_log)
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for d, c, lab in [(df, "#2a78d6", "real"), (nt, "#eb6834", "sim (nominal)")]:
        ax[0].plot(d.time, d.cmd_F if lab == "real" else d.cmd_F, color=c, lw=1, ls="--" if lab != "real" else "-", label=f"F command ({lab})")
        ax[1].plot(d.time, d.wrist_angle_deg, color=c, lw=1.4, label=lab)
        ax[2].plot(d.time, d.force_N, color=c, lw=1.0, label=lab)
    ons = np.load(SIM / f"nom_F0_{task}/trace.npz")["onsets"]
    for a in ax:
        for t in ons:
            a.axvline(t, color="#999", lw=0.5, alpha=0.5)
        a.grid(alpha=0.25, lw=0.5)
    ax[0].set_ylabel("F cmd [MPa]"); ax[1].set_ylabel("wrist [deg]"); ax[2].set_ylabel("force [N]")
    ax[1].legend(frameon=False, fontsize=8); ax[2].set_xlabel("time [s]"); ax[2].set_xlim(2.3, 3.6)
    fig.suptitle("F0, double160, same command: gray lines = targets", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "c1b_example.png", dpi=200); fig.savefig(OUT / "c1b_example.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
