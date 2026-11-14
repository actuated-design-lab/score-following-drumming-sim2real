"""analyze_probe.py — 段 1（ff_sim.py probe）の集計（Isaac も GPU も使わない）

入力: <dir>/probe_envs.npz・params.csv・probe_meta.json
出力: <dir>/summary.md・probe_table.csv・fig_probe.png

1 打ごとに、発火（指令を振り下ろしに切り替えた時刻）から最初の打撃（力のピーク ≥ 1 N、採点と同じしきい値）までの時間を L、
そのピーク力を F とする。最初のピークから 150 ms 以内のピークの数を接触の回数（2 以上 = 跳ね返り）とする。
力は env のログの force_z と同じ量（ff_sim.py）。

  python scripts/tools/user0/ff/analyze_probe.py --dir data/user0/ff/probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff_lib as FL  # noqa: E402

THR_N = FL.MIN_FRAC * FL.TARGET_N     # 1 N
PEAK_DIST_S = 0.03
BOUNCE_WIN_S = 0.15
SEARCH_S = 0.6


def strike_after(t, f, t_from, t_to):
    """[t_from, t_to) の最初の打撃：(L [s], ピーク力, 接触の回数)。無ければ (nan, 0, 0)"""
    m = (t >= t_from) & (t < t_to)
    if not m.any():
        return np.nan, 0.0, 0
    tt, ff = t[m], f[m]
    dt = float(np.median(np.diff(t)))
    idx, _ = find_peaks(ff, height=THR_N, distance=max(1, int(round(PEAK_DIST_S / dt))))
    if len(idx) == 0:
        return np.nan, 0.0, 0
    t1 = tt[idx[0]]
    n = int(np.sum(tt[idx] <= t1 + BOUNCE_WIN_S))
    return float(t1 - t_from), float(ff[idx[0]]), n


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/user0/ff/probe")
    a = ap.parse_args()
    d = Path(a.dir)
    Z = np.load(d / "probe_envs.npz")
    P = pd.read_csv(d / "params.csv")
    meta = json.loads((d / "probe_meta.json").read_text(encoding="utf-8"))
    t = Z["time"]
    fz, fm, wr = Z["force_z"], Z["force_mag"], Z["wrist_deg"]
    # 符号の確認：force_z（採点に使う量）が打撃で正に出ているか
    sign_note = ""
    if fz.max() < THR_N <= fm.max():
        sign_note = "**force_z が 1 N を超えない（力の大きさは超える）。z の向きが逆の可能性があるので、力の大きさで集計した。**"
        fz = fm
    rows = []
    dt = float(np.median(np.diff(t)))
    for r in P.itertuples():
        fires = [round(x / FL.CTRL_DT) * FL.CTRL_DT for x in json.loads(r.fires)]
        w0 = float(wr[np.searchsorted(t, fires[0]) - 1, r.env])
        base = dict(env=r.env, group=r.group, ioi_ms=r.ioi_ms, a_up=r.a_up, a_dn=r.a_dn, h=r.h, g=r.g,
                    wrist_at_fire_deg=round(w0, 1))
        if len(fires) == 1:
            L, F, nc = strike_after(t, fz[:, r.env], fires[0], fires[0] + SEARCH_S)
            rows.append(dict(**base, strike=1, L_ms=round(L * 1000) if np.isfinite(L) else np.nan, F_N=round(F, 1),
                             contacts=nc))
            continue
        # 2 打：1 打目の発火から 2 打目の発火 + SEARCH_S までのピーク（採点と同じ 1 N）を時刻順に並べ、
        # 最初のピークを 1 打目、それより後の最初のピークを 2 打目とする（跳ね返りと 2 打目は区別しない。採点も区別しない）
        m = (t >= fires[0]) & (t < fires[1] + SEARCH_S)
        idx, _ = find_peaks(fz[m, r.env], height=THR_N, distance=max(1, int(round(PEAK_DIST_S / dt))))
        tp, Fp = t[m][idx], fz[m, r.env][idx]
        for i in range(2):
            ok = i < len(tp)
            rows.append(dict(**base, strike=i + 1, L_ms=round((tp[i] - fires[i]) * 1000) if ok else np.nan,
                             F_N=round(float(Fp[i]), 1) if ok else 0.0, contacts=len(tp)))
    S = pd.DataFrame(rows)
    S.to_csv(d / "probe_table.csv", index=False)

    A = S[S.group == "A"]
    hitA = A[A.F_N >= THR_N]
    L_by = A.pivot_table(index="a_dn", columns="a_up", values="L_ms", aggfunc="median").round(0)
    F_by = A.pivot_table(index="a_dn", columns="h", values="F_N", aggfunc="median").round(1)
    hit_by = A.assign(hit=A.F_N >= THR_N).pivot_table(index="a_dn", columns="a_up", values="hit", aggfunc="mean").round(2)
    near20 = A[(A.F_N >= 15) & (A.F_N <= 25)].sort_values("L_ms")
    B = S[S.group == "B"]
    B2 = B[B.strike == 2].copy()
    B1 = B[B.strike == 1].set_index("env")
    B2["F1_N"] = B2.env.map(B1.F_N)
    B2["二打とも"] = (B2.F_N >= THR_N) & (B2.F1_N >= THR_N)

    # 図
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for an, c in zip(sorted(A.a_dn.unique()), ("#0072B2", "#E69F00", "#D55E00")):
        s = hitA[hitA.a_dn == an]
        ax[0].scatter(s.wrist_at_fire_deg, s.L_ms, s=18, color=c, label=f"a_dn {an}")
        ax[1].scatter(s.h * 1000, s.F_N, s=18, color=c, label=f"a_dn {an}")
    ax[0].set_xlabel("発火時の手首角 [deg]"); ax[0].set_ylabel("L（発火→打撃）[ms]"); ax[0].legend(fontsize=8)
    ax[1].axhline(FL.TARGET_N, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel("保持時間 h [ms]"); ax[1].set_ylabel("ピーク力 [N]"); ax[1].legend(fontsize=8)
    for (au, h), s in B2.groupby(["a_up", "h"]):
        ax[2].plot(s.ioi_ms, s.F_N, marker="o", label=f"a_up {au}・h {h}")
    ax[2].axhline(FL.TARGET_N, color="k", lw=0.8, ls=":")
    ax[2].set_xlabel("2 打の間隔 [ms]"); ax[2].set_ylabel("2 打目のピーク力 [N]"); ax[2].legend(fontsize=8)
    plt.suptitle("段 1：Gen2（DR なし）で 1 打・2 打を叩かせたときの L と打撃力")
    plt.tight_layout()
    plt.savefig(d / "fig_probe.png", dpi=110)
    plt.close()

    Lq = hitA.L_ms.quantile([0.1, 0.5, 0.9]).round(0).tolist() if len(hitA) else [np.nan] * 3
    out = [f"# FF 段 1：指令から打撃までの時間 L と打撃力（{meta['env_cfg']}）", "",
           f"env {meta['num_envs']} 本、k_t {meta['impact_force_scale']}、物理 {meta['dt_phys'] * 1000:.0f} ms 刻み。"
           f"発火 = F を振り下ろしに切り替えた指令の時刻（{meta['t_settle']} s、待機で姿勢を落ち着かせたあと）。"
           f"L = 発火 → 最初の打撃（ピーク ≥ 1 N）。接触 = 最初のピークから 150 ms 以内のピークの数。"
           "値の意味は `ff_lib.Prim`（a_up・a_dn・g は MPa、h は s）", ""]
    if sign_note:
        out += [sign_note, ""]
    out += ["## 群 A：1 打（待機の a_up × 振り下ろしの a_dn × 保持 h × G の g）", "",
            f"打撃が出た組 {len(hitA)} / {len(A)}。L の 10 / 50 / 90 % 点 = {Lq[0]:.0f} / {Lq[1]:.0f} / {Lq[2]:.0f} ms。"
            f"跳ね返り（接触 2 以上）の割合 {float((hitA.contacts >= 2).mean()) if len(hitA) else float('nan'):.2f}", "",
            "打撃が出た割合（行 a_dn、列 a_up。h・g をまとめて）", "", md_table(hit_by.reset_index()), "",
            "L の中央値 [ms]（行 a_dn、列 a_up）", "", md_table(L_by.reset_index()), "",
            "ピーク力の中央値 [N]（行 a_dn、列 h）", "", md_table(F_by.reset_index()), "",
            f"ピーク力が 15〜25 N の組（{len(near20)} 組、L の短い順に最大 15 組）", "",
            md_table(near20.head(15)[["a_up", "a_dn", "h", "g", "wrist_at_fire_deg", "L_ms", "F_N", "contacts"]]), "",
            "## 群 B：2 打（間隔 IOI）", "",
            "1 打目の発火から 2 打目の発火 + 0.6 s までの打撃のピーク（1 N 以上）を時刻順に並べ、最初を 1 打目、次を 2 打目とした。"
            "跳ね返りと 2 打目は区別していない（採点も区別しない）。L2 は 2 打目の発火から（負なら 2 打目の指令より前に当たっている）。"
            "contacts はこの区間のピークの数", "",
            md_table(B2[["a_up", "a_dn", "h", "g", "ioi_ms", "F1_N", "L_ms", "F_N", "contacts", "二打とも"]]
                     .rename(columns={"L_ms": "L2_ms", "F_N": "F2_N"})), "",
            "![](fig_probe.png)", "",
            "全組の値は `probe_table.csv`。"]
    (d / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main()
