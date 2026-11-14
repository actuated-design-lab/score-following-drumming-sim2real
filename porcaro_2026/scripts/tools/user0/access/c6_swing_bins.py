"""c6_swing_bins.py — 「F0 の区分 2（前打間隔 200〜320 ms）の a_dn = 0.058 MPa は実機では弱すぎる」仮説の確かめ（2026-11-05）

実機の F0（access2026v2 の本表 15 本）と、同じ F0 指令（opt2 r0 favorite）の sim（公称・DR なし、diag_opt2.sh の
r0_favorite_<task>/trace.npz）で、打点ごとに次を出し、FF のプリミティブの区分（前打間隔、ff_lib.ioi_bins）で集計する。

  ω_dn   ：手首の振り下ろしの角速度のピーク [deg/s]（窓の中の最上点から最下点までの −dθ/dt の最大。θ は 25 ms の移動平均）
  A      ：手首の角度の振れ幅 [deg]（窓の中で、最下点より前の最大 − 最下点）
  θ_top  ：振れ幅の上端 [deg]（振り下ろしを始めた高さ）
  P_F    ：F の実測圧のピーク [MPa]（発火 t_k − L から t_k + 30 ms まで）
  窓：t_k − 250 ms から t_k + 40 ms。ただし直前の目標時刻より前には広げない

区分は sim の trace の onsets（ff_lib が指令を作るときと同じ 20 ms 格子の目標時刻）から決め、実機の打点とは番号で対応させる。
実機の指令の F が、各打点の発火の窓の中で区分の a_dn と一致するかも数える（区分の対応の確認）。

実機の圧力の列：ログの meas_pres_F / meas_pres_G のうち、指令の F に（遅れを許して）よく追従する方を F の実測圧とする。
11/03 以降のログでは meas_pres_G の方が指令の F に追従する（config/pressure_slots.yaml の並びと合っていない疑い。別に報告）。

  python scripts/tools/user0/access/c6_swing_bins.py
出力：data/user0/eval/access_1104/c6/{strikes.csv, summary.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
import ff_lib as FL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--jetson", default=str(ROOT.parent / "jetson_project"))
ap.add_argument("--real", default="data/user0/access2026v2/core2026_score/strikes.csv", help="jetson_project からの相対パス")
ap.add_argument("--sim_dir", default="data/user0/ff/opt2/diag")
ap.add_argument("--sim_prefix", default="r0_favorite_")
ap.add_argument("--ff", default="data/user0/ff/opt2/r0/favorite.json")
ap.add_argument("--out", default="data/user0/eval/access_1104/c6")
ap.add_argument("--real_only", action="store_true",
                help="実機だけ集計する（sim の trace は目標時刻と区分を取るためだけに使う。既定で 11/04 の nom_F0_*、git に入っている）。"
                     "出力は real_only.md と strikes_real.csv")
a = ap.parse_args()

if a.real_only and a.sim_dir == "data/user0/ff/opt2/diag":
    a.sim_dir, a.sim_prefix = "data/user0/eval/access_1104/sim", "nom_F0_"     # 目標時刻は MIDI だけで決まるので、どの F0 の trace でも同じ
J = Path(a.jetson)
sys.path.insert(0, str(J / "tools"))
from core_score import playback_time  # noqa: E402  （FF のログの時刻を core_score と同じ定義に）

PR = [FL.Prim.from_dict(d).clipped() for d in json.load(open(ROOT / a.ff))["prims"]]
NB = len(PR)
BIN_NAME = ["< 120", "120〜200", "200〜320", "≥ 320"]
PRE, POST, FWIN = 0.250, 0.040, 0.030


def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def per_strike(t, th, pF, onsets, bins):
    """打点ごとの ω_dn・A・θ_top・P_F。t は等間隔"""
    dt = float(np.median(np.diff(t)))
    ths = smooth(th, max(1, int(round(0.025 / dt))))
    om = -np.gradient(ths, dt)
    rows = []
    for k, (tk, b) in enumerate(zip(onsets, bins)):
        t0 = tk - PRE if k == 0 else max(tk - PRE, onsets[k - 1])
        w = (t >= t0) & (t <= tk + POST)
        if w.sum() < 3:
            rows.append(dict(omega_dn=np.nan, A=np.nan, th_top=np.nan, th_bot=np.nan, P_F=np.nan))
            continue
        xw, ow = ths[w], om[w]
        ib = int(np.argmin(xw))
        it = int(np.argmax(xw[: ib + 1]))
        top = float(xw[it])
        fw = (t >= tk - PR[b].L) & (t <= tk + FWIN)
        rows.append(dict(omega_dn=float(np.max(ow[it: ib + 1])), A=top - float(xw[ib]), th_top=top, th_bot=float(xw[ib]),
                         P_F=float(np.max(pF[fw])) if fw.any() else np.nan))
    return pd.DataFrame(rows)


def lagged_r(c, m, max_lag=40):
    return max(np.corrcoef(c[: len(c) - l], m[l:])[0, 1] for l in range(0, max_lag, 2))


# --------------------------------------------------------------------------- sim
sim, ons_by_task = [], {}
for d in sorted((ROOT / a.sim_dir).glob(a.sim_prefix + "*")):
    f = d / "trace.npz"
    if not f.exists():
        continue
    task = d.name[len(a.sim_prefix):]
    z = np.load(f)
    on = np.asarray(z["onsets"], float)
    b = FL.ioi_bins(on)
    ons_by_task[task] = (on, b)
    s = per_strike(z["time"], z["q"][:, 0].astype(float), z["P_out"][:, 1].astype(float), on, b)
    sim.append(s.assign(src="sim", task=task, strike_idx=np.arange(len(on)), bin=b, file=str(f.relative_to(ROOT))))
if not sim:
    raise SystemExit(f"sim の trace がない：{ROOT / a.sim_dir}/{a.sim_prefix}*/trace.npz（WS で diag_opt2.sh を回したあとに実行）")
S = pd.concat(sim, ignore_index=True)

# --------------------------------------------------------------------------- 実機
RS = pd.read_csv(J / a.real)
RS = RS[(RS.method == "F0") & (~RS.bridge.astype(bool))]
real, chk, colinfo = [], [], []
for fn, x in RS.groupby("file"):
    task = x.task.iloc[0]
    if task not in ons_by_task:
        print(f"[c6] sim に {task} がないので飛ばす")
        continue
    on_sim, b = ons_by_task[task]
    x = x.sort_values("strike_idx")
    if len(x) != len(on_sim):
        print(f"[c6] 打点の数が合わない（実機 {len(x)}、sim {len(on_sim)}）：{fn}")
        continue
    df = playback_time(pd.read_csv(J / fn))
    t = df["time"].to_numpy(float)
    rF, rG = lagged_r(df.cmd_F.to_numpy(float), df.meas_pres_F.to_numpy(float)), lagged_r(df.cmd_F.to_numpy(float), df.meas_pres_G.to_numpy(float))
    col = "meas_pres_F" if rF >= rG else "meas_pres_G"
    colinfo.append(dict(file=fn, r_cmdF_measF=rF, r_cmdF_measG=rG, used=col))
    on = x.target_time.to_numpy(float)
    s = per_strike(t, df.wrist_angle_deg.to_numpy(float), df[col].to_numpy(float), on, b)
    # 区分の対応の確認：発火の窓の中ほどの指令の F が、その区分の a_dn と合うか
    cf = df.cmd_F.to_numpy(float)
    for k, (tk, bk) in enumerate(zip(on, b)):
        i = np.searchsorted(t, tk - PR[bk].L + min(PR[bk].h, PR[bk].L) / 2)
        chk.append(abs(cf[min(i, len(cf) - 1)] - PR[bk].a_dn) < 0.01)
    real.append(s.assign(src="real", task=task, strike_idx=x.strike_idx.to_numpy(), bin=b, file=fn,
                         success=x.success.to_numpy(), peak_force=x.peak_force.to_numpy()))
R = pd.concat(real, ignore_index=True)
CI = pd.DataFrame(colinfo)

out = ROOT / a.out
out.mkdir(parents=True, exist_ok=True)
R["prev_bin"] = R.sort_values(["file", "strike_idx"]).groupby("file").bin.shift(1)
if a.real_only:
    R.to_csv(out / "strikes_real.csv", index=False)
    L = ["# F0 の区分ごとの振り下ろし：実機だけ（c6_swing_bins.py --real_only）", "",
         f"実機：jetson_project `{a.real}` の F0（本表 {R.file.nunique()} 本、{len(R)} 打点）。区分は FF が指令を作るときの区分（ff_lib.ioi_bins、20 ms 格子の目標時刻）。"
         f"区分の対応の確認：指令の F が発火の窓の中ほどで区分の a_dn と一致した打点 {np.mean(chk):.3f}", "",
         f"F の実測圧の列：{CI.used.value_counts().to_dict()}（指令の F との相関の中央値 meas_pres_F {CI.r_cmdF_measF.median():.2f}、meas_pres_G {CI.r_cmdF_measG.median():.2f}）", "",
         "## 区分ごと（中央値）", "", "| 区分 | a_dn [MPa] | 打点 | ω_dn [deg/s] | A [deg] | θ_top [deg] | θ_bot [deg] | P_F [MPa] | 成功率 |", "|---|---|---|---|---|---|---|---|---|"]
    for b, r in R.groupby("bin"):
        L.append(f"| {b}（{BIN_NAME[b]}） | {PR[b].a_dn:.3f} | {len(r)} | {r.omega_dn.median():.0f} | {r.A.median():.1f} | {r.th_top.median():.1f} | "
                 f"{r.th_bot.median():.1f} | {r.P_F.median():.3f} | {r.success.mean():.3f} |")
    L += ["", "## 区分 × 直前の打点の区分（中央値）", "", "| 区分 | 直前の区分 | 打点 | ω_dn [deg/s] | A [deg] | θ_top [deg] | 成功率 |", "|---|---|---|---|---|---|---|"]
    for (b, pb), r in R.groupby(["bin", "prev_bin"]):
        L.append(f"| {b}（{BIN_NAME[b]}） | {int(pb)}（{BIN_NAME[int(pb)]}） | {len(r)} | {r.omega_dn.median():.0f} | {r.A.median():.1f} | {r.th_top.median():.1f} | {r.success.mean():.3f} |")
    L += ["", "最初の打点（直前なし）は下の表に入れていない"]
    open(out / "real_only.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    raise SystemExit(0)
pd.concat([R, S], ignore_index=True).to_csv(out / "strikes.csv", index=False)


def med_iqr(v, nd=1):
    v = pd.Series(v).dropna()
    if v.empty:
        return "—"
    return f"{v.median():.{nd}f} [{v.quantile(.25):.{nd}f}, {v.quantile(.75):.{nd}f}]"


L = ["# F0 の区分ごとの振り下ろし：実機と sim（c6_swing_bins.py）", "",
     f"実機：jetson_project `{a.real}` の F0（本表 {R.file.nunique()} 本、{len(R)} 打点）。sim：`{a.sim_dir}/{a.sim_prefix}*`（公称・DR なし、{S.task.nunique()} タスク、{len(S)} 打点）。"
     f"指令：`{a.ff}`", "",
     f"区分の対応の確認：実機の指令の F が発火の窓の中ほどで区分の a_dn と一致した打点 {np.mean(chk):.3f}（{len(chk)} 打点）", "",
     f"実機の F の実測圧の列：{CI.used.value_counts().to_dict()}（指令の F との相関 meas_pres_F {CI.r_cmdF_measF.median():.2f}、"
     f"meas_pres_G {CI.r_cmdF_measG.median():.2f}、中央値）", "",
     "値は中央値 [四分位]。ω_dn = 振り下ろしの角速度のピーク、A = 角度の振れ幅、θ_top = 振り下ろしを始めた高さ、P_F = 発火から t_k + 30 ms までの F の実測圧のピーク", "",
     "| 区分（前打間隔 ms） | a_dn [MPa] | L / h [ms] | 打点 実機 / sim | ω_dn 実機 [deg/s] | ω_dn sim | 比 | A 実機 [deg] | A sim | 比 | θ_top 実機 | θ_top sim | P_F 実機 [MPa] | P_F sim | 成功率 実機 |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for b in range(NB):
    r, s = R[R.bin == b], S[S.bin == b]
    if r.empty and s.empty:
        continue
    L.append(f"| {b}（{BIN_NAME[b]}） | {PR[b].a_dn:.3f} | {PR[b].L * 1000:.0f} / {PR[b].h * 1000:.0f} | {len(r)} / {len(s)} | "
             f"{med_iqr(r.omega_dn, 0)} | {med_iqr(s.omega_dn, 0)} | {r.omega_dn.median() / s.omega_dn.median():.2f} | "
             f"{med_iqr(r.A)} | {med_iqr(s.A)} | {r.A.median() / s.A.median():.2f} | {med_iqr(r.th_top)} | {med_iqr(s.th_top)} | "
             f"{med_iqr(r.P_F, 3)} | {med_iqr(s.P_F, 3)} | {r.success.mean():.3f} |")
L += ["", "## 区分 × タスク（中央値。比 = 実機 / sim）", "",
      "| 区分 | タスク | 打点 実機 / sim | ω_dn 実機 / sim | 比 | A 実機 / sim | 比 | P_F 実機 / sim |", "|---|---|---|---|---|---|---|---|"]
for (b, t), r in R.groupby(["bin", "task"]):
    s = S[(S.bin == b) & (S.task == t)]
    if s.empty:
        continue
    L.append(f"| {b} | {t} | {len(r)} / {len(s)} | {r.omega_dn.median():.0f} / {s.omega_dn.median():.0f} | {r.omega_dn.median() / s.omega_dn.median():.2f} | "
             f"{r.A.median():.1f} / {s.A.median():.1f} | {r.A.median() / s.A.median():.2f} | {r.P_F.median():.3f} / {s.P_F.median():.3f} |")
open(out / "summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
