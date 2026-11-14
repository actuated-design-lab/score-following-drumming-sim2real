"""hys_loops.py — ヒステリシスのループを実機と sim で比べる（run_hys_sweep.py の出力を読む）

指標
  tm_A_quasistatic（DF・F の 0.05 Hz 三角波）の DF 三角波 2 周（tri0, tri1）で、手首角 θ と DF 圧力 P のループについて
    ・ΔP@θ : 同じ角度での「下げの枝 − 上げの枝」の圧力差の中央値 [kPa]（ループの横幅）
    ・Δθ@P : 同じ圧力での「上げの枝 − 下げの枝」の角度差の中央値 [deg]（ループの縦幅）
    ・θ の範囲、手首角 RMSE（その三角波の区間）
  全動作（stage2）：手首角の RMSE、平均ずれ（静的）、平均ずれを引いた RMSE（動的）。最初の 2 s は除く
実機は jetson_project の実測ログ（scripts/tools/user0/jfps2026/real_log.py で DF/F の入れ替わりを戻し、エコーで時刻を合わせる）

使い方:
  python analysis/eval/hys_loops.py                       # data/user0/jfps2026/replay/hys/ を全部読む
  python analysis/eval/hys_loops.py --fig play_w20_cp15   # ループの図に重ねる play の条件
出力（<hys_dir>/summary/）: loops.csv, angles.csv, summary.md, loops_tm_A.png
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools", "user0", "jfps2026"))
import real_log as RL  # noqa: E402

DT = 0.005
SKIP_S = 2.0
NAME_RE = re.compile(r"sim_(?P<motion>.+?)__(?P<pm>[^_].*?)__(?P<hys>.+)\.csv$")


def load_real(jetson, motion):
    cmd = pd.read_csv(RL.jetson_signal_path(jetson, motion))
    c3 = cmd[["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].values
    A, _ = RL.align_real_log(pd.read_csv(RL.find_real_log(jetson, motion)), c3, DT,
                             columns=RL.PRES_COLS + ("wrist_angle_deg",))
    real = pd.DataFrame(A, columns=["P_DF", "P_F", "P_G", "wrist"])
    seg_path = RL.jetson_signal_path(jetson, motion).replace(".csv", "_annotated.csv")
    seg = pd.read_csv(seg_path).segment.values if os.path.exists(seg_path) else None
    if seg is not None:
        real["segment"] = seg[np.clip((np.arange(len(real)) * DT / RL.CMD_DT).astype(int), 0, len(seg) - 1)]
    return real


def smooth(x, n=21):
    return np.convolve(x, np.ones(n) / n, "same")


def loop_width(p, a):
    """ループの横幅 ΔP@θ（下げ−上げ）と縦幅 Δθ@P（上げ−下げ）"""
    p, a = smooth(p), smooth(a); dp = np.gradient(p); up, dn = dp > 0, dp < 0
    ga = np.linspace(np.percentile(a, 15), np.percentile(a, 85), 25)
    gp = np.linspace(np.percentile(p, 15), np.percentile(p, 85), 25)

    def br(x, y, m, g):
        o = np.argsort(x[m]); return np.interp(g, x[m][o], y[m][o])
    dP = np.median(br(a, p, dn, ga) - br(a, p, up, ga))
    dA = np.median(br(p, a, up, gp) - br(p, a, dn, gp))
    return dP * 1000, dA


def to_md(df):
    """markdown の表（tabulate が無い環境では to_string() で代用）"""
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hys_dir", default=os.path.join(ROOT, "data/user0/jfps2026/replay/hys"))
    ap.add_argument("--jetson", default=os.environ.get("JETSON_PROJECT", os.path.join(os.path.dirname(ROOT), "jetson_project")))
    ap.add_argument("--fig", default=None, help="ループの図に重ねる play の条件（例 play_w20_cp15）")
    a = ap.parse_args()
    out = os.path.join(a.hys_dir, "summary"); os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.hys_dir, "sim_*__*__*.csv")))
    if not files:
        raise SystemExit(f"{a.hys_dir} に sim_<動作>__<圧力モデル>__<ヒステリシス>.csv が無い")
    reals, loops, angs, sims_A = {}, [], [], {}
    for f in files:
        mm = NAME_RE.search(os.path.basename(f)); motion, pm, hys = mm["motion"], mm["pm"], mm["hys"]
        if motion not in reals:
            reals[motion] = load_real(a.jetson, motion)
        r = reals[motion]; s = pd.read_csv(f); n = min(len(s), len(r)); i0 = int(SKIP_S / DT)
        e = s.wrist_angle_deg.values[i0:n] - r.wrist.values[i0:n]
        angs.append(dict(motion=motion, pmodel=pm, hys=hys, rmse=np.sqrt(np.mean(e ** 2)), bias=e.mean(),
                         rmse_dyn=np.sqrt(np.mean((e - e.mean()) ** 2))))
        if motion == "tm_A_quasistatic" and "segment" in r:
            sims_A[(pm, hys)] = s
            for k in (0, 1):
                ii = np.flatnonzero(r.segment.values[:n] == f"A_DF_tri_{k}")
                dP_s, dA_s = loop_width(s.P_out_DF.values[ii], s.wrist_angle_deg.values[ii])
                loops.append(dict(pmodel=pm, hys=hys, tri=k, dP_kPa=dP_s, dA_deg=dA_s,
                                  theta_min=s.wrist_angle_deg.values[ii].min(), theta_max=s.wrist_angle_deg.values[ii].max(),
                                  rmse=np.sqrt(np.mean((s.wrist_angle_deg.values[ii] - r.wrist.values[ii]) ** 2))))
    if "tm_A_quasistatic" in reals and "segment" in reals["tm_A_quasistatic"]:
        r = reals["tm_A_quasistatic"]
        for k in (0, 1):
            ii = np.flatnonzero(r.segment.values == f"A_DF_tri_{k}")
            dP, dA = loop_width(r.P_DF.values[ii], r.wrist.values[ii])
            loops.append(dict(pmodel="real", hys="real", tri=k, dP_kPa=dP, dA_deg=dA,
                              theta_min=r.wrist.values[ii].min(), theta_max=r.wrist.values[ii].max(), rmse=0.0))
    L = pd.DataFrame(loops); A = pd.DataFrame(angs)
    L.to_csv(os.path.join(out, "loops.csv"), index=False, float_format="%.2f")
    A.to_csv(os.path.join(out, "angles.csv"), index=False, float_format="%.2f")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fo:
        fo.write("# ヒステリシスの比較（hys_loops.py）\n\n## tm_A：DF 三角波のループ（tri0 / tri1）\n\n")
        if len(L):
            P = L.pivot_table(index=["pmodel", "hys"], columns="tri", values=["dP_kPa", "dA_deg", "rmse"]).round(1)
            fo.write(to_md(P) + "\n\n")
        fo.write("## 手首角（全動作）\n\n")
        fo.write(to_md(A.pivot_table(index="motion", columns=["pmodel", "hys"], values="rmse").round(1)) + "\n")
    print(open(os.path.join(out, "summary.md"), encoding="utf-8").read())
    if len(sims_A):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
                             "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
        r = reals["tm_A_quasistatic"]; ii = np.flatnonzero(r.segment.values == "A_DF_tri_1")
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        ax.plot(smooth(r.P_DF.values[ii]), smooth(r.wrist.values[ii]), color="#1b2430", lw=1.6, label="Measured")
        show = [(("orificeV2", "relay"), "#4a5568", "-", "Sim, rate-based, flow-limited P"),
                (("measured_raw", "relay"), "#8a8985", "--", "Sim, rate-based, measured P")]
        if a.fig:
            show.append((("measured_raw", a.fig), "#c2410c", "-", "Sim, play, measured P"))
        for key, col, ls, lab in show:
            if key in sims_A:
                s = sims_A[key]
                ax.plot(smooth(s.P_out_DF.values[ii]), smooth(s.wrist_angle_deg.values[ii]), color=col, ls=ls, lw=1.1, label=lab)
        ax.set_xlabel("DF pressure [MPa]"); ax.set_ylabel("Wrist angle [deg]"); ax.grid(alpha=0.25, lw=0.5)
        h_, l_ = ax.get_legend_handles_labels()   # 凡例は図の上の外に置く（中に置くとどこでも線に重なる）
        fig.legend(h_, l_, frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.6, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"loops_tm_A.{ext}"), dpi=200)


if __name__ == "__main__":
    main()
