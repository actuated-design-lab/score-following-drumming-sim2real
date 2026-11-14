"""compare_hys_modes.py — PAM 摩擦の向きの決め方（play＝圧力 / play_length＝筋の長さ）で再生した sim を、実機と手首角で比べる

入力: --dir 以下の sim_<動作>__<タグ>.csv（replay_open_loop.py の出力。サブフォルダも探す）と、同名の .json（real_log の場所）
      例: sim_tm_B_steps__play_p.csv、static_len003/sim_id_static_cocontraction__len003.csv
指標（手首角、最初の --skip_s 秒は除く）:
  RMSE、平均ずれ（sim − 実機）、止まっているとき（実機の角速度 < 5 deg/s）の RMSE、動いているとき（> 50 deg/s）の RMSE
  遅れ lag_ms：sim を時間方向にずらして RMSE が最小になるずれ（±300 ms、正 = sim が実機より遅い）と、そのときの RMSE
  振れ幅の比 amp：sim と実機の手首角の標準偏差の比（1 より大きい = sim の方が大きく振れる）
  グリップ角：RMSE、平均ずれ、振れ幅の比（grip_rmse / grip_offset / grip_amp）
出力: <dir>/summary.md（動作 × タグの表）、<dir>/metrics.csv
  --plot タグ,タグ,...：その動作ごとに、実機とそのタグの sim の手首角を重ねた図（<dir>/fig_<動作>.png、--plot_t0 から 6 s）

使い方:
  python analysis/eval/compare_hys_modes.py --dir data/user0/jfps2026/replay/hys_length
  python analysis/eval/compare_hys_modes.py --dir data/user0/jfps2026/replay/hys_length --plot play_p,measDFk,kd02
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools", "user0", "jfps2026"))
import real_log as RL  # noqa: E402

DT = 0.005


def load_real(jetson, motion, real_log):
    cmd = pd.read_csv(RL.jetson_signal_path(jetson, motion))
    c3 = cmd[["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].values
    path = real_log if real_log and os.path.exists(real_log) else RL.find_real_log(jetson, motion, os.path.basename(real_log or "") or None)
    A, _ = RL.align_real_log(pd.read_csv(path), c3, DT, columns=RL.PRES_COLS + ("wrist_angle_deg", "grip_angle_deg"))
    return A[:, 3], path, A[:, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--jetson", default=os.environ.get("JETSON_PROJECT", os.path.join(os.path.dirname(ROOT), "jetson_project")))
    ap.add_argument("--skip_s", type=float, default=2.0)
    ap.add_argument("--plot", default="")
    ap.add_argument("--plot_t0", type=float, default=10.0)
    a = ap.parse_args()
    rows, cache, sims = [], {}, {}
    for f in sorted(glob.glob(os.path.join(a.dir, "**", "sim_*__*.csv"), recursive=True)):
        name = os.path.basename(f)[4:-4]
        motion, tag = name.split("__", 1)
        meta = os.path.splitext(f)[0] + ".json"
        rl = json.load(open(meta, encoding="utf-8")).get("real_log") if os.path.exists(meta) else None
        if motion not in cache:
            cache[motion] = load_real(a.jetson, motion, rl)
        real, path, real_g = cache[motion]
        sd = pd.read_csv(f)
        s = sd.wrist_angle_deg.values
        sg = sd.grip_angle_deg.values if "grip_angle_deg" in sd else np.full(len(s), np.nan)
        n = min(len(s), len(real)); i0 = int(a.skip_s / DT)
        e = (s[:n] - real[:n])[i0:]
        w = np.abs(np.gradient(np.convolve(real[:n], np.ones(9) / 9, "same"), DT))[i0:]
        rest, mov = w < 5, w > 50
        rm = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan
        # 遅れ：sim[t] と real[t − lag] を比べる（lag > 0 = sim が遅い）
        best = (np.inf, 0)
        for k in range(-int(0.3 / DT), int(0.3 / DT) + 1):
            a0, b0 = max(i0, k), min(n, n + k)
            if b0 - a0 < 100:
                continue
            v = rm(s[a0:b0] - real[a0 - k:b0 - k])
            best = min(best, (v, k))
        rows.append(dict(motion=motion, tag=tag, rmse=rm(e), offset=float(e.mean()), rmse_rest=rm(e[rest]),
                         rmse_moving=rm(e[mov]), lag_ms=best[1] * DT * 1000, rmse_lagged=best[0],
                         amp=float(np.std(s[i0:n]) / np.std(real[i0:n])),
                         grip_rmse=rm(sg[i0:n] - real_g[i0:n]), grip_offset=float(np.mean(sg[i0:n] - real_g[i0:n])),
                         grip_amp=float(np.std(sg[i0:n]) / np.std(real_g[i0:n])) if np.std(real_g[i0:n]) > 0 else np.nan,
                         rest_pct=100 * rest.mean(), moving_pct=100 * mov.mean(), real_log=os.path.basename(path)))
        sims[(motion, tag)] = s
    M = pd.DataFrame(rows)
    os.makedirs(a.dir, exist_ok=True)
    M.to_csv(os.path.join(a.dir, "metrics.csv"), index=False, float_format="%.3f")
    lines = ["# PAM 摩擦の向きの決め方の比較（compare_hys_modes.py）", "",
             f"手首角 [deg]、最初の {a.skip_s:g} s を除く。止まっている＝実機の角速度 < 5 deg/s、動いている＝ > 50 deg/s", ""]
    for col, lab in (("rmse", "RMSE"), ("rmse_rest", "止まっているときの RMSE"), ("rmse_moving", "動いているときの RMSE"),
                     ("offset", "平均ずれ（sim − 実機）"), ("lag_ms", "遅れ [ms]（正 = sim が遅い）"),
                     ("rmse_lagged", "遅れを取り除いたときの RMSE"), ("amp", "振れ幅の比（sim ÷ 実機）"),
                     ("grip_rmse", "グリップ角の RMSE"), ("grip_offset", "グリップ角の平均ずれ（sim − 実機）"),
                     ("grip_amp", "グリップ角の振れ幅の比（sim ÷ 実機）")):
        lines += [f"## {lab}", "", "```", M.pivot(index="motion", columns="tag", values=col).round(2).to_string(), "```", ""]
    open(os.path.join(a.dir, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    tags = [t for t in a.plot.split(",") if t]
    if tags:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0"]
        for motion, (real, _, _) in cache.items():
            have = [t for t in tags if (motion, t) in sims]
            if not have:
                continue
            t = np.arange(len(real)) * DT
            i0 = min(int(a.plot_t0 / DT), max(len(real) - int(6 / DT), 0))
            i1 = min(i0 + int(6 / DT), len(real))
            fig, ax = plt.subplots(figsize=(10, 3.6), facecolor="#fcfcfb")
            ax.set_facecolor("#fcfcfb")
            ax.plot(t[i0:i1], real[i0:i1], color="#0b0b0b", lw=2, label="real")
            for c, tg in zip(cols, have):
                s = sims[(motion, tg)]
                j1 = min(i1, len(s), len(t))
                ax.plot(t[i0:j1], s[i0:j1], color=c, lw=1.5, label=f"sim {tg}")
            ax.grid(True, color="#e1e0d9", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_xlabel("time [s]", fontsize=9)
            ax.set_ylabel("wrist angle [deg]", fontsize=9)
            ax.set_title(motion, fontsize=10, loc="left")
            ax.legend(fontsize=8, frameon=False, ncol=len(have) + 1)
            fig.tight_layout()
            fig.savefig(os.path.join(a.dir, f"fig_{motion}.png"), dpi=150)
            plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
