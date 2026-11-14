"""import_real_core.py — 実機コア（2026-11、Access 用）の材料を porcaro_2026 から取り込み、実行の台本を作る

porcaro_2026 の scripts/tools/user0/eval/real_core.py が作ったもの（data/user0/eval/real_core/）を読むだけ（porcaro_2026 には書かない）。
  songs/core2026/        評価のタスクの MIDI（ホールドアウト 2・GMD 138・170 の切り直し・ダブル 160）
  songs/core2026_calib/  較正の列（F3・RL+Δ の Δ を測る。評価には使わない）
  songs/core2026_rld/    RL+Δ の MIDI（目標を Δ だけ前にずらしたもの。real_core.py shift のあと）
  signals/core2026/      FF の指令（F0_*、F3_*）
  core2026_runs.csv      実行順（run_order.csv）に、実機で打つコマンドを付けたもの

  python3 tools/import_real_core.py --porcaro ../porcaro_2026
  python3 tools/import_real_core.py --porcaro ../porcaro_2026 --v 2   # 押し付けの判定を直してやり直す FF（2026-11-04）：
      指令 signals_v2/ → signals/core2026v2/、実行順 run_order_v2.csv → core2026_runs_v2.csv（FF フェーズの 40 本）
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--porcaro", default=os.path.join(ROOT, "..", "porcaro_2026"))
    ap.add_argument("--v", type=int, default=1)
    a = ap.parse_args()
    src = os.path.join(a.porcaro, "data/user0/eval/real_core")
    SIG = "core2026" if a.v == 1 else f"core2026v{a.v}"          # signals/ の下のフォルダ
    RUNS = "core2026_runs.csv" if a.v == 1 else f"core2026_runs_v{a.v}.csv"
    ORDER = "run_order.csv" if a.v == 1 else f"run_order_v{a.v}.csv"
    pairs = [("songs", "songs/core2026", "*.mid"), ("calib", "songs/core2026_calib", "*.mid"),
             ("songs_rl_delta", "songs/core2026_rld", "*.mid"),
             ("signals" if a.v == 1 else f"signals_v{a.v}", f"signals/{SIG}", "*.csv")]
    for s, d, pat in pairs:
        fs = sorted(glob.glob(os.path.join(src, s, pat)))
        if not fs:
            print(f"[--] {s}：まだない")
            continue
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
        for f in fs:
            shutil.copy2(f, os.path.join(ROOT, d, os.path.basename(f)))
        print(f"[OK] {s} → {d}（{len(fs)} 個）")
    for f in ("tasks.csv", "calib.csv", "README.md"):
        shutil.copy2(os.path.join(src, f), os.path.join(ROOT, "songs/core2026", f))
    tasks = pd.read_csv(os.path.join(src, "tasks.csv")).set_index("task")
    od = pd.read_csv(os.path.join(src, ORDER))
    rld = sorted(glob.glob(os.path.join(ROOT, "songs/core2026_rld", "*.mid")))
    cmds = []
    for r in od.itertuples():
        f = tasks.loc[r.task, "file"]
        trial = int(getattr(r, "trial", r.rep))      # 日の差の橋渡し（bridge = 1）は試行番号 4（11/03 の 1〜3 と区別）
        stem = os.path.splitext(f)[0]
        if r.method in ("B", "E"):
            c = f"python3 src/deploy_policy.py --model GEN2B/{r.method} --midi songs/core2026/{f} --trial {trial}"
        elif r.method == "RL+Δ":
            base = stem.rsplit("_bpm", 1)[0]
            m = [x for x in rld if os.path.basename(x).startswith(base + "_rld")]
            c = (f"python3 src/deploy_policy.py --model GEN2B/B --midi songs/core2026_rld/{os.path.basename(m[0])} --trial {trial}"
                 if m else "（RL+Δ の MIDI がまだない：較正のあと real_core.py shift → このスクリプトをもう一度）")
        else:
            ok = os.path.exists(os.path.join(ROOT, f"signals/{SIG}", f"{r.method}_{stem}.csv"))
            c = (f"python3 tools/run_signal_playback.py {SIG}/{r.method}_{stem}" if ok
                 else f"（{r.method} の指令がまだない：real_core.py ff / shift → このスクリプトをもう一度）")
        cmds.append(c)
    od["command"] = cmds
    od.to_csv(os.path.join(ROOT, RUNS), index=False)
    print(f"[OK] {RUNS}（{len(od)} 本。未準備 {sum(c.startswith('（') for c in cmds)} 本）")
    cal = pd.read_csv(os.path.join(src, "calib.csv"))
    print("\n較正（先に叩く）：")
    for f in cal.file:
        print(f"  python3 src/deploy_policy.py --model GEN2B/B --midi songs/core2026_calib/{f} --trial 1")
        stem = os.path.splitext(f)[0]
        if os.path.exists(os.path.join(ROOT, f"signals/{SIG}", f"F0_{stem}.csv")):
            print(f"  python3 tools/run_signal_playback.py {SIG}/F0_{stem}")
    print("  → python3 tools/core_score.py <ログのフォルダ> --calib  で Δ を出す")


if __name__ == "__main__":
    main()
