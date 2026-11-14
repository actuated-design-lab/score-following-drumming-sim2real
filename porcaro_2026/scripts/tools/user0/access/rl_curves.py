"""rl_curves.py — B・E 全シードの学習曲線（iteration × 平均エピソード報酬）を TensorBoard のログから抜き出す（CPU、Isaac 不要）

入力: logs/user0/rsl_rl/porcaro_rslrl_{lstm,mlp}_gen2/<日時>_<run>/events.out.tfevents.*（rsl_rl の SummaryWriter）
      run は data/user0/gen2/train/runs_20261101_1217_b.md の 7 本（B_b_seed1–3、E_b_seed1–3、B_b_obsv1_seed1）
出力: data/user0/eval/access_1110/rl_curves/curves.csv（run, model, seed, obs, iteration, mean_reward, mean_episode_length）
      data/user0/eval/access_1110/rl_curves/summary.md（run ごとの行数・最後の iteration・最後 50 it の平均、使ったタグ）

  python scripts/tools/user0/access/rl_curves.py
  python scripts/tools/user0/access/rl_curves.py --log_root logs/rsl_rl     # ログの場所が違うとき
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

RUNS = ["B_b_seed1", "B_b_seed2", "B_b_seed3", "E_b_seed1", "E_b_seed2", "E_b_seed3", "B_b_obsv1_seed1"]
REWARD_TAGS = ("Train/mean_reward", "Train/mean_reward/time")
LEN_TAGS = ("Train/mean_episode_length",)


def find_run_dirs(roots, run):
    hits = []
    for r in roots:
        for exp in ("porcaro_rslrl_lstm_gen2", "porcaro_rslrl_mlp_gen2"):
            hits += glob.glob(os.path.join(r, exp, f"*_{run}"))
    return sorted(set(hits))


def read_scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    rt = next((t for t in REWARD_TAGS if t in tags), None)
    lt = next((t for t in LEN_TAGS if t in tags), None)
    if rt is None:
        cand = [t for t in tags if "reward" in t.lower()]
        raise SystemExit(f"!!! {run_dir}: 報酬のタグが見つからない（reward を含むタグ: {cand}）")
    rew = pd.DataFrame([(e.step, e.value) for e in ea.Scalars(rt)], columns=["iteration", "mean_reward"])
    if lt:
        ln = pd.DataFrame([(e.step, e.value) for e in ea.Scalars(lt)], columns=["iteration", "mean_episode_length"])
        rew = rew.merge(ln, on="iteration", how="left")
    # 同じ iteration が複数あれば（再開など）最後の値
    rew = rew.drop_duplicates("iteration", keep="last").sort_values("iteration")
    return rew, rt, lt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_root", nargs="+", default=["logs/user0/rsl_rl", "logs/rsl_rl"])
    ap.add_argument("--out", default="data/user0/eval/access_1110/rl_curves")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, lines = [], []
    for run in RUNS:
        dirs = find_run_dirs(a.log_root, run)
        if len(dirs) != 1:
            lines.append(f"| {run} | — | — | — | — | ログのフォルダが {len(dirs)} 個: {dirs} |")
            continue
        d = dirs[0]
        df, rt, lt = read_scalars(d)
        m = re.match(r"(B|E)_b(_obsv1)?_seed(\d)", run)
        df.insert(0, "run", run)
        df.insert(1, "model", m.group(1))
        df.insert(2, "seed", int(m.group(3)))
        df.insert(3, "obs", "v1" if m.group(2) else "v2")
        rows.append(df)
        last50 = df[df.iteration >= df.iteration.max() - 49].mean_reward.mean()
        lines.append(f"| {run} | {len(df)} | {int(df.iteration.min())}–{int(df.iteration.max())} | {df.mean_reward.iloc[-1]:.2f} | "
                     f"{last50:.2f} | {os.path.relpath(d)}（タグ {rt}{'、' + lt if lt else ''}） |")
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(os.path.join(a.out, "curves.csv"), index=False)
    md = ["# RL の学習曲線（rl_curves.py）", "",
          "平均エピソード報酬 = rsl_rl が記録する、直近に終わったエピソードの報酬の合計の平均（Train/mean_reward）。",
          "カリキュラムは iteration 300・750 で切り替わる（GEN2.md）。", "",
          "| run | 行数 | iteration | 最後の値 | 最後 50 it の平均 | ログ |", "|---|---|---|---|---|---|", *lines, "",
          f"CSV: {os.path.join(a.out, 'curves.csv')}"]
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
