"""train_dist_return.py — 選んだチェックポイント（B seed3・E seed2）を学習分布（Gen2_B、カリキュラム最終段、DR あり）で 1 エピソードずつ回し、
エピソード報酬と、その上限（全音符で R_match = 1、休符のボーナス満額、罰 0）に対する達成率を出す（WS、Isaac）

上限（エピソードごと）= 音符の数 × 1.0 ＋ 0.005 × (BPM/120) × 休符の step 数（目標波形 < 2 N の step、rewards/reward.py の rest_threshold）
  音符の数 = その env の最初のエピソードの目標波形の山（> 1 N、preview_random_rhythm.py と同じ数え方）で、エピソードの長さの中にあるもの
方策は決定的（get_inference_policy）。学習中の Train/mean_reward は行動ノイズありなので、同じではない。
env e は最初のエピソードだけ数える（終わった env はその後の報酬を足さない）。

  python scripts/tools/user0/access/train_dist_return.py --headless --model B \\
      --ckpt logs/user0/rsl_rl/porcaro_rslrl_lstm_gen2/2026-11-02_07-13-45_B_b_seed3/model_1499.pt --name B_b_seed3
  python scripts/tools/user0/access/train_dist_return.py --headless --model E \\
      --ckpt logs/user0/rsl_rl/porcaro_rslrl_mlp_gen2/<日時>_E_b_seed2/model_1499.pt --name E_b_seed2
出力: data/user0/eval/access_1110/train_dist/<name>/{episodes.csv, summary.json}、summary.md（両方そろったら表）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--model", choices=["B", "E"], required=True)
p.add_argument("--name", required=True)
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--seed", type=int, default=7100)
p.add_argument("--level", type=int, default=2, help="カリキュラムの段（2 = 最終段 = 学習の 750 it 以降の分布）")
p.add_argument("--out", default="data/user0/eval/access_1110/train_dist")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
from porcaro_2026.tasks.direct.porcaro_2026.user0 import porcaro_2026_env_cfg as C  # noqa: E402

TASK = "Porcaro-Gen2-B-user0"
REST_N, HIT_N, W_MATCH, W_REST = 2.0, 1.0, 1.0, 0.005


def main():
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner
    import porcaro_2026.tasks  # noqa: F401

    cfg = C.Porcaro2026EnvCfg_Gen2_B()
    cfg.scene.num_envs = args.num_envs
    cfg.logging.enabled = False
    cfg.seed = args.seed
    if args.model == "E":
        cfg.use_frame_stacking = True
        cfg.frame_stack_k = 5
    dt_ctrl = cfg.sim.dt * cfg.decimation
    base = C.obs_base_dim(cfg, int(cfg.lookahead_horizon / dt_ctrl))
    cfg.observation_space = base * cfg.frame_stack_k if cfg.use_frame_stacking else base
    # 重みの確認（付録 A の値と同じか）
    assert cfg.rewards.weight_match == W_MATCH and cfg.rewards.weight_rest == W_REST, (cfg.rewards.weight_match, cfg.rewards.weight_rest)

    entry = "rsl_rl_lstm_cfg_entry_point" if args.model == "B" else "rsl_rl_mlp_cfg_entry_point"
    agent_cfg = load_cfg_from_registry(TASK, entry)
    envg = gym.make(TASK, cfg=cfg)
    env = envg.unwrapped
    # カリキュラムを最終段に固定（ラッパーの最初の reset より前に。env の step はしきい値で段を決め直すので、しきい値も 0 にする）
    env.curriculum_thresholds = [-1, -1]
    env.rhythm_generator.set_curriculum_level(args.level)
    wrapper = RslRlVecEnvWrapper(envg, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.device)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed_all(args.seed + 1)
    wrapper.reset()
    gen = env.rhythm_generator
    assert gen.curriculum_level == args.level, gen.curriculum_level

    # 最初のエピソードの中身（上限の計算用）
    N = env.num_envs
    traj = gen.target_trajectories.detach().cpu().numpy().copy()
    bpm = gen.current_bpms.detach().cpu().numpy().copy()
    steps = env.episode_duration_steps.detach().cpu().numpy().copy()
    notes = np.zeros(N, int)
    rest = np.zeros(N, int)
    for e in range(N):
        tr = traj[e, : steps[e]]
        k = np.nonzero((tr[1:-1] > tr[:-2]) & (tr[1:-1] >= tr[2:]) & (tr[1:-1] > HIT_N))[0] + 1
        notes[e] = len(k)
        rest[e] = int((tr < REST_N).sum())
    ceiling = W_MATCH * notes + W_REST * (bpm / 120.0) * rest

    ret = torch.zeros(N, device=env.device)
    alive = torch.ones(N, dtype=torch.bool, device=env.device)
    ep_len = torch.zeros(N, device=env.device)
    t0 = time.time()
    with torch.inference_mode():
        o = wrapper.get_observations()
        obs = o[0] if isinstance(o, tuple) else o
        if hasattr(policy, "reset_memory"):
            policy.reset_memory()
        for _ in range(int(steps.max()) + 5):
            act = policy(obs)
            obs, rew, dones, _ = wrapper.step(act)
            ret += torch.where(alive, rew.view(-1), torch.zeros_like(ret))
            ep_len += alive.float()
            alive &= ~dones.view(-1).bool()
            if not alive.any():
                break
    print(f"[train_dist_return] {N} env、{time.time() - t0:.0f} s、終わっていない env {int(alive.sum())}")
    ret = ret.cpu().numpy()
    ep_len = ep_len.cpu().numpy()
    done = ~alive.cpu().numpy()

    import pandas as pd
    out = ROOT / args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(dict(env=np.arange(N), bpm=bpm, steps=steps, ep_len=ep_len, notes=notes, rest_steps=rest,
                           ceiling=ceiling, ret=ret, done=done))
    df.to_csv(out / "episodes.csv", index=False)
    d = df[df.done]
    summ = dict(name=args.name, ckpt=args.ckpt, model=args.model, level=args.level, seed=args.seed, n_env=N, n_done=int(len(d)),
                notes_mean=float(d.notes.mean()), notes_sd=float(d.notes.std()),
                rest_bonus_mean=float((d.ceiling - d.notes).mean()), ceiling_mean=float(d.ceiling.mean()),
                return_mean=float(d.ret.mean()), return_sd=float(d.ret.std()),
                attain_ratio_of_means=float(d.ret.mean() / d.ceiling.mean()),
                attain_mean_of_ratios=float((d.ret / d.ceiling).mean()),
                attain_ci95=float(1.96 * (d.ret / d.ceiling).std() / np.sqrt(len(d))))
    json.dump(summ, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False))

    # そろった分の表
    rows = []
    for f in sorted((ROOT / args.out).glob("*/summary.json")):
        s = json.load(open(f))
        rows.append(f"| {s['name']} | {s['n_done']} | {s['notes_mean']:.2f} ± {s['notes_sd']:.2f} | {s['ceiling_mean']:.2f} | "
                    f"{s['return_mean']:.2f} ± {s['return_sd']:.2f} | {s['attain_ratio_of_means']:.3f} | "
                    f"{s['attain_mean_of_ratios']:.3f} ± {s['attain_ci95']:.3f} |")
    md = ["# 学習分布での達成率（train_dist_return.py）", "",
          f"Gen2_B（DR あり）、カリキュラム段 {args.level}、env ごとに最初の 1 エピソード、決定的な方策。上限 = 音符 × 1.0 ＋ 休符ボーナス満額。", "",
          "| run | エピソード | 音符/エピソード | 上限の平均 | 報酬の平均 ± SD | 達成率（平均÷平均） | 達成率（比の平均 ± 95 % CI） |",
          "|---|---|---|---|---|---|---|", *rows]
    open(ROOT / args.out / "summary.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        os.write(2, ("\n[train_dist_return] エラーで止まりました:\n" + traceback.format_exc()).encode("utf-8", "replace"))
        os._exit(1)
    app.close()
