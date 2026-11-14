"""eval_b.py — (b) の列（b_sets.py）を、DR の引きを共通にして RL と FF を同じ式で採点する（WS、Isaac）

1 回の起動で 1 つの方策（RL のチェックポイント、または FF のプリミティブ）を評価する。
env の数 = 列の数 × DR の引きの数（既定 16 × 32 = 512）。env e は列 e % n_seq、引き e // n_seq。
DR（起動時の質量・摩擦・重心・手首の粘性・摩擦、リセット時の k・τ・L・通信の遅れ）は cfg.seed で決まるので、
同じ --dr_seed・同じ cfg・同じ env の数なら、方策が違っても同じ引きになる（ペア）。
  注意：観測 v1 と v2 の cfg は起動時の乱数の使い方が違う（v2 は圧力のずれを先に引く）ので、v1（保険）と v2 の間はペアにならない。
        シードの選択（B どうし、E どうし）と FF（v2 の cfg で回す）と v2 の RL の比較はペアになる。
採点は ff_score（analysis/harness/strike_extract.extract_strikes、±30 ms・1 N、1 対 1）。
力の列は play_sim_* の simulation_log.csv と同じ：k_t を掛けた接触力の z 成分を 5 ms ごと、目標はその tick の目標波形。

  python scripts/tools/user0/eval/eval_b.py --headless --set val --model B \\
      --ckpt logs/user0/rsl_rl/porcaro_rslrl_lstm_gen2/<日時>_B_b_seed1/model_1499.pt --name B_b_seed1
  python scripts/tools/user0/eval/eval_b.py --headless --set val --ff data/user0/ff/opt/<解>.json --name F0_x
  python scripts/tools/user0/eval/eval_b.py --headless --midi_dir data/user0/eval/real_core/songs --model B --ckpt ... --name B_core
出力：data/user0/eval/b_eval/<set>/<name>/strikes.csv（目標ごと）、summary.json
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
p.add_argument("--set", default="val", help="b_sets.py の val / holdout")
p.add_argument("--sets_dir", default="data/user0/eval/b_sets")
p.add_argument("--midi_dir", default=None, help="MIDI のフォルダ（例 data/user0/eval/real_core/songs）。--set の代わりに、その中の MIDI 全部を列にする")
p.add_argument("--ckpt", default=None)
p.add_argument("--model", choices=["B", "E"], default=None)
p.add_argument("--ff", default=None, help="FF のプリミティブ（json：{'prims': [{L, a_dn, h, a_up, g} × 4]}）")
p.add_argument("--obs", choices=["v2", "v1"], default="v2", help="env の cfg（v2 = Gen2_B、v1 = Gen2_B_ObsV1）。FF は v2")
p.add_argument("--n_dr", type=int, default=32)
p.add_argument("--dr_seed", type=int, default=7000)
p.add_argument("--name", required=True)
p.add_argument("--out", default="data/user0/eval/b_eval")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
sys.path.insert(0, str(ROOT / "scripts/tools/user0/eval"))
import ff_score  # noqa: E402

from porcaro_2026.tasks.direct.porcaro_2026.user0 import porcaro_2026_env_cfg as C  # noqa: E402
from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env import Porcaro2026Env  # noqa: E402

TASK = {"v2": "Porcaro-Gen2-B-user0", "v1": "Porcaro-Gen2-B-obsv1-user0"}


def check_params(cfg, prm):
    got = dict(bpm_range=list(cfg.rand_bpm_range), note_probs=[round(x, 6) for x in cfg.rand_note_probs], rest_prob=cfg.rand_rest_prob,
               min_interval_s=cfg.rand_min_interval_s, lead_in_s=cfg.rand_lead_in_s, episode_beats=cfg.rand_episode_beats, tail_s=cfg.rand_tail_s)
    want = dict(bpm_range=prm["bpm_range"], note_probs=[round(x, 6) for x in prm["note_probs"]], rest_prob=prm["rest_prob"],
                min_interval_s=prm["min_interval_s"], lead_in_s=prm["lead_in_s"], episode_beats=prm["episode_beats"], tail_s=prm["tail_s"])
    if got != want:
        raise SystemExit(f"!!! b_sets の値と cfg が違う\n  cfg  {got}\n  sets {want}")


def main():
    if (args.ckpt is None) == (args.ff is None):
        raise SystemExit("--ckpt（＋ --model）か --ff のどちらか一方")
    if args.midi_dir:
        # MIDI のタスク（実機コアなど）：play_sim_midi の MidiInjector と同じ目標波形（ff_lib.midi_target）
        import ff_lib
        fs = sorted(Path(ROOT / args.midi_dir).glob("*.mid"))
        tg = [ff_lib.midi_target(f) for f in fs]
        Tm = max(len(x[0]) for x in tg)
        traj = np.zeros((len(fs), max(Tm, 5000)), np.float32)
        for i, (x, _) in enumerate(tg):
            traj[i, :len(x)] = x
        bpm = np.array([b for _, b in tg])
        ons = [ff_lib.onsets_from_target(x, b) for x, b in tg]
        dur = np.array([o[-1] + 1.0 for o in ons])
        S = {"onset_steps": np.array([np.round(o / 0.02).astype(int) for o in ons], dtype=object)}
        prm = None
        args.set = Path(args.midi_dir).name
        print("[eval_b] MIDI:", [f.name for f in fs])
    else:
        S = np.load(ROOT / args.sets_dir / f"{args.set}.npz", allow_pickle=True)
        prm = json.loads(str(S["params"]))
        traj, bpm, dur = S["traj"], S["bpm"], S["dur_s"]
    n_seq = len(bpm)
    N = n_seq * args.n_dr
    cfg = (C.Porcaro2026EnvCfg_Gen2_B if args.obs == "v2" else C.Porcaro2026EnvCfg_Gen2_B_ObsV1)()
    if prm is not None:
        check_params(cfg, prm)
    cfg.scene.num_envs = N
    cfg.logging.enabled = False
    cfg.seed = args.dr_seed
    if args.model == "E":
        cfg.use_frame_stacking = True
        cfg.frame_stack_k = 5
    dt_ctrl = cfg.sim.dt * cfg.decimation
    base = C.obs_base_dim(cfg, int(cfg.lookahead_horizon / dt_ctrl))
    cfg.observation_space = base * cfg.frame_stack_k if cfg.use_frame_stacking else base

    policy, wrapper = None, None
    if args.ckpt:
        import gymnasium as gym
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner
        import porcaro_2026.tasks  # noqa: F401
        entry = "rsl_rl_lstm_cfg_entry_point" if args.model == "B" else "rsl_rl_mlp_cfg_entry_point"
        agent_cfg = load_cfg_from_registry(TASK[args.obs], entry)
        envg = gym.make(TASK[args.obs], cfg=cfg)
        wrapper = RslRlVecEnvWrapper(envg, clip_actions=agent_cfg.clip_actions)
        env = envg.unwrapped
        runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.ckpt)
        policy = runner.get_inference_policy(device=env.device)
    else:
        env = Porcaro2026Env(cfg)
    # リセット時の DR（k・τ・L・通信の遅れ、初期関節角）を、方策の作り方や、それまでのリセットの回数によらずそろえる
    torch.manual_seed(args.dr_seed + 1)
    torch.cuda.manual_seed_all(args.dr_seed + 1)
    (wrapper if wrapper is not None else env).reset()
    dev = env.device
    gen = env.rhythm_generator

    # 列を env に入れる（リセットのあとで上書き。エピソードが途中で切れないよう長さを最大にする）
    seq_of = np.arange(N) % n_seq
    T = gen.target_trajectories.shape[1]
    tr = torch.zeros((N, T), device=dev)
    Tn = min(T, traj.shape[1])
    tr[:, :Tn] = torch.as_tensor(traj[seq_of, :Tn], device=dev)
    gen.target_trajectories = tr
    gen.current_bpms[:] = torch.as_tensor(bpm[seq_of], device=dev, dtype=torch.float32)
    env.episode_duration_steps[:] = env.max_episode_length + 10
    n_steps = int(np.ceil(dur.max() / dt_ctrl)) + 2
    if hasattr(env, "obs_history"):
        env.obs_history[:] = 0.0

    # 力の記録：scene.update のたびに（物理 5 ms ごと）k_t を掛けた z 成分
    D = cfg.decimation
    Fz = torch.zeros((n_steps * D, N), device=dev)
    ptr = [0]
    orig_update = env.scene.update

    def rec_update(*a, **k):
        r = orig_update(*a, **k)
        f = env.stick_sensor.data.net_forces_w
        f = f[:, 0, :] if f.dim() == 3 else f
        if ptr[0] < Fz.shape[0]:
            Fz[ptr[0]] = f[:, 2] * env.force_scale_sim_to_real
        ptr[0] += 1
        return r
    env.scene.update = rec_update

    # FF の指令
    if args.ff:
        import ff_lib
        spec = json.load(open(args.ff))
        prims = [ff_lib.Prim.from_dict(d) for d in spec["prims"]]
        cmds = []
        for i in range(n_seq):
            on = np.asarray(S["onset_steps"][i]) * dt_ctrl
            cmds.append(ff_lib.synth(on, prims, n_steps))
        A = torch.as_tensor(np.stack(cmds)[seq_of], device=dev, dtype=torch.float32)      # (N, n_steps, 3) MPa
        A = A / cfg.controller.Pmax * 2.0 - 1.0

    t0 = time.time()
    with torch.inference_mode():
        if policy is not None:
            o = wrapper.get_observations()
            obs = o[0] if isinstance(o, tuple) else o
            if hasattr(policy, "reset_memory"):
                policy.reset_memory()
        for k in range(n_steps):
            if policy is not None:
                act = policy(obs)
                obs, _, dones, _ = wrapper.step(act)
            else:
                env.step(A[:, k])
    env.scene.update = orig_update
    print(f"[eval_b] {n_steps} step × {N} env、{time.time() - t0:.0f} s")

    # 採点：env ごとに simulation_log と同じ形にして extract_strikes
    Fz = Fz[: ptr[0]].cpu().numpy()
    tt = (np.arange(Fz.shape[0]) + 1) * cfg.sim.dt
    rows, press = [], []
    for e in range(N):
        s = seq_of[e]
        tgt = np.repeat(traj[s, :n_steps], D)[: len(tt)]
        ts = pd.DataFrame({"time_s": tt, "target_force": tgt, "force_z": Fz[:, e]})
        ts = ts[ts.time_s <= dur[s] + 0.3]
        st = ff_score.score_sim(ts, float(bpm[s]))
        press.append(ff_score.excess_contact_frac(ts.time_s.to_numpy(), ts.force_z.to_numpy(), train_gap_s=ff_score.PRESS_TRAIN_GAP_S))
        st["env"], st["seq"], st["dr"] = e, s, e // n_seq
        rows.append(st)
    st = pd.concat(rows, ignore_index=True)
    out = ROOT / args.out / args.set / args.name
    out.mkdir(parents=True, exist_ok=True)
    st.to_csv(out / "strikes.csv", index=False)
    per_dr = st.groupby("dr").success.mean()
    summ = dict(name=args.name, ckpt=args.ckpt, ff=args.ff, model=args.model, obs=args.obs, set=args.set, n_seq=n_seq, n_dr=args.n_dr,
                dr_seed=args.dr_seed, n_targets=int(len(st)), **ff_score.summarize(st),
                success_dr_mean=float(per_dr.mean()), success_dr_sd=float(per_dr.std()),
                success_dr_ci95=float(1.96 * per_dr.std() / np.sqrt(len(per_dr))),
                train_gap_s=ff_score.TRAIN_GAP_S,
                press_train_frac=float(np.mean(press)),      # 押し付けの診断：60 ms の連なりで 100 ms を超えた接触の時間の割合（env の平均）
                n_expected_targets=int(sum(len(S["onset_steps"][s]) for s in seq_of)))
    json.dump(summ, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        os.write(2, ("\n[eval_b] エラーで止まりました:\n" + traceback.format_exc()).encode("utf-8", "replace"))
        os._exit(1)
    app.close()
