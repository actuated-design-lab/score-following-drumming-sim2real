"""play_sim_midi.py — 方策を MIDI の打撃列で sim 上で動かし、物理ステップごとのログを書く（環境 1 つ）。

目標の波形は MIDI のノートから作り、rhythm_generator の目標を差し替える（学習の生成器は使わない）。
出力：<eval_logs_root>/<run_tag>/<ckpt>/<MIDI 名>_trial<t>/simulation_log.csv
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import re

# 同じフォルダの cli_args・obs_mask を import するため
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from isaaclab.app import AppLauncher
import cli_args
import obs_mask

parser = argparse.ArgumentParser(description="Play RL agent with MIDI Input (Injection Mode).")
parser.add_argument("--midi", type=str, required=True, help="Path to MIDI file.")
parser.add_argument("--force_scale", type=float, default=20.0, help="Target Force [N].")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=30000, help="Length of video (steps).")

def add_arg_if_missing(parser, arg_name, **kwargs):
    existing_opts = [opt for action in parser._actions for opt in action.option_strings]
    if arg_name not in existing_opts:
        parser.add_argument(arg_name, **kwargs)

add_arg_if_missing(parser, "--load_checkpoint", type=str, default="model_.*.pt", help="Checkpoint file name pattern.")
add_arg_if_missing(parser, "--load_run", type=str, default=None, help="Specific run folder name to load.")
add_arg_if_missing(parser, "--checkpoint", type=str, default=None, help="Path to specific checkpoint file.")
add_arg_if_missing(parser, "--experiment", type=str, default=None, help="Experiment folder name.")
add_arg_if_missing(parser, "--task", type=str, default=None, help="Name of the task.")
add_arg_if_missing(parser, "--num_envs", type=int, default=None, help="Number of environments.")
add_arg_if_missing(parser, "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config.")
add_arg_if_missing(parser, "--seed", type=int, default=None, help="Seed.")
add_arg_if_missing(parser, "--use_pretrained_checkpoint", action="store_true", help="Use pre-trained checkpoint.")
add_arg_if_missing(parser, "--trial", type=int, default=0,
                    help="Trial index for this MIDI condition; used to separate output CSVs "
                         "and to derive a reproducible per-trial seed.")
add_arg_if_missing(parser, "--lookahead_horizon", type=float, default=None,
                    help="Override lookahead horizon (e.g. 0.1, 0.5, 1.0). Must match the "
                         "checkpoint's training-time value or policy loading will fail with an "
                         "observation-space size mismatch.")
add_arg_if_missing(parser, "--use_frame_stacking", action="store_true",
                    help="Enable frame-stacking (finite history) observation.")
add_arg_if_missing(parser, "--frame_stack_k", type=int, default=5,
                    help="Number of frames to stack.")
add_arg_if_missing(parser, "--pam_tau_scale", type=float, default=None,
                    help="Fix PAM time-constant multiplier to this single value, overriding "
                         "pam_tau_scale_range DR sampling entirely. Must match the checkpoint's "
                         "training-time value for the tau sweep (e.g. 0.5 / 1.0 / 2.0).")
add_arg_if_missing(parser, "--eval_logs_root", type=str, default="eval_logs",
                    help="Root directory for eval_logs/{run_tag}/{ckpt}/{condition}_trial{t}/ output "
                         "(override to keep tau-sweep/non-DR/etc. eval runs from mixing with the "
                         "main experiment matrix's eval_logs/).")
add_arg_if_missing(parser, "--mask_mode", type=str, default="none", choices=list(obs_mask.MASK_MODES),
                    help="Far-future lookahead ablation: overwrite the [--mask_lo_s, --mask_hi_s) "
                         "window of the lookahead observation buffer with zeros/uniform noise/a "
                         "random within-window shuffle, applied every step after env.reset()/step().")
add_arg_if_missing(parser, "--mask_lo_s", type=float, default=0.5,
                    help="Start of the masked lookahead time window (seconds).")
add_arg_if_missing(parser, "--mask_hi_s", type=float, default=1.0,
                    help="End of the masked lookahead time window (seconds).")

try:
    cli_args.add_rsl_rl_args(parser)
except argparse.ArgumentError:
    pass
except Exception as e:
    print(f"[Warning] Failed to add cli_args: {e}")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import porcaro_2026.tasks

try:
    import mido
except ImportError:
    print("\n[Error] 'mido' library is not installed.")
    sys.exit(1)

class MidiInjector:
    """MIDI のノートを、学習と同じスーパーガウシアンの目標波形にして env に入れる。"""

    def __init__(self, midi_path, dt, device, target_force=20.0):
        self.device = device
        self.dt = dt
        self.target_force = target_force

        mid = mido.MidiFile(midi_path)
        tempo = 500000 # Default 120 BPM

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break

        # ファイル名に bpmNNN があれば、MIDI のテンポよりそちらを使う
        match = re.search(r'bpm(\d+)', midi_path.lower())
        if match:
            extracted_bpm = float(match.group(1))
            tempo = mido.bpm2tempo(extracted_bpm)
            print(f"[MIDI] Overriding tempo from filename: BPM {extracted_bpm}")

        self.bpm = mido.tempo2bpm(tempo)

        current_time = 0.0
        spikes = []
        for msg in mid.merged_track:
            time_delta = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            current_time += time_delta
            if msg.type == 'note_on' and msg.velocity > 0:
                spikes.append(current_time)

        self.duration_sec = current_time + 2.0
        total_steps = int(self.duration_sec / dt) + 100

        spike_tensor = torch.zeros((1, 1, total_steps), device=device)
        for t in spikes:
            idx = int(t / dt)
            if idx < total_steps:
                spike_tensor[0, 0, idx] = 1.0

        width_sec = 0.035
        sigma = width_sec / 2.0
        radius = int(width_sec / dt)
        t_vals = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) * dt
        kernel = (target_force * torch.exp(-0.5 * (t_vals / sigma) ** 4)).view(1, 1, -1)

        with torch.no_grad():
            traj = F.conv1d(spike_tensor, kernel, padding=radius)

        self.trajectory = traj.view(-1)
        print(f"[MIDI] Loaded {midi_path}: BPM={self.bpm:.1f}, Duration={self.duration_sec:.1f}s, Steps={total_steps}")

    def inject_to_env(self, env):
        raw_env = env.unwrapped
        if not hasattr(raw_env, "rhythm_generator"):
            return
        gen = raw_env.rhythm_generator
        num_envs = raw_env.num_envs
        if hasattr(gen, "current_bpms"):
            gen.current_bpms[:] = self.bpm
        midi_len = self.trajectory.shape[0]
        new_traj_buffer = self.trajectory.unsqueeze(0).expand(num_envs, -1).clone()
        gen.target_trajectories = new_traj_buffer
        gen.max_steps = midi_len

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    checkpoint_path = getattr(args_cli, "checkpoint", None)
    load_run = getattr(args_cli, "load_run", None)
    experiment_name = args_cli.experiment if args_cli.experiment else agent_cfg.experiment_name
    run_dir_arg = load_run if load_run else ".*"

    if checkpoint_path:
        resume_path = retrieve_file_path(checkpoint_path)
    else:
        log_root_path = os.path.abspath(os.path.join(cli_args.user_logs_root(args_cli.task), experiment_name))
        resume_path = get_checkpoint_path(log_root_path, run_dir_arg, args_cli.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 出力先：eval_logs/{run_tag}/{ckpt_name}/{MIDI 名}_trial{t}/
    ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]   # 例: model_1499
    run_tag = os.path.basename(os.path.dirname(resume_path))          # 例: 2026-03-01_08-00-23
    midi_stem = os.path.splitext(os.path.basename(args_cli.midi))[0]  # 例: gmd_01_low_bpm80
    condition_tag = f"{midi_stem}_trial{args_cli.trial}"

    eval_out_dir = os.path.join(args_cli.eval_logs_root, run_tag, ckpt_name, condition_tag)
    os.makedirs(eval_out_dir, exist_ok=True)

    env_cfg.scene.num_envs = 1
    if hasattr(args_cli, "device") and args_cli.device:
        env_cfg.sim.device = args_cli.device
    # --seed を env_cfg に渡す（未指定なら env_cfg.seed の既定値のまま）
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 300.0
    env_cfg.log_dir = log_dir
    if hasattr(env_cfg, "logging"):
        env_cfg.logging.enabled = True
        env_cfg.logging.filepath = os.path.join(eval_out_dir, "simulation_log.csv")
        print("[INFO] Play mode detected: Logging enabled (force).")

    # 学習時の lookahead_horizon / use_frame_stacking / frame_stack_k に合わせる（train.py と同じ）。
    # 違うと OnPolicyRunner.load() で観測の次元が合わずに落ちる
    if args_cli.lookahead_horizon is not None:
        env_cfg.lookahead_horizon = args_cli.lookahead_horizon
        print(f"[Config] lookahead_horizon overridden to {args_cli.lookahead_horizon}")

    if args_cli.use_frame_stacking:
        env_cfg.use_frame_stacking = True
        env_cfg.frame_stack_k = args_cli.frame_stack_k

    if args_cli.pam_tau_scale is not None:
        env_cfg.pam_tau_scale_range = (args_cli.pam_tau_scale, args_cli.pam_tau_scale)
        print(f"[Config] pam_tau_scale_range fixed to "
              f"({args_cli.pam_tau_scale}, {args_cli.pam_tau_scale}) for tau sweep eval")

    dt_ctrl = env_cfg.sim.dt * env_cfg.decimation
    lookahead_steps = int(env_cfg.lookahead_horizon / dt_ctrl)
    # 観測の版で 1 フレームの次元が変わる（v1 = 10 + L、v2 = 12 + L。user0/porcaro_2026_env_cfg.obs_base_dim と同じ式）
    base_obs_dim = (12 if getattr(env_cfg, "obs_version", "v1") == "v2" else 10) + lookahead_steps
    env_cfg.observation_space = (
        base_obs_dim * env_cfg.frame_stack_k if env_cfg.use_frame_stacking else base_obs_dim
    )
    print(f"[Config Override] lookahead={env_cfg.lookahead_horizon} "
        f"frame_stacking={env_cfg.use_frame_stacking} k={env_cfg.frame_stack_k} "
        f"-> observation_space={env_cfg.observation_space}")

    mask_lo_idx, mask_hi_idx = obs_mask.compute_mask_range(
        dt_ctrl, lookahead_steps, args_cli.mask_lo_s, args_cli.mask_hi_s
    )
    if args_cli.mask_mode != "none":
        print(f"[Mask] mode={args_cli.mask_mode} window=[{args_cli.mask_lo_s}s, {args_cli.mask_hi_s}s) "
              f"-> obs columns [{mask_lo_idx}, {mask_hi_idx}) of {env_cfg.observation_space}")
        if env_cfg.use_frame_stacking:
            raise ValueError(
                "--mask_mode is only implemented for the non-frame-stacked observation layout "
                "(obs_mask.compute_mask_range assumes obs_single, not the frame-stacking reshape)."
            )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)


    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_midi"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt_ctrl = env.unwrapped.dt_ctrl_step
    midi_injector = MidiInjector(args_cli.midi, dt_ctrl, env.unwrapped.device, args_cli.force_scale)

    obs, _ = env.reset()

    # --pam_tau_scale が圧力チャネル（pam.py の PAMChannel.current_tau_scale）に届いているか確かめる。
    # 値は reset() / reset_idx() で入るので、env.reset() の後で見る
    if args_cli.pam_tau_scale is not None:
        raw_env = env.unwrapped
        ctrl = raw_env.action_controller
        expected = args_cli.pam_tau_scale
        for ch_name in ("ch_DF", "ch_F", "ch_G"):
            ch = getattr(ctrl, ch_name)
            actual = ch.current_tau_scale
            assert actual is not None, f"{ch_name}.current_tau_scale is None after env.reset()"
            assert torch.allclose(actual, torch.full_like(actual, expected)), (
                f"[Tau Safety Check] {ch_name}.current_tau_scale={actual.tolist()} != expected {expected} "
                f"-- pam_tau_scale_range override did NOT reach the actuator."
            )
        tau_p_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        tau_vals = [0.043, 0.045, 0.060, 0.066, 0.094, 0.131]
        print(f"[Tau Safety Check] PASSED: ch_DF/F/G.current_tau_scale == {expected} for all envs")
        for p, base in zip(tau_p_axis[::2], tau_vals[::2]):
            print(f"  illustrative: P_cmd={p} tau_base={base:.4f} "
                  f"tau_final(expected)={expected * base:.4f}  (tau_final = tau_scale * tau_base)")

    if args_cli.mask_mode != "none":
        obs = obs_mask.apply_mask(obs, args_cli.mask_mode, mask_lo_idx, mask_hi_idx)
        obs_t = obs_mask.policy_tensor(obs)
        print(f"[Mask] obs.shape={tuple(obs_t.shape)} post-mask sample obs[0, {mask_lo_idx}:{mask_lo_idx+3}]="
              f"{obs_t[0, mask_lo_idx:mask_lo_idx+3].tolist()}")

    if hasattr(policy, "reset_memory"): policy.reset_memory()
    midi_injector.inject_to_env(env)

    print("="*60)
    print(f" Sim-Verification Started (MIDI Mode)")
    step_count = 0
    max_steps = midi_injector.trajectory.shape[0]

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if args_cli.mask_mode != "none":
                    obs = obs_mask.apply_mask(obs, args_cli.mask_mode, mask_lo_idx, mask_hi_idx)
                step_count += 1

                # 曲の終わり（2 step の余裕）を先に判定する
                if step_count >= (max_steps - 2):
                    print(f"Song finished successfully ({step_count} steps). Exiting...")
                    break

                if dones.any():
                    print(f"Env reset detected at step {step_count} (Fall or Timeout). Exiting...")
                    break

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()