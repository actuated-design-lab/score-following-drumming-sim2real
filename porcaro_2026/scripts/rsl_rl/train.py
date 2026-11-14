# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--lookahead_horizon", type=float, default=None,
                    help="Override lookahead horizon (e.g. 0.1, 0.5, 1.0).")
parser.add_argument("--use_frame_stacking", action="store_true", default=False,
                    help="Enable frame-stacking (finite history) observation.")
parser.add_argument("--frame_stack_k", type=int, default=5,
                    help="Number of frames to stack.")
parser.add_argument("--pam_tau_scale", type=float, default=None,
                    help="Fix PAM time-constant multiplier to this single value, "
                         "overriding pam_tau_scale_range DR sampling entirely "
                         "(e.g. 0.5 / 1.0 / 2.0 for the tau sweep).")
# 既存の学習済みモデルから「新しいラン」として学習を始める（追加学習）。指定しなければ何も変わらない。
# --resume と違い、別の experiment_name のフォルダにあるチェックポイントも使える
# オプティマイザは初期化し、iteration は 0 から数える。
parser.add_argument("--init_checkpoint", type=str, default=None,
                    help="Initialize the policy from this checkpoint (.pt) and train as a new run "
                         "(fresh optimizer, iteration counter reset). Observation/action shapes must match.")
parser.add_argument("--init_action_std", type=float, default=None,
                    help="With --init_checkpoint: reset the policy's action noise std to this value "
                         "(a converged policy has a small std, which leaves little exploration).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import math
import torch
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import porcaro_2026.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    logs_root_dir = "rsl_rl"
    if args_cli.pam_tau_scale is not None:
        # τ を固定した学習は logs/<user>/rsl_rl_tau_sweep/ に分ける（通常の学習ログと混ぜない）
        logs_root_dir = "rsl_rl_tau_sweep"
        lh_for_tag = args_cli.lookahead_horizon if args_cli.lookahead_horizon is not None else env_cfg.lookahead_horizon
        tau_tag = f"tau{args_cli.pam_tau_scale}_lh{lh_for_tag}"
        agent_cfg.run_name = f"{tau_tag}_{agent_cfg.run_name}" if agent_cfg.run_name else tau_tag
        print(f"[Config] Tau sweep detected (--pam_tau_scale={args_cli.pam_tau_scale}): "
              f"isolating logs under {cli_args.user_logs_root(args_cli.task, logs_root_dir)}/, run_name -> {agent_cfg.run_name}")

    # logs/<userN>/<logs_root_dir>/<experiment_name>/ - the user comes from the task (see cli_args.user_logs_root)
    log_root_path = os.path.join(cli_args.user_logs_root(args_cli.task, logs_root_dir), agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # 学習のときは物理ステップごとのログを書かない（遅くなるため）
    if hasattr(env_cfg, "logging"):
        print("[INFO] Training mode: Forcing data logging OFF to improve performance.")
        env_cfg.logging.enabled = False

    if args_cli.lookahead_horizon is not None:
        env_cfg.lookahead_horizon = args_cli.lookahead_horizon
        print(f"[Config] lookahead_horizon overridden to {args_cli.lookahead_horizon}")

    if args_cli.use_frame_stacking:
        env_cfg.use_frame_stacking = True
        env_cfg.frame_stack_k = args_cli.frame_stack_k

    if args_cli.pam_tau_scale is not None:
        env_cfg.pam_tau_scale_range = (args_cli.pam_tau_scale, args_cli.pam_tau_scale)
        print(f"[Config] pam_tau_scale_range fixed to "
              f"({args_cli.pam_tau_scale}, {args_cli.pam_tau_scale}) for tau sweep")

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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    elif args_cli.init_checkpoint:
        _init_from_checkpoint(runner, args_cli.init_checkpoint, args_cli.init_action_std)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


def _init_from_checkpoint(runner, checkpoint: str, action_std: float | None) -> None:
    """--init_checkpoint: 方策の重み（と観測の正規化）だけを読み込み、新しいランとして学習を始める。"""
    path = retrieve_file_path(checkpoint)
    print(f"[INFO]: Initializing policy from checkpoint: {path} (new run, fresh optimizer)")
    runner.load(path, load_optimizer=False)
    runner.current_learning_iteration = 0
    if action_std is not None:
        policy = runner.alg.policy
        if hasattr(policy, "log_std"):
            policy.log_std.data.fill_(math.log(action_std))
        elif hasattr(policy, "std"):
            policy.std.data.fill_(action_std)
        else:
            raise RuntimeError("--init_action_std: policy has neither 'std' nor 'log_std'")
        print(f"[INFO]: Reset action noise std to {action_std}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
