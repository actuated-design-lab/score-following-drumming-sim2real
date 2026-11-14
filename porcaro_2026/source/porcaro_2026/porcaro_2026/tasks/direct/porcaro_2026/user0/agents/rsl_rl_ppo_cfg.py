# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL の PPO 設定（LSTM）。rsl_rl_ppo_lstm_cfg.py と同じ中身で、rsl_rl_cfg_entry_point の既定として使う。
MLP は rsl_rl_ppo_mlp_cfg.py。実験ごとに experiment_name を分ける。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticRecurrentCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 48 → 120 step（2.4 s）。BPM 60 で 2 拍以上、BPM 120 で 1 小節
    num_steps_per_env = 120
    max_iterations = 1500
    save_interval = 50
    experiment_name = "porcaro_rslrl_lstm_modelB_DR"

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
