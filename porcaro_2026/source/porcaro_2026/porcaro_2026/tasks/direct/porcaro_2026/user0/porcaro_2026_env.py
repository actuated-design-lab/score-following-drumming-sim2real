# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import argparse
import numpy as np
import torch
import math
from collections.abc import Sequence
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.managers import EventManager

from .porcaro_2026_env_cfg import Porcaro2026EnvCfg, obs_base_dim
from ..common.actions.base import ActionController
from ..common.actions.torque import TorqueActionController
from .logging.logging_manager import LoggingManager
from .rewards.reward import RewardManager
from .rhythm_generator import RhythmGenerator
from .cfg.assets import WRIST_J0, GRIP_J0



class Porcaro2026Env(DirectRLEnv):
    """Porcaro の打撃タスクの環境（DirectRLEnv）。"""

    cfg: Porcaro2026EnvCfg

    def __init__(self, cfg: Porcaro2026EnvCfg, render_mode: str | None = None, **kwargs):
        self.robot: Articulation = None
        self.drum: RigidObject = None
        self.stick_sensor: ContactSensor = None
        self.drum_sensor: ContactSensor = None

        self.action_controller: ActionController = None
        self.logging_manager: LoggingManager = None
        self.reward_manager: RewardManager = None
        self.actuator_net = None


        # カリキュラム：累積 env step で切り替える（cfg.curriculum_iters があれば下で iteration に置き換える）
        self.total_env_steps = 0
        self.curriculum_thresholds = [25_000_000, 100_000_000]

        # 打撃力の倍率 k_t。物理は変えず、報酬・観測・ログが見る力だけを実機のスケールに合わせる（Gen1 は 3.0）
        self.force_scale_sim_to_real = float(getattr(cfg, "impact_force_scale", 3.0))

        # BPM ごと・パターンごとの報酬（直近 20 回のエピソード平均。TensorBoard 用）
        self.bpm_reward_history = {
            bpm: deque(maxlen=20)
            for bpm in [60, 80, 100, 120, 140, 160]
        }

        self.pattern_keys = ["single_4", "single_8", "double", "rest"]
        self.pattern_reward_history = {
            pat: deque(maxlen=20) for pat in self.pattern_keys
        }

        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

        if len(self.dof_idx) != 2:
            raise ValueError(f"Expected 2 DOFs (wrist, grip), but found {len(self.dof_idx)} based on {self.cfg.dof_names}")
        self.joint_ids_tuple: tuple[int, int] = (self.dof_idx[0], self.dof_idx[1])

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.prev_actions = torch.zeros_like(self.actions)

        # 物理サブステップの間の力の最大と積算
        self.max_force_z_buffer = torch.zeros(self.num_envs, device=self.device)
        self.force_sum_buffer = torch.zeros(self.num_envs, device=self.device)

        # エピソードの長さ（BPM で変わる）
        self.episode_duration_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        dt_ctrl = self.cfg.sim.dt
        ctrl_cfg = self.cfg.controller

        pam_tau_scale_range = getattr(self.cfg, "pam_tau_scale_range", (1.0, 1.0))

        self.action_controller = TorqueActionController(
            dt_ctrl=dt_ctrl,
            control_mode=ctrl_cfg.control_mode,
            r=ctrl_cfg.r,
            L=ctrl_cfg.L,
            theta_t_DF_deg=ctrl_cfg.theta_t_DF_deg,
            theta_t_F_deg=ctrl_cfg.theta_t_F_deg,
            theta_t_G_deg=ctrl_cfg.theta_t_G_deg,
            Pmax=ctrl_cfg.Pmax,
            tau=ctrl_cfg.tau,
            dead_time=ctrl_cfg.dead_time,
            N=ctrl_cfg.N,
            force_map_csv=ctrl_cfg.force_map_csv,
            force_scale=ctrl_cfg.force_scale,
            h0_map_csv=ctrl_cfg.h0_map_csv,
            use_pressure_dependent_tau=ctrl_cfg.use_pressure_dependent_tau,
            geometric_cfg=self.cfg.pam_geometric_cfg,
            pam_tau_scale_range=pam_tau_scale_range,
            pam_viscosity=getattr(ctrl_cfg, "pam_viscosity", 0.0),
            pam_hys_const=getattr(ctrl_cfg, "pam_hys_const", 0.5),
            pam_hys_coef_p=getattr(ctrl_cfg, "pam_hys_coef_p", 15.0),
            pam_contract_gain=getattr(ctrl_cfg, "pam_contract_gain", 1.5),
            pam_extend_gain=getattr(ctrl_cfg, "pam_extend_gain", 1.0),
            pam_p_dot_scale=getattr(ctrl_cfg, "pam_p_dot_scale", 100.0),
            transition_width=getattr(ctrl_cfg, "transition_width", 0.0),
            pam_hys_mode=getattr(ctrl_cfg, "pam_hys_mode", "relay"),
            pam_hys_play_widths=getattr(ctrl_cfg, "pam_hys_play_widths", (0.02,)),
            pam_hys_play_weights=getattr(ctrl_cfg, "pam_hys_play_weights", None),
            pam_hys_play_widths_h=getattr(ctrl_cfg, "pam_hys_play_widths_h", (0.003,)),
            pam_engage_width_deg=getattr(ctrl_cfg, "pam_engage_width_deg", (0.0, 0.0, 0.0)),
            pam_map_play_w=getattr(ctrl_cfg, "pam_map_play_w", 0.0),
            pressure_model=getattr(ctrl_cfg, "pressure_model", "table"),
            shaped_params=getattr(ctrl_cfg, "shaped_params", None),
            pressure_tau_scale_range=getattr(ctrl_cfg, "pressure_tau_scale_range", (1.0, 1.0)),
            pressure_L_scale_range=getattr(ctrl_cfg, "pressure_L_scale_range", (1.0, 1.0)),
            cmd_delay_s=getattr(ctrl_cfg, "cmd_delay_s", 0.0),
            cmd_delay_range_s=getattr(ctrl_cfg, "cmd_delay_range_s", None),
            contraction_scale_range=getattr(ctrl_cfg, "contraction_scale_range", (1.0, 1.0)),
            theta_t_offset_range_deg=getattr(ctrl_cfg, "theta_t_offset_range_deg", ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))),
        )
        self.action_controller.reset(self.num_envs, self.device)

        print("=" * 80)
        print(f"[INFO] Logging Configuration Check:")
        print(f"  - Enabled : {self.cfg.logging.enabled}")
        print(f"  - Filepath: {self.cfg.logging.filepath}")
        print("=" * 80)

        self.logging_manager = LoggingManager(
            env=self,
            dt=self.cfg.sim.dt,
            log_filepath=self.cfg.logging.filepath,
            enable_logging=self.cfg.logging.enabled
        )

        self.reward_manager = RewardManager(
            cfg=self.cfg.rewards,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.dt_ctrl_step = self.cfg.sim.dt * self.cfg.decimation
        self.target_hit_force = getattr(self.cfg, "target_hit_force", 20.0)

        self.rhythm_generator = RhythmGenerator(
            num_envs=self.num_envs,
            device=self.device,
            dt=self.dt_ctrl_step,
            max_episode_length=self.max_episode_length,  # cfg で確保した最大長
            bpm_range=self.cfg.bpm_range,
            target_force=self.target_hit_force
        )

        # use_simple_rhythm = True で固定パターン・固定 BPM（検証用）
        use_simple = getattr(self.cfg, "use_simple_rhythm", False)
        if use_simple:
            mode = getattr(self.cfg, "simple_rhythm_mode", "double")
            bpm = getattr(self.cfg, "simple_rhythm_bpm", 160.0)
            print(f"[INFO] RhythmGenerator: Test Mode Enabled (Pattern: {mode}, BPM: {bpm})")
            self.rhythm_generator.set_test_mode(enabled=True, bpm=bpm, pattern=mode)
        else:
            print(f"[INFO] RhythmGenerator: Training Mode (Random BPM & Rudiments)")
            self.rhythm_generator.set_test_mode(enabled=False)
        if getattr(self.cfg, "rhythm_mode", "grid") == "random":
            # 周期的でない打撃列（rhythm_generator.set_random_mode、user0/GEN2.md）。評価（use_simple_rhythm や MIDI）には効かない
            self.rhythm_generator.set_random_mode(
                True, bpm_range=self.cfg.rand_bpm_range, note_beats=self.cfg.rand_note_beats,
                note_probs=self.cfg.rand_note_probs, rest_prob=self.cfg.rand_rest_prob,
                min_interval_s=self.cfg.rand_min_interval_s, curriculum=self.cfg.rand_curriculum,
                lead_in_s=getattr(self.cfg, "rand_lead_in_s", -1.0), episode_beats=getattr(self.cfg, "rand_episode_beats", 16),
                tail_s=getattr(self.cfg, "rand_tail_s", 0.5))
            # 打撃列の乱数を学習の seed で固定する（None なら固定しない）。
            # 検証用・評価用のホールドアウト列は、これと重ならない種で別に生成する
            seed = getattr(self.cfg, "seed", None)
            if seed is not None:
                self.rhythm_generator._rng = np.random.default_rng(int(seed))
            print(f"[INFO] RhythmGenerator: random onsets bpm={self.cfg.rand_bpm_range} notes={self.cfg.rand_note_beats} "
                  f"probs={self.cfg.rand_note_probs} rng_seed={seed}")

        lookahead_horizon = getattr(self.cfg, "lookahead_horizon", 0.5)
        self.lookahead_steps = int(lookahead_horizon / self.dt_ctrl_step)

        self.use_frame_stacking = getattr(self.cfg, "use_frame_stacking", False)
        self.frame_stack_k = getattr(self.cfg, "frame_stack_k", 1)
        # 観測の版（cfg.obs_version、porcaro_2026_env_cfg.obs_base_dim）
        self.obs_version = getattr(self.cfg, "obs_version", "v1")
        self.base_obs_dim = obs_base_dim(self.cfg, self.lookahead_steps)
        if self.obs_version == "v2":
            # センサの経路：圧力（MPa）と接触（0/1）を tick ごとに標本化し、obs_sensor_delay_steps tick 遅らせて観測に入れる
            D = max(0, int(getattr(self.cfg, "obs_sensor_delay_steps", 1)))
            self.sensor_delay_steps = D
            self.sensor_p_buf = torch.zeros((self.num_envs, D + 1, 3), device=self.device)
            self.sensor_c_buf = torch.zeros((self.num_envs, D + 1), device=self.device)
            self.p_bias = torch.zeros((self.num_envs, 3), device=self.device)
            self.contact_latch = torch.zeros(self.num_envs, device=self.device)
            self.obs_p_scale = float(self.cfg.controller.Pmax)
            self._resample_p_bias(torch.arange(self.num_envs, device=self.device))
        # カリキュラム：iteration で固定（cfg.curriculum_iters）。None なら上の累積ステップのまま（Gen1）
        ci = getattr(self.cfg, "curriculum_iters", None)
        if ci is not None:
            spi = int(getattr(self.cfg, "curriculum_steps_per_iter", 120))
            self.curriculum_thresholds = [int(i) * spi * self.num_envs for i in ci]
            print(f"[Curriculum] iteration {tuple(ci)} で切り替え（env step 累積 {self.curriculum_thresholds}）")
        print(f"[Obs] obs_version={self.obs_version} base_obs_dim={self.base_obs_dim}")

        if self.use_frame_stacking:
            self.obs_history = torch.zeros(
                (self.num_envs, self.frame_stack_k, self.base_obs_dim),
                device=self.device, dtype=torch.float32
            )

        print("=" * 80)
        print(f"[DEBUG DIAGNOSTIC] 環境設定値の確認")
        print(f"  - sim.dt (物理ステップ): {self.cfg.sim.dt}")
        print(f"  - decimation (間引き数): {self.cfg.decimation}")
        print(f"  - dt_ctrl (制御周期)   : {self.dt_ctrl_step:.5f} 秒")
        print("=" * 80)

        if self.cfg.events is not None:
            self.event_manager = EventManager(self.cfg.events, self)
            self.event_manager.apply(mode="startup")
        else:
            self.event_manager = None

        all_ids = torch.arange(self.num_envs, device=self.device)
        self.rhythm_generator.reset(all_ids)

    # 座標系：sim は下向きが正、このプロジェクト（実機・ログ）は上向きが正。手首とグリップの符号を反転する
    def _get_corrected_joint_state(self):
        """手首・グリップの (q, qd) [N, 2] を、上向きを正にして返す。"""
        q_full = self.robot.data.joint_pos
        qd_full = self.robot.data.joint_vel
        q = q_full[:, self.dof_idx].clone()
        qd = qd_full[:, self.dof_idx].clone()
        q *= -1.0
        qd *= -1.0

        return q, qd

    def _get_corrected_full_state(self):
        """全関節の (q, qd)。手首・グリップだけ符号を反転する（コントローラとログ用）。"""
        q_full = self.robot.data.joint_pos.clone()
        qd_full = self.robot.data.joint_vel.clone()
        wrist_idx, grip_idx = self.joint_ids_tuple
        q_full[:, [wrist_idx, grip_idx]] *= -1.0
        qd_full[:, [wrist_idx, grip_idx]] *= -1.0

        return q_full, qd_full

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.drum = RigidObject(self.cfg.drum_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["drum"] = self.drum

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.stick_sensor = ContactSensor(self.cfg.stick_contact_cfg)
        self.drum_sensor = ContactSensor(self.cfg.drum_contact_cfg)
        self.scene.sensors["stick_contact"] = self.stick_sensor
        self.scene.sensors["drum_contact"] = self.drum_sensor

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # カリキュラムのレベル（累積 env step で判定）
        self.total_env_steps += self.num_envs
        current_level = 0
        if self.total_env_steps > self.curriculum_thresholds[1]:
            current_level = 2
        elif self.total_env_steps > self.curriculum_thresholds[0]:
            current_level = 1

        if hasattr(self, "rhythm_generator"):
            if self.rhythm_generator.curriculum_level != current_level:
                print(f"[Curriculum] Level Up! {self.rhythm_generator.curriculum_level} -> {current_level} (Steps: {self.total_env_steps})")
                self.rhythm_generator.set_curriculum_level(current_level)

        self._pre_physics_step(actions)

        self.max_force_z_buffer[:] = 0.0

        # サブステップごとの接触力 [N, decimation, 3]
        force_history_tensor = torch.zeros(
            (self.num_envs, self.cfg.decimation, 3),
            device=self.device,
            dtype=torch.float32
        )

        # P_cmd などログ用の値は、コントローラの get_last_telemetry から取る
        for i in range(self.cfg.decimation):
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.cfg.sim.dt)

            if self.stick_sensor.data.net_forces_w.dim() == 3:
                current_force_vec = self.stick_sensor.data.net_forces_w[:, 0, :]
            else:
                current_force_vec = self.stick_sensor.data.net_forces_w

            current_force_vec = current_force_vec * self.force_scale_sim_to_real

            force_history_tensor[:, i, :] = current_force_vec
            # 力は z 成分ではなく大きさで見る
            current_force_mag = torch.norm(current_force_vec, dim=-1)

            self.max_force_z_buffer = torch.max(self.max_force_z_buffer, current_force_mag)

            self.force_sum_buffer += current_force_mag

            # 物理ステップごとのログ
            if self.logging_manager.enable_logging:
                self.logging_manager.update_time(self.cfg.sim.dt)
                current_steps = self.episode_length_buf
                tgt_val = self.rhythm_generator.get_current_target(current_steps)[0].item()
                tgt_bpm = self.rhythm_generator.current_bpms[0].item()

                telemetry = self.action_controller.get_last_telemetry()
                if telemetry is None: telemetry = {}
                q_log, qd_log = self._get_corrected_full_state()

                if self.actuator_net is not None:
                     nan_tensor = torch.full((self.num_envs, 3), float('nan'), device=self.device)
                     if "P_cmd" not in telemetry: telemetry["P_cmd"] = nan_tensor
                     if "P_out" not in telemetry: telemetry["P_out"] = nan_tensor

                q_log, qd_log = self._get_corrected_full_state()
                self.logging_manager.buffer_step_data(
                    q_full=q_log,
                    qd_full=qd_log,
                    telemetry=telemetry,
                    actions=self.actions,
                    current_sim_time=self.sim.current_time,
                    target_force=self.rhythm_generator.get_current_target(self.episode_length_buf)[0].item(),
                    target_bpm=self.rhythm_generator.current_bpms[0].item()
                )

        self.episode_length_buf += 1

        if self.obs_version == "v2":
            self._sample_sensors()

        obs = self._get_observations()

        # 観測と報酬の NaN/Inf を 0 にする（物理が一瞬発散しても学習全体が落ちないように）
        if isinstance(obs, dict):
            for key, val in obs.items():
                if isinstance(val, torch.Tensor):
                    if torch.isnan(val).any() or torch.isinf(val).any():
                        obs[key] = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        elif isinstance(obs, torch.Tensor):
            if torch.isnan(obs).any() or torch.isinf(obs).any():
                obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        target_force_tensor = torch.full((self.num_envs,), self.target_hit_force, device=self.device)
        rew, reward_terms = self._get_rewards(force_max=self.max_force_z_buffer, target_ref=target_force_tensor)

        if torch.isnan(rew).any() or torch.isinf(rew).any():
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)

        self.reset_terminated[:] = False
        terminated, time_outs = self._get_dones()

        reset_mask = terminated | time_outs

        if self.logging_manager.enable_logging:
            f1_val = torch.zeros_like(rew)
            if hasattr(self.reward_manager, "get_first_hit_force"):
                 f1_val = self.reward_manager.get_first_hit_force()

            self.logging_manager.finalize_log_step(
                peak_force=force_history_tensor,
                f1_force=f1_val,
                step_reward=rew
            )

        # RSL-RL のログ
        if not hasattr(self, "extras"): self.extras = {}
        self.extras["log"] = {}

        if not hasattr(self, "episode_sums"):
            self.episode_sums = {
                k: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for k in reward_terms.keys()
            }
            self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.episode_sums["total"] += rew
        for key, value in reward_terms.items():
            self.episode_sums[key] += value

        if reset_mask.any():
            env_ids = reset_mask.nonzero(as_tuple=False).flatten()
            if hasattr(self, "rhythm_generator"):
                done_bpms = self.rhythm_generator.current_bpms[env_ids]
                done_pat_idxs = self.rhythm_generator.current_pattern_idxs[env_ids]
                done_rewards = self.episode_sums["total"][env_ids]


                for t_bpm in [60.0, 80.0, 100.0, 120.0, 140.0, 160.0]:
                    bpm_mask = (torch.abs(done_bpms - t_bpm) < 1.0)
                    if bpm_mask.any():
                        avg_bpm_reward = done_rewards[bpm_mask].mean().item()
                        self.bpm_reward_history[int(t_bpm)].append(avg_bpm_reward)

                for pat_idx, pat_name in enumerate(self.pattern_keys):
                    pat_mask = (done_pat_idxs == pat_idx)
                    if pat_mask.any():
                        avg_pat_reward = done_rewards[pat_mask].mean().item()
                        self.pattern_reward_history[pat_name].append(avg_pat_reward)
            episode_info = {}
            episode_info["reward"] = self.episode_sums["total"][env_ids]
            for key, count in self.episode_sums.items():
                if key != "total":
                    episode_info[key] = count[env_ids]

            self.extras["episode"] = episode_info
            self.episode_sums["total"][env_ids] = 0.0
            for key in self.episode_sums.keys():
                self.episode_sums[key][env_ids] = 0.0
            self._reset_idx(env_ids)
        else:
            if "episode" in self.extras:
                self.extras.pop("episode")

        # BPM 別・パターン別の報酬の移動平均を毎ステップ載せる
        for t_bpm, history in self.bpm_reward_history.items():
            if len(history) > 0:
                self.extras["log"][f"Episode_Reward/BPM_{t_bpm}"] = sum(history) / len(history)
            else:
                self.extras["log"][f"Episode_Reward/BPM_{t_bpm}"] = 0.0

        for pat_name, history in self.pattern_reward_history.items():
            if len(history) > 0:
                self.extras["log"][f"Episode_Reward/Pattern_{pat_name}"] = sum(history) / len(history)
            else:
                self.extras["log"][f"Episode_Reward/Pattern_{pat_name}"] = 0.0

        for k, v in reward_terms.items():
             self.extras["log"][f"Step_Reward/{k}"] = torch.mean(v)

        self.extras["time_outs"] = time_outs
        self.extras["force/max_force_pooled"] = self.max_force_z_buffer.mean()

        current_len = self.episode_length_buf[0].item() if isinstance(self.episode_length_buf, torch.Tensor) else self.episode_length_buf
        should_save = (current_len % 50 == 0) or (reset_mask.any().item())

        if self.logging_manager.enable_logging and should_save:
             if self.logging_manager.logger is not None:
                 self.logging_manager.logger.save()

        self.prev_actions[:] = self.actions

        return obs, rew, terminated, time_outs, self.extras

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is not None:
            self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """行動をコントローラに渡してトルクを掛ける（物理ステップごと）。"""
        # p_cmd_3d [N, 3] は今は使っていない
        p_cmd_3d = None
        if hasattr(self.action_controller, "compute_pressure"):
            with torch.no_grad():
                p_cmd_3d = self.action_controller.compute_pressure(self.actions)

        processed_actions = self.actions
        q_full_corrected, _ = self._get_corrected_full_state()

        self.action_controller.apply(
            actions=processed_actions,
            q=q_full_corrected,
            joint_ids=self.joint_ids_tuple,
            robot=self.robot
        )


    def _resample_p_bias(self, env_ids):
        lo, hi = getattr(self.cfg, "obs_pressure_bias_range", (0.0, 0.0))
        self.p_bias[env_ids] = lo + (hi - lo) * torch.rand((len(env_ids), 3), device=self.device)

    def _sample_sensors(self):
        """v2：この tick の圧力と接触を標本化し、遅れのバッファに入れ、接触のラッチを更新する（step で 1 回だけ呼ぶ）。

        圧力：コントローラの P_out（ITV 出口の圧力のモデル。200 Hz の物理ステップの最後の値）に、エピソードごとのずれと白色ノイズを足す。
        接触：直前の tick のあいだ（物理 4 ステップ）に、k_t を掛けた力の大きさが閾値以上になったか（実機：その間に受けた 200 Hz の
              パケットのどれかでロードセルが閾値以上）。
        ラッチ：遅らせた接触を OR で溜め、目標打撃の contact_flag_reset_lead_steps tick 前に 0 に戻す（順番は OR → 戻す）。
              目標打撃の時刻は目標波形がピークの contact_flag_onset_frac 倍以上の tick（学習の生成器・MIDI・実機で同じ規則）。
              フラグは「いまの目標（またはその直前の目標）で接触したか」で、目標の時刻をまたぐと 0 から数え直す。
        """
        tel = self.action_controller.get_last_telemetry() or {}
        p = tel.get("P_out")
        if p is None:
            p = torch.zeros((self.num_envs, 3), device=self.device)
        std = float(getattr(self.cfg, "obs_pressure_noise_std", 0.0))
        p = p + self.p_bias + (std * torch.randn_like(p) if std > 0 else 0.0)
        c = (self.max_force_z_buffer >= float(getattr(self.cfg, "contact_flag_threshold_n", 1.0))).float()
        # 遅れ：buf[:, 0] が最新、buf[:, D] が D tick 前
        self.sensor_p_buf = torch.roll(self.sensor_p_buf, shifts=1, dims=1)
        self.sensor_c_buf = torch.roll(self.sensor_c_buf, shifts=1, dims=1)
        self.sensor_p_buf[:, 0] = p
        self.sensor_c_buf[:, 0] = c
        c_obs = self.sensor_c_buf[:, self.sensor_delay_steps]
        # ラッチ
        self.contact_latch = torch.maximum(self.contact_latch, c_obs)
        # 戻す時刻は接触の信号と同じ時間軸でそろえる（sim は D tick 遅らせた信号なので k − D + lead。実機は遅れを足さないので k + lead）。
        # 目標 j（目標波形のピークが tick j）の窓は、物理の時刻で tick j − lead から tick j+1 − lead までの区間の接触から始まる
        # （reward は tick j − 1 → j の区間の力を目標 j と比べる。lead = 2 なら 20 ms 早い打撃まで目標 j に数える）
        lead = int(getattr(self.cfg, "contact_flag_reset_lead_steps", 2))
        k = (self.episode_length_buf - self.sensor_delay_steps + lead).clamp(0, self.rhythm_generator.max_steps - 1).unsqueeze(1)
        tgt = torch.gather(self.rhythm_generator.target_trajectories, 1, k).squeeze(1)
        onset = tgt >= float(getattr(self.cfg, "contact_flag_onset_frac", 0.75)) * self.target_hit_force
        self.contact_latch = torch.where(onset, torch.zeros_like(self.contact_latch), self.contact_latch)

    def _get_observations(self) -> dict:
        q, qd = self._get_corrected_joint_state()

        bpm_val = self.rhythm_generator.current_bpms.view(-1, 1)
        bpm_obs = bpm_val / 180.0

        # 拍の位相（1 拍で 2π）。v1 だけが使う
        time_s = self.episode_length_buf.float().unsqueeze(1) * self.dt_ctrl_step
        phase = time_s * (bpm_val / 60.0) * (2 * math.pi)
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)

        # 目標の力の先読み（/目標力）
        current_steps = self.episode_length_buf
        rhythm_buf = self.rhythm_generator.get_lookahead(current_steps, self.lookahead_steps)
        rhythm_buf = rhythm_buf / self.target_hit_force

        # v1：q(2)+qd(2)+前回指令(3)+sin(1)+cos(1)+BPM(1)+先読み(25) = 35 次元
        if self.obs_version == "v2":
            # v2：拍位相を外し、遅らせた実測圧力（/Pmax）と接触のラッチフラグを足す（実機の obs_v2.py と同じ並び）
            p_obs = self.sensor_p_buf[:, self.sensor_delay_steps] / self.obs_p_scale
            obs_single = torch.cat((q, qd, self.prev_actions, bpm_obs, rhythm_buf, p_obs,
                                    self.contact_latch.unsqueeze(1)), dim=-1)
        else:
            obs_single = torch.cat((q, qd, self.prev_actions, sin_phase, cos_phase, bpm_obs, rhythm_buf), dim=-1)

        if self.use_frame_stacking:
            self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
            self.obs_history[:, -1, :] = obs_single
            obs = self.obs_history.reshape(self.num_envs, -1)
        else:
            obs = obs_single

        return {"policy": obs}

    def _get_rewards(self, force_max: torch.Tensor = None, target_ref: torch.Tensor = None) -> torch.Tensor:
        if force_max is None: force_max = self.max_force_z_buffer
        dt_step = self.cfg.sim.dt * self.cfg.decimation
        current_steps = self.episode_length_buf
        target_trace = self.rhythm_generator.get_current_target(current_steps).view(-1)
        if target_ref is None:
            target_ref = torch.full((self.num_envs,), self.target_hit_force, device=self.device)
        q_corr, _ = self._get_corrected_joint_state()

        current_bpms = self.rhythm_generator.current_bpms if hasattr(self, "rhythm_generator") else None

        telemetry = self.action_controller.get_last_telemetry()
        if telemetry is not None and "P_out" in telemetry:
            p_out = telemetry["P_out"]
        else:
            p_out = torch.zeros((self.num_envs, 3), device=self.device)

        total_reward, reward_terms = self.reward_manager.compute_rewards(
            actions=self.actions,
            p_out=p_out,
            joint_pos=q_corr,
            force_z=force_max,
            target_force_trace=target_trace,
            target_force_ref=target_ref,
            dt=dt_step,
            current_bpm=current_bpms
        )
        if not hasattr(self, "extras"): self.extras = {}
        return total_reward, reward_terms

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # エピソードの長さ（16 拍、BPM で決まる）を超えたらタイムアウト。バッファの上限でも切る
        time_outs = self.episode_length_buf >= self.episode_duration_steps
        time_outs |= (self.episode_length_buf >= self.max_episode_length - 1)

        # 関節角の発散（|q| > 4 rad ≈ 230° か NaN）。計算しているが終了には使っていない（terminated は常に False）
        q, _ = self._get_corrected_joint_state()
        is_exploded = torch.any(torch.abs(q) > 4.0, dim=-1)
        is_exploded |= torch.any(torch.isnan(q), dim=-1)

        self.reset_terminated[:] = False
        return self.reset_terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        # 初期姿勢は assets.py の default_joint_pos をそのまま書く（符号の反転はしない）。
        # EventManager の reset はそのあとに呼び、初期関節角の DR がこの値を基準に乗るようにする
        q_target = self.robot.data.default_joint_pos[env_ids].clone()
        qd_target = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_joint_state_to_sim(
            position=q_target,
            velocity=qd_target,
            env_ids=env_ids
        )
        if hasattr(self, "event_manager") and self.event_manager is not None:
            self.event_manager.reset(env_ids)

        # リズムを引き直し、エピソードの長さを決める
        if hasattr(self, "rhythm_generator"):
            self.rhythm_generator.reset(env_ids)

            # 16 拍分の時間 [s]。助走を固定するとき（rand_lead_in_s ≥ 0）は生成器が決めた長さ
            bpms = self.rhythm_generator.current_bpms[env_ids]
            durations_s = self.rhythm_generator.episode_duration_s(env_ids)
            if durations_s is None:
                durations_s = 16.0 * (60.0 / bpms)
            steps = (durations_s / self.dt_ctrl_step).long()
            self.episode_duration_steps[env_ids] = steps

        # Gen2：エピソードの頭で圧力チャネルを 0 から始め、コントローラ側の DR（k・θt・圧力の τ/L・通信の遅れ）を抽選し直す。
        # Gen1 はここを呼んでいなかった（pam_tau_scale_range の抽選も効いていなかった）ので、既定では呼ばない
        if getattr(self.cfg, "reset_controller_on_episode", False):
            self.action_controller.reset_idx(env_ids)

        self.prev_actions[env_ids] = 0.0

        if hasattr(self, "obs_history"):
            self.obs_history[env_ids] = 0.0
        if getattr(self, "obs_version", "v1") == "v2":
            self.sensor_p_buf[env_ids] = 0.0
            self.sensor_c_buf[env_ids] = 0.0
            self.contact_latch[env_ids] = 0.0
            self._resample_p_bias(env_ids)
        if hasattr(self, "reward_manager"):
            self.reward_manager.reset_idx(env_ids)
        if hasattr(self, "logging_manager"):
            self.logging_manager.reset_idx(env_ids)
        # 2 回目のリズムのリセット：エピソードの長さ（上で 1 回目の BPM から計算）と、実際に流れるリズムの BPM がずれる。
        # Gen1 からこのまま。rhythm_reset_once = True で呼ばない（GEN2.md）
        if hasattr(self, "rhythm_generator") and not getattr(self.cfg, "rhythm_reset_once", False):
            self.rhythm_generator.reset(env_ids)
            # 助走を固定するとき（rand_lead_in_s ≥ 0）は、実際に流れるリズムの長さでエピソードを決め直す（2 回目のリセットのずれを持ち込まない）
            d2 = self.rhythm_generator.episode_duration_s(env_ids)
            if d2 is not None:
                self.episode_duration_steps[env_ids] = (d2 / self.dt_ctrl_step).long()


        if hasattr(self, "last_pressure_est"):
            self.last_pressure_est[env_ids] = 0.0

    def close(self):
        if hasattr(self, "logging_manager") and self.logging_manager is not None:
            self.logging_manager.save_on_exit()
        super().close()


# 単体での動作確認用
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="UI なしで実行")
    p.add_argument("--num_envs", type=int, default=128, help="並列環境数")
    p.add_argument("--max_steps", type=int, default=1000, help="実行ステップ数")
    return p.parse_args()

def main():
    args = _parse_args()
    app = AppLauncher(headless=args.headless).app
    try:
        cfg = Porcaro2026EnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = Porcaro2026Env(cfg)
        print("[INFO] 環境をリセットします...")
        _ = env.reset()
        print(f"[INFO] {args.max_steps} ステップのシミュレーションを実行します。")
        for i in range(args.max_steps):
            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            obs, rew, terminated, truncated, info = env.step(actions)
            if (terminated | truncated).any():
                env.reset_done(terminated | truncated)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and hasattr(env, 'logging_manager') and env.logging_manager:
            env.logging_manager.save_on_exit()
        app.close()

if __name__ == "__main__":
    main()
