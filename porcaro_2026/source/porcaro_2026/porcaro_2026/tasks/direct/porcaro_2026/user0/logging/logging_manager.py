from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING, Any, List, Sequence
from .datalogger import DataLogger

if TYPE_CHECKING:
    from ..porcaro_2026_env import Porcaro2026Env

class LoggingManager:
    """環境 0 の物理ステップごとの値を CSV に書く。行は step の中で貯め、力と報酬が決まってから書く。"""

    def __init__(self,
                 env: Porcaro2026Env,
                 dt: float,
                 log_filepath: str | None = None,
                 enable_logging: bool = False):

        self.env = env
        self.device = env.device
        self.dt = dt  # 物理ステップ（sim.dt）
        self.num_envs = env.num_envs
        self.enable_logging = enable_logging

        # 環境ごとのエピソード内の時刻 [s]
        self.current_time_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # 1 回の step（decimation 回の物理ステップ）の行。finalize_log_step で書いてクリアする
        self.step_data_buffer: list[list[float]] = []

        self.logger: DataLogger | None = None
        if enable_logging and log_filepath is not None:
            headers = [
                "time_s",
                "action_0", "action_1", "action_2",
                "q_wrist_deg", "q_grip_deg",
                "qd_wrist_deg", "qd_grip_deg",
                "force_z", "f1_score",
                "target_force", "target_bpm",
                "P_cmd_DF", "P_cmd_F", "P_cmd_G",
                "P_out_DF", "P_out_F", "P_out_G",
                "step_reward"
            ]
            self.logger = DataLogger(filepath=log_filepath, headers=headers)
            print(f"[LoggingManager] Logging enabled -> {log_filepath}")

    def reset_idx(self, env_ids: torch.Tensor):
        if self.enable_logging:
            self.current_time_s[env_ids] = 0.0

    def update_time(self, dt_step: float):
        if self.enable_logging:
            self.current_time_s += dt_step

    def buffer_step_data(self,
                         q_full: torch.Tensor,
                         qd_full: torch.Tensor,
                         telemetry: dict,
                         actions: torch.Tensor,
                         current_sim_time: float,  # 使っていない
                         target_force: float = 0.0,
                         target_bpm: float = 0.0):
        """物理ステップ 1 回分の行を貯める（力と報酬はあとで finalize_log_step が埋める）。"""
        if not self.enable_logging:
            return

        env_idx = 0
        t = self.current_time_s[env_idx].item()
        act = actions[env_idx].detach().cpu().numpy().tolist()
        # 関節は deg で書く
        wrist_deg = np.degrees(q_full[env_idx, self.env.dof_idx[0]].item())
        grip_deg  = np.degrees(q_full[env_idx, self.env.dof_idx[1]].item())
        wrist_vel_deg = np.degrees(qd_full[env_idx, self.env.dof_idx[0]].item())
        grip_vel_deg  = np.degrees(qd_full[env_idx, self.env.dof_idx[1]].item())

        def get_tel(key, idx):
            val = telemetry.get(key, None)
            if val is not None:
                return val[env_idx, idx].item()
            return 0.0

        p_cmd = [get_tel("P_cmd", i) for i in range(3)]
        p_out = [get_tel("P_out", i) for i in range(3)]

        # 力（列 8）、F1（列 9）、報酬（最後の列）は 0 にしておく
        row = [
            t,
            *act,
            wrist_deg, grip_deg,
            wrist_vel_deg, grip_vel_deg,
            0.0, 0.0,
            target_force, target_bpm,
            *p_cmd,
            *p_out,
            0.0
        ]
        self.step_data_buffer.append(row)

    def finalize_log_step(self, peak_force: torch.Tensor, f1_force: torch.Tensor, step_reward: torch.Tensor):
        """step の最後に、貯めた行に力・F1・報酬を入れて DataLogger に渡す。

        peak_force は物理ステップごとの接触力 (N, decimation, 3)。f1_force・step_reward は (N,) で、全行に同じ値を入れる。
        """
        if not self.enable_logging or self.logger is None:
            self.step_data_buffer.clear()
            return

        env_idx = 0
        num_buffered = len(self.step_data_buffer)

        forces_cpu = peak_force[env_idx].detach().cpu()  # [decimation, 3]
        # 行数と力の履歴の長さがずれたときは短い方に合わせる
        limit = min(num_buffered, forces_cpu.shape[0])

        reward_val = step_reward[env_idx].item()
        f1_val = f1_force[env_idx].item()

        rows_to_write = []

        for i in range(limit):
            row = self.step_data_buffer[i]
            # force_z は z 成分（報酬は力の大きさを使う）
            f_z = forces_cpu[i, 2].item()
            row[8] = f_z
            row[9] = f1_val
            row[-1] = reward_val
            rows_to_write.append(row)

        if rows_to_write:
            self.logger.add_data_batch(rows_to_write)
        self.step_data_buffer.clear()

    def save_on_exit(self):
        if self.logger:
            self.logger.save()
            print(f"[LoggingManager] Logs saved to {self.logger.filepath}")