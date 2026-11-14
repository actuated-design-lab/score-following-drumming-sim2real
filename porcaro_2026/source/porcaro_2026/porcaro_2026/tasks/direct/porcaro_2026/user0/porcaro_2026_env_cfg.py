# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from pathlib import Path
import math
import torch
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from .cfg.assets import ROBOT_CFG, DRUM_CFG, DATA_DIR
from .cfg.sensors import contact_forces_stick_at_drum_cfg, drum_vs_stick_cfg
from .cfg.controller_cfg import TorqueControllerCfg
from .cfg.logging_cfg import LoggingCfg, RewardLoggingCfg
from .cfg.rewards_cfg import RewardsCfg
from ..common.cfg.actuator_cfg import PamGeometricCfg

def obs_base_dim(cfg, lookahead_steps: int) -> int:
    """1 フレームの観測の次元（obs_version で変わる）。train / play / export_onnx もこれと同じ式を使う"""
    v = getattr(cfg, "obs_version", "v1")
    if v == "v1":
        return 10 + lookahead_steps
    if v == "v2":
        return 12 + lookahead_steps
    raise ValueError(f"obs_version は v1 か v2: {v}")


@configclass
class Porcaro2026EnvCfg(DirectRLEnvCfg):
    """環境の基底の設定。"""

    decimation: int = 4  # 制御 50 Hz
    episode_length_s: float = 100.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,     # 物理 200 Hz（5 ms）
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.0, dynamic_friction=0.0, restitution=0.8,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=3.0,
        replicate_physics=True,
    )

    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    drum_cfg:  RigidObjectCfg  = DRUM_CFG.replace(prim_path="/World/envs/env_.*/Drum")

    stick_contact_cfg: ContactSensorCfg = contact_forces_stick_at_drum_cfg
    drum_contact_cfg: ContactSensorCfg = drum_vs_stick_cfg

    action_space: int = 3
    observation_space: int = 60
    state_space: int = 0
    dof_names: list[str] = ["Base_link_Wrist_joint", "Hand_link_Grip_joint"]

    pam_geometric_cfg: PamGeometricCfg = PamGeometricCfg(
        natural_length=0.150,
        use_absolute_geometry = False,
        wire_slack_offsets=(0.0, 0.0, 0.0),
    )

    use_simple_rhythm: bool = False
    simple_rhythm_mode: str = "single_8"
    simple_rhythm_bpm: float = 120.0
    target_hit_force: float = 20.0
    lookahead_horizon: float = 0.5
    bpm_range: tuple[float, float] = (60.0, 160.0)
    # リズムのリセットを 1 回にする（False = Gen1 のまま：_reset_idx で 2 回引き直し、エピソードの長さと BPM がずれる）
    rhythm_reset_once: bool = False
    # 打撃列（"grid" = 4 パターン × 6 テンポ、"random" = 周期的でない列、rhythm_generator.set_random_mode）
    rhythm_mode: str = "grid"
    rand_bpm_range: tuple[float, float] = (60.0, 160.0)
    rand_note_beats: tuple = (0.25, 1.0 / 3.0, 0.5, 0.75, 1.0, 1.5, 2.0)    # 16 分・8 分 3 連・8 分・付点 8 分・4 分・付点 4 分・2 分
    rand_note_probs: tuple = (0.20, 0.10, 0.25, 0.10, 0.20, 0.10, 0.05)
    rand_rest_prob: float = 0.15
    rand_min_interval_s: float = 0.08
    rand_curriculum: bool = True
    # random の助走：< 0 なら最初の 1 小節を休みにする。≥ 0 なら最初の打撃をこの時刻 [s] に置き、
    # rand_episode_beats 拍ぶん並べ、エピソードは rand_lead_in_s + rand_episode_beats 拍 + rand_tail_s [s]
    rand_lead_in_s: float = -1.0
    rand_episode_beats: int = 16
    rand_tail_s: float = 0.5

    # カリキュラムの切り替え：None = Gen1 のまま（累積 25M / 100M env step）。
    # (i1, i2) を与えると、iteration i1 からレベル 1、i2 からレベル 2（1 iteration = curriculum_steps_per_iter 回の env.step）。
    # レベルが効くのは、その後にリセットした（新しく始まった）エピソードから
    curriculum_iters: tuple | None = None
    curriculum_steps_per_iter: int = 120

    # 観測の版
    #   "v1" = Gen1：q(2)+qd(2)+前回指令(3)+sin/cos 拍位相(2)+BPM(1)+先読み(L)         → 10+L（L=25 で 35）
    #   "v2" = 拍位相を外し、実測圧力(3)と接触のラッチフラグ(1)を足す：
    #          q(2)+qd(2)+前回指令(3)+BPM(1)+先読み(L)+圧力(3)+接触(1)            → 12+L（L=25 で 37）
    obs_version: str = "v1"
    # v2 のセンサの経路（実機は 200 Hz の受信の最新値を 50 Hz の tick で読む）
    obs_sensor_delay_steps: int = 1              # 圧力・接触を何 tick 遅らせて観測に入れるか【仮：1 tick = 20 ms】
    obs_pressure_noise_std: float = 0.005        # 圧力の白色ノイズ [MPa]【仮】
    obs_pressure_bias_range: tuple = (-0.01, 0.01)   # 圧力のずれ [MPa]、エピソードごと・筋ごとに一様【仮】
    contact_flag_threshold_n: float = 1.0        # 接触とみなす力 [N]（sim は k_t を掛けた力、実機はロードセル）
    contact_flag_onset_frac: float = 0.75        # 目標波形がピークのこの割合以上の tick を「目標打撃の時刻」とみなす
    contact_flag_reset_lead_steps: int = 2       # 目標打撃の何 tick 前からを、その打撃の窓にするか（2 = 20 ms 早い打撃まで。env._sample_sensors）

    use_frame_stacking: bool = False
    frame_stack_k: int = 5

    pam_tau_scale_range: tuple[float, float] = (1.0, 1.0)

    # 打撃力の倍率 k_t（sim の接触力に掛けて、報酬・ログが見る力にする）。Gen1 は 3.0
    impact_force_scale: float = 3.0
    # エピソードの頭でコントローラ（圧力チャネル・コントローラ側の DR）をリセットするか。Gen1 は False
    reset_controller_on_episode: bool = False

    controller: TorqueControllerCfg = TorqueControllerCfg()
    logging: LoggingCfg = LoggingCfg(enabled=False)
    rewards: RewardsCfg = RewardsCfg()
    reward_logging: RewardLoggingCfg = RewardLoggingCfg()
    events: PorcaroEventCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.rewards, "target_force_fd"):
            self.rewards.target_force_fd = self.target_hit_force

        # observation_space は制御周期と先読みの長さから決める
        dt_ctrl = self.sim.dt * self.decimation
        lookahead_steps = int(self.lookahead_horizon / dt_ctrl)

        # ベース次元：v1 = 10 + L（q2+qd2+prev_act3+sin1+cos1+bpm1）、v2 = 12 + L（上の obs_version を参照）
        base_obs_dim = obs_base_dim(self, lookahead_steps)

        if self.use_frame_stacking:
            self.observation_space = base_obs_dim * self.frame_stack_k
        else:
            self.observation_space = base_obs_dim

        print(f"[Porcaro2026EnvCfg] lookahead_horizon={self.lookahead_horizon}s "
              f"base_obs_dim={base_obs_dim} use_frame_stacking={self.use_frame_stacking} "
              f"frame_stack_k={self.frame_stack_k if self.use_frame_stacking else 1} "
              f"-> observation_space={self.observation_space}")


@configclass
class PorcaroEventCfg:
    randomize_mass: EventTerm = None
    randomize_material: EventTerm = None
    reset_robot_joints: EventTerm = None

def apply_domain_randomization(cfg: Porcaro2026EnvCfg):
    cfg.events = PorcaroEventCfg()
    cfg.events.randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (0.95, 1.05), "operation": "scale"},
    )
    cfg.events.randomize_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "static_friction_range": (0.95, 1.05), "dynamic_friction_range": (0.95, 1.05), "restitution_range": (0.0, 0.0), "num_buckets": 64},
    )
    cfg.events.reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.95, 1.05), "velocity_range": (0, 0)},
    )


# Gen1 の設定（再現用に残す）。Default = DR なし、DR = DR あり
@configclass
class Porcaro2026EnvCfg_Default(Porcaro2026EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.controller.tau = 0.09
        self.controller.use_pressure_dependent_tau = False
        self.controller.pam_viscosity = 0.0

@configclass
class Porcaro2026EnvCfg_DR(Porcaro2026EnvCfg_Default):
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)
        self.pam_tau_scale_range = (0.8, 1.2)


# Gen2（2026/10/29）：機構側を実測で作り直して凍結した sim。中身と根拠は user0/GEN2.md
GEN2_FORCE_MAP_CSV = str(DATA_DIR / "pam_force_map_measured_DF.csv")
GEN2_H0_MAP_CSV = str(DATA_DIR / "pam_force_0_map_measured_DF.csv")


@configclass
class Porcaro2026EnvCfg_Gen2(Porcaro2026EnvCfg_Default):
    """Gen2 の基準（DR なし）。検証・再生用。値は設計値と実測で、合わせ込みの倍率を持たない"""

    def __post_init__(self):
        super().__post_init__()
        c = self.controller
        # 力：ロボットに付いていた DF の単体試験（10/26）のマップ、倍率なし（3 筋共通）
        c.force_map_csv = GEN2_FORCE_MAP_CSV
        c.h0_map_csv = GEN2_H0_MAP_CSV
        c.force_scale = 1.0
        c.transition_width = 0.003
        # ワイヤの張り始め角：設計値
        c.theta_t_DF_deg, c.theta_t_F_deg, c.theta_t_G_deg = 0.0, 70.0, 45.0
        # ヒステリシス力の向き：圧力の遊び（幅 40 kPa）
        c.pam_hys_mode = "play"
        c.pam_hys_play_widths = (0.04,)
        c.pam_hys_coef_p = 15.0
        # 圧力モデル：再同定モデル（通信を除いて同定、common/actions/itv.py）＋通信の遅れ
        c.pressure_model = "shaped"
        # 通信の遅れ【仮】：USB 変換の設定後の往復 約 8 ms（エコー計測）を、観測と指令のループ全体の遅れとして指令側に入れる
        c.cmd_delay_s = 0.008
        # 関節：手首の粘性 0.2（可動域・質量・重心は USD に入っている）
        self.robot_cfg.actuators["wrist"].damping = 0.2
        # 打撃力の倍率【仮】：打撃ありの再生（10/29）でピーク力の中央値を合わせると 1.6（Gen1 は 3.0）
        self.impact_force_scale = 1.6
        self.reset_controller_on_episode = True
        self.pam_tau_scale_range = (1.0, 1.0)     # table 用（Gen2 では使わない）


@configclass
class Gen2EventCfg:
    # Gen1 と同じ：全リンクの質量 ±5 %、接触の摩擦 ±5 %、初期関節角 ±5 %
    randomize_mass: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (0.95, 1.05), "operation": "scale"},
    )
    randomize_material: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "static_friction_range": (0.95, 1.05),
                "dynamic_friction_range": (0.95, 1.05), "restitution_range": (0.0, 0.0), "num_buckets": 64},
    )
    reset_robot_joints: EventTerm = EventTerm(
        func=mdp.reset_joints_by_scale, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.95, 1.05), "velocity_range": (0, 0)},
    )
    # Gen2 で追加【範囲は仮】。スティックの質量は上の ±5 % を上書きする（質量の DR は既定値から掛け直すため、後に書いた方が効く）
    randomize_stick_mass: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Stick_link"]), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )
    randomize_stick_com: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Stick_link"]),
                "com_range": {"x": (-0.002, 0.002), "y": (-0.002, 0.002), "z": (-0.002, 0.002)}},
    )
    randomize_wrist_damping: EventTerm = EventTerm(
        func=mdp.randomize_actuator_gains, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Base_link_Wrist_joint"]),
                "damping_distribution_params": (0.5, 1.5), "operation": "scale"},
    )
    randomize_wrist_friction: EventTerm = EventTerm(
        func=mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Base_link_Wrist_joint"]),
                "friction_distribution_params": (0.5, 1.5), "operation": "scale"},
    )


@configclass
class Porcaro2026EnvCfg_Gen2_DR(Porcaro2026EnvCfg_Gen2):
    """Gen2 の学習用。基準は Gen2 のまま、観測したばらつきを DR で覆う"""

    def __post_init__(self):
        super().__post_init__()
        self.events = Gen2EventCfg()
        c = self.controller
        # 筋の縮み方 k：筋ごとに 0.75〜1.25（静的に合う k が 0.78）
        c.contraction_scale_range = (0.75, 1.25)
        # 圧力モデルの τ・L：×0.8〜1.2（Gen1 の τ と同じ幅）
        c.pressure_tau_scale_range = (0.8, 1.2)
        c.pressure_L_scale_range = (0.8, 1.2)
        # 通信の遅れ【仮】：4〜12 ms
        c.cmd_delay_range_s = (0.004, 0.012)
        # ワイヤの張り始め角のずらし：入れていない（入れるなら例：((-3, 3), (-3, 3), (-3, 3))）
        c.theta_t_offset_range_deg = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


# 学習に使ったタスク（2026-11-01）：周期的でない打撃列＋観測 v2。機構側・DR は Gen2_DR のまま
@configclass
class Porcaro2026EnvCfg_Gen2_B(Porcaro2026EnvCfg_Gen2_DR):
    """観測 v2（拍位相なし、実測圧力 3、接触のラッチフラグ 1）。"""

    def __post_init__(self):
        # 分布の値。super().__post_init__ の観測次元の計算より前に入れる
        self.rhythm_mode = "random"
        self.rhythm_reset_once = True
        self.rand_bpm_range = (60.0, 160.0)
        self.rand_note_beats = (0.25, 1.0 / 3.0, 0.5, 0.75, 1.0, 1.5, 2.0)   # 16 分・8 分 3 連・8 分・付点 8 分・4 分・付点 4 分・2 分
        # 16 分は当初の 0.28 から 0.45 に上げた（0.28 だと IOI < 120 ms がレベル 2 で 5.4 %、学習全体で 3.3 % と、
        # 目安の約 10 % を大きく下回ったため）。ほかの音価は 0.28 のときの比を保って縮めた
        self.rand_note_probs = (0.45, 0.076, 0.183, 0.061, 0.122, 0.061, 0.047)
        self.rand_rest_prob = 0.15
        self.rand_min_interval_s = 0.08
        self.rand_curriculum = True
        self.curriculum_iters = (300, 750)        # 0〜299：テンポ 60〜93・8 分より短い音価なし／300〜749：60〜126・16 分なし／750〜：全部
        self.curriculum_steps_per_iter = 120      # agents の num_steps_per_env と同じ
        # 助走 2.0 s（冒頭 1 小節の休みの代わり。評価の MIDI と同じ）＋ 32 拍 ＋ 余白 0.5 s
        self.rand_lead_in_s = 2.0
        self.rand_episode_beats = 32
        self.rand_tail_s = 0.5
        self.obs_version = "v2"
        super().__post_init__()


@configclass
class Porcaro2026EnvCfg_Gen2_B_ObsV1(Porcaro2026EnvCfg_Gen2_B):
    """分布は Gen2_B と同じ、観測は Gen1 と同じ v1。"""

    def __post_init__(self):
        super().__post_init__()
        self.obs_version = "v1"
        dt_ctrl = self.sim.dt * self.decimation
        base = obs_base_dim(self, int(self.lookahead_horizon / dt_ctrl))
        self.observation_space = base * self.frame_stack_k if self.use_frame_stacking else base
