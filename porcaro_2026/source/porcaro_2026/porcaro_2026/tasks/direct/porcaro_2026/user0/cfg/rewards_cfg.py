from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    """報酬の重み（rewards/reward.py）。"""

    scale_reward_by_force_magnitude: bool = False

    # 有効な振りとみなす手首の高さ [deg]。0 = 手首が 0° より上がってから打てば有効
    # （grip 0 のとき手首は約 9° で打面に当たる）
    swing_amplitude_threshold_deg: float = 0.0

    # *_s の時間は使っていない。reward.py が BPM（16 分音符の長さ）から計算する

    # 打撃の一致
    weight_match: float = 1.0
    impact_window_s: float = 0.04

    # 休みを守る報酬と、休み中の接触の罰
    weight_rest: float = 0.005
    weight_rest_penalty: float = -0.01

    # 長すぎる接触（押し付け）の罰
    weight_contact_continuous: float = -0.2
    max_contact_duration_s: float = 0.04

    weight_joint_limits: float = 0.0
    weight_miss: float = -1.0
    # クールタイム内の連打と、音符の外の打撃
    weight_double_hit: float = -0.5

    # 圧力の項。手首の共収縮（DF と F が同時に高い）は 0 で無効
    weight_wrist_co_contract: float = 0.0
    # G を握り続けると罰
    weight_grip_penalty: float = -0.5

    target_force_fd: float = 20.0  # 目標の打撃力 [N]
    sigma_force: float = 15.0      # 力の一致度 exp(-|F - 目標| / sigma) の幅 [N]

    limit_wrist_range: tuple[float, float] = (-100.0, 120.0)
