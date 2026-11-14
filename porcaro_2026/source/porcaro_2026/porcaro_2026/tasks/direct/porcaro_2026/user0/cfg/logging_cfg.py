from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class LoggingCfg:
    """物理ステップごとのログ（関節・指令・圧力・力）"""
    enabled: bool = True
    filepath: str = "simulation_log.csv"

@configclass
class RewardLoggingCfg:
    """報酬の項ごとのログ"""
    enabled: bool = True
    filepath: str = "reward_log.csv"