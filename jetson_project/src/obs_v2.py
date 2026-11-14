"""obs_v2.py — 観測 v2（2026-11-01、(b) の学習）の実機側の部品

sim（porcaro_2026 リポジトリの user0/porcaro_2026_env.py::_sample_sensors）と同じ規則。数値は sim の cfg と同じにすること:
  contact_flag_threshold_n = 1.0、contact_flag_onset_frac = 0.75、contact_flag_reset_lead_steps = 2

接触のラッチフラグ（1 ビット）：
  tick ごとに、前の tick から受けた 200 Hz の全パケットのロードセル force_N の最大が閾値以上なら「接触あり」。
  それを OR で溜め、そのあと、目標波形の lead tick 先（先読みの lead 番目、0〜1 に正規化）が onset_frac 以上なら 0 に戻す。
  sim は通信の遅れの代わりに接触を 1 tick 遅らせ、戻す位置も同じだけずらしている（k − 1 + lead）。実機は遅れが物理的に入るので k + lead。
  どちらも「目標 j の窓は、目標の時刻の 20 ms 前の区間の接触から始まる」になる。
"""
from __future__ import annotations

import numpy as np

CONTACT_THRESHOLD_N = 1.0
ONSET_FRAC = 0.75
RESET_LEAD_STEPS = 2


class ContactLatch:
    def __init__(self, threshold_n: float = CONTACT_THRESHOLD_N, onset_frac: float = ONSET_FRAC,
                 lead_steps: int = RESET_LEAD_STEPS):
        self.threshold_n = float(threshold_n)
        self.onset_frac = float(onset_frac)
        self.lead_steps = int(lead_steps)
        self.latch = 0.0

    def reset(self):
        self.latch = 0.0

    def update(self, f_max: float, traj_norm: np.ndarray) -> float:
        """f_max：この tick のあいだの force_N の最大（パケットがなければ -inf）。traj_norm：先読み（0〜1、0 番目 = いまの tick）"""
        if f_max >= self.threshold_n:
            self.latch = 1.0
        if len(traj_norm) > self.lead_steps and float(traj_norm[self.lead_steps]) >= self.onset_frac:
            self.latch = 0.0
        return self.latch

    def describe(self) -> dict:
        return {"contact_threshold_n": self.threshold_n, "onset_frac": self.onset_frac, "reset_lead_steps": self.lead_steps}
