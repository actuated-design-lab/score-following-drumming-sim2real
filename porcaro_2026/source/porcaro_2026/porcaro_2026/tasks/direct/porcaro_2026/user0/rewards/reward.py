from __future__ import annotations
import torch
import math
from ..cfg.rewards_cfg import RewardsCfg

class RewardManager:
    def __init__(self, cfg: RewardsCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.hit_threshold = getattr(cfg, "hit_threshold_force", 1.0)
        self.target_ref_val = float(cfg.target_force_fd)

        self.default_bpm = 120.0        # current_bpm が渡されないときの BPM
        # 目標波形が目標力の 10 % 未満の区間を休みとみなす
        self.rest_threshold = self.target_ref_val * 0.10

        # hit_state：0 = 非接触、1 = 有効な振りからの接触、2 = 振りが足りない接触
        self.hit_state = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_peak_force = torch.zeros(num_envs, device=self.device)
        self.contact_duration = torch.zeros(num_envs, device=self.device)
        self.peak_timing_scale = torch.zeros(num_envs, device=self.device)
        self.last_reward_time = torch.full((num_envs,), -1.0, device=self.device)
        self.prev_is_rest = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.current_time_s = torch.zeros(num_envs, device=self.device)
        self.max_height_since_last_hit = torch.full((num_envs,), -1.0, device=self.device)

        self.has_hit_current_note = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        threshold_deg = getattr(cfg, "swing_amplitude_threshold_deg", 0.5)
        self.swing_threshold = threshold_deg * (math.pi / 180.0)

        self.episode_sums = {
            "match": torch.zeros(num_envs, device=self.device),
            "rest_reward": torch.zeros(num_envs, device=self.device),
            "rest_penalty": torch.zeros(num_envs, device=self.device),
            "miss_penalty": torch.zeros(num_envs, device=self.device),
            "double_hit_penalty": torch.zeros(num_envs, device=self.device),
            "contact_limit": torch.zeros(num_envs, device=self.device),
            "joint_limit": torch.zeros(num_envs, device=self.device),
            "wrist_co_contract": torch.zeros(num_envs, device=self.device),
            "grip_penalty": torch.zeros(num_envs, device=self.device)
        }

    def reset_idx(self, env_ids: torch.Tensor):
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.contact_duration[env_ids] = 0.0
        self.peak_timing_scale[env_ids] = 0.0
        self.last_reward_time[env_ids] = -1.0
        self.prev_is_rest[env_ids] = True
        self.current_time_s[env_ids] = 0.0
        self.max_height_since_last_hit[env_ids] = -1.0
        self.has_hit_current_note[env_ids] = False
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

    def compute_rewards(self,
                        actions: torch.Tensor,
                        p_out: torch.Tensor,
                        joint_pos: torch.Tensor,
                        force_z: torch.Tensor,
                        target_force_trace: torch.Tensor,
                        target_force_ref: torch.Tensor,
                        dt: float,
                        current_bpm: torch.Tensor = None) -> dict[str, torch.Tensor]:

        self.current_time_s += dt

        wrist_pos = joint_pos[:, 0]
        self.max_height_since_last_hit = torch.max(self.max_height_since_last_hit, wrist_pos)

        if current_bpm is None:
             safe_bpm = torch.full((self.num_envs,), self.default_bpm, device=self.device)
        else:
             safe_bpm = current_bpm.clamp(min=10.0)

        t_16th = 15.0 / safe_bpm

        # 時間の閾値は 16 分音符の長さに比例させる
        dyn_max_contact   = t_16th * 0.40   # これを超える接触は contact_limit の対象
        dyn_cooltime      = t_16th * 0.25   # 報酬を出してからこの時間は次の打撃に報酬を出さない（ダブルの 2 打目はこれより後）
        dyn_impact_window = t_16th * 0.40
        dyn_miss_thresh   = t_16th * 2.00

        match_scale_factor = torch.ones_like(safe_bpm)
        time_scale_factor = (safe_bpm / 120.0)

        terms = {}
        is_touching = (force_z > self.hit_threshold)
        is_rest_period = (target_force_trace < self.rest_threshold)

        self.has_hit_current_note[is_rest_period] = False

        # 未打：音符の区間が終わった時点で、その音符に報酬が出ていなければ罰
        note_offset = (~self.prev_is_rest) & is_rest_period
        miss_penalty_term = torch.zeros(self.num_envs, device=self.device)
        if note_offset.any():
            ids = torch.where(note_offset)[0]
            time_diff = self.current_time_s[ids] - self.last_reward_time[ids]
            missed = (time_diff > dyn_miss_thresh[ids]) | (self.last_reward_time[ids] < 0.0)
            real_miss = missed & (self.hit_state[ids] == 0)
            if real_miss.any():
                miss_penalty_term[ids[real_miss]] = 1.0

        terms["miss_penalty"] = miss_penalty_term * match_scale_factor

        self.contact_duration = torch.where(
            is_touching,
            self.contact_duration + dt,
            torch.zeros_like(self.contact_duration)
        )

        # 打撃の報酬：接触の立ち上がり → 接触中 → 離れたときに確定する
        match_reward = torch.zeros(self.num_envs, device=self.device)
        double_hit_penalty_term = torch.zeros(self.num_envs, device=self.device)

        # 接触の立ち上がり
        rising = (self.hit_state == 0) & is_touching
        if rising.any():
            ids = torch.where(rising)[0]

            # 前の打撃のあとに手首が swing_threshold より上がっていれば有効な振り
            valid_swing = self.max_height_since_last_hit[ids] > self.swing_threshold

            self.hit_state[ids] = torch.where(valid_swing, torch.tensor(1, device=self.device), torch.tensor(2, device=self.device))

            self.current_peak_force[ids] = force_z[ids]
            self.peak_timing_scale[ids] = (target_force_trace[ids] / self.target_ref_val).clamp(0.0, 1.0)
            self.max_height_since_last_hit[ids] = -10.0

        # 接触中：impact_window の間のピーク力と、目標波形の大きさの最大を取る
        sustain = ((self.hit_state == 1) | (self.hit_state == 2)) & is_touching
        if sustain.any():
            ids = torch.where(sustain)[0]
            in_window = (self.contact_duration[ids] <= dyn_impact_window[ids])
            if in_window.any():
                upd_ids = ids[in_window]
                current_scale = (target_force_trace[upd_ids] / self.target_ref_val).clamp(0.0, 1.0)
                self.peak_timing_scale[upd_ids] = torch.max(self.peak_timing_scale[upd_ids], current_scale)

                curr_force = force_z[upd_ids]
                is_new_peak = (curr_force > self.current_peak_force[upd_ids])
                if is_new_peak.any():
                    self.current_peak_force[upd_ids[is_new_peak]] = curr_force[is_new_peak]

        # 離れたとき：報酬と罰を確定する
        falling = (self.hit_state != 0) & (~is_touching)
        if falling.any():
            ids = torch.where(falling)[0]

            hit_in_note = (self.peak_timing_scale[ids] > 0.005)
            time_since_last_reward = self.current_time_s[ids] - self.last_reward_time[ids]

            is_cooled_down = (time_since_last_reward > dyn_cooltime[ids])
            is_not_locked = ~self.has_hit_current_note[ids]

            # 報酬は有効な振りからの接触（hit_state == 1）だけ。1 音符に 1 回
            valid_hits = (self.hit_state[ids] == 1)
            rewardable_mask = valid_hits & hit_in_note & is_cooled_down & is_not_locked
            success_ids = ids[rewardable_mask]

            if len(success_ids) > 0:
                accuracy_score, magnitude_score = self._evaluate_hit(
                    self.current_peak_force[success_ids],
                    target_force_ref[success_ids]
                )
                base_reward = 0.2 + (0.4 * magnitude_score) + (0.4 * accuracy_score)
                match_reward[success_ids] = base_reward * match_scale_factor[success_ids]

                self.last_reward_time[success_ids] = self.current_time_s[success_ids]
                self.has_hit_current_note[success_ids] = True

            # 罰はすべての接触が対象。音符の中だがクールタイム内の連打
            too_fast_mask = hit_in_note & (~is_cooled_down)
            too_fast_ids = ids[too_fast_mask]
            if len(too_fast_ids) > 0:
                double_hit_penalty_term[too_fast_ids] = 1.0 * match_scale_factor[too_fast_ids]

            # 音符の外の打撃（休みの間の接触、またはタイミングを大きく外した打撃）
            rest_hit_mask = ~hit_in_note
            rest_hit_ids = ids[rest_hit_mask]
            if len(rest_hit_ids) > 0:
                double_hit_penalty_term[rest_hit_ids] = 1.0 * match_scale_factor[rest_hit_ids]

            self.hit_state[ids] = 0
            self.current_peak_force[ids] = 0.0
            self.peak_timing_scale[ids] = 0.0

        terms["match"] = match_reward
        terms["double_hit_penalty"] = double_hit_penalty_term

        # 毎ステップの項：休みを守る報酬、休み中の接触、長すぎる接触
        compliance = is_rest_period & (~is_touching)
        rest_base = torch.where(compliance, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))
        terms["rest_reward"] = rest_base * time_scale_factor

        violation = is_rest_period & is_touching

        # 休み中の接触は 1 N を超えた分だけ罰する（かすりは許す）
        force_excess = (force_z - 1.0).clamp(min=0.0)
        rest_pen_base = torch.where(violation, (force_excess / 10.0), torch.zeros(self.num_envs, device=self.device))
        terms["rest_penalty"] = rest_pen_base * time_scale_factor

        over_time = (self.contact_duration - dyn_max_contact).clamp(min=0.0)
        terms["contact_limit"] = over_time * time_scale_factor

        terms["joint_limit"] = torch.zeros(self.num_envs, device=self.device)

        # 圧力の項。p_out [MPa] を 0.6 MPa で割って 0〜1 にする
        P_MAX = 0.6
        act_df = (p_out[:, 0] / P_MAX).clamp(0.0, 1.0)
        act_f  = (p_out[:, 1] / P_MAX).clamp(0.0, 1.0)
        act_g  = (p_out[:, 2] / P_MAX).clamp(0.0, 1.0)

        # 手首の共収縮：DF と F に同時に圧が入っていると罰
        terms["wrist_co_contract"] = (act_df * act_f) * time_scale_factor* dt

        # グリップ：G を握り続けると罰（2 乗なので短い握りは小さい）
        terms["grip_penalty"] = (act_g ** 2) * time_scale_factor *dt

        w_miss = getattr(self.cfg, "weight_miss", -10.0)
        w_double = getattr(self.cfg, "weight_double_hit", -5.0)
        w_wrist_co = getattr(self.cfg, "weight_wrist_co_contract", -0.05)
        w_grip_pen = getattr(self.cfg, "weight_grip_penalty", -0.01)

        total_reward = (
            terms["match"] * self.cfg.weight_match +
            terms["rest_reward"] * self.cfg.weight_rest +
            terms["rest_penalty"] * self.cfg.weight_rest_penalty +
            terms["contact_limit"] * self.cfg.weight_contact_continuous +
            terms["joint_limit"] * self.cfg.weight_joint_limits +
            terms["miss_penalty"] * w_miss +
            terms["double_hit_penalty"] * w_double +
            terms["wrist_co_contract"] * w_wrist_co +
            terms["grip_penalty"] * w_grip_pen
        )

        self.prev_is_rest = is_rest_period.clone()
        return total_reward, terms

    def _evaluate_hit(self, peak_force, target_val):
        force_error = torch.abs(peak_force - target_val)
        accuracy_score = torch.exp(-force_error / self.cfg.sigma_force)
        magnitude_score = (peak_force / target_val).clamp(0.0, 1.0)
        return accuracy_score, magnitude_score
