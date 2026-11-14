import torch
import torch.nn.functional as F
import numpy as np

class RhythmGenerator:
    """目標の力の波形（打撃列をスーパーガウシアンで畳み込んだもの）を環境ごとに作る。学習と sim の検証で共通。"""
    def __init__(self, num_envs, device, dt, max_episode_length,
                 bpm_range=None, target_force=0.0):

        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_episode_length
        self.target_peak_force = target_force

        self.bpm_options = torch.tensor([60.0, 80.0, 100.0, 120.0, 140.0, 160.0], device=device)

        # カリキュラム：レベルごとに使う BPM の上限（bpm_options の添字）
        self.curriculum_level = 0
        self.max_bpm_idx_per_level = [1, 3, 5]

        # 1 小節の中の打撃の位置（16 分音符単位）
        self.rudiments = {
            "single_4":  [0, 8],
            "single_8":  [0, 4, 8, 12],
            "double":    [0, 1, 4, 5, 8, 9, 12, 13],
            "rest":      []
        }
        self.pattern_keys = ["single_4", "single_8", "double", "rest"]

        self.current_bpms = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.current_pattern_idxs = torch.zeros(num_envs, device=device, dtype=torch.long)

        self.target_trajectories = torch.zeros(
            (num_envs, max_episode_length), device=device, dtype=torch.float32
        )

        self.test_mode = False
        self.test_bpm = 120.0
        self.test_pattern = "single_8"

        # 目標の波形のカーネル
        width_sec = 0.035
        sigma = width_sec / 2.0
        kernel_radius = int(width_sec / dt)
        t_vals = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32) * dt

        # 指数 4 のスーパーガウシアン。ピーク付近が平らで裾が急に落ちるので、高 BPM でも隣の打撃と重なりにくい
        kernel = self.target_peak_force * torch.exp(-0.5 * (t_vals / sigma) ** 4)
        self.kernel = kernel.view(1, 1, -1)
        self.kernel_padding = kernel_radius

    def set_test_mode(self, enabled: bool, bpm: float = 120.0, pattern: str = "single_4"):
        self.test_mode = enabled
        self.test_bpm = bpm
        self.test_pattern = pattern
        if pattern not in self.rudiments and pattern != "random":
             if "double" in pattern: self.test_pattern = "double"
             elif "single" in pattern: self.test_pattern = "single_8"
             else: self.test_pattern = "single_4"

    def set_curriculum_level(self, level: int):
        self.curriculum_level = min(level, 2)

    # 周期的でない打撃列（既定は使わない＝グリッドのまま）
    def set_random_mode(self, enabled: bool, bpm_range=(60.0, 160.0),
                        note_beats=(0.25, 1.0 / 3.0, 0.5, 0.75, 1.0, 1.5, 2.0),
                        note_probs=(0.20, 0.10, 0.25, 0.10, 0.20, 0.10, 0.05),
                        rest_prob=0.15, min_interval_s=0.08, curriculum=True,
                        lead_in_s=-1.0, episode_beats=16, tail_s=0.5):
        """有効にすると、学習時（test_mode でないとき）のリズムを周期的でない打撃列にする。

        エピソードの長さ（16 拍）・最初の 1 小節が休み・目標の力の波形（スーパーガウシアン）はグリッドと同じ。
        テンポはエピソードごとに bpm_range から連続値で一様に引き、エピソードの中では一定（拍の位相の観測と矛盾させないため）。
        2 小節目の頭から、打撃の間隔を音価 note_beats [拍] から note_probs の確率で選んでつないでいく。
        確率 rest_prob でその音符を休みにする（間隔だけ進む）。間隔が min_interval_s 未満になる音価は選ばない。
        curriculum=True なら、レベル 0：テンポは範囲の下 1/3 まで・8 分音符より短い音価なし、
        レベル 1：下 2/3 まで・16 分なし（3 連は可）、レベル 2：全部（グリッドのカリキュラムと同じ 3 段）。

        lead_in_s < 0（既定）：上のとおり、最初の 1 小節（4 拍）が休みで、16 拍でエピソードが終わる（episode_beats・tail_s は使わない）。
        lead_in_s ≥ 0：休みの小節をやめ、最初の打撃を lead_in_s [s] に置く（テンポによらない助走）。
          そこから episode_beats 拍ぶん音符を並べ、エピソードの長さは lead_in_s + episode_beats 拍 + tail_s [s]
          （tail_s は最後の音符の打撃と報酬がエピソードの中に収まるための余白）。長さは episode_duration_s() で env に渡す。
          拍の位相の観測はエピソードの頭から数えるので、評価の MIDI（最初の打を 2.0 s に置いたもの）と同じ関係になる。
        """
        self.random_mode = bool(enabled)
        self.rand_bpm_range = (float(bpm_range[0]), float(bpm_range[1]))
        self.rand_note_beats = np.asarray(note_beats, dtype=np.float64)
        p = np.asarray(note_probs, dtype=np.float64)
        if len(p) != len(self.rand_note_beats):
            raise ValueError("note_beats と note_probs の長さが違う")
        self.rand_note_probs = p / p.sum()
        self.rand_rest_prob = float(rest_prob)
        self.rand_min_interval_s = float(min_interval_s)
        self.rand_curriculum = bool(curriculum)
        self.rand_lead_in_s = float(lead_in_s)
        self.rand_episode_beats = float(episode_beats)
        self.rand_tail_s = float(tail_s)
        self.rand_ep_duration_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._rng = np.random.default_rng()

    def uses_lead_in(self) -> bool:
        return bool(getattr(self, "random_mode", False)) and not self.test_mode and getattr(self, "rand_lead_in_s", -1.0) >= 0.0

    def episode_duration_s(self, env_ids):
        """lead_in を使うとき、直近の reset で決めたエピソードの長さ [s]（使わないときは None）"""
        return self.rand_ep_duration_s[env_ids] if self.uses_lead_in() else None

    def _random_levels(self):
        """カリキュラムのレベルに応じた (テンポの上限, 使える音価の下限 [拍])"""
        lo, hi = self.rand_bpm_range
        if not self.rand_curriculum or self.curriculum_level >= 2:
            return hi, 0.0
        if self.curriculum_level == 0:
            return lo + (hi - lo) / 3.0, 0.5
        return lo + (hi - lo) * 2.0 / 3.0, 0.3       # 3 連（1/3 拍）は使える、16 分（1/4 拍）はまだ

    def _random_spikes(self, num_reset):
        """(bpms, spikes) を返す。spikes は (num_reset, max_steps) の 0/1"""
        bpm_hi, min_beat = self._random_levels()
        lo = self.rand_bpm_range[0]
        bpms = self._rng.uniform(lo, max(bpm_hi, lo), num_reset)
        spikes = np.zeros((num_reset, self.max_steps), dtype=np.float32)
        lead = self.rand_lead_in_s if getattr(self, "rand_lead_in_s", -1.0) >= 0.0 else None
        durs = np.zeros(num_reset, dtype=np.float32)
        for e in range(num_reset):
            beat_s = 60.0 / bpms[e]
            ok = (self.rand_note_beats >= min_beat - 1e-9) & (self.rand_note_beats * beat_s >= self.rand_min_interval_s)
            if not ok.any():
                ok = self.rand_note_beats == self.rand_note_beats.max()
            beats = self.rand_note_beats[ok]
            probs = self.rand_note_probs[ok] / self.rand_note_probs[ok].sum()
            if lead is None:
                t, end, t0_s = 4.0, 16.0, 0.0           # [拍]。最初の 1 小節は休み、16 拍でエピソードが終わる
            else:
                t, end, t0_s = 0.0, self.rand_episode_beats, lead   # 最初の打撃を lead [s] に置き、episode_beats 拍並べる
                durs[e] = lead + end * beat_s + self.rand_tail_s
            first = True
            while t < end - 1e-9:
                hit = first or (self._rng.random() >= self.rand_rest_prob)
                if hit:
                    k = int(round((t0_s + t * beat_s) / self.dt))
                    if k < self.max_steps:
                        spikes[e, k] = 1.0
                first = False
                t += float(self._rng.choice(beats, p=probs))
        self._last_durs = torch.as_tensor(durs, device=self.device)
        return (torch.as_tensor(bpms, device=self.device, dtype=torch.float32),
                torch.as_tensor(spikes, device=self.device))

    def reset(self, env_ids):
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # BPM とパターン（エピソードの中では固定）
        use_random = getattr(self, "random_mode", False) and not self.test_mode
        if use_random:
            bpms, rand_spikes = self._random_spikes(num_reset)
            if hasattr(self, "rand_ep_duration_s"):
                self.rand_ep_duration_s[env_ids] = self._last_durs
            ep_pat_idxs = torch.full((num_reset,), len(self.pattern_keys), device=self.device, dtype=torch.long)
            ep_patterns = ["random"] * num_reset
        elif self.test_mode:
            bpms = torch.full((num_reset,), self.test_bpm, device=self.device)
            pat_idx = self.pattern_keys.index(self.test_pattern) if self.test_pattern in self.pattern_keys else 0
            ep_pat_idxs = torch.full((num_reset,), pat_idx, device=self.device, dtype=torch.long)
            ep_patterns = [self.pattern_keys[pat_idx]] * num_reset
        else:
            max_idx = self.max_bpm_idx_per_level[self.curriculum_level]
            idxs = torch.randint(0, max_idx + 1, (num_reset,), device=self.device)
            bpms = self.bpm_options[idxs]

            if self.curriculum_level == 0:
                probs = torch.tensor([0.4, 0.45, 0.05, 0.1], device=self.device)
            elif self.curriculum_level == 1:
                probs = torch.tensor([0.2, 0.4, 0.3, 0.1], device=self.device)
            else:
                probs = torch.tensor([0.2, 0.2, 0.5, 0.1], device=self.device)

            ep_pat_idxs = torch.multinomial(probs, num_reset, replacement=True)
            ep_patterns = [self.pattern_keys[i] for i in ep_pat_idxs.tolist()]

        self.current_bpms[env_ids] = bpms
        self.current_pattern_idxs[env_ids] = ep_pat_idxs

        # グリッド：4 小節（1 小節目は休み）
        steps_per_16th = (15.0 / bpms / self.dt)
        spikes = torch.zeros((num_reset, self.max_steps), device=self.device)
        for bar_idx in range(0 if use_random else 4):
            bar_start_steps = bar_idx * 16 * steps_per_16th

            if bar_idx == 0:
                selected_patterns = ["rest"] * num_reset
            else:
                selected_patterns = ep_patterns

            unique_patterns = set(selected_patterns)

            for pat_name in unique_patterns:
                env_mask_list = [p == pat_name for p in selected_patterns]
                env_mask = torch.tensor(env_mask_list, device=self.device, dtype=torch.bool)

                if not env_mask.any():
                    continue

                offsets = self.rudiments.get(pat_name, [])
                if not offsets:
                    continue

                target_local_indices = torch.where(env_mask)[0]
                base = bar_start_steps[target_local_indices].unsqueeze(1)
                step_unit = steps_per_16th[target_local_indices].unsqueeze(1)
                off_tensor = torch.tensor(offsets, device=self.device).unsqueeze(0)
                hit_times_float = base + off_tensor * step_unit
                hit_times = torch.round(hit_times_float).long()
                valid_hits = hit_times < self.max_steps

                for i in range(len(target_local_indices)):
                    local_env_idx = target_local_indices[i]
                    valid_times = hit_times[i][valid_hits[i]]
                    spikes[local_env_idx, valid_times] = 1.0

        if use_random:
            spikes = rand_spikes

        # 打撃列をカーネルで畳み込んで目標の波形にする
        spikes_reshaped = spikes.unsqueeze(1)
        traj = F.conv1d(spikes_reshaped, self.kernel, padding=self.kernel_padding)
        traj = traj.squeeze(1)

        if traj.shape[1] > self.max_steps:
            traj = traj[:, :self.max_steps]
        elif traj.shape[1] < self.max_steps:
            traj = F.pad(traj, (0, self.max_steps - traj.shape[1]))

        self.target_trajectories[env_ids] = traj

    def get_lookahead(self, current_time_step_indices, horizon_steps):
        offsets = torch.arange(horizon_steps, device=self.device)
        indices = current_time_step_indices.unsqueeze(1) + offsets.unsqueeze(0)
        safe_indices = indices.clamp(max=self.max_steps - 1)
        vals = torch.gather(self.target_trajectories, 1, safe_indices)
        valid_mask = (indices < self.max_steps).float()
        return vals * valid_mask

    def get_current_target(self, current_time_step_indices):
        indices = current_time_step_indices.unsqueeze(1).clamp(max=self.max_steps-1)
        return torch.gather(self.target_trajectories, 1, indices).squeeze(1)