from __future__ import annotations
import torch
import math
from .base import ActionController, RobotLike
from .pam import (
    PAMChannel, calculate_effective_contraction,
    PamForceMap, H0Map,
    apply_soft_engagement,
    calculate_absolute_contraction,
    apply_model_a_force,
    calculate_simple_latched_friction,
    calculate_play_friction,
    play_direction,
    play_operator_update,
)
from .itv import ShapedOrificeChannel, EnvDelay
from ..cfg.actuator_cfg import PamGeometricCfg

def _stack3(a, b, c, n_envs: int) -> torch.Tensor:
    """3 本の筋の値を (n_envs, 3) にまとめる（記録用）。環境 1 つのときに 0 次元になる値もそろえる。"""
    return torch.stack([torch.as_tensor(x, dtype=torch.float32).reshape(-1).expand(n_envs) for x in (a, b, c)], dim=1)


class TorqueActionController(ActionController):
    """行動 → 指令圧 → 圧力チャネル → PAM の力 → 関節トルク。手首は DF と F の拮抗、グリップは G。"""

    def __init__(self,
                 dt_ctrl: float,
                 control_mode: str = "pressure",
                 r: float = 0.014, L: float = 0.150,
                 theta_t_DF_deg: float = 0.0,
                 theta_t_F_deg:  float = 0.0,
                 theta_t_G_deg:  float = 0.0,
                 Pmax: float = 0.6,
                 tau: float = 0.09, dead_time: float = 0.03,
                 N: float = 630.0,
                 pam_viscosity: float = 0.0,
                 pam_hys_const: float = 0.5,
                 pam_hys_coef_p: float = 15,
                 force_map_csv: str | None = None,
                 force_scale: float | tuple[float, float, float] = 1.0,
                 h0_map_csv: str | None = None,
                 use_pressure_dependent_tau: bool = True,
                 geometric_cfg: PamGeometricCfg | None = None,
                 transition_width: float = 0.0,
                 pressure_shrink_gain: float = 0.0,
                 pam_p_dot_scale: float = 100,
                 pam_contract_gain: float = 1.5,
                 pam_extend_gain: float = 1.0,
                 pam_tau_scale_range: tuple[float, float] = (1.0, 1.0),
                 pam_hys_mode: str = "relay",
                 pam_hys_play_widths: tuple[float, ...] = (0.02,),
                 pam_hys_play_weights: tuple[float, ...] | None = None,
                 pam_hys_play_widths_h: tuple[float, ...] = (0.003,),
                 pam_engage_width_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 pam_map_play_w: float = 0.0,
                 # Gen2 で足した引数。既定値は Gen1 と同じ動き
                 pressure_model: str = "table",
                 shaped_params: dict | None = None,
                 pressure_tau_scale_range: tuple[float, float] = (1.0, 1.0),
                 pressure_L_scale_range: tuple[float, float] = (1.0, 1.0),
                 cmd_delay_s: float = 0.0,
                 cmd_delay_range_s: tuple[float, float] | None = None,
                 contraction_scale_range: tuple[float, float] = (1.0, 1.0),
                 theta_t_offset_range_deg: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
                     (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
                 ):

        self.dt_ctrl = float(dt_ctrl)
        self.control_mode = control_mode
        self.r, self.L = float(r), float(L)

        # ワイヤの張り始め角 [rad]。reset() で環境ごとの Tensor にする
        self.theta_t_rad = {
            "DF": math.radians(theta_t_DF_deg),
            "F":  math.radians(theta_t_F_deg),
            "G":  math.radians(theta_t_G_deg)
        }

        self.theta_t_rad_tensors = None

        self.Pmax = float(Pmax)
        self.N = float(N)
        self.pam_viscosity = float(pam_viscosity)
        self.pam_hys_const = float(pam_hys_const)
        self.pam_hys_coef_p = float(pam_hys_coef_p)
        # 力マップの倍率。スカラー（3筋共通、Gen1）か (DF, F, G) の組（筋ごと）
        if isinstance(force_scale, (list, tuple)):
            if len(force_scale) != 3:
                raise ValueError(f"force_scale は スカラー か (DF, F, G) の3要素: {force_scale}")
            self.force_scale_DF, self.force_scale_F, self.force_scale_G = (float(v) for v in force_scale)
        else:
            self.force_scale_DF = self.force_scale_F = self.force_scale_G = float(force_scale)
        self.force_scale = self.force_scale_DF   # 単一値を読むコード用
        self.transition_width = float(transition_width)
        # ワイヤが張り始めてから張り切るまでの角度幅 [deg]（DF, F, G）。0 = θt で即座に張る。
        # θt からたるむ側へ w 度の範囲で、筋の力に 0→1 の係数を線形に掛ける（θt で 1、θt から w 度で 0）。
        # 例: F を θt_F = 70°、w = 47° にすると、23° で張り始めて 70° で張り切る（10/14 静的試験の当てはめ）
        if len(pam_engage_width_deg) != 3:
            raise ValueError(f"pam_engage_width_deg は (DF, F, G) の3要素: {pam_engage_width_deg}")
        self.engage_width_rad = tuple(math.radians(float(v)) for v in pam_engage_width_deg)
        # 力マップ・自由収縮マップに入れる圧力を遊び作用素に通す。片側の幅 [MPa]、0 = 実圧力のまま。
        # 加圧の途中はマップが w 低い圧力を、減圧の途中は w 高い圧力を見る → 同じ圧力で上げと下げの収縮率が変わる
        # （PAM 単体の等張性で、上げ・下げの差が圧力の幅で 25〜70 kPa）。0 MPa 付近では上限を 2P にして、0 MPa で力が残らないようにする。
        # 注意：長さを固定したときの力の差も ∂F/∂P × 2w だけ出る（単体の等尺性で測った差 3.5〜9 N より大きくなりやすい）。
        # 摩擦（pam_hys_*）も長さのヒステリシスを作るので、両方を入れると二重になる
        if pam_map_play_w < 0:
            raise ValueError(f"pam_map_play_w は 0 以上: {pam_map_play_w}")
        self.pam_map_play_w = float(pam_map_play_w)
        self.map_play_z = None                          # (n_envs, 3)
        self.map_play_needs_init = None

        self.pam_p_dot_scale = float(pam_p_dot_scale)
        self.pam_contract_gain = float(pam_contract_gain)
        self.pam_extend_gain = float(pam_extend_gain)
        self.pam_tau_scale_range = pam_tau_scale_range

        # ヒステリシス力の向きの決め方
        #   "relay"（既定・Gen1）: 圧力の変化率 dP/dt の符号で即座に切り替える（calculate_simple_latched_friction）
        #   "play"                : 圧力の遊び作用素で決める。幅 w_i [MPa] 以上戻ったときに連続に切り替わる（速さに依らない）
        #   "play_length"         : 筋の長さ（収縮率 h）の遊び作用素で決める。幅は pam_hys_play_widths_h
        #                               [収縮率、0.003 = 0.3%、手首 1.8° 相当]。筋が縮む向きで d = +1（圧力の play の加圧の枝と同じ扱い）。
        #                               圧力一定でも関節が動けば向きが変わる（ISIE 静的負荷試験で、圧力一定でもループが出ることに対応）
        if pam_hys_mode not in ("relay", "play", "play_length"):
            raise ValueError(f"pam_hys_mode は 'relay' か 'play' か 'play_length': {pam_hys_mode}")
        self.pam_hys_mode = pam_hys_mode
        if pam_hys_mode == "play_length":
            pam_hys_play_widths = pam_hys_play_widths_h
        w = [float(v) for v in (pam_hys_play_widths if isinstance(pam_hys_play_widths, (list, tuple)) else [pam_hys_play_widths])]
        if any(v <= 0 for v in w):
            raise ValueError(f"pam_hys_play_widths は正の値: {w}")
        a = [float(v) for v in pam_hys_play_weights] if pam_hys_play_weights is not None else [1.0 / len(w)] * len(w)
        if len(a) != len(w) or abs(sum(a) - 1.0) > 1e-6:
            raise ValueError(f"pam_hys_play_weights は widths と同じ長さで合計 1: {a}")
        self._play_widths_list, self._play_weights_list = w, a
        self.play_widths = self.play_weights = None     # reset() でデバイスに載せる
        self.play_state = None                           # (n_envs, 3, K)
        self.play_needs_init = None                      # (n_envs,) True の環境は次の apply で z = P に揃える

        print("-" * 60)
        print(f"[TorqueActionController] Initializing PAM Force Model...")
        if force_map_csv:
            try:
                self.force_map = PamForceMap.from_csv(force_map_csv)
                print(f"  >>> SUCCESS: Loaded Real Force Map")
            except Exception as e:
                print(f"  >>> ERROR: Failed to load Force Map: {e}")
                raise e
        else:
            self.force_map = None
            print(f"  >>> WARNING: No CSV provided. Using Ideal Quasi-static Model")

        if h0_map_csv:
            try:
                self.h0_map = H0Map.from_csv(h0_map_csv)
                print(f"  >>> SUCCESS: Loaded Real h0 Map")
            except Exception as e:
                print(f"  >>> ERROR: Failed to load h0 Map: {e}")
                raise e
        else:
            self.h0_map = None

        # 幾何：use_absolute_geometry = True で Model A（|θ − θt| から収縮率）、False で Model B（有効収縮率とたるみ）
        if geometric_cfg is not None:
            self.slack_offsets = torch.tensor(geometric_cfg.wire_slack_offsets)
            self.L0_sim = geometric_cfg.natural_length
            self.use_absolute_geometry = getattr(geometric_cfg, "use_absolute_geometry", False)
        else:
            self.slack_offsets = torch.zeros(3)
            self.L0_sim = self.L
            self.use_absolute_geometry = False

        use_2d = not self.use_absolute_geometry

        tau_lut = None
        if use_pressure_dependent_tau:
            tau_P_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            tau_vals   = [0.043,0.045,0.060,0.066,0.094,0.131]
            tau_lut = (tau_P_axis, tau_vals)

        # 圧力モデル
        #   "table" : Gen1。始点圧×指令圧の時定数テーブル＋むだ時間（PAMChannel）。むだ時間に通信の遅れも含んでいた
        #   "shaped": 再同定モデル（itv.py）。通信を除いて同定したので、通信の遅れは cmd_delay_s で別に入れる
        if pressure_model not in ("table", "shaped"):
            raise ValueError(f"pressure_model は 'table' か 'shaped': {pressure_model}")
        self.pressure_model = pressure_model
        if pressure_model == "shaped":
            prm = dict(shaped_params or {})
            self.ch_DF, self.ch_F, self.ch_G = (
                ShapedOrificeChannel(dt_ctrl, Pmax=self.Pmax, tau_scale_range=pressure_tau_scale_range,
                                     L_scale_range=pressure_L_scale_range, **prm) for _ in range(3))
        else:
            self.ch_DF = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax,
                                    tau_lut=tau_lut, use_2d_dynamics=use_2d,
                                    tau_scale_range=self.pam_tau_scale_range)
            self.ch_F  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax,
                                    tau_lut=tau_lut, use_2d_dynamics=use_2d,
                                    tau_scale_range=self.pam_tau_scale_range)
            self.ch_G  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax,
                                    tau_lut=tau_lut, use_2d_dynamics=use_2d,
                                    tau_scale_range=self.pam_tau_scale_range)

        # 通信の遅れ（Jetson → MicroLabBox → ITV に指令が届くまで）[s]。圧力チャネルの手前で指令を遅らせる。
        # cmd_delay_range_s を与えると reset_idx のたびに環境ごとに一様に抽選する（None なら cmd_delay_s で固定）
        self.cmd_delay_s = float(cmd_delay_s)
        self.cmd_delay_range_s = None if cmd_delay_range_s is None else tuple(float(v) for v in cmd_delay_range_s)
        d_max = max(self.cmd_delay_s, self.cmd_delay_range_s[1] if self.cmd_delay_range_s else 0.0)
        self.cmd_delay = EnvDelay(dt_ctrl, d_max + 0.01) if d_max > 0.0 else None
        self.cmd_delay_L = None                          # (n_envs, 3)

        # 筋の縮み方の倍率 k（ドメインランダム化、筋ごとに独立に抽選）。F_k(P, h) = F(P, h / k)、h0_k(P) = k · h0(P)
        # k < 1：同じ圧力・同じ力でも k 倍しか縮まない。(1, 1) で DR なし
        self.contraction_scale_range = tuple(float(v) for v in contraction_scale_range)
        self.k_contr = None                              # (n_envs, 3)
        # ワイヤの張り始め角のずらし [deg]（DF, F, G の順に (下限, 上限)）。reset_idx のたびに抽選
        if len(theta_t_offset_range_deg) != 3:
            raise ValueError(f"theta_t_offset_range_deg は (DF, F, G) の3組: {theta_t_offset_range_deg}")
        self.theta_t_offset_range_rad = tuple((math.radians(float(a)), math.radians(float(b)))
                                              for a, b in theta_t_offset_range_deg)

        self._last_telemetry: dict | None = None
        self.pressure_shrink_gain = float(pressure_shrink_gain)

        self.prev_P_stack = None
        self.prev_direction_stack = None

    def reset(self, n_envs: int, device: str | torch.device):
        self.ch_DF.reset(n_envs, device)
        self.ch_F.reset(n_envs, device)
        self.ch_G.reset(n_envs, device)
        if self.force_map is not None: self.force_map.to(device)
        if self.h0_map is not None: self.h0_map.to(device)

        self.slack_offsets = self.slack_offsets.to(device)

        # 環境ごとの張り始め角（DR の範囲の中央で始め、reset_idx で抽選し直す）
        self.theta_t_rad_tensors = {
            k: torch.full((n_envs,), v + sum(self.theta_t_offset_range_rad[i]) / 2, device=device, dtype=torch.float32)
            for i, (k, v) in enumerate(self.theta_t_rad.items())
        }
        self.k_contr = torch.full((n_envs, 3), sum(self.contraction_scale_range) / 2, device=device, dtype=torch.float32)
        if self.cmd_delay is not None:
            self.cmd_delay.reset((n_envs, 3), device)
            d0 = sum(self.cmd_delay_range_s) / 2 if self.cmd_delay_range_s else self.cmd_delay_s
            self.cmd_delay_L = torch.full((n_envs, 3), d0, device=device, dtype=torch.float32)

        self._last_telemetry = None
        self.prev_P_stack = None
        self.prev_direction_stack = torch.zeros((n_envs, 3), device=device, dtype=torch.float32)
        K = len(self._play_widths_list)
        self.play_widths = torch.tensor(self._play_widths_list, device=device, dtype=torch.float32)
        self.play_weights = torch.tensor(self._play_weights_list, device=device, dtype=torch.float32)
        self.play_state = torch.zeros((n_envs, 3, K), device=device, dtype=torch.float32)
        self.play_needs_init = torch.ones(n_envs, device=device, dtype=torch.bool)
        self.map_play_z = torch.zeros((n_envs, 3), device=device, dtype=torch.float32)
        self.map_play_needs_init = torch.ones(n_envs, device=device, dtype=torch.bool)

    def _map_pressure(self, P_DF, P_F, P_G):
        """マップに入れる圧力。pam_map_play_w = 0 ならそのまま。"""
        if self.pam_map_play_w <= 0.0:
            return P_DF, P_F, P_G
        P = torch.stack([P_DF, P_F, P_G], dim=1)
        if self.map_play_z is None or self.map_play_z.shape != P.shape:
            self.map_play_z = P.clone()
            self.map_play_needs_init = torch.zeros(P.shape[0], device=P.device, dtype=torch.bool)
        if bool(self.map_play_needs_init.any()):
            m = self.map_play_needs_init
            self.map_play_z[m] = P[m]
            self.map_play_needs_init[m] = False
        w = self.pam_map_play_w
        self.map_play_z = torch.minimum(torch.maximum(self.map_play_z, P - w), P + w)
        Pm = torch.clamp(self.map_play_z, min=0.0)
        Pm = torch.minimum(Pm, 2.0 * P)               # 0 MPa に近いとき（P < w）は 2P を上限に
        return Pm[:, 0], Pm[:, 1], Pm[:, 2]

    def reset_idx(self, env_ids: torch.Tensor):
        self.ch_DF.reset_idx(env_ids)
        self.ch_F.reset_idx(env_ids)
        self.ch_G.reset_idx(env_ids)
        if self.prev_P_stack is not None:
            self.prev_P_stack[env_ids] = 0.0
        if self.prev_direction_stack is not None:
            self.prev_direction_stack[env_ids] = 0.0
        if self.play_needs_init is not None:
            self.play_needs_init[env_ids] = True
        if self.map_play_needs_init is not None:
            self.map_play_needs_init[env_ids] = True
        # Gen2 のドメインランダム化（範囲が (a, a) なら何もしない）
        dev = self.k_contr.device if self.k_contr is not None else None
        n = len(env_ids)
        if dev is not None and n > 0:
            lo, hi = self.contraction_scale_range
            if hi > lo:
                self.k_contr[env_ids] = lo + (hi - lo) * torch.rand((n, 3), device=dev)
            for i, k in enumerate(("DF", "F", "G")):
                lo, hi = self.theta_t_offset_range_rad[i]
                if hi > lo:
                    self.theta_t_rad_tensors[k][env_ids] = self.theta_t_rad[k] + lo + (hi - lo) * torch.rand(n, device=dev)
            if self.cmd_delay is not None:
                self.cmd_delay.reset_idx(env_ids)
                if self.cmd_delay_range_s and self.cmd_delay_range_s[1] > self.cmd_delay_range_s[0]:
                    lo, hi = self.cmd_delay_range_s
                    self.cmd_delay_L[env_ids] = (lo + (hi - lo) * torch.rand(n, device=dev)).unsqueeze(1).expand(-1, 3)

    def compute_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.nan_to_num(actions, nan=0.0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        return self._compute_command_pressure(actions)

    def _compute_command_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        # pressure：各筋の指令圧 = (a + 1) / 2 · Pmax。ep：a0 = 手首の差圧、a1 = 手首の基準圧、a2 = G
        if self.control_mode == "pressure":
            P_cmd_unscaled = (actions + 1.0) * 0.5
            P_cmd_stack = P_cmd_unscaled * self.Pmax
        elif self.control_mode == "ep":
            MAX_P_BASE = self.Pmax * 0.5
            MAX_P_DIFF = self.Pmax * 0.5
            a = actions
            P_base = MAX_P_BASE * (a[:, 1] + 1.0) * 0.5
            P_diff = MAX_P_DIFF * a[:, 0]
            P_cmd_DF = torch.clamp(P_base + P_diff, 0.0, self.Pmax)
            P_cmd_F  = torch.clamp(P_base - P_diff, 0.0, self.Pmax)
            P_cmd_G = self.Pmax * (a[:, 2] + 1.0) * 0.5
            P_cmd_stack = torch.stack([P_cmd_DF, P_cmd_F, P_cmd_G], dim=-1)
        else:
            P_cmd_stack = torch.zeros_like(actions)

        return torch.clamp(P_cmd_stack, 0.0, self.Pmax)

    @torch.no_grad()
    def apply(self, *, actions: torch.Tensor, q: torch.Tensor,
              joint_ids: tuple[int, int], robot: RobotLike) -> None:
        wid, gid = joint_ids
        n_envs = int(q.shape[0])

        # robot.data.joint_vel は [rad/s]、sim は下向きが正なので符号を反転する。q は呼び出し側で反転済み
        dq_sim = robot.data.joint_vel
        dq_wrist_rad = -dq_sim[:, wid]
        dq_grip_rad  = -dq_sim[:, gid]

        if self.theta_t_rad_tensors is None:
            self.reset(n_envs, q.device)

        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, nan=0.0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)

        P_cmd_stack = self._compute_command_pressure(actions)

        # 通信の遅れ → 圧力チャネル
        P_in = self.cmd_delay.step(P_cmd_stack, self.cmd_delay_L) if self.cmd_delay is not None else P_cmd_stack
        P_DF = self.ch_DF.step(P_in[:, 0])
        P_F  = self.ch_F.step(P_in[:, 1])
        P_G  = self.ch_G.step(P_in[:, 2])

        # dP/dt（relay のヒステリシスの向き）
        P_current_stack = torch.stack([P_DF, P_F, P_G], dim=1)
        if self.prev_P_stack is None:
            self.prev_P_stack = torch.zeros_like(P_current_stack)
        if self.prev_direction_stack is None:
             self.prev_direction_stack = torch.zeros_like(P_current_stack)

        P_dot_stack = (P_current_stack - self.prev_P_stack) / self.dt_ctrl
        self.prev_P_stack = P_current_stack.clone()

        q_wrist_rad = q[:, joint_ids[0]]
        q_grip_rad  = q[:, joint_ids[1]]

        SIGN_DF =  1.0
        SIGN_F  = -1.0
        SIGN_G  = -1.0

        # 力マップ・自由収縮マップに入れる圧力（pam_map_play_w > 0 なら遊び作用素を通したもの）
        Pm_DF, Pm_F, Pm_G = self._map_pressure(P_DF, P_F, P_G)

        th_DF = self.theta_t_rad_tensors["DF"]
        th_F  = self.theta_t_rad_tensors["F"]
        th_G  = self.theta_t_rad_tensors["G"]

        if self.use_absolute_geometry:
            # Model A
            eps_DF = calculate_absolute_contraction(q_wrist_rad, th_DF, self.r, self.L0_sim)
            eps_F  = calculate_absolute_contraction(q_wrist_rad, th_F,  self.r, self.L0_sim)
            eps_G  = calculate_absolute_contraction(q_grip_rad,  th_G,  self.r, self.L0_sim)

            if self.force_map is not None and self.h0_map is not None:
                F_DF = apply_model_a_force(self.force_map, self.h0_map, Pm_DF, eps_DF) * self.force_scale_DF
                F_F  = apply_model_a_force(self.force_map, self.h0_map, Pm_F,  eps_F)  * self.force_scale_F
                F_G  = apply_model_a_force(self.force_map, self.h0_map, Pm_G,  eps_G)  * self.force_scale_G

            # 自由収縮率 h0 を超えたら力 0
            if self.h0_map is not None:
                h0_DF, h0_F, h0_G = self.h0_map(Pm_DF), self.h0_map(Pm_F), self.h0_map(Pm_G)
                F_DF = torch.where(eps_DF <= h0_DF, F_DF, torch.zeros_like(F_DF))
                F_F  = torch.where(eps_F  <= h0_F,  F_F,  torch.zeros_like(F_F))
                F_G  = torch.where(eps_G  <= h0_G,  F_G,  torch.zeros_like(F_G))

        else:
            # Model B：有効収縮率 h
            h_DF = calculate_effective_contraction(
                    q_wrist_rad, th_DF, self.r, self.L0_sim, self.slack_offsets[0],
                    pressure=P_DF, shrink_gain=self.pressure_shrink_gain, clamp=False, sign=SIGN_DF)
            h_F = calculate_effective_contraction(
                    q_wrist_rad, th_F, self.r, self.L0_sim, self.slack_offsets[1],
                    pressure=P_F, shrink_gain=self.pressure_shrink_gain, clamp=False, sign=SIGN_F)
            h_G = calculate_effective_contraction(
                    q_grip_rad, th_G, self.r, self.L0_sim, self.slack_offsets[2],
                    pressure=P_G, shrink_gain=self.pressure_shrink_gain, clamp=False, sign=SIGN_G)

            # 静的な力（力マップ）。縮み方の倍率 k（Gen2 の DR、既定 1）：F(P, h / k)
            kc = self.k_contr
            F_DF_static = self.force_map(Pm_DF, h_DF / kc[:, 0]) * self.force_scale_DF
            F_F_static  = self.force_map(Pm_F,  h_F / kc[:, 1])  * self.force_scale_F
            F_G_static  = self.force_map(Pm_G,  h_G / kc[:, 2])  * self.force_scale_G

            # 収縮速度
            def calculate_h_dot(dq_rad, r, sign, L0):
                return (r * sign * dq_rad) / L0

            h_dot_DF = calculate_h_dot(dq_wrist_rad, self.r, SIGN_DF, self.L0_sim)
            h_dot_F  = calculate_h_dot(dq_wrist_rad, self.r, SIGN_F,  self.L0_sim)
            h_dot_G  = calculate_h_dot(dq_grip_rad,  self.r, SIGN_G,  self.L0_sim)

            # ヒステリシス力
            if self.pam_hys_mode in ("play", "play_length"):
                if self.pam_hys_mode == "play":
                    P_now = torch.stack([P_DF, P_F, P_G], dim=1)                   # (n_envs, 3) 遊び作用素の入力＝圧力
                else:
                    P_now = torch.stack([h_DF, h_F, h_G], dim=1)                   # 遊び作用素の入力＝筋の収縮率（縮むと増える）
                if self.play_needs_init is not None and bool(self.play_needs_init.any()):
                    m = self.play_needs_init
                    self.play_state[m] = P_now[m].unsqueeze(-1).expand(-1, -1, self.play_state.shape[-1])
                    self.play_needs_init[m] = False
                self.play_state = play_operator_update(self.play_state, P_now, self.play_widths)
                d = play_direction(self.play_state, P_now, self.play_widths, self.play_weights)
                fric_DF = calculate_play_friction(h_dot_DF, d[:, 0], P_DF, self.pam_viscosity, self.pam_hys_coef_p,
                                                  self.pam_hys_const, self.pam_contract_gain, self.pam_extend_gain)
                fric_F = calculate_play_friction(h_dot_F, d[:, 1], P_F, self.pam_viscosity, self.pam_hys_coef_p,
                                                 self.pam_hys_const, self.pam_contract_gain, self.pam_extend_gain)
                fric_G = calculate_play_friction(h_dot_G, d[:, 2], P_G, self.pam_viscosity, self.pam_hys_coef_p,
                                                 self.pam_hys_const, self.pam_contract_gain, self.pam_extend_gain)
                self.prev_direction_stack = d
            else:
                fric_DF, new_dir_DF = calculate_simple_latched_friction(
                    h_dot_DF, P_dot_stack[:, 0], P_DF,
                    self.prev_direction_stack[:, 0],
                    self.pam_viscosity, self.pam_hys_coef_p, self.pam_hys_const,
                    p_dot_scale=self.pam_p_dot_scale,
                    contract_gain=self.pam_contract_gain,
                    extend_gain=self.pam_extend_gain
                )
                fric_F, new_dir_F = calculate_simple_latched_friction(
                    h_dot_F, P_dot_stack[:, 1], P_F,
                    self.prev_direction_stack[:, 1],
                    self.pam_viscosity, self.pam_hys_coef_p, self.pam_hys_const,
                    p_dot_scale=self.pam_p_dot_scale,
                    contract_gain=self.pam_contract_gain,
                    extend_gain=self.pam_extend_gain
                )
                fric_G, new_dir_G = calculate_simple_latched_friction(
                    h_dot_G, P_dot_stack[:, 2], P_G,
                    self.prev_direction_stack[:, 2],
                    self.pam_viscosity, self.pam_hys_coef_p, self.pam_hys_const,
                    p_dot_scale=self.pam_p_dot_scale,
                    contract_gain=self.pam_contract_gain,
                    extend_gain=self.pam_extend_gain
                )

                self.prev_direction_stack = torch.stack([new_dir_DF, new_dir_F, new_dir_G], dim=1)

            # 合力（負の力は 0）
            F_DF_total_raw = F_DF_static + fric_DF
            F_F_total_raw  = F_F_static  + fric_F
            F_G_total_raw  = F_G_static  + fric_G

            F_DF_total_raw = torch.clamp(F_DF_total_raw, min=0.0)
            F_F_total_raw  = torch.clamp(F_F_total_raw,  min=0.0)
            F_G_total_raw  = torch.clamp(F_G_total_raw,  min=0.0)

            # たるみ → 張りの遷移（h0 の手前 transition_width で 0 → 1）
            if self.h0_map is not None:
                h0_DF, h0_F, h0_G = self.h0_map(Pm_DF) * kc[:, 0], self.h0_map(Pm_F) * kc[:, 1], self.h0_map(Pm_G) * kc[:, 2]

            self._tel_extra = {"F_static": _stack3(F_DF_static, F_F_static, F_G_static, n_envs),
                               "fric": _stack3(fric_DF, fric_F, fric_G, n_envs)}
            F_DF = apply_soft_engagement(F_DF_total_raw, h_DF, h0_DF, self.transition_width)
            F_F  = apply_soft_engagement(F_F_total_raw,  h_F,  h0_F,  self.transition_width)
            F_G  = apply_soft_engagement(F_G_total_raw,  h_G,  h0_G,  self.transition_width)

        # ワイヤの張り始め（pam_engage_width_deg > 0 の筋だけ）
        if any(w > 0.0 for w in self.engage_width_rad):
            def _engage(F, q_rad, th_t, sign, w):
                if w <= 0.0:
                    return F
                slack = sign * (q_rad - th_t)            # θt からたるむ側へ何 rad 入っているか（幾何の縮み × L/r）
                return F * torch.clamp(1.0 - slack / w, min=0.0, max=1.0)
            F_DF = _engage(F_DF, q_wrist_rad, th_DF, SIGN_DF, self.engage_width_rad[0])
            F_F  = _engage(F_F,  q_wrist_rad, th_F,  SIGN_F,  self.engage_width_rad[1])
            F_G  = _engage(F_G,  q_grip_rad,  th_G,  SIGN_G,  self.engage_width_rad[2])

        # トルク（sim の向きに戻す）
        tau_w = self.r * (F_DF - F_F)
        tau_g = self.r * (- F_G)

        tau_full = torch.zeros(n_envs, robot.num_joints, device=q.device, dtype=q.dtype)
        tau_full[:, wid] = -tau_w
        tau_full[:, gid] = -tau_g
        robot.set_joint_effort_target(tau_full)

        P_act_stack = torch.stack([P_DF, P_F, P_G], dim=1)
        self._last_telemetry = {
            "P_cmd": P_cmd_stack.clone(),
            "P_out": P_act_stack.clone(),
            "tau_w": tau_w.clone(),
            "tau_g": tau_g.clone(),
            "F_mus": _stack3(F_DF, F_F, F_G, n_envs).detach().clone(),   # 筋の力（摩擦・たるみ込み）[N]
        }
        for k_, v_ in getattr(self, "_tel_extra", {}).items():                 # Model B のみ：マップの力と摩擦
            self._last_telemetry[k_] = v_.detach().clone()

    def get_last_telemetry(self):
        return self._last_telemetry