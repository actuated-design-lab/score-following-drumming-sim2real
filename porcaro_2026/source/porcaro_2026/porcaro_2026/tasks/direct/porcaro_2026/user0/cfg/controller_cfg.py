from __future__ import annotations
from isaaclab.utils import configclass

from .assets import FORCE_MAP_CSV, H0_MAP_CSV

@configclass
class TorqueControllerCfg:
    """TorqueActionController（common/actions/torque.py）の設定。既定値は Gen1。"""
    # 制御モード（"ep" か "pressure"）
    control_mode: str = "pressure"

    r: float = 0.014
    L: float = 0.150
    theta_t_DF_deg: float = 7.0
    theta_t_F_deg: float = 70.0
    theta_t_G_deg: float = 45.0
    Pmax: float = 0.6  # [MPa]。pneumatic.py の表の最大と同じ
    tau: float = 0.09
    dead_time: float = 0.03
    N: float = 630.0  # 簡易式 Fpam_quasi_static 用（力マップの CSV を使うときは使わない）
    pam_viscosity: float = 0.0
    force_map_csv: str = FORCE_MAP_CSV
    # 力マップの倍率。スカラー（3筋共通）か (DF, F, G) の組（筋ごと）。Gen1 は 0.2
    force_scale: float | tuple[float, float, float] = 0.2
    h0_map_csv: str = H0_MAP_CSV
    use_pressure_dependent_tau: bool = True

    # PAM のヒステリシス力と、たるみの境目
    pam_hys_const: float = 0.5          # c0 [N]
    pam_hys_coef_p: float = 15.0        # cP [N/MPa]
    pam_contract_gain: float = 1.5      # 収縮側の非対称係数
    pam_extend_gain: float = 1.0        # 伸長側の非対称係数
    pam_p_dot_scale: float = 100.0      # 向き d = tanh(clip(scale·dP/dt)/0.1) の scale
    transition_width: float = 0.0       # たるみ→張りの遷移幅（0 = 階段）
    # ワイヤが張り始めてから張り切るまでの角度幅 [deg]（DF, F, G）。0 = θt で即座に張る
    #   θt からたるむ側へこの幅の範囲で、筋の力に 0→1 の係数を線形に掛ける
    pam_engage_width_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 力マップ・自由収縮マップに入れる圧力の遊び作用素の片側の幅 [MPa]。0 = 実圧力のまま
    pam_map_play_w: float = 0.0

    # ヒステリシス力の向きの決め方
    #   relay: 圧力の変化率の符号で即座に切り替える（Gen1）
    #   play : 圧力の遊び作用素。幅 w_i [MPa] 以上戻ったときに連続に切り替わる（速さに依らない。Gen2）
    #   play_length : 筋の長さ（収縮率 h）の遊び作用素。圧力一定でも関節が動けば向きが変わる
    pam_hys_mode: str = "relay"
    pam_hys_play_widths: tuple[float, ...] = (0.02,)    # 片側の幅 w_i [MPa]。複数並べると Prandtl–Ishlinskii 型
    pam_hys_play_weights: tuple[float, ...] | None = None   # 各要素の重み（合計 1）。None = 等分
    # pam_hys_mode = "play_length" のとき：筋の収縮率 h の遊びの幅（0.003 = 0.3%、手首 1.8° 相当）
    pam_hys_play_widths_h: tuple[float, ...] = (0.003,)

    # Gen2 で足した項目。既定値は Gen1 と同じ動き（user0/GEN2.md）
    # 圧力モデル："table"（Gen1）/ "shaped"（再同定モデル、common/actions/itv.py）
    pressure_model: str = "table"
    shaped_params: dict | None = None                       # shaped のパラメータの上書き（None = itv.py の DEFAULTS）
    pressure_tau_scale_range: tuple[float, float] = (1.0, 1.0)  # shaped の τ の倍率（DR）
    pressure_L_scale_range: tuple[float, float] = (1.0, 1.0)    # shaped のむだ時間 L の倍率（DR）
    # 通信の遅れ [s]（指令が ITV に届くまで。shaped は通信を除いて同定したので別に入れる）。range を与えると DR
    cmd_delay_s: float = 0.0
    cmd_delay_range_s: tuple[float, float] | None = None
    # 筋の縮み方の倍率 k の範囲（筋ごとに独立に抽選。F(P, h/k)、h0·k）。(1, 1) で DR なし
    contraction_scale_range: tuple[float, float] = (1.0, 1.0)
    # ワイヤの張り始め角のずらし [deg]：(DF, F, G) それぞれ (下限, 上限)
    theta_t_offset_range_deg: tuple = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
