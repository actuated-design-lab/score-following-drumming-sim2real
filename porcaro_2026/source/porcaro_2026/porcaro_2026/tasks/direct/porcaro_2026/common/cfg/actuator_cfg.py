from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class PamDelayModelCfg:
    """空気圧の遅れ（可変むだ時間＋可変一次遅れ）の設定。値は pneumatic.py の 1D の表と同じ。"""

    tau_pressure_axis: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)     # [MPa]
    tau_values: tuple[float, ...] = (0.043, 0.045, 0.060, 0.066, 0.094, 0.131)  # 時定数 [s]（TAU_TAB）

    deadtime_pressure_axis: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)   # [MPa]
    deadtime_values: tuple[float, ...] = (0.038, 0.035, 0.032, 0.030, 0.023, 0.023)  # むだ時間 [s]（L_TAB）。圧力が高いほど短い

    max_delay_time: float = 0.1     # 遅延バッファの長さ [s]

    # 表を使うときは無視される
    delay_time: float = 0.04
    time_constant: float = 0.15


@configclass
class ActuatorNetModelCfg:
    """ActuatorNet（データ駆動モデル）の設定"""
    input_dim: int = 4
    output_dim: int = 1
    hidden_units: list[int] = (64, 64)
    model_path: str | None = None  # 学習済みの重み（.pt）

@configclass
class PamGeometricCfg:
    """PAM の幾何（自然長、ワイヤのたるみ）"""

    # ワイヤの長さのずれ [m]、(DF, F, G)。正 = たるみ（力の出始めが遅れる）、負 = 初期張力
    wire_slack_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)

    natural_length: float = 0.150   # 筋の自然長 L0 [m]
    use_absolute_geometry: bool = False

@configclass
class PamModelA_GeometricCfg(PamGeometricCfg):
    """Model A：|θ − θt| から収縮率、たるみなし"""
    use_absolute_geometry: bool = True
    wire_slack_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)
@configclass
class PamModelA_DynamicsCfg(PamDelayModelCfg):
    """Model A：固定の時定数＋1D のむだ時間"""
    use_2d_dynamics: bool = False
    fixed_time_constant: float = 0.09   # [s]
    use_1d_deadtime: bool = True
