"""
replay_open_loop.py — 実機と同じ圧力指令をシミュレータに開ループで流し、圧力と関節角を記録する

作り直した圧力モデルをシミュレータに入れて、実機の動作を再現できるかを見る。
user0 の環境（--env_cfg、既定は Gen1 の Porcaro2026EnvCfg_Default、DR なし）をそのまま使い、
方策の代わりに CSV の圧力指令を行動に変換して入れる。

圧力モデル（--pmodel）:
  table    : 今のシミュレータ（始点圧×指令圧の 2 次元テーブル, pneumatic.py）
  lag      : 対称な一次遅れ＋むだ時間（9/29–30 のエコー付きデータで同定）
  orifice  : lag に、給気・排気のオリフィス流量の上限を足したもの（同上）
  shaped   : orifice の前段で、指令の小さく速い振動を実測の振幅比マップで縮める（ITV のしきい値・現象論。scripts/tools/user0/jfps2026/itv_models.py）
  pilot    : パイロット段（積分）＋重なりを持つ主弁の2状態モデル（ITV のしきい値・物理寄り。scripts/tools/user0/jfps2026/itv_models.py）
  pilot_leak : pilot の主弁が重なりの中でも少し漏れるようにしたもの（小振幅の削りすぎを修正。同上）
  measured : 実機で測った圧力をそのまま入れる（--real_log 必須）。
             「圧力→角度」だけを比べ、機構側（力マップ・摩擦・たるみ）の誤差を切り分ける
             実測圧力は移動平均（--meas_lpf_ms, 既定 50 ms）と遊び（--meas_play_kpa, 既定 ±10 kPa）を
             かけてから入れる。生のままだと圧力ノイズでヒステリシス力の向きが毎ステップ反転し、
             sim 側のヒステリシスが実質消える

関節の上書き（--joint）: Isaac の actuator の damping / friction（例: --joint '{"wrist_damping":0.0}'）
  と、関節の可動域 <wrist|grip>_<lower|upper>_deg（プロジェクトの向き＝手首を上げると正。例: '{"wrist_upper_deg":80}'）。
  可動域は USD の値を起動時に表示する

機械側の上書き（--ctrl）:
  cfg.controller の項目を JSON で上書きする（force_scale は (DF, F, G) の組も可）。
  例: --ctrl '{"force_scale":[0.2,0.15,0.2],"theta_t_F_deg":60}'

入力:
  --signal  test_signals/tm_*.csv（time, cmd_pressure_DF, cmd_pressure_F, cmd_pressure_G, 50 Hz）
  --real_log  measured モードのときの実機ログ（run_signal_playback.py の出力）
              実機ログは DF/F の圧力列が入れ替わっている（2026/9/29 判明）ので、ここで戻す。
              時刻は flag 列（MicroLabBox が受け取った DF 指令のエコー）で指令に合わせる（scripts/tools/user0/jfps2026/real_log.py）。

出力（--out, 200 Hz, 物理ステップごと）:
  time, P_cmd_DF/F/G, P_out_DF/F/G, wrist_angle_deg, grip_angle_deg, force_N
  角度は実機と同じ向き（Up+）に直したもの。力はスティック接触力の大きさ（×3.0 は環境と同じ）。

DR の束（10/31 追加）: --env_cfg Gen2_DR --num_envs 64 --seed 0 で、DR を引いた 64 本に同じ指令を流す。
  CSV は env 0、全 env の角度と力は <out>_envs.npz（抽選された DR の値も dr_* として入る）
打撃の解析（10/31 追加）: --record_contact（接触力の成分・スティックの速度の列を足す）、--sim_hz 1000、
  --restitution 0.4（ロボットと打面の全形状の反発係数）、--impact_force_scale、--print_materials

使い方（Isaac Lab の python で）:
  python scripts/tools/user0/jfps2026/replay_open_loop.py --signal ../jetson_project/test_signals/tm_C_sine.csv \
      --pmodel table --no_drum --headless --out out/sim_tm_C_sine_table.csv
  python scripts/tools/user0/jfps2026/replay_open_loop.py --signal ... --pmodel measured \
      --real_log ../jetson_project/test_signals/data_tm_C_sine_XXXX.csv --no_drum --headless --out ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description="開ループ再生（圧力指令 → シミュレータ）")
p.add_argument("--signal", required=True)
p.add_argument("--pmodel", choices=["cfg", "table", "lag", "orifice", "shaped", "pilot", "pilot_leak", "measured"], default="table",
               help="cfg = 環境の設定クラスの圧力モデルをそのまま使う（Gen2 なら shaped＋通信の遅れ。学習用 sim と同じ）")
p.add_argument("--env_cfg", choices=["Default", "Gen2", "Gen2_DR"], default="Default",
               help="環境の設定クラス。Default = Gen1、Gen2 = 凍結した学習用 sim の DR なし版（user0/GEN2.md）、"
                    "Gen2_DR = 学習用（DR あり。--num_envs 本の env に同じ指令を流し、DR の束を見る）")
p.add_argument("--num_envs", type=int, default=1,
               help="env の数。Gen2_DR で DR の束を見るとき用。2 以上なら env ごとの角度を <out>_envs.npz に保存")
p.add_argument("--seed", type=int, default=0, help="DR の抽選の種（Gen2_DR のとき）")
p.add_argument("--sim_hz", type=float, default=None,
               help="物理の刻み [Hz]（既定は cfg の 200）。制御 50 Hz は保ったまま decimation を合わせる。打撃の解析用")
p.add_argument("--restitution", type=float, default=None,
               help="ロボットと打面の全形状の反発係数をこの値にする（起動後に物理エンジンへ書き込む）。打撃の解析用")
p.add_argument("--robot_restitution", type=float, default=None,
               help="ロボット側の形状だけ反発係数を上書き（打面は cfg のまま）。0 で学習（Gen2_DR：ロボットの反発を DR で 0）と同じ")
p.add_argument("--impact_force_scale", type=float, default=None, help="k_t の上書き（既定は cfg の値）")
p.add_argument("--record_contact", action="store_true",
               help="打撃の解析用の列を足す：接触力の成分 fx/fy/fz（×k_t 前）とスティック重心の速度 vx/vy/vz・角速度")
p.add_argument("--print_materials", action="store_true",
               help="起動時に、ロボットと打面の形状の摩擦・反発係数（物理エンジンの値）と USD の材料（combine mode）を表示して json に残す")
p.add_argument("--real_log", default=None)
p.add_argument("--params", default=None,
               help="lag/orifice/shaped/pilot/pilot_leak のパラメータを JSON で上書き（例: '{\"tau\":0.09,\"L\":0.045}'）。"
                    "shaped/pilot/pilot_leak の既定値は scripts/tools/user0/jfps2026/itv_models.py の DEFAULTS")
p.add_argument("--no_drum", action="store_true", help="打面を横へ 2 m 退避（打面なしの実機試験と揃える）")
p.add_argument("--ctrl", default=None,
               help="機械側（張力・関節）の設定を JSON で上書き。cfg.controller の項目名で指定。"
                    "例: '{\"force_scale\":[0.2,0.15,0.2],\"theta_t_F_deg\":60,\"pam_hys_const\":1.0}'")
p.add_argument("--joint", default=None,
               help="関節（Isaac の actuator）の設定を JSON で上書き。キーは <wrist|grip>_<damping|friction>。"
                    "例: '{\"wrist_damping\":0.0}'（既定は assets.py：手首 damping 0.1 / friction 0.001、グリップ 0.001 / 0.00001）")
p.add_argument("--meas_lpf_ms", type=float, default=50.0,
               help="measured モードで実測圧力にかける移動平均の幅 [ms]（中心合わせ・遅れなし）。0 で無効")
p.add_argument("--meas_play_kpa", type=float, default=10.0,
               help="移動平均のあとにかける遊び（バックラッシュ）の半幅 [kPa]。0 で無効。"
                    "生のままだと圧力ノイズでヒステリシス力の向き d が毎ステップの56%%で反転する（2026/10/6 判明）。"
                    "既定の 50 ms + ±10 kPa で反転 0.4%%/step（tm_B で確認）")
p.add_argument("--t_end", type=float, default=None, help="最初のこの秒数だけ再生する（既定は信号の最後まで）")
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

# ここから Isaac Lab のモジュールを import できる
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import torch                  # noqa: E402

from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env import Porcaro2026Env        # noqa: E402
import sys as _sys                                                                              # noqa: E402
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itv_models                                                                               # noqa: E402
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gen2"))
from dr_probe import dr_samples                                                                 # noqa: E402
from real_log import load_real_aligned  # 実機ログの読み込みと時刻合わせ（analysis/eval/compare_replay.py と共通）  # noqa: E402
from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env_cfg import (  # noqa: E402
    Porcaro2026EnvCfg_Default, Porcaro2026EnvCfg_Gen2, Porcaro2026EnvCfg_Gen2_DR)

PA = 0.1013  # 大気圧 [MPa]

# 9/29–30 のエコー付きデータ（通信を含まない）で同定した値。analysis/compare_pressure_models.py
DEFAULT_PARAMS = {
    "lag": {"tau": 0.0836, "L": 0.0446},
    "orifice": {"tau": 0.0856, "L": 0.0442, "c_in": 5.17, "c_out": 11.33, "ps": 0.578, "b": 0.053},
}


# 圧力チャネル（PAMChannel と同じインタフェース: reset / reset_idx / step）
class _DelayBuf:
    def __init__(self, dt, L):
        self.dt, self.L = dt, float(L)
        self.D = self.L / dt
        self.M = int(math.floor(self.D)); self.mu = self.D - self.M
        self.K = self.M + 3
        self.buf = None; self.wp = 0

    def reset(self, shape, device):
        self.buf = torch.zeros((self.K, *shape), device=device); self.wp = 0
        self._primed = False

    def step(self, x):
        if not self._primed:           # 最初の値で埋める（開始時の段差を作らない）
            self.buf[:] = x; self._primed = True
        self.buf[self.wp] = x
        a = self.buf[(self.wp - self.M) % self.K]
        b = self.buf[(self.wp - self.M - 1) % self.K]
        self.wp = (self.wp + 1) % self.K
        return (1 - self.mu) * a + self.mu * b


def _g(pu, pd, b):
    """ISO 6358 型の流量関数（pu 上流・pd 下流, 絶対圧）"""
    r = torch.clamp(pd / pu, max=1.0)
    x = torch.clamp((r - b) / (1 - b), min=0.0)
    return torch.where(r <= b, pu, pu * torch.sqrt(torch.clamp(1 - x * x, min=0.0)))


class LagChannel:
    def __init__(self, dt, tau, L, Pmax=0.6):
        self.dt, self.tau, self.Pmax = dt, tau, Pmax
        self.delay = _DelayBuf(dt, L); self.p = None

    def reset(self, n_envs, device):
        self.delay.reset((n_envs,), device); self.p = torch.zeros(n_envs, device=device)

    def reset_idx(self, env_ids):
        if self.p is not None:
            self.p[env_ids] = 0.0

    def step(self, P_cmd):
        u = self.delay.step(torch.clamp(P_cmd, 0.0, self.Pmax))
        self.p = self.p + self.dt / (self.tau + self.dt) * (u - self.p)
        return self.p


class OrificeChannel(LagChannel):
    def __init__(self, dt, tau, L, c_in, c_out, ps, b, Pmax=0.6):
        super().__init__(dt, tau, L, Pmax)
        self.c_in, self.c_out, self.ps, self.b = c_in, c_out, ps, b

    def step(self, P_cmd):
        u = self.delay.step(torch.clamp(P_cmd, 0.0, self.Pmax))
        q = (u - self.p) / self.tau
        pab = self.p + PA
        lim_in = self.c_in * _g(torch.full_like(pab, self.ps + PA), pab, self.b)
        lim_in = torch.where(self.ps + PA > pab, lim_in, torch.zeros_like(lim_in))
        lim_out = self.c_out * _g(pab, torch.full_like(pab, PA), self.b)
        lim_out = torch.where(pab > PA, lim_out, torch.zeros_like(lim_out))
        q = torch.where(q > 0, torch.minimum(q, lim_in), torch.maximum(q, -lim_out))
        self.p = self.p + q * self.dt
        return self.p


class MeasuredChannel:
    """実機で測った圧力を 1 物理ステップ（5 ms）ずつ返す"""
    def __init__(self, seq):
        self.seq = np.asarray(seq, np.float32); self.i = 0; self.dev = "cpu"

    def reset(self, n_envs, device):
        self.i = 0; self.n = n_envs; self.dev = device

    def reset_idx(self, env_ids):
        pass

    def step(self, P_cmd):
        v = float(self.seq[min(self.i, len(self.seq) - 1)]); self.i += 1
        return torch.full((self.n,), v, device=self.dev)


# 実機ログの読み込み（measured モード）
def smooth_centered(x, width_ms, dt):
    """中心合わせの移動平均（遅れなし）。端は端の値で延長"""
    k = int(round(width_ms / 1000.0 / dt))
    if k <= 1:
        return x
    k += (k + 1) % 2                                  # 奇数に
    pad = k // 2
    xp = np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])]
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def play_operator(x, half_width):
    """遊び（バックラッシュ）：入力が ±half_width の帯の中で揺れている間は出力を保つ"""
    if half_width <= 0:
        return x
    y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = min(max(y[i - 1], x[i] - half_width), x[i] + half_width)
    return y


# 接触の材料（DR の値は scripts/tools/user0/gen2/dr_probe.py）
def set_and_report_materials(env):
    """--restitution があれば全形状に書き込み、ロボットと打面の材料（物理エンジンの値）と USD の材料を返す"""
    rep = {}
    for name, asset in (("robot", env.robot), ("drum", env.drum)):
        view = asset.root_physx_view
        m = view.get_material_properties()                     # (num_envs, num_shapes, 3) = 静止摩擦・動摩擦・反発
        r = args.restitution if args.restitution is not None else (args.robot_restitution if name == "robot" else None)
        if r is not None:
            m[..., 2] = float(r)
            view.set_material_properties(m, torch.arange(m.shape[0]))
            m = view.get_material_properties()
        rep[name] = dict(static_friction=sorted(set(np.round(m[0, :, 0].numpy(), 4).tolist())),
                         dynamic_friction=sorted(set(np.round(m[0, :, 1].numpy(), 4).tolist())),
                         restitution=sorted(set(np.round(m[0, :, 2].numpy(), 4).tolist())))
    usd = []
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            path = str(prim.GetPath())
            if path.startswith("/World/envs/env_") and not path.startswith("/World/envs/env_0/") and path != "/World/envs/env_0":
                continue
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                d = {"path": path}
                for a in prim.GetAttributes():
                    n = a.GetName()
                    if "estitution" in n or "riction" in n or "ombine" in n:
                        v = a.Get()
                        d[n] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
                usd.append(d)
    except Exception as e:  # noqa: BLE001
        usd.append({"error": repr(e)})
    rep["usd_materials"] = usd
    print("[replay] 接触の材料（物理エンジン）:", json.dumps({k: v for k, v in rep.items() if k != "usd_materials"}, ensure_ascii=False))
    for d in usd:
        print("[replay]   USD:", d)
    print("[replay]   PhysX の既定の組み合わせは average（USD の combine mode が無いとき）。"
          "実際に効く反発 = 2 つの形状の値を combine mode で組み合わせたもの")
    return rep


def main():
    sig = pd.read_csv(args.signal)
    cmd = sig[["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].values.astype(np.float32)
    n_steps = len(cmd)
    if args.t_end:                                   # 最初の t_end 秒だけ回す（実機との位置合わせは信号全体で行う）
        n_steps = min(n_steps, int(round(args.t_end / 0.02)))

    cfg = {"Default": Porcaro2026EnvCfg_Default, "Gen2": Porcaro2026EnvCfg_Gen2,
           "Gen2_DR": Porcaro2026EnvCfg_Gen2_DR}[args.env_cfg]()
    cfg.scene.num_envs = args.num_envs
    is_dr = args.env_cfg == "Gen2_DR"
    if is_dr:
        cfg.seed = args.seed
        torch.manual_seed(args.seed)
    else:
        cfg.events = None                   # DR なし
    if args.sim_hz:
        dec = int(round(0.02 * args.sim_hz))
        cfg.sim.dt = 0.02 / dec
        cfg.decimation = dec
        cfg.sim.render_interval = dec
        print(f"[replay] 物理の刻み: {1 / cfg.sim.dt:.0f} Hz（decimation {dec}、制御 50 Hz）")
    if args.impact_force_scale is not None:
        cfg.impact_force_scale = args.impact_force_scale
    cfg.pam_tau_scale_range = (1.0, 1.0)
    cfg.logging.enabled = False
    cfg.episode_length_s = n_steps * 0.02 + 10.0   # 途中でエピソードが切れないように
    ctrl_over = json.loads(args.ctrl) if args.ctrl else {}
    for k, v in ctrl_over.items():
        if not hasattr(cfg.controller, k):
            raise SystemExit(f"--ctrl: cfg.controller に '{k}' はありません")
        setattr(cfg.controller, k, tuple(v) if isinstance(v, list) else v)
    if ctrl_over:
        print(f"[replay] 機械側の上書き: {ctrl_over}")
    joint_over = json.loads(args.joint) if args.joint else {}
    limit_over = {}                                   # (関節番号 0 = 手首 / 1 = グリップ, "lower"/"upper") → deg
    for k, v in joint_over.items():
        jn, _, attr = k.partition("_")
        if jn in ("wrist", "grip") and attr in ("lower_deg", "upper_deg"):
            limit_over[(0 if jn == "wrist" else 1, attr[:5])] = float(v)
            continue
        if jn not in cfg.robot_cfg.actuators or attr not in ("damping", "friction"):
            raise SystemExit(f"--joint: '{k}' は <wrist|grip>_<damping|friction> か <wrist|grip>_<lower|upper>_deg で指定")
        setattr(cfg.robot_cfg.actuators[jn], attr, float(v))
    if joint_over:
        print(f"[replay] 関節の上書き: {joint_over}")
    if args.no_drum:
        x, y, z = cfg.drum_cfg.init_state.pos
        cfg.drum_cfg.init_state.pos = (x + 2.0, y, z)   # 下へ動かすと床に埋まるので横へ

    env = Porcaro2026Env(cfg)
    dt_phys = cfg.sim.dt
    N = env.num_envs
    all_ids = torch.arange(N, device=env.device)
    materials = set_and_report_materials(env) if (args.print_materials or args.restitution is not None
                                                       or args.robot_restitution is not None) else None

    # 関節の可動域（USD の値を表示し、--joint の指定があれば書き換える）。sim の向きはプロジェクトと逆（q_sim = −θ）----
    lim = env.robot.data.joint_pos_limits.clone()     # (num_envs, num_joints, 2) [rad]、sim の向き
    for j, name in enumerate(("wrist", "grip")):
        di = int(env.dof_idx[j])
        lo, hi = -float(lim[0, di, 1]), -float(lim[0, di, 0])
        print(f"[replay] {name} の可動域（USD）: {np.degrees(lo):.1f}〜{np.degrees(hi):.1f} deg（プロジェクトの向き）")
    if limit_over:
        for (j, side), deg in limit_over.items():
            di = int(env.dof_idx[j])
            # プロジェクトの lower = sim の upper の符号反転、upper = sim の lower の符号反転
            lim[:, di, 1 if side == "lower" else 0] = -np.radians(deg)
        if hasattr(env.robot, "write_joint_position_limit_to_sim"):
            env.robot.write_joint_position_limit_to_sim(lim, warn_limit_violation=False)
        else:
            env.robot.write_joint_limits_to_sim(lim)
        for j, name in enumerate(("wrist", "grip")):
            di = int(env.dof_idx[j])
            lo, hi = -float(lim[0, di, 1]), -float(lim[0, di, 0])
            print(f"[replay] {name} の可動域（上書き後）: {np.degrees(lo):.1f}〜{np.degrees(hi):.1f} deg")
    Pmax = env.action_controller.Pmax
    ctrl = env.action_controller

    # 圧力モデルの差し替え
    prm = dict(DEFAULT_PARAMS.get(args.pmodel, {}))
    if args.params:
        prm.update(json.loads(args.params))
    if args.pmodel == "lag":
        ctrl.ch_DF, ctrl.ch_F, ctrl.ch_G = (LagChannel(dt_phys, prm["tau"], prm["L"], Pmax) for _ in range(3))
    elif args.pmodel == "orifice":
        ctrl.ch_DF, ctrl.ch_F, ctrl.ch_G = (OrificeChannel(dt_phys, prm["tau"], prm["L"], prm["c_in"], prm["c_out"],
                                                           prm["ps"], prm["b"], Pmax) for _ in range(3))
    elif args.pmodel in ("shaped", "pilot", "pilot_leak"):
        ctrl.ch_DF, ctrl.ch_F, ctrl.ch_G = itv_models.make_channels(args.pmodel, dt_phys, Pmax=Pmax, **prm)
        prm = itv_models.resolved_params(args.pmodel, **prm)
    elif args.pmodel in ("cfg", "table"):
        pass                                          # 環境が作ったチャネルのまま（table は Default の既定）
    elif args.pmodel == "measured":
        if not args.real_log:
            raise SystemExit("--pmodel measured には --real_log が必要です")
        Pm = load_real_aligned(args.real_log, cmd, dt_phys)
        Pm = np.stack([play_operator(smooth_centered(Pm[:, k], args.meas_lpf_ms, dt_phys), args.meas_play_kpa / 1000.0)
                       for k in range(3)], 1)
        ctrl.ch_DF, ctrl.ch_F, ctrl.ch_G = (MeasuredChannel(Pm[:, k]) for k in range(3))
    ctrl.reset(env.num_envs, env.device)
    if is_dr:
        ctrl.reset_idx(all_ids)                     # reset() は DR の範囲の中央に戻すので、ここで抽選し直す
    print(f"[replay] pmodel={args.pmodel} params={prm} steps={n_steps} ({n_steps * 0.02:.1f} s) no_drum={args.no_drum}")

    # 物理ステップごとの記録（_apply_action の直後 = そのステップの圧力と、その時点の関節状態） ----
    rec, rec_envs = [], []
    stick_id = env.robot.find_bodies("Stick_link")[0][0]
    orig_apply = env._apply_action

    def patched_apply():
        orig_apply()
        tel = ctrl.get_last_telemetry()
        q, qd = env._get_corrected_joint_state()
        fv = env.stick_sensor.data.net_forces_w
        fv = fv[:, 0, :] if fv.dim() == 3 else fv
        f = torch.norm(fv * env.force_scale_sim_to_real, dim=-1)
        # GPU の上で 1 行にまとめて溜め、最後に 1 回だけ CPU へ移す（毎ステップの同期を避ける）
        z3 = torch.zeros(3, device=q.device, dtype=torch.float32)
        g = lambda key: tel[key][0].float() if key in tel else z3
        row = [tel["P_cmd"][0].float(), tel["P_out"][0].float(), torch.rad2deg(q[0]).float(),
               f[0:1].float(), torch.rad2deg(qd[0, :1]).float(), tel["tau_w"][0:1].float(),
               g("F_mus"), g("F_static"), g("fric")]
        if args.record_contact:
            row += [fv[0].float(), env.robot.data.body_lin_vel_w[0, stick_id].float(),
                    env.robot.data.body_ang_vel_w[0, stick_id].float()]
        rec.append(torch.cat(row))
        if N > 1:
            rec_envs.append(torch.cat([torch.rad2deg(q[:, :2]).float(), f.view(-1, 1).float()], 1))

    env._apply_action = patched_apply

    env.reset()
    ctrl.reset(env.num_envs, env.device)                 # reset 後にチャネルを初期化し直す
    if is_dr:
        torch.manual_seed(args.seed)                     # 同じ --seed なら、動作が違っても同じ DR の組になる
        ctrl.reset_idx(all_ids)                          # reset() で範囲の中央に戻ったものを抽選し直す
    big = torch.full_like(env.episode_duration_steps, 10 ** 9)
    t_wall = time.time()
    for k in range(n_steps):
        env.episode_duration_steps[:] = big              # BPM 由来のタイムアウトを無効化
        a = torch.as_tensor(cmd[k] / Pmax * 2.0 - 1.0, device=env.device).view(1, 3)
        env.step(a)
        if k in (0, 1) or (k + 1) % 250 == 0 or k + 1 == n_steps:   # 進み具合（sim 5 s ごと）
            print(f"[replay] {(k + 1) * 0.02:6.1f} / {n_steps * 0.02:.1f} s（経過 {time.time() - t_wall:.1f} s）", flush=True)

    r = torch.stack(rec).cpu().numpy()
    cols = ["P_cmd_DF", "P_cmd_F", "P_cmd_G", "P_out_DF", "P_out_F", "P_out_G",
            "wrist_angle_deg", "grip_angle_deg", "force_N", "wrist_vel_dps", "tau_w",
            "F_DF", "F_F", "F_G", "Fmap_DF", "Fmap_F", "Fmap_G",
            "fric_DF", "fric_F", "fric_G"]
    if args.record_contact:
        cols += ["fx_raw", "fy_raw", "fz_raw", "stick_vx", "stick_vy", "stick_vz", "stick_wx", "stick_wy", "stick_wz"]
    out = pd.DataFrame(r, columns=cols)
    out.insert(0, "time", np.arange(len(out)) * dt_phys)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    dr = None
    if N > 1:
        E = torch.stack(rec_envs).cpu().numpy()                    # (T, N, 3)
        dr = dr_samples(env, ctrl)
        np.savez_compressed(os.path.splitext(args.out)[0] + "_envs.npz", time=np.arange(len(E)) * dt_phys,
                            wrist_angle_deg=E[:, :, 0], grip_angle_deg=E[:, :, 1], force_N=E[:, :, 2],
                            **{f"dr_{k}": np.asarray(v) for k, v in dr.items()})
        dr = {k: dict(min=float(np.min(v)), max=float(np.max(v)), mean=float(np.mean(v))) for k, v in dr.items()}
    meta = dict(signal=os.path.abspath(args.signal), pmodel=args.pmodel, params=prm, no_drum=args.no_drum,
                real_log=args.real_log, dt=dt_phys,
                env_cfg=f"Porcaro2026EnvCfg_{args.env_cfg}" + ("" if is_dr else " (DRなし)"),
                num_envs=N, seed=args.seed if is_dr else None, dr=dr,
                impact_force_scale=float(env.force_scale_sim_to_real), restitution=args.restitution,
                robot_restitution=args.robot_restitution,
                materials=materials, record_contact=args.record_contact,
                ctrl=ctrl_over, joint=joint_over,
                meas_filter=(dict(lpf_ms=args.meas_lpf_ms, play_kpa=args.meas_play_kpa) if args.pmodel == "measured" else None))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[replay] 保存: {args.out}  ({len(out)} 行)")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # 例外のときは app.close() が止まって、エラーの表示が出ないまま固まることがある（2026/10/29）。
        # Isaac が sys.stderr を差し替えていて、そのまま終了すると表示が消えるので、
        # 端末（ファイル記述子 2）へ直接書き、replay_error.log にも残してから、閉じずに終了する
        import traceback
        msg = "\n[replay] エラーで止まりました:\n" + traceback.format_exc()
        os.write(2, msg.encode("utf-8", "replace"))
        with open("replay_error.log", "w", encoding="utf-8") as fh:
            fh.write(msg)
        os._exit(1)
    app.close()
