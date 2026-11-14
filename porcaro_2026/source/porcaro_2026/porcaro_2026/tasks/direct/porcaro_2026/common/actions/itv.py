"""itv.py — 電空レギュレータ（ITV）の圧力チャネル「shaped」（Gen2 の学習用 sim で使う）

JFPS 2026 で作り直した圧力モデルを、学習用 sim 本体に入れたもの。再生専用の
scripts/tools/user0/jfps2026/itv_models.py の ShapedOrificeChannel と同じ式で、違いは次の 2 点だけ:
  ・時定数 τ とむだ時間 L に、環境ごとの倍率（ドメインランダム化）を掛けられる。reset_idx のたびに抽選する
  ・むだ時間を環境ごとに変えられる遅延バッファにした（倍率がすべて 1 なら itv_models.py と同じ出力）

モデル（指令 u → 圧力 p、物理ステップ dt = 5 ms ごと）:
  1) 指令の中心 c（一次遅れ tc）からのずれ dev = u − c について、振幅 A と周波数 f をその場で推定し、
     実測の振幅比マップから作った表 G(f, A) で dev を縮める：u' = c + G·dev（小さく速い振動が通らない ITV の特性）
  2) u' をむだ時間 L だけ遅らせる
  3) 一次遅れ τ に、給気・排気のオリフィス流量の上限をかけて p を進める

パラメータ（DEFAULTS）は 2026/10/8、jetson_project の DF チャネルのエコー付きデータで同定（通信の遅れを含まない）。
通信の遅れは TorqueActionController の cmd_delay_s で別に入れる。
"""
from __future__ import annotations

import math

import torch

PA = 0.1013  # 大気圧 [MPa]

DEFAULTS = {
    "orifice": dict(tau=0.08330, L=0.045, c_in=4.8542, c_out=9.9620, ps=0.5776, b=0.1323),
    "shape": dict(tc=0.6, W=0.6, hk=0.2, h0=0.003),
}

# G(f, A) = 実機の振幅比 / 流量上限モデルの振幅比（jetson_project analysis/itv/model_A.py）
# 行 = 周波数 [Hz]（0 は「振動していない」= 1）、列 = 振幅 [MPa]
TABLE_F = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
TABLE_A = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.3]
TABLE_G = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.37, 1.284, 1.248, 1.243, 1.213, 1.131, 1.114, 1.074, 1.035, 1.0],
    [1.305, 1.43, 1.434, 1.395, 1.356, 1.281, 1.216, 1.152, 1.076, 1.0],
    [0.678, 1.25, 1.229, 1.2, 1.208, 1.195, 1.13, 1.119, 1.109, 1.0],
    [0.736, 0.263, 0.743, 0.876, 1.009, 1.1, 1.082, 1.063, 1.032, 1.0],
    [0.47, 0.47, 0.47, 0.47, 0.612, 0.754, 0.896, 0.966, 1.037, 1.0],
    [0.366, 0.183, 0.146, 0.177, 0.168, 0.241, 0.568, 0.585, 0.925, 1.0],
    [0.3, 0.331, 0.168, 0.167, 0.166, 0.096, 0.193, 0.291, 0.645, 1.0],
    [0.263, 0.333, 0.132, 0.134, 0.167, 0.118, 0.089, 0.234, 0.379, 1.0],
    [0.156, 0.156, 0.156, 0.156, 0.141, 0.126, 0.111, 0.121, 0.13, 1.0],
]


def _g(pu, pd, b):
    """ISO 6358 型の流量関数（絶対圧）"""
    r = torch.clamp(pd / pu, max=1.0)
    x = torch.clamp((r - b) / (1 - b), min=0.0)
    return torch.where(r <= b, pu, pu * torch.sqrt(torch.clamp(1 - x * x, min=0.0)))


def _orifice_limit(q, p, c_in, c_out, ps, b):
    pab = p + PA
    lim_in = c_in * _g(torch.full_like(pab, 1.0) * (ps + PA), pab, b)
    lim_in = torch.where(ps + PA > pab, lim_in, torch.zeros_like(lim_in))
    lim_out = c_out * _g(pab, torch.full_like(pab, PA), b)
    lim_out = torch.where(pab > PA, lim_out, torch.zeros_like(lim_out))
    return torch.where(q > 0, torch.minimum(q, lim_in), torch.maximum(q, -lim_out))


class EnvDelay:
    """環境ごとにむだ時間を変えられる遅延バッファ（線形補間で端数も扱う）。

    x: (n_envs,) または (n_envs, C)、L: x と同じ形 [s]。reset 後の最初の step で、バッファをその値で埋める
    （開始時の段差を作らない。itv_models.py の _DelayBuf と同じ）。reset_idx した環境は 0 で埋める。
    """

    def __init__(self, dt: float, L_max: float):
        self.dt = float(dt)
        self.K = int(math.ceil(L_max / self.dt)) + 3
        self.buf = None
        self.wp = 0
        self._primed = False

    def reset(self, shape, device):
        self.buf = torch.zeros((self.K, *shape), device=device, dtype=torch.float32)
        self.wp = 0
        self._primed = False

    def reset_idx(self, env_ids):
        if self.buf is not None:
            self.buf[:, env_ids] = 0.0

    def step(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        if self.buf is None:
            self.reset(x.shape, x.device)
        if not self._primed:
            self.buf[:] = x
            self._primed = True
        self.buf[self.wp] = x
        D = torch.clamp(L / self.dt, min=0.0, max=self.K - 2.0)
        M = torch.floor(D).to(torch.int64)
        mu = D - M
        i0 = (self.wp - M) % self.K
        i1 = (self.wp - M - 1) % self.K
        a = torch.gather(self.buf, 0, i0.unsqueeze(0)).squeeze(0)
        b = torch.gather(self.buf, 0, i1.unsqueeze(0)).squeeze(0)
        self.wp = (self.wp + 1) % self.K
        return (1 - mu) * a + mu * b


class GainTable:
    """G(f, A) の双一次補間（f は線形、A は対数）"""

    def __init__(self, fg=TABLE_F, ag=TABLE_A, G=TABLE_G, device="cpu"):
        self.fg = torch.as_tensor(fg, dtype=torch.float32, device=device)
        self.lag = torch.log(torch.as_tensor(ag, dtype=torch.float32, device=device))
        self.G = torch.as_tensor(G, dtype=torch.float32, device=device)

    def to(self, device):
        self.fg, self.lag, self.G = self.fg.to(device), self.lag.to(device), self.G.to(device)
        return self

    def __call__(self, f, A):
        fg, lag, G = self.fg, self.lag, self.G
        fq = torch.clamp(f, fg[0], fg[-1])
        i = torch.clamp(torch.searchsorted(fg, fq, right=True) - 1, 0, len(fg) - 2)
        wf = (fq - fg[i]) / (fg[i + 1] - fg[i])
        la = torch.clamp(torch.log(torch.clamp(A, min=1e-4)), lag[0], lag[-1])
        j = torch.clamp(torch.searchsorted(lag, la, right=True) - 1, 0, len(lag) - 2)
        wa = (la - lag[j]) / (lag[j + 1] - lag[j])
        g0 = G[i, j] * (1 - wa) + G[i, j + 1] * wa
        g1 = G[i + 1, j] * (1 - wa) + G[i + 1, j + 1] * wa
        return g0 * (1 - wf) + g1 * wf


class ShapedOrificeChannel:
    """1 本の筋の圧力チャネル。PAMChannel と同じインタフェース（reset / reset_idx / step）。

    tau_scale_range / L_scale_range：reset_idx のたびに環境ごとに一様に抽選する倍率（(1, 1) で DR なし）。
    reset() の直後は範囲の中央の値。
    """

    def __init__(self, dt, tau=None, L=None, c_in=None, c_out=None, ps=None, b=None, tc=None, W=None, hk=None, h0=None,
                 Pmax=0.6, table: GainTable | None = None,
                 tau_scale_range=(1.0, 1.0), L_scale_range=(1.0, 1.0)):
        o, s = DEFAULTS["orifice"], DEFAULTS["shape"]
        self.dt, self.Pmax = float(dt), float(Pmax)
        self.tau0 = float(o["tau"] if tau is None else tau)
        self.L0 = float(o["L"] if L is None else L)
        self.c_in = float(o["c_in"] if c_in is None else c_in)
        self.c_out = float(o["c_out"] if c_out is None else c_out)
        self.ps = float(o["ps"] if ps is None else ps)
        self.b = float(o["b"] if b is None else b)
        self.tc = float(s["tc"] if tc is None else tc)
        self.W = float(s["W"] if W is None else W)
        self.hk = float(s["hk"] if hk is None else hk)
        self.h0 = float(s["h0"] if h0 is None else h0)
        self.table = table if table is not None else GainTable()
        self.tau_scale_range = tuple(float(v) for v in tau_scale_range)
        self.L_scale_range = tuple(float(v) for v in L_scale_range)
        self.delay = EnvDelay(self.dt, self.L0 * max(self.L_scale_range[1], 1.0) + 0.02)
        self.p = self.c = self.amp = self.zr = self.sgn = None
        self.tau = self.L = None

    def reset(self, n_envs, device):
        self.table.to(device)
        self.delay.reset((n_envs,), device)
        z = lambda: torch.zeros(n_envs, device=device, dtype=torch.float32)
        self.p, self.c, self.amp, self.zr = z(), z(), z(), z()
        self.sgn = torch.ones(n_envs, device=device)
        self.tau = torch.full((n_envs,), self.tau0 * sum(self.tau_scale_range) / 2, device=device)
        self.L = torch.full((n_envs,), self.L0 * sum(self.L_scale_range) / 2, device=device)
        self._primed = False

    def reset_idx(self, env_ids):
        if self.p is None:
            return
        for t in (self.p, self.c, self.amp, self.zr):
            t[env_ids] = 0.0
        self.sgn[env_ids] = 1.0
        self.delay.reset_idx(env_ids)
        n = len(env_ids) if not isinstance(env_ids, slice) else self.p.shape[0]
        for name, base, (lo, hi) in (("tau", self.tau0, self.tau_scale_range), ("L", self.L0, self.L_scale_range)):
            if hi > lo:
                getattr(self, name)[env_ids] = base * (lo + (hi - lo) * torch.rand(n, device=self.p.device))

    def shape(self, u):
        if not self._primed:
            self.c = u.clone()
            self._primed = True
        dt = self.dt
        self.c = self.c + (u - self.c) * dt / self.tc
        dev = u - self.c
        self.amp = self.amp + (dev.abs() - self.amp) * dt / self.W
        A = 1.5708 * self.amp
        h = self.hk * A + self.h0
        down = (self.sgn > 0) & (dev < -h)
        up = (self.sgn < 0) & (dev > h)
        ev = (down | up).float()
        self.sgn = torch.where(down, -torch.ones_like(self.sgn), torch.where(up, torch.ones_like(self.sgn), self.sgn))
        self.zr = self.zr + (ev / dt - self.zr) * dt / self.W
        g = self.table(self.zr / 2.0, A)
        return self.c + g * dev

    @torch.no_grad()
    def step(self, P_cmd):
        u = self.shape(torch.clamp(P_cmd, 0.0, self.Pmax))
        u = self.delay.step(u, self.L)
        q = (u - self.p) / self.tau
        q = _orifice_limit(q, self.p, self.c_in, self.c_out, self.ps, self.b)
        self.p = self.p + q * self.dt
        return self.p
