"""itv_models.py — ITV（電空レギュレータ）の小振幅・高周波のしきい値を入れた圧力チャネル（torch）

replay_open_loop.py の LagChannel / OrificeChannel と同じインタフェース:
    ch.reset(n_envs, device); ch.reset_idx(env_ids); p = ch.step(P_cmd)   # 物理ステップ（5 ms）ごと

  ShapedOrificeChannel（モデルA・現象論）
      指令の中心 c（一次遅れ tc）からのずれ dev = u - c について、振幅 A と周波数 f をその場で推定し、
      実測の振幅比マップから作った表 G(f, A)（= 実機 / 流量上限モデル O の振幅比）で dev を縮める。
      u' = c + G·dev を、流量上限モデル O（むだ時間＋一次遅れ＋給気・排気の流量上限）に通す。

  PilotChannel（モデルB・物理寄り）
      e = u(t-L) - p,  π' = ki·e,  x = π + kpp·e - p
      主弁は x が重なり d を超えたときだけ開く：p' = km·(x-d)（給気）, km_dn·(x+d)（排気）
      それぞれ給気・排気のオリフィス流量で頭打ち（O と同じ式）

  PilotChannel(kl>0)（モデルB＋漏れ・物理寄り、make_channels('pilot_leak')）
      重なりの中でも主弁が少し漏れる：|x|<=d で p' = kl·x、外側は p' = ±kl·d + km·(x∓d)
      （B の小振幅での削りすぎを直したもの。重なり d≈24 kPa、漏れ kl≈5.2 /s、kpp→0）

パラメータは DEFAULTS（2026/10/8、jetson_project の DF チャネルのエコー付きデータで同定。
同定・評価のスクリプトは jetson_project/analysis/itv/）。

既知の限界（2026/10/8 時点）
  ・3チャネルとも DF の同定値を使っている。F は排気が DF より遅く、2 Hz・小振幅では F・G とも実機はほぼ素通りだが、
    A は振幅比 0.7、B＋漏れは 0.5 まで削る（3〜4 Hz はおおむね合う）
  ・B＋漏れは、1 s 保持したあとの孤立した小ステップを遅くしすぎる（DF の下げ 10〜20 kPa の t50：実機 60〜65 ms、
    B＋漏れ 160 ms）。A は孤立ステップでは O と同じ（約 100 ms）
パラメータはスカラーでも (n_envs,) のテンソルでもよい（ドメインランダム化用）。
"""
from __future__ import annotations
import math
import torch

PA = 0.1013

DEFAULTS = {
    # O（流量上限モデル）: エコー付きステップで同定（struct_vs_data.py）。同定時のむだ時間は 9 サンプル（45 ms）に丸めていたので、その値
    "orifice": dict(tau=0.08330, L=0.045, c_in=4.8542, c_out=9.9620, ps=0.5776, b=0.1323),
    # A の整形部（model_A.py）
    "shape": dict(tc=0.6, W=0.6, hk=0.2, h0=0.003),
    # B（model_B.py, w_map=0.7）
    "pilot": dict(L=0.0202, ki=11.18, kpp=0.0235, d=0.0156, km=30.99, km_dn=25.52,
                  c_in=3.774, c_out=19.41, ps=0.5882, b=0.0500),
    # B＋漏れ（model_BL.py, w_map=0.7）。c_out はデータからほぼ決まらない（排気側の上限はほとんど効かない）
    "pilot_leak": dict(L=0.02138, ki=10.924, kpp=0.0, d=0.02409, km=38.764, km_dn=41.149, kl=5.2285,
                       c_in=4.1407, c_out=44.465, ps=0.57475, b=0.0500),
}


def _g(pu, pd, b):
    """ISO 6358 型の流量関数（絶対圧）"""
    r = torch.clamp(pd / pu, max=1.0)
    x = torch.clamp((r - b) / (1 - b), min=0.0)
    return torch.where(r <= b, pu, pu * torch.sqrt(torch.clamp(1 - x * x, min=0.0)))


class _DelayBuf:
    def __init__(self, dt, L):
        self.dt, self.L = dt, float(L)
        D = self.L / dt; self.M = int(math.floor(D)); self.mu = D - self.M; self.K = self.M + 3
        self.buf = None; self.wp = 0; self._primed = False

    def reset(self, shape, device):
        self.buf = torch.zeros((self.K, *shape), device=device); self.wp = 0; self._primed = False

    def reset_idx(self, env_ids):
        if self.buf is not None:
            self.buf[:, env_ids] = 0.0

    def step(self, x):
        if not self._primed:
            self.buf[:] = x; self._primed = True
        self.buf[self.wp] = x
        a = self.buf[(self.wp - self.M) % self.K]; b = self.buf[(self.wp - self.M - 1) % self.K]
        self.wp = (self.wp + 1) % self.K
        return (1 - self.mu) * a + self.mu * b


def _orifice_limit(q, p, c_in, c_out, ps, b):
    pab = p + PA
    lim_in = c_in * _g(torch.full_like(pab, 1.0) * (ps + PA), pab, b)
    lim_in = torch.where(ps + PA > pab, lim_in, torch.zeros_like(lim_in))
    lim_out = c_out * _g(pab, torch.full_like(pab, PA), b)
    lim_out = torch.where(pab > PA, lim_out, torch.zeros_like(lim_out))
    return torch.where(q > 0, torch.minimum(q, lim_in), torch.maximum(q, -lim_out))


# ----------------------------------------------------------------------------- A
class GainTable:
    """G(f, A) の双一次補間（f は線形、A は対数）"""
    def __init__(self, fg, ag, G, device="cpu"):
        self.fg = torch.as_tensor(fg, dtype=torch.float32, device=device)
        self.lag = torch.log(torch.as_tensor(ag, dtype=torch.float32, device=device))
        self.G = torch.as_tensor(G, dtype=torch.float32, device=device)

    def to(self, device):
        self.fg, self.lag, self.G = self.fg.to(device), self.lag.to(device), self.G.to(device); return self

    def __call__(self, f, A):
        fg, lag, G = self.fg, self.lag, self.G
        fq = torch.clamp(f, fg[0], fg[-1]); i = torch.clamp(torch.searchsorted(fg, fq, right=True) - 1, 0, len(fg) - 2)
        wf = (fq - fg[i]) / (fg[i + 1] - fg[i])
        la = torch.clamp(torch.log(torch.clamp(A, min=1e-4)), lag[0], lag[-1])
        j = torch.clamp(torch.searchsorted(lag, la, right=True) - 1, 0, len(lag) - 2)
        wa = (la - lag[j]) / (lag[j + 1] - lag[j])
        g0 = G[i, j] * (1 - wa) + G[i, j + 1] * wa
        g1 = G[i + 1, j] * (1 - wa) + G[i + 1, j + 1] * wa
        return g0 * (1 - wf) + g1 * wf


class ShapedOrificeChannel:
    def __init__(self, dt, table: GainTable, tau, L, c_in, c_out, ps, b, tc=0.6, W=0.6, hk=0.2, h0=0.003, Pmax=0.6):
        self.dt, self.table, self.Pmax = dt, table, Pmax
        self.tau, self.c_in, self.c_out, self.ps, self.b = tau, c_in, c_out, ps, b
        self.tc, self.W, self.hk, self.h0 = tc, W, hk, h0
        self.delay = _DelayBuf(dt, L)
        self.p = self.c = self.amp = self.zr = self.sgn = None

    def reset(self, n_envs, device):
        self.table.to(device); self.delay.reset((n_envs,), device)
        z = lambda: torch.zeros(n_envs, device=device)
        self.p, self.c, self.amp, self.zr = z(), z(), z(), z(); self.sgn = torch.ones(n_envs, device=device)
        self._primed = False

    def reset_idx(self, env_ids):
        if self.p is None:
            return
        for t in (self.p, self.c, self.amp, self.zr):
            t[env_ids] = 0.0
        self.sgn[env_ids] = 1.0; self.delay.reset_idx(env_ids)

    def shape(self, u):
        if not self._primed:
            self.c = u.clone(); self._primed = True
        dt = self.dt
        self.c = self.c + (u - self.c) * dt / self.tc
        dev = u - self.c
        self.amp = self.amp + (dev.abs() - self.amp) * dt / self.W
        A = 1.5708 * self.amp; h = self.hk * A + self.h0
        down = (self.sgn > 0) & (dev < -h); up = (self.sgn < 0) & (dev > h)
        ev = (down | up).float()
        self.sgn = torch.where(down, -torch.ones_like(self.sgn), torch.where(up, torch.ones_like(self.sgn), self.sgn))
        self.zr = self.zr + (ev / dt - self.zr) * dt / self.W
        g = self.table(self.zr / 2.0, A)
        return self.c + g * dev

    def step(self, P_cmd):
        u = self.shape(torch.clamp(P_cmd, 0.0, self.Pmax))
        u = self.delay.step(u)
        q = (u - self.p) / self.tau
        q = _orifice_limit(q, self.p, self.c_in, self.c_out, self.ps, self.b)
        self.p = self.p + q * self.dt
        return self.p


# ----------------------------------------------------------------------------- B
class PilotChannel:
    def __init__(self, dt, L, ki, kpp, d, km, km_dn, c_in, c_out, ps, b, kl=0.0, Pmax=0.6):
        self.dt, self.Pmax, self.kl = dt, Pmax, kl
        self.ki, self.kpp, self.d, self.km, self.km_dn = ki, kpp, d, km, km_dn
        self.c_in, self.c_out, self.ps, self.b = c_in, c_out, ps, b
        self.delay = _DelayBuf(dt, L); self.p = self.pi = None

    def reset(self, n_envs, device):
        self.delay.reset((n_envs,), device)
        self.p = torch.zeros(n_envs, device=device); self.pi = torch.zeros(n_envs, device=device)

    def reset_idx(self, env_ids):
        if self.p is not None:
            self.p[env_ids] = 0.0; self.pi[env_ids] = 0.0; self.delay.reset_idx(env_ids)

    def step(self, P_cmd):
        u = self.delay.step(torch.clamp(P_cmd, 0.0, self.Pmax))
        e = u - self.p
        self.pi = torch.clamp(self.pi + self.ki * e * self.dt, min=0.0)
        self.pi = torch.minimum(self.pi, torch.full_like(self.pi, 1.0) * self.ps)
        x = self.pi + self.kpp * e - self.p
        kl, d = self.kl, self.d
        q = torch.where(x > d, kl * d + self.km * (x - d),
                        torch.where(x < -d, -kl * d + self.km_dn * (x + d), kl * x))
        q = _orifice_limit(q, self.p, self.c_in, self.c_out, self.ps, self.b)
        self.p = self.p + q * self.dt
        return self.p

# A の表 G(f, A) = 実機の振幅比 / O の振幅比（model_A.py が 9/30 振幅スイープ・9/30 正弦・10/6 正弦から作成）
# 行 = 周波数 [Hz]（0 は「振動していない」= 1）、列 = 振幅 [MPa]（0.30 は打撃の切り替え級 = O どおり通す）
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
    [0.156, 0.156, 0.156, 0.156, 0.141, 0.126, 0.111, 0.121, 0.13, 1.0]
]


def default_table(device="cpu"):
    return GainTable(TABLE_F, TABLE_A, TABLE_G, device)


def resolved_params(kind, **over):
    """記録用：既定値に上書きを重ねたパラメータ"""
    base = {**DEFAULTS["orifice"], **DEFAULTS["shape"]} if kind == "shaped" else dict(DEFAULTS[kind])
    return {**base, **over}


def make_channels(kind, dt, table=None, Pmax=0.6, **over):
    """kind = 'shaped'（A）, 'pilot'（B）, 'pilot_leak'（B＋漏れ）。DF/F/G の3チャネルを返す"""
    if kind == "shaped":
        prm = {**DEFAULTS["orifice"], **DEFAULTS["shape"], **over}
        table = table or default_table()
        return tuple(ShapedOrificeChannel(dt, table, Pmax=Pmax, **prm) for _ in range(3))
    if kind in ("pilot", "pilot_leak"):
        prm = {**DEFAULTS[kind], **over}
        return tuple(PilotChannel(dt, Pmax=Pmax, **prm) for _ in range(3))
    raise ValueError(kind)
