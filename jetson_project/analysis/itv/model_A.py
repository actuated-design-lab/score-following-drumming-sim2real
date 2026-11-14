"""モデルA：指令の振動成分を、実測の振幅比マップで縮めてから流量上限モデル O に渡す（現象論）

  c   : 指令の中心（一次遅れ、時定数 tc）
  dev = u - c
  A   : 振動の振幅の推定  = π/2 × EMA(|dev|)          （正弦なら平均|dev| = 2A/π）
  f   : 振動の周波数の推定 = EMA(dev のゼロクロス数/s) / 2 （ヒステリシス h で小さな揺れを無視）
  G   : G(A, f) = 実機の振幅比 / O の振幅比（実測マップから作る表を補間）
  u'  = c + G × dev  →  O モデル

どれも1ステップずつの更新なので、Isaac Lab の各環境でテンソル演算としてそのまま回せる。

使い方: python model_A.py
"""
import os, sys, json, time
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pmeval as E, fit_common as F
from base_eval import sim_O, _g

DT = 0.005; PA = 0.1013


def build_table(D, O, keys=("ampsweep", "sine", "sine2")):
    preds = {k: sim_O(D[k]["u"], float(np.median(D[k]["m"][:20])), *O) for k in keys}
    T = E.sine_table(D, preds, keys=keys)
    T = T[T.a >= 0.02].assign(G=lambda t: t.real / t.model)
    P = T.groupby(["f", "a"]).G.mean().unstack("a")
    fg = np.array(sorted(set(P.index) | {0.0}), float); ag = np.array(sorted(P.columns), float)
    P = P.reindex(index=fg)
    P.loc[0.0] = 1.0                                   # 振動していない＝そのまま通す
    P[0.30] = 1.0                                      # 大振幅（打撃の切り替え級）は O どおり通す
    ag = np.array(sorted(P.columns), float); P = P[ag]
    P = P.interpolate(axis=1, limit_direction="both").interpolate(axis=0, limit_direction="both")
    return fg, ag, np.clip(P.values, 0.0, 1.6), T


@njit(cache=False)
def _interp2(fg, ag, G, f, a):
    if f <= fg[0]: i = 0; wf = 0.0
    elif f >= fg[-1]: i = len(fg) - 2; wf = 1.0
    else:
        i = np.searchsorted(fg, f) - 1; wf = (f - fg[i]) / (fg[i + 1] - fg[i])
    la = np.log(max(a, 1e-4)); lag = np.log(ag)
    if la <= lag[0]: j = 0; wa = 0.0
    elif la >= lag[-1]: j = len(ag) - 2; wa = 1.0
    else:
        j = np.searchsorted(lag, la) - 1; wa = (la - lag[j]) / (lag[j + 1] - lag[j])
    g0 = G[i, j] * (1 - wa) + G[i, j + 1] * wa
    g1 = G[i + 1, j] * (1 - wa) + G[i + 1, j + 1] * wa
    return g0 * (1 - wf) + g1 * wf


@njit(cache=False)
def shape_cmd(cmd, tc, W, hk, h0, fg, ag, G):
    """指令を整形して返す（O の前段）"""
    n = len(cmd); out = np.empty(n)
    c = cmd[0]; amp = 0.0; zr = 0.0; sgn = 1.0
    for i in range(n):
        u = cmd[i]
        c += (u - c) * DT / tc
        dev = u - c
        amp += (abs(dev) - amp) * DT / W
        A = 1.5708 * amp
        h = hk * A + h0
        ev = 0.0
        if sgn > 0 and dev < -h:
            sgn = -1.0; ev = 1.0
        elif sgn < 0 and dev > h:
            sgn = 1.0; ev = 1.0
        zr += (ev / DT - zr) * DT / W
        f = zr / 2.0
        g = _interp2(fg, ag, G, f, A)
        out[i] = c + g * dev
    return out


def make_sim(O, tc, W, hk, h0, tab):
    fg, ag, G, _ = tab
    return lambda u, p0: sim_O(shape_cmd(u, tc, W, hk, h0, fg, ag, G), p0, *O)


if __name__ == "__main__":
    D = E.load_all()
    O = np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "struct_vs_data_fits.json")))["step"]["O"])
    tab = build_table(D, O)
    print("G 表（行=周波数 Hz, 列=振幅 MPa）"); print(pd.DataFrame(tab[2], index=tab[0], columns=tab[1]).round(2).to_string())
    M = F.prep_map(D)
    res = None; rows = []
    for tc in (0.15, 0.3, 0.6):
        for W in (0.3, 0.6, 1.0):
            for hk in (0.2, 0.5):
                sim = make_sim(O, tc, W, hk, 0.003, tab)
                r = F.make_residual(D, M, lambda th: sim, w_map=0.7)(None)
                rows.append((float(np.sum(r ** 2) / 2), tc, W, hk))
    rows.sort(); print("上位", rows[:5])
    _, tc, W, hk = rows[0]
    json.dump(dict(tc=tc, W=W, hk=hk, h0=0.003, fg=list(tab[0]), ag=list(tab[1]), G=tab[2].tolist(), O=list(O)),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_A.json"), "w"), indent=1)
    ra = E.evaluate(D, make_sim(O, tc, W, hk, 0.003, tab))
    ro = E.evaluate(D, lambda u, p0: sim_O(u, p0, *O))
    print(pd.DataFrame([E.summary(ro, "O"), E.summary(ra, f"A tc={tc} W={W} hk={hk}")]).to_string())
    print("t50 real", {k: round(v) for k, v in sorted(ra["t50_real"].items())})
    print("t50 A   ", {k: round(v) for k, v in sorted(ra["t50_model"].items())})
