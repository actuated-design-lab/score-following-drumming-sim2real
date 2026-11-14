import os
"""同定の共通部品：時間波形の残差＋振幅比マップの残差"""
import numpy as np, pandas as pd
import pmeval as E

DT = E.DT
TIME_KEYS = ("grid", "steps1", "steps2", "bigstep", "smallstep", "rev")
MAP_KEYS = ("ampsweep", "sine", "sine2")


def prep_map(D, keys=MAP_KEYS, amin=0.02):
    """正弦区間ごとのロックイン用ベクトルを前計算"""
    out = []
    for k in keys:
        dd = D[k]; seg = dd["seg"]
        for L in pd.unique(seg):
            mm = E.SINE_RE.match(L)
            if not mm:
                continue
            c, a, f = float(mm[1]), float(mm[2]), float(mm[3])
            if a < amin:
                continue
            ii = np.flatnonzero(seg == L); ii = ii[int(len(ii) * 0.3):]
            if len(ii) < int(2 / f / DT):
                continue
            tt = ii * DT; cs = np.cos(2 * np.pi * f * tt); sn = np.sin(2 * np.pi * f * tt)
            def amp(x, cs=cs, sn=sn):
                x = x - x.mean(); return np.hypot(2 * np.mean(x * cs), 2 * np.mean(x * sn))
            au = amp(dd["u"][ii]); real = amp(dd["m"][ii]) / au
            out.append((k, ii, cs, sn, au, real, c, a, f))
    return out


def map_ratios(M, preds):
    r = []
    for k, ii, cs, sn, au, real, c, a, f in M:
        x = preds[k][ii]; x = x - x.mean()
        r.append(np.hypot(2 * np.mean(x * cs), 2 * np.mean(x * sn)) / au)
    return np.array(r)


def make_residual(D, M, sim_of_theta, w_map=1.0, step=8):
    reals = np.array([m[5] for m in M])
    nm = len(M)
    def res(th):
        sim = sim_of_theta(th); rs = []; preds = {}
        for k in TIME_KEYS:
            d = D[k]; p = sim(d["u"], float(np.median(d["m"][:20]))); preds[k] = p
            r = (p - d["m"])[::step] / np.ptp(d["m"]); rs.append(r / np.sqrt(len(r)))
        for k in set(m[0] for m in M):
            d = D[k]; preds[k] = sim(d["u"], float(np.median(d["m"][:20])))
        mr = map_ratios(M, preds)
        rs.append(w_map * (mr - reals) / np.sqrt(nm))
        out = np.concatenate(rs)
        return np.where(np.isfinite(out), out, 10.0)
    return res
