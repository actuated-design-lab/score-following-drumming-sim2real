"""pmeval.py — ITV（DF チャネル）の圧力モデルを、同定用・検証用データで共通に評価する

使い方（他のスクリプトから）:
    import pmeval as E
    D = E.load_all(JP)                 # データ読み込み（エコーの指令 u, 実測 m, 区間ラベル seg, 200 Hz）
    r = E.evaluate(D, sim)             # sim(u, p0) -> p を渡すと、各指標を返す

指標
  ・振幅比マップ：正弦区間ごとに、指令の基本波に対する出力の振幅比（ロックイン）。実機とモデルの差
  ・時間波形：NRMSE（実測のレンジで割る）
  ・小ステップ：大きさ別の t50（50% に届く時間）
  ・2段ステップ：2段目の 250 ms 後の到達割合
"""
from __future__ import annotations
import os, re, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from repo_paths import find_measured, signal_path  # noqa: E402

DT = 0.005
# 既定はこのリポジトリのルート（analysis/itv/ の2つ上）。別の場所のデータを読むときは JETSON_PROJECT で指定
JP_DEFAULT = os.environ.get("JETSON_PROJECT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

FILES = {
    # 同定用
    "grid":      ("echo_grid",      "data_echo_grid_1790668422.csv",      "fit"),
    "steps1":    ("echo_steps",     "data_echo_steps_1790667361.csv",     "fit"),
    "steps2":    ("echo_steps",     "data_echo_steps_1790666997.csv",     "fit"),
    "bigstep":   ("echo_bigstep",   "data_echo_bigstep_1790749923.csv",   "fit"),
    "ampsweep":  ("echo_ampsweep",  "data_echo_ampsweep_1790754857.csv",  "fit"),
    "sine":      ("echo_sine",      "data_echo_sine_1790749662.csv",      "fit"),
    "sine2":     ("pm_sine2",       "data_pm_sine2_1791266975.csv",       "fit"),
    "smallstep": ("pm_smallstep",   "data_pm_smallstep_1791263970.csv",   "fit"),
    "rev":       ("pm_rev",         "data_pm_rev_1791269781.csv",         "fit"),
    # 検証用（同定に使わない）
    "rand":      ("pm_rand",        "data_pm_rand_1791267082.csv",        "val"),
    "ramp":      ("pm_ramp",        "data_pm_ramp_1791266789.csv",        "val"),
    "stairs":    ("pm_stairs",      "data_pm_stairs_1791267251.csv",      "val"),
    "tmC":       ("tm_C_sine",      "data_tm_C_sine_1791183590.csv",      "val"),
    "tmD":       ("tm_D_antagonist", "data_tm_D_antagonist_1791183665.csv", "val"),
    "tmE2":      ("tm_E_dbl160_seed2", "data_tm_E_dbl160_seed2_1791183686.csv", "val"),
    "tmE3":      ("tm_E_dbl160_seed3", "data_tm_E_dbl160_seed3_1791183707.csv", "val"),
}


def _echo_lag(flag, cmd50):
    n = len(flag)
    cmd = cmd50[np.clip((np.arange(n) * DT / 0.02).astype(int), 0, len(cmd50) - 1)]
    best = (1e9, 0)
    for lag in range(-100, 400):
        a, b = (cmd[: n - lag], flag[lag:]) if lag >= 0 else (cmd[-lag:], flag[: n + lag])
        m = min(len(a), len(b))
        if m < 200:
            continue
        e = float(np.mean((a[:m] - b[:m]) ** 2))
        if e < best[0]:
            best = (e, lag)
    return best[1]


def load(JP, key):
    sig_name, data_name, kind = FILES[key]
    d = pd.read_csv(find_measured(data_name, JP))
    ann = signal_path(f"{sig_name}_annotated", JP)
    sig = pd.read_csv(ann) if os.path.exists(ann) else pd.read_csv(signal_path(sig_name, JP))
    fl = np.nan_to_num(d.flag.values.astype(float))
    bad = np.abs(fl) > 1
    if bad.any():
        fl[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), fl[~bad])
    m = d.meas_pres_F.values.astype(float)          # 物理的な DF（列が入れ替わっている）
    seg = None
    if "segment" in sig:
        lag = _echo_lag(fl, sig.cmd_pressure_DF.values)
        ix = np.clip(((np.arange(len(d)) - lag) * DT / 0.02).astype(int), 0, len(sig) - 1)
        seg = sig.segment.values[ix].astype(str)
    i0 = int(np.argmax(fl > 0.02))
    return dict(key=key, kind=kind, u=fl[i0:].copy(), m=m[i0:].copy(), seg=None if seg is None else seg[i0:])


def load_all(JP=JP_DEFAULT):
    return {k: load(JP, k) for k in FILES}


# ------------------------------------------------------------------ 指標
SINE_RE = re.compile(r"(?:amp|sine)_c([\d.]+)_a([\d.]+)_f([\d.]+)")


def lock(x, tt, f):
    c = np.cos(2 * np.pi * f * tt); s = np.sin(2 * np.pi * f * tt); x = x - x.mean()
    return np.hypot(2 * np.mean(x * c), 2 * np.mean(x * s))


def sine_table(D, preds, keys=("ampsweep", "sine", "sine2", "tmC")):
    """正弦区間ごとの振幅比。preds[key] = モデル出力"""
    rows = []
    for k in keys:
        dd = D[k]; seg = dd["seg"]
        for L in pd.unique(seg):
            mm = SINE_RE.match(L)
            if not mm:
                continue
            c, a, f = float(mm[1]), float(mm[2]), float(mm[3])
            ii = np.flatnonzero(seg == L)
            ii = ii[int(len(ii) * 0.3):]            # 立ち上がりを捨てる
            if len(ii) < int(2 / f / DT):
                continue
            tt = ii * DT; au = lock(dd["u"][ii], tt, f)
            if au < 1e-3:
                continue
            rows.append(dict(data=k, seg=L, c=c, a=a, f=f, real=lock(dd["m"][ii], tt, f) / au,
                             model=lock(preds[k][ii], tt, f) / au))
    T = pd.DataFrame(rows)
    return T.groupby(["data", "c", "a", "f"], as_index=False)[["real", "model"]].mean()


def nrmse(p, m):
    return float(np.sqrt(np.mean((p - m) ** 2)) / np.ptp(m) * 100)


def smallstep_t50(u, x, seg):
    """step_up/step_dn 区間ごとに、50% に届くまでの時間 [ms]"""
    xs = np.convolve(x, np.ones(5) / 5, mode="same"); out = {}
    for L in pd.unique(seg):
        mm = re.match(r"step_(up|dn)_c([\d.]+)_d([\d.]+)_r\d", L)
        if not mm:
            continue
        ii = np.flatnonzero(seg == L)
        w = u[ii[0] - 20: ii[0] + 20]; k = np.flatnonzero(np.abs(np.diff(w)) > 0.002)
        on = ii[0] - 20 + k[-1] + 1 if len(k) else ii[0]
        d = float(mm[3]) * (1 if mm[1] == "up" else -1)
        p0 = np.median(xs[on - 40:on]); seg_ = xs[on:on + 200]
        cr = np.flatnonzero((seg_ - p0) * np.sign(d) >= 0.5 * abs(d))
        out.setdefault((mm[1], abs(d)), []).append(cr[0] * DT * 1000 if len(cr) else np.nan)
    return {k: np.nanmedian(v) for k, v in out.items()}


def rev_frac(u, x, seg, tau=0.25):
    xs = np.convolve(x, np.ones(5) / 5, mode="same"); rows = []
    for L in pd.unique(seg):
        mm = re.match(r"(rev|cont)_(up|dn)_c([\d.]+)_d([\d.]+)_h([\d.]+)_r\d_s2$", L)
        if not mm:
            continue
        ii = np.flatnonzero(seg == L)
        w = u[ii[0] - 20: ii[0] + 20]; k = np.flatnonzero(np.abs(np.diff(w)) > 0.003)
        on = ii[0] - 20 + k[-1] + 1 if len(k) else ii[0]
        dd = float(mm[4]); first = 1 if mm[2] == "up" else -1; second = -first if mm[1] == "rev" else first
        n = int(tau / DT)
        rows.append(dict(kind=mm[1], d=dd * 1000, h=float(mm[5]) * 1000, frac=(xs[on + n] - xs[on]) / (second * dd)))
    return pd.DataFrame(rows)


def evaluate(D, sim, verbose=False):
    """sim(u, p0) -> p。全データで回して指標をまとめる"""
    preds = {k: sim(d["u"], float(np.median(d["m"][:20]))) for k, d in D.items()}
    res = {"nrmse": {k: nrmse(preds[k], D[k]["m"]) for k in D}}
    T = sine_table(D, preds)
    T = T[T.a >= 0.02]
    res["map_fit"] = float(np.mean(np.abs(T[T.data != "tmC"].model - T[T.data != "tmC"].real)))
    res["map_val"] = float(np.mean(np.abs(T[T.data == "tmC"].model - T[T.data == "tmC"].real)))
    res["map"] = T
    s = D["smallstep"]
    res["t50_real"] = smallstep_t50(s["u"], s["m"], s["seg"]); res["t50_model"] = smallstep_t50(s["u"], preds["smallstep"], s["seg"])
    r = D["rev"]
    a = rev_frac(r["u"], r["m"], r["seg"]); b = rev_frac(r["u"], preds["rev"], r["seg"])
    a["model"] = b["frac"].values; res["rev"] = a
    res["preds"] = preds
    return res


def summary(res, name):
    n = res["nrmse"]
    fit = np.mean([n[k] for k in ("grid", "steps1", "steps2", "bigstep", "smallstep", "rev")])
    val = {k: n[k] for k in ("rand", "ramp", "stairs", "tmC", "tmD", "tmE2", "tmE3")}
    R = res["rev"]; R = R.assign(diff=R.model - R.frac)
    small = R[(R.d <= 20) & (R.h <= 200)]["diff"].mean()
    return dict(model=name, 同定ステップ平均=round(fit, 2), map同定=round(res["map_fit"], 3), map検証tmC=round(res["map_val"], 3),
                **{k: round(v, 2) for k, v in val.items()}, 反転小_モデル差=round(float(small), 2))
