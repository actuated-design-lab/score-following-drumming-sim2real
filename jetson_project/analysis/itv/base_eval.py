import os, sys, json, numpy as np, pandas as pd, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pmeval as E
from numba import njit
DT = 0.005; PA = 0.1013

@njit(cache=False)
def _g(pu, pd_, b):
    r = pd_ / pu
    if r <= b: return pu
    x = (r - b) / (1 - b)
    return pu * np.sqrt(max(1 - x * x, 0.0))

@njit(cache=False)
def sim_O(cmd, p0, tau, L, cin, cout, ps, b):
    D = int(round(L / DT)); p = p0; out = np.empty(len(cmd))
    for i in range(len(cmd)):
        u = cmd[i - D] if i >= D else cmd[0]
        q = (u - p) / tau; pab = p + PA
        if q > 0:
            lim = cin * _g(ps + PA, pab, b) if ps + PA > pab else 0.0; q = min(q, lim)
        else:
            lim = cout * _g(pab, PA, b) if pab > PA else 0.0; q = max(q, -lim)
        p += q * DT; out[i] = p
    return out

if __name__ == "__main__":
    t = time.time(); D = E.load_all(); print("load %.1f s" % (time.time() - t), {k: len(v["u"]) for k, v in D.items()})
    O = np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "struct_vs_data_fits.json")))["step"]["O"])
    t = time.time(); r = E.evaluate(D, lambda u, p0: sim_O(u, p0, *O)); print("eval %.1f s" % (time.time() - t))
    print(pd.DataFrame([E.summary(r, "O（ステップ同定）")]).to_string())
    T = r["map"]; print(T[T.data.isin(["ampsweep", "sine2"]) & (T.c == 0.3)].round(2).to_string())
    print("t50 real", {k: round(v) for k, v in sorted(r["t50_real"].items())}); print("t50 O   ", {k: round(v) for k, v in sorted(r["t50_model"].items())})
