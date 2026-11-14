"""access_table2.py — IEEE Access 表 II・§6.1：学習に使った圧力段（モデル A = shaped、sim と同じパラメータ）の検証 NRMSE（2026-11-05）

モデル：porcaro_2026 common/actions/itv.py の ShapedOrificeChannel と同じ式とパラメータ
  流量上限つき一次遅れ（tau 83.30 ms、L 45 ms、c_in 4.8542、c_out 9.9620、p_s 0.5776 MPa、b 0.1323）
  ＋小振幅の補正 G(f, A)（tc 0.6、W 0.6、hk 0.2、h0 0.003、表は fit_A.json = itv.py の TABLE_G）
  注：fit_A.json の L は 42.7 ms。sim の itv.py は 45 ms に丸めているので、ここは 45 ms で回す（--fitL で 42.7 ms）

データ（どれも同定に使っていない）
  (a) 検証用の録音 7 本（pmeval.FILES の "val"）：ランダムな段、ランプ、階段、正弦（tm_C）、拮抗（tm_D）、打撃指令の再生 2 本（tm_E dbl160 s2・s3）。
      DF チャネル、入力はエコーの指令（通信の遅れを含まない）。pmeval.evaluate と同じ NRMSE（RMSE ÷ 実測の範囲）
  (b) 打撃中の実機ログ 105 本（以前の研究の実機デプロイ、ral_20260803。このリポジトリには入っていない）の 3 チャネル。エコーがないので、一定の遅れ δ（0〜80 ms）を全データ共通で 1 つだけ許す
      （table1_models.py と同じ扱い）。当時のログは列名と中身が違う（物理 DF = meas_pres_F、物理 F = meas_pres_DF）

  python3 analysis/itv/access_table2.py
出力：analysis/itv/access_table2.csv（1 行 = データ）、access_table2_ral.csv（105 本 × 3 チャネル）
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pmeval as E  # noqa: E402
from base_eval import sim_O  # noqa: E402
from model_A import shape_cmd  # noqa: E402

A = json.load(open(os.path.join(_HERE, "fit_A.json")))
O = list(A["O"])
if "--fitL" not in sys.argv:
    O[1] = 0.045
fg, ag, G = np.array(A["fg"]), np.array(A["ag"]), np.array(A["G"])


def model(u, p0):
    return sim_O(shape_cmd(np.asarray(u, float), A["tc"], A["W"], A["hk"], A["h0"], fg, ag, G), p0, *O)


def nrmse(p, m):
    return float(np.sqrt(np.mean((p - m) ** 2)) / (np.ptp(m) + 1e-12) * 100)


def shift(p, s):
    return p if s == 0 else np.concatenate([np.full(s, p[0]), p[:-s]])


rows = []
# (a) 検証用の録音
D = E.load_all()
res = E.evaluate(D, model)
LABEL = {"rand": "Random steps", "ramp": "Ramp", "stairs": "Staircase", "tmC": "Sinusoid", "tmD": "Antagonistic motion",
         "tmE2": "Replay of strike commands 1", "tmE3": "Replay of strike commands 2"}
for k, lab in LABEL.items():
    assert E.FILES[k][2] == "val"
    rows.append(dict(data=lab, recording=E.FILES[k][1], channel="DF", n=1, nrmse_pct=res["nrmse"][k], iqr_pct="", delay_ms=0))

# (b) 打撃中の 105 本
ral = []
for f in sorted(glob.glob(os.path.join(E.JP_DEFAULT, "data/user0/ral2026/ral_20260803/*.csv"))):
    if "summary" in f:
        continue
    d = pd.read_csv(f)
    meas = {"DF": d.meas_pres_F.values, "F": d.meas_pres_DF.values, "G": d.meas_pres_G.values}
    for c in ("DF", "F", "G"):
        ral.append((os.path.basename(f), c, d[f"cmd_{c}"].values.astype(float), meas[c].astype(float)))
preds = [(model(u, float(np.median(m[:20]))), m) for _, _, u, m in ral]
errs = [np.mean([np.mean((shift(p, s) - m) ** 2) for p, m in preds]) for s in range(17)]
s = int(np.argmin(errs))
R = pd.DataFrame([dict(file=f, channel=c, nrmse_pct=nrmse(shift(p, s), m)) for (f, c, _, _), (p, m) in zip(ral, preds)])
n_runs = R.file.nunique()
for c in ("DF", "F", "G"):
    v = R[R.channel == c].nrmse_pct
    rows.append(dict(data=f"Strike commands ({n_runs} runs), median", recording="ral_20260803", channel=c, n=len(v),
                     nrmse_pct=float(v.median()), iqr_pct="%.1f–%.1f" % tuple(np.percentile(v, [25, 75])), delay_ms=s * 5))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(_HERE, "access_table2.csv"), index=False, float_format="%.2f")
R.to_csv(os.path.join(_HERE, "access_table2_ral.csv"), index=False, float_format="%.3f")
print(f"モデル A、パラメータ O = {np.round(O, 4).tolist()}")
print(T.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
