"""realcore.py — 実機コア（access2026、11/03・11/04）のログを、採点（jetson_project tools/core_score.py）と同じ時刻で読む

- 時刻：deploy（RL）は strike_metrics.apply_time_source(df, "rate_corrected")、playback（FF）は core_score.playback_time。
  採点の strikes.csv の peak_time と同じ時刻になる
- 圧力：11/03 以降のログは pres_mapped = 1（列名＝筋の名前）。それ以外のログは読まない（入れ替わりの扱いが要る）
- 接触イベント：strike_metrics.contact_events（1 N で始まり 0.5 N 未満で終わる、100 ms 以内）。採点と同じ定義
- 単位：圧力 MPa、角度 deg（実機の向き）、力 N

置き場所の前提：porcaro_2026 と jetson_project が同じ親フォルダにある（--jetson で変えられる）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
JETSON = Path(os.environ.get("JETSON_PROJECT", ROOT.parent / "jetson_project"))
SCORE = JETSON / "data/user0/access2026/core2026_score"

COLS = ["cmd_DF", "cmd_F", "cmd_G", "meas_pres_DF", "meas_pres_F", "meas_pres_G",
        "wrist_angle_deg", "grip_angle_deg", "force_N"]
DT = 0.005


def _core_score():
    sys.path.insert(0, str(JETSON / "tools"))
    sys.path.insert(0, str(JETSON / "analysis"))
    import core_score  # noqa: E402
    return core_score


def runs(main_only=True):
    r = pd.read_csv(SCORE / "runs.csv")
    if main_only and "bridge" in r:
        r = r[~r.bridge.astype(bool)]
    return r.reset_index(drop=True)


def strikes():
    return pd.read_csv(SCORE / "strikes.csv")


def load(file, kind_log):
    """採点と同じ時刻の DataFrame（COLS と time、あれば obs_contact）"""
    cs = _core_score()
    df = pd.read_csv(JETSON / file)
    if "pres_mapped" not in df or int(df["pres_mapped"].iloc[0]) != 1:
        raise ValueError(f"pres_mapped = 1 でないログ：{file}")
    if kind_log == "deploy":
        df = cs.apply_time_source(df, "rate_corrected")
    else:
        df = cs.playback_time(df)
    return df


def contact_events(t, f, on_n=1.0):
    """(ピークの添字, 立ち上がり（初めて on_n 以上）の添字)"""
    sys.path.insert(0, str(JETSON / "analysis"))
    from strike_metrics import contact_events as ce
    pk = ce(t, f, on_n)
    on = []
    for k in pk:
        i = k
        while i > 0 and f[i - 1] >= on_n:
            i -= 1
        on.append(i)
    return np.asarray(pk, int), np.asarray(on, int)


def lock(df, t0s, cols, lo=-0.30, hi=0.20, dt=DT):
    """t0s のまわりを dt の格子に補間して (n_event, n_t) の配列の dict と格子を返す"""
    g = np.round(np.arange(lo, hi + dt / 2, dt), 6)
    t = df["time"].to_numpy(float)
    out = {}
    for c in cols:
        y = df[c].to_numpy(float)
        out[c] = np.stack([np.interp(t0 + g, t, y, left=np.nan, right=np.nan) for t0 in t0s]) if len(t0s) else np.zeros((0, len(g)))
    return g, out
