"""
real_log.py — 実機ログ（jetson_project の run_signal_playback.py の出力）の読み込みと、指令の時間軸への位置合わせ

Isaac Lab に依存しない（numpy / pandas だけ）。replay_open_loop.py の measured モードと
analysis/eval/compare_replay.py が同じ処理を使う。

- 実機ログは DF/F の圧力列が入れ替わっている（2026/9/29 判明）：物理 DF = meas_pres_F、物理 F = meas_pres_DF
- 時刻は flag 列（MicroLabBox が受け取った DF 指令のエコー）を指令 DF に合わせて揃える
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REAL_DT = 0.005   # 実機ログの刻み [s]
CMD_DT = 0.02     # 指令（test_signals/tm_*.csv）の刻み [s]

# 物理 DF / F / G の順に並べた実機ログの圧力列（入れ替わりを戻す）。〜2026/10/26 のログの並び
PRES_COLS = ("meas_pres_F", "meas_pres_DF", "meas_pres_G")
# 2026/10/29 以降、jetson_project はログの列名を筋の名前にそろえて保存し、pres_mapped = 1 の列を付ける
# （config/pressure_slots.yaml）。その列があるログは名前どおりに読む
# 注意：10/26〜10/29 昼（pres_mapped なし）のログは並びが [G, F, DF] で、PRES_COLS では正しく読めない（grip_release の 1 本だけ。角度しか使っていない）
MAPPED_COLS = ("meas_pres_DF", "meas_pres_F", "meas_pres_G")


def pres_columns(d, columns):
    """d が pres_mapped のログなら、columns のうち PRES_COLS の部分を名前どおりの列に置き換える"""
    if "pres_mapped" in d.columns and int(d["pres_mapped"].iloc[0]) == 1:
        rep = dict(zip(PRES_COLS, MAPPED_COLS))
        return tuple(rep.get(c, c) for c in columns)
    return tuple(columns)


def echo_lag(flag, cmd_df_50hz, search=(-100, 400)):
    """実機ログ上で flag（DF 指令のエコー）が指令 DF より何行ずれているか（行数, 1 行 = REAL_DT）"""
    flag = np.nan_to_num(np.asarray(flag, float))
    t_real = np.arange(len(flag)) * REAL_DT
    # 指令DF（50 Hz, ZOH）を 200 Hz に展開
    cmd_df = np.asarray(cmd_df_50hz)[np.clip((t_real / CMD_DT).astype(int), 0, len(cmd_df_50hz) - 1)]
    best = (1e9, 0)
    for lag in range(*search):
        if lag >= 0:
            a = cmd_df[: len(cmd_df) - lag]; b = flag[lag: lag + len(a)]
        else:
            b = flag[: len(flag) + lag]; a = cmd_df[-lag: -lag + len(b)]
        n = min(len(a), len(b))
        if n < 100:
            continue
        e = np.mean((a[:n] - b[:n]) ** 2)
        if e < best[0]:
            best = (e, lag)
    return best[1]


def align_real_log(d, cmd_50hz, dt, columns=PRES_COLS):
    """実機ログ d（DataFrame）の columns を、指令の時間軸（0 から指令の最後まで, dt 刻み）に合わせて返す

    戻り値: (n, len(columns)) の配列, ずれ [行]
    """
    columns = pres_columns(d, columns)
    lag = echo_lag(d["flag"].values, np.asarray(cmd_50hz)[:, 0])
    t_real = np.arange(len(d)) * REAL_DT
    t_cmd = np.arange(len(cmd_50hz)) * CMD_DT
    t_sim = np.arange(int(round(t_cmd[-1] / dt)) + 1) * dt
    out = np.stack([np.interp(t_sim + lag * REAL_DT, t_real, d[c].values.astype(float)) for c in columns], 1)
    return out, lag


def load_real_aligned(path, cmd_50hz, dt):
    """実機ログを、エコー（flag）で指令の時間軸に合わせ、dt 刻みの DF/F/G 圧力列にして返す"""
    out, lag = align_real_log(pd.read_csv(path), cmd_50hz, dt)
    print(f"[real] エコーで合わせたずれ: {lag * 5} ms（flag が指令に一致する位置）")
    return out


# jetson_project の置き場所
#   今  : 入力信号 signals/<name>.csv、実測 data/<user>/<venue>/playback_<日付>/data_<name>_<unixtime>.csv
#   以前: どちらも test_signals/
def jetson_signal_path(jetson, name):
    """入力信号（指令, 50 Hz）のパス。name は tm_C_sine など（.csv なし）"""
    import os
    for d in ("signals", "test_signals"):
        p = os.path.join(jetson, d, f"{name}.csv")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name}.csv が {jetson}/signals/ にも test_signals/ にも無い")


def jetson_real_logs(jetson, name):
    """実測ログ data_<name>_<数字>.csv の一覧（_drum などの別信号は除く）"""
    import glob, os
    pats = [os.path.join(jetson, "data", "*", "*", "playback_*", f"data_{name}_*.csv"),
            os.path.join(jetson, "test_signals", f"data_{name}_*.csv")]
    out = []
    for pat in pats:
        out += [f for f in glob.glob(pat) if os.path.basename(f)[len(f"data_{name}_"):-4].isdigit()]
    return sorted(set(out))


def find_real_log(jetson, name, hint=None):
    """実測ログを1本に決める。hint（json に書かれたパス）があれば、そのファイル名を優先して探す"""
    import os
    logs = jetson_real_logs(jetson, name)
    if hint:
        hit = [f for f in logs if os.path.basename(f) == os.path.basename(hint)]
        if hit:
            return hit[0]
    if len(logs) != 1:
        raise FileNotFoundError(f"{name}: 実測ログが {len(logs)} 本: {logs}")
    return logs[0]
