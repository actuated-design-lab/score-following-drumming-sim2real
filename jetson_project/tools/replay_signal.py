"""replay_signal.py — RL のテイク（deploy_policy のログ）で送った圧力の指令を、そのまま再生できる信号 CSV にする（2026-11-11）

ロードセルのないドラム（動画用）で方策をそのまま回すと、観測 v2 の接触フラグ（ロードセルの力 ≥ 1 N）がずっと 0 になり、
学習のときにない入力になる。そこで、ロードセルの上で回したテイクの指令（20 ms ごとの cmd_DF・cmd_F・cmd_G）を記録どおりに
送り直す（方策なし、run_signal_playback.py）。始まりは deploy_policy と同じ（圧力 0 で 2 s 待ってから 1 つめの指令）。

  python3 tools/replay_signal.py data/user0/okui2026/gen2b_20261111/deploy_okui_both_bpm200_GEN2B-E_trial01_<unixtime>.csv
  → signals/replay/<ログの名前>.csv（run_signal_playback.py replay/<ログの名前> で再生。demo_take.py --replay <ログ> が呼ぶ）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "signals/replay")
DT = 0.02


def build(log: str) -> str:
    """ログ → signals/replay/<名前>.csv。戻り値は run_signal_playback に渡す名前（replay/<名前>）"""
    x = pd.read_csv(log)
    need = {"t_send", "cmd_DF", "cmd_F", "cmd_G"}
    if not need <= set(x.columns):
        sys.exit(f"deploy_policy のログではない（{sorted(need - set(x.columns))} の列がない）：{log}")
    g = x.dropna(subset=["t_send"]).drop_duplicates("t_send").sort_values("t_send")   # 制御の 1 tick = 送信 1 回
    d = np.diff(g.t_send.to_numpy())
    if len(d) and (np.median(d) < 0.015 or np.median(d) > 0.025):
        sys.exit(f"送信の間隔が 20 ms でない（中央値 {np.median(d) * 1000:.1f} ms）：{log}")
    S = pd.DataFrame({"time": np.arange(len(g)) * DT,
                      "cmd_pressure_DF": g.cmd_DF.to_numpy(float),
                      "cmd_pressure_F": g.cmd_F.to_numpy(float),
                      "cmd_pressure_G": g.cmd_G.to_numpy(float)})
    os.makedirs(OUT, exist_ok=True)
    name = os.path.splitext(os.path.basename(log))[0]
    S.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    late = int((d > 0.025).sum())
    print(f"[replay] {len(S)} tick（{len(S) * DT:.2f} s）→ signals/replay/{name}.csv"
          + (f"　※元のテイクで送信が 25 ms 以上あいた tick が {late} 回（再生は 20 ms ちょうどで送る）" if late else ""))
    return f"replay/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="deploy_policy のログ（data/<user>/<venue>/gen2b_<日付>/deploy_*.csv）")
    build(ap.parse_args().log)


if __name__ == "__main__":
    main()
