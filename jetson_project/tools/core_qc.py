"""core_qc.py — 実機コアの 1 run のログから、機械的に決まる品質のフラグを出す（run の直後に core_runner.py が呼ぶ）

閾値は過去の正常な実機ログの範囲から決めた（根拠：analysis の集計、PR の本文）。どれも結果（成功率）を見ずに決まる量。
  HARD（除外して取り直す）
    comm_yield      パケットの受信率が 0.99 未満か 1.01 超（以前の実機ログ 427 本で 0.996〜1.004）
    comm_gap        受信の間隔の最大が 100 ms 超（同 427 本の最大 49.9 ms。10/14・10/29 の再生 19 本は最大 12 ms）
    comm_resync     ヘッダの再同期で捨てたバイトが 57 超 ＝ 再同期 2 回以上（1 回の読み飛ばしは最大 57 バイト = パケット長 58 − 1。
                    同 427 本で最大 26 = 1 回、14 本）。2026-11-04 に 26 → 57 に直した（26 では 1 回でも 27〜57 バイトなら HARD になり、
                    「2 回以上」という意図と合っていなかった。FF の再生のログで、同期の直後の 1 回（35・56 バイト）が HARD になって気づいた）。
                    FF の再生は同期の直後の読み飛ばしを bytes_discarded_at_sync に別に記録する（数えない）
    lc_offset       ロードセルの休み（最初の打の前、0.5〜1.4 s）の中央値が ±2 N の外（正常 −0.42〜+0.06 N。
                    10/29 の grip_release は −18 N に張り付いていた ＝ アンプが入っていない状態）
    lc_frozen       休みの SD が 0.05 N 未満（値が凍っている。正常 0.12〜0.17 N。上側は、最初の打の前に打面に触れる run があるので見ない）
    lc_stuck        同じ値が 20 サンプル（100 ms）以上続く（正常は最大 3）
  SUSPECT（直後のチェック動作で確かめる。これだけでは除外しない）
    no_contact_long 目標があるのに 1 N 以上の接触がない区間が 8 s 以上（以前の実機ログで E は最大 6.0 s、B は 99 % 点 8.2 s）
    miss_streak     連続して打たなかった目標が 10 以上（E は最大 7、B は 99 % 点 22。B は長い外れが稀にある）
  壊れた行（再同期の直後の 1 行など）は数えるだけで、採点からは落とす

  python3 tools/core_qc.py <ログ.csv> [--midi <元の MIDI>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = dict(yield_lo=0.99, yield_hi=1.01, gap_ms=100.0, resync_bytes=57, lc_off=2.0, lc_sd_lo=0.05, lc_sd_hi=0.5, lc_stuck=20,
          no_contact_s=8.0, miss_streak=10, contact_n=1.0)
PHYS = dict(force_N=1000, wrist_angle_deg=300, grip_angle_deg=300, meas_pres_DF=2, meas_pres_F=2, meas_pres_G=2)


def qc(csv, onsets=None):
    df = pd.read_csv(csv)
    meta = {}
    js = os.path.splitext(csv)[0] + ".json"
    if os.path.exists(js):
        meta = json.load(open(js, encoding="utf-8"))
    bad = np.zeros(len(df), bool)
    for c, lim in PHYS.items():
        if c in df:
            bad |= ~(df[c].abs() < lim)
    out = dict(file=os.path.relpath(csv, ROOT), n_rows=len(df), n_corrupt=int(bad.sum()))
    d = df[~bad]
    tcol = "t_recv" if "t_recv" in d else ("timestamp_pc" if "timestamp_pc" in d else None)
    if tcol:
        tr = d[tcol].to_numpy(float)
        span = tr[-1] - tr[0]
        out["comm_yield"] = float(meta.get("packet_yield", len(df) / (span * 200 + 1)))
        out["comm_gap_ms"] = float(np.diff(tr).max() * 1000)
    out["comm_resync_bytes"] = int(meta.get("bytes_discarded_on_resync", 0) or 0)
    f = d.force_N.to_numpy(float)
    t = d.time.to_numpy(float)
    rest = (t >= 0.5) & (t < 1.4)
    out["lc_rest_med"] = float(np.median(f[rest])) if rest.any() else np.nan
    out["lc_rest_sd"] = float(np.std(f[rest])) if rest.any() else np.nan
    same = np.r_[False, np.diff(f) == 0]
    run = mx = 0
    for s in same:
        run = run + 1 if s else 0
        mx = max(mx, run)
    out["lc_stuck_max"] = int(mx + 1 if mx else 1)
    # 接触：目標の時刻（onsets [s]）があれば、目標ごとの打撃の有無と、目標があるのに接触がない区間
    if onsets is not None and len(onsets):
        on = np.asarray(onsets, float)
        c = f >= TH["contact_n"]
        tc = t[c]
        hit = np.array([np.any((tc >= o - 0.06) & (tc < (on[i + 1] - 0.06 if i + 1 < len(on) else o + 0.5))) for i, o in enumerate(on)])
        run = mx = 0
        for h in hit:
            run = 0 if h else run + 1
            mx = max(mx, run)
        out["miss_streak"] = int(mx)
        ev = np.sort(np.r_[on[0], tc[(tc >= on[0]) & (tc <= on[-1])], on[-1]])
        out["no_contact_max_s"] = float(np.diff(ev).max()) if len(ev) > 1 else np.nan
        out["miss_frac"] = float(1 - hit.mean())
    hard, susp = [], []
    if "comm_yield" in out and not (TH["yield_lo"] <= out["comm_yield"] <= TH["yield_hi"]):
        hard.append(f"comm_yield {out['comm_yield']:.4f}")
    if out.get("comm_gap_ms", 0) > TH["gap_ms"]:
        hard.append(f"comm_gap {out['comm_gap_ms']:.0f} ms")
    if out["comm_resync_bytes"] > TH["resync_bytes"]:
        hard.append(f"comm_resync {out['comm_resync_bytes']} B")
    if not abs(out["lc_rest_med"]) <= TH["lc_off"]:
        hard.append(f"lc_offset {out['lc_rest_med']:+.2f} N")
    if not out["lc_rest_sd"] >= TH["lc_sd_lo"]:
        hard.append(f"lc_frozen SD {out['lc_rest_sd']:.3f} N")
    if out["lc_stuck_max"] >= TH["lc_stuck"]:
        hard.append(f"lc_stuck {out['lc_stuck_max']} サンプル")
    if out.get("no_contact_max_s", 0) >= TH["no_contact_s"]:
        susp.append(f"no_contact_long {out['no_contact_max_s']:.1f} s")
    if out.get("miss_streak", 0) >= TH["miss_streak"]:
        susp.append(f"miss_streak {out['miss_streak']}")
    out["hard"] = "; ".join(hard)
    out["suspect"] = "; ".join(susp)
    out["qc"] = "HARD" if hard else ("SUSPECT" if susp else "OK")
    return out


def onsets_from_midi(path):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import torch
    from midi_rhythm_generator import MidiRhythmGenerator
    g = MidiRhythmGenerator(path, device=torch.device("cpu"), dt=0.02, target_force=20.0, lookahead_steps=1)
    b = g.trajectory_buffer.cpu().numpy()
    hi = b >= 0.75 * b.max()
    return np.nonzero(hi & ~np.r_[False, hi[:-1]])[0] * 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--midi", default=None)
    a = ap.parse_args()
    r = qc(a.log, onsets_from_midi(a.midi) if a.midi else None)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
