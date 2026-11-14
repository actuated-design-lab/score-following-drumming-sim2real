"""core_check.py — 実機コア（core2026）の run の間に流す短いチェック動作と、その判定（腱ワイヤの外れ・センサ・供給圧の目安）

チェック動作（signals/core2026/check_motion.csv、50 Hz、12.5 s。再生の前に 2 s の 0 圧がつく）
  0.0–1.5 s  休み      DF 0.10 / F 0.10 / G 0.30
  1.5–3.0 s  持ち上げ  DF 0.40 / F 0.05 / G 0.60     → 手首が上がる（DF のワイヤ）
  3.0–3.24 s 振り下ろし DF 0.00 / F 0.45 / G 0.60    → 手首が落ちて打面を打つ（F のワイヤ・G・ロードセル）
  3.24–4.5 s 休み
  4.5–6.24 s 2 回目（同じ）、6.24–7.5 s 休み
  7.5–12.5 s G の確認（手首を DF 0.40 で持ち上げたまま、スティックを宙に浮かせて重力をかける）
             7.5–8.0 G 0.60（整える）→ 8.0–9.0 G 0.00 → 9.0–10.0 G 0.60 → 10.0–11.0 G 0.00 → 11.0–12.0 G 0.60 → 12.0–12.5 休み
  G のワイヤはプーリーの溝から脱線して外れることが多い（端はネジ止めで外れない。2026-11-03 本人）。脱線すると、張っている部分が抜けて
  グリップの可動域が広がり、G を加圧しても効かなくなる。そこで G は「伝達の喪失」を次の 3 つで見る：
    grip_resp     G 0.60 と G 0.00 のグリップ角の差（2 往復の平均）。基準の半分未満に縮んだら FAIL（加圧が効かない）
    grip_G0       G 0.00 のときのグリップ角（重力で垂れる量）。基準から ±tol 以上ずれたら FAIL（遊びが増えて垂れ方が変わる）
    grip_strike   振り下ろしの打撃のあいだのグリップ角の振れ（最大 − 最小）。基準より tol 以上大きければ FAIL（打った反動で回る）
  （打撃の値は FF 段 1 の probe で 15〜25 N に入った組：a_up 0.4、a_dn 0.45、h 0.24、g 0.6）

判定：その日の基準（最初に流した N 回の中央値。--make_baseline）と比べる。許容幅は TOL（--tol で上書き）。
  外れ（FAIL）：G の 3 つの量だけ（2026-11-03 13:28 から。手首の角度・打撃の有無は表示だけ）
  情報（INFO）：持ち上げ中の DF の実測圧（供給圧が下がると指令 0.40 に届かなくなる）、休みのロードセルの値

  python3 tools/core_check.py make_signal
  python3 tools/core_check.py analyze <ログ.csv> [--baseline core2026_check_baseline.json]
  python3 tools/core_check.py make_baseline <ログ1.csv> <ログ2.csv> <ログ3.csv>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = os.path.join(ROOT, "signals", "core2026", "check_motion.csv")
BASE = os.path.join(ROOT, "core2026_check_baseline.json")
DT = 0.02
REST, LIFT, STRIKE, G0, G5 = (0.10, 0.10, 0.30), (0.40, 0.05, 0.60), (0.00, 0.45, 0.60), (0.10, 0.10, 0.00), (0.10, 0.10, 0.50)
LG6, LG0 = (0.40, 0.05, 0.60), (0.40, 0.05, 0.00)
PLAN = [(0.0, 1.5, REST), (1.5, 3.0, LIFT), (3.0, 3.24, STRIKE), (3.24, 4.5, REST), (4.5, 6.0, LIFT), (6.0, 6.24, STRIKE),
        (6.24, 7.5, REST), (7.5, 8.0, LG6), (8.0, 9.0, LG0), (9.0, 10.0, LG6), (10.0, 11.0, LG0), (11.0, 12.0, LG6), (12.0, 12.5, REST)]
T_END = 12.5
# 許容幅（2026-11-03 に実機で決め直した。access2026/playback_20261103 の 9 回：正常 6 回・G の脱線 3 回）
#   G の脱線：打撃のあいだのグリップの振れ 正常 3.6 ± 0.4° → 脱線 41〜49°、G の区間の振れ 正常 2.0〜3.1° → 脱線 52〜75°、
#             G 0 のグリップ角 正常 0.0° → 脱線 −0.2〜69°（重い脱線のときだけ動く）
#   G の応答（G 0 と 0.6 の差）は正常で −0.4° しかなく使えない（判定から外した）
#   打撃の時刻は正常 6 回で 190〜330 ms、力は 4〜21 N とばらつくので判定に使わない（表示のみ）。2 回とも打たないときだけ FAIL
#   手首の持ち上げは、その日の最初の 1 回が低い（32.6° 対 2 回目以降 39〜43°）ので幅を 10° にし、基準の前に暖機を 1 回流す
TOL = dict(wrist_rest=4.0, wrist_lift=10.0, wrist_lift_min_rise=15.0, hit_min_n=1.0,
           grip_strike_add=10.0, grip_strike_abs=15.0, grip_range_add=10.0, grip_range_abs=15.0, grip_G0=10.0)


def make_signal():
    t = np.arange(int(round(T_END / DT))) * DT
    cmd = np.zeros((len(t), 3))
    for a, b, v in PLAN:
        cmd[(t >= a - 1e-9) & (t < b - 1e-9)] = v
    os.makedirs(os.path.dirname(SIG), exist_ok=True)
    pd.DataFrame({"time": np.round(t, 4), "cmd_pressure_DF": cmd[:, 0], "cmd_pressure_F": cmd[:, 1], "cmd_pressure_G": cmd[:, 2]}).to_csv(SIG, index=False)
    print(f"→ {SIG}（{t[-1] + DT:.1f} s）")


def seg(df, a, b):
    return df[(df.time >= a) & (df.time < b)]


def metrics(csv):
    df = pd.read_csv(csv)
    df = df[(df.force_N.abs() < 1000) & (df.wrist_angle_deg.abs() < 300)]          # 再同期の壊れた行を落とす
    m = {}
    m["wrist_rest"] = float(seg(df, 1.0, 1.5).wrist_angle_deg.mean())
    m["grip_rest"] = float(seg(df, 1.0, 1.5).grip_angle_deg.mean())
    m["force_rest_med"] = float(seg(df, 0.2, 1.4).force_N.median())
    m["force_rest_sd"] = float(seg(df, 0.2, 1.4).force_N.std())
    for k, t0 in (("1", 1.5), ("2", 4.5)):
        lift = seg(df, t0 + 1.2, t0 + 1.5)
        m[f"wrist_lift{k}"] = float(lift.wrist_angle_deg.mean())
        m[f"pDF_lift{k}"] = float(lift.meas_pres_DF.mean()) if "meas_pres_DF" in df else np.nan
        s = seg(df, t0 + 1.5, t0 + 2.3)
        i = int(s.force_N.to_numpy().argmax()) if len(s) else 0
        m[f"peak{k}_N"] = float(s.force_N.max()) if len(s) else np.nan
        m[f"peak{k}_ms"] = float((s.time.iloc[i] - (t0 + 1.5)) * 1000) if len(s) else np.nan
        g = s.grip_angle_deg
        m[f"grip_strike{k}"] = float(g.max() - g.min()) if len(g) else np.nan
    g0 = [seg(df, a + 0.7, a + 1.0).grip_angle_deg.mean() for a in (8.0, 10.0)]
    g6 = [seg(df, a + 0.7, a + 1.0).grip_angle_deg.mean() for a in (9.0, 11.0)]
    m["grip_G0"] = float(np.mean(g0))
    m["grip_G6"] = float(np.mean(g6))
    m["grip_resp"] = m["grip_G6"] - m["grip_G0"]
    gg = seg(df, 7.5, 12.0).grip_angle_deg
    m["grip_range"] = float(gg.max() - gg.min())
    m["file"] = os.path.relpath(csv, ROOT)
    return m


def judge(m, base, tol=TOL):
    """2026-11-03 の実機（正常 6 回・G の脱線 3 回、data/user0/access2026/playback_20261103/）で決め直した判定。
    FAIL にするのは、正常と外れがはっきり分かれた量だけ。打撃の力と時刻は正常でもばらつくので（下の TOL の注）表示だけにする。"""
    fails, info = [], []
    if base is None:
        return "NO_BASELINE", ["その日の基準がまだない（make_baseline）"], []
    b = base["median"]
    # 欠けたログ（受信エラーなどで NaN）は判定できない → ERROR（PASS にしない。2026-11-03 19:00、no 33 のあとのチェックで受信が切れて NaN のまま PASS になった）
    bad = [k for k in ("grip_strike1", "grip_strike2", "grip_range", "grip_G0", "wrist_lift1", "wrist_lift2", "peak1_N", "peak2_N")
           if k in m and not np.isfinite(m[k])]
    if bad:
        return "ERROR", [f"ログが欠けていて判定できない（NaN：{', '.join(bad)}）→ 受信エラー。もう一度チェック"], []
    # G の脱線：打撃の反動でグリップが大きく回る・G の区間で大きく振れる・G 0 で垂れる
    for k in ("1", "2"):
        if m[f"grip_strike{k}"] > max(b[f"grip_strike{k}"] + tol["grip_strike_add"], tol["grip_strike_abs"]):
            fails.append(f"打撃 {k} のあいだのグリップ角の振れ {m[f'grip_strike{k}']:.1f}°（基準 {b[f'grip_strike{k}']:.1f}°）→ G のワイヤ（脱線）")
    if m["grip_range"] > max(b["grip_range"] + tol["grip_range_add"], tol["grip_range_abs"]):
        fails.append(f"G の区間のグリップ角の振れ {m['grip_range']:.1f}°（基準 {b['grip_range']:.1f}°）→ G のワイヤ（脱線）")
    if abs(m["grip_G0"] - b["grip_G0"]) > tol["grip_G0"]:
        fails.append(f"G 0 のときのグリップ角 {m['grip_G0']:.1f}°（基準 {b['grip_G0']:.1f}° ± {tol['grip_G0']}）→ G のワイヤ（遊びが増えた）")
    # DF・F・打撃の有無は表示だけ（2026-11-03 13:28 本人：問題になるのは G だけ。手首の持ち上げ角は正常でも 40〜51° を行き来し、
    # 角度だけの FAIL で取り直しが続いたため。ロードセルの異常は run の QC の HARD が見る）
    info.append(f"手首：持ち上げ {m['wrist_lift1']:.1f}°・{m['wrist_lift2']:.1f}°（基準 {b['wrist_lift1']:.1f}・{b['wrist_lift2']:.1f}）、"
                f"休み {m['wrist_rest']:.1f}°（基準 {b['wrist_rest']:.1f}）")
    for k in ("1", "2"):
        info.append(f"打撃 {k}：{m[f'peak{k}_N']:.1f} N・{m[f'peak{k}_ms']:.0f} ms（基準 {b[f'peak{k}_N']:.1f} N・{b[f'peak{k}_ms']:.0f} ms）、"
                    f"DF の実測圧 {m[f'pDF_lift{k}']:.3f} MPa（基準 {b[f'pDF_lift{k}']:.3f}）")
    info.append(f"G：打撃の振れ {m['grip_strike1']:.1f}°・{m['grip_strike2']:.1f}°（基準 {b['grip_strike1']:.1f}・{b['grip_strike2']:.1f}）、"
                f"G の区間の振れ {m['grip_range']:.1f}°（基準 {b['grip_range']:.1f}）、G 0 で {m['grip_G0']:.1f}°（基準 {b['grip_G0']:.1f}）")
    info.append(f"休みのロードセル：中央値 {m['force_rest_med']:+.2f} N、SD {m['force_rest_sd']:.2f} N")
    return ("FAIL" if fails else "PASS"), fails, info


def load_base(path=BASE):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make_signal", "analyze", "make_baseline"])
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--baseline", default=BASE)
    a = ap.parse_args()
    if a.cmd == "make_signal":
        return make_signal()
    M = [metrics(f) for f in a.logs]
    if a.cmd == "make_baseline":
        D = pd.DataFrame(M).drop(columns="file")
        base = {"median": D.median().to_dict(), "sd": D.std().to_dict(), "files": [m["file"] for m in M], "n": len(M)}
        json.dump(base, open(a.baseline, "w"), indent=1, ensure_ascii=False)
        print(D.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
        print(f"→ {a.baseline}（{len(M)} 回の中央値）")
        return
    base = load_base(a.baseline)
    for m in M:
        v, fails, info = judge(m, base)
        print(f"[{v}] {m['file']}")
        for s in fails:
            print("   ✗ " + s)
        for s in info:
            print("   · " + s)


if __name__ == "__main__":
    main()
