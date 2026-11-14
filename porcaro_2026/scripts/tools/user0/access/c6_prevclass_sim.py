"""c6_prevclass_sim.py — §6.4 の FF の失敗機構の sim 確認：区分（bin）× 直前の区分（prev_bin）ごとの振れ幅と成功率を、実機と sim（公称・DR なし）で並べる（CPU、Isaac 不要）

入力:
  data/user0/eval/access_1104/c6/strikes.csv       src=sim の 306 行（task, strike_idx, bin, A, th_top, omega_dn）
  data/user0/eval/access_1104/c6/strikes_real.csv  実機 918 打点（success あり。pkg_1105.py item6 と同じく実機側はこちら）
  data/user0/ff/opt2/diag/r0_favorite_<task>/{trace.npz, strikes.csv}   sim の打点ごとの成功（opt2 r0 favorite と同じ指令）
手順:
  a. sim の打点に成功を付ける（(task, strike_idx) で対応。trace.npz の onsets[strike_idx] と strikes.csv の target_time の差が全打点で 20 ms 以下でなければ止める）
  b. prev_bin：pkg_1105.py item6 と同じ（src・file ごとに strike_idx 順に並べ bin を 1 つずらす。曲の最初の打撃は外す）
  c. src × bin × prev_bin ごとに 打点数・θ_top 中央値・A 中央値・成功率（prev_bin = all の行も）
  d. 実機の値が tab_prevclass と一致するか（200–320 × 直前 ≥320：123 打点、A 20.7°、成功率 0.593）
  e. 判定（下の CRIT。summary.md には集計の前に書く）

  python scripts/tools/user0/access/c6_prevclass_sim.py
出力: data/user0/eval/access_1110/c6_prev/{cells.csv, summary.md}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
C6 = ROOT / "data/user0/eval/access_1104/c6"
DIAG = ROOT / "data/user0/ff/opt2/diag"
OUT = ROOT / "data/user0/eval/access_1110/c6_prev"
BIN = ["<120", "120-200", "200-320", ">=320"]  # pkg_1105.py と同じ
TASKS = ["b_holdout_00_bpm90", "b_holdout_01_bpm96", "double_lead2_bpm160",
         "gmd_legacy_gmd_03_high_recut_bpm138", "gmd_legacy_gmd_04_extreme_recut_bpm170"]
TOL_S, MIN_N = 0.020, 5
FAIL_CELLS = [("200-320", "<120"), ("200-320", "120-200"), ("120-200", "120-200")]
REST_CELL = ("200-320", ">=320")

CRIT = f"""## 判定の基準（集計の前に決めたもの）

- 次の 3 区分で、sim が A ≤ 3° かつ成功率 ≤ 0.3 なら「振れない・失敗する」、A ≤ 3° で成功率 > 0.3 なら「振れないが当たる」（A > 3° ならどちらでもない）
  200–320 × 直前 <120、200–320 × 直前 120–200、120–200 × 直前 120–200
- 200–320 × 直前 ≥320 で sim の A ≥ 10° なら、「休みのあとは振れる」が sim にもある
- 打点数が {MIN_N} 未満の区分は判定に使わない（表には出し、判定の欄にそう書く）
"""


def die(msg):
    print("!!! " + msg)
    sys.exit(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / "summary.md"
    head = ["# §6.4 FF の失敗機構の sim 確認（c6_prevclass_sim.py）", "",
            "sim = 公称・DR なし、opt2 r0 favorite と同じ指令（data/user0/ff/opt2/diag/r0_favorite_<task>）。"
            "実機 = c6/strikes_real.csv（918 打点）。区分 = 打点の間隔（ms）、prev_bin = 同じ曲の 1 つ前の打点の区分。", "", CRIT]
    md_path.write_text("\n".join(head) + "\n", encoding="utf-8")  # 判定の基準を先に書く

    # 入力の確認
    need = [C6 / "strikes.csv", C6 / "strikes_real.csv"] + [DIAG / f"r0_favorite_{t}" / f for t in TASKS for f in ("trace.npz", "strikes.csv")]
    miss = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if miss:
        die(f"入力がない: {miss}")

    c6 = pd.read_csv(C6 / "strikes.csv")
    sim = c6[c6.src == "sim"].drop(columns=["success", "peak_force", "prev_bin"], errors="ignore").copy()
    if len(sim) != 306:
        die(f"c6/strikes.csv の sim 行が {len(sim)}（306 のはず）")

    # a. 成功を付ける
    parts, chk = [], []
    for t in TASKS:
        d = DIAG / f"r0_favorite_{t}"
        st = pd.read_csv(d / "strikes.csv")
        on = np.load(d / "trace.npz")["onsets"]
        if len(st) != len(on) or (st.strike_idx.values != np.arange(len(on))).any():
            die(f"{t}: strikes.csv {len(st)} 行と onsets {len(on)} 個が対応しない")
        dt = np.abs(on[st.strike_idx.values] - st.target_time.values)
        chk.append(dict(task=t, n=len(st), max_dt_ms=1e3 * dt.max()))
        if dt.max() > TOL_S:
            die(f"{t}: onsets と target_time の差が最大 {1e3 * dt.max():.1f} ms（> 20 ms）")
        s = sim[sim.task == t]
        if len(s) != len(st):
            die(f"{t}: c6 の sim 行 {len(s)} と strikes.csv {len(st)} 行の数が違う")
        parts.append(s.merge(st[["strike_idx", "success", "peak_force", "timing_err_ms"]], on="strike_idx", how="left", validate="one_to_one"))
    sim = pd.concat(parts, ignore_index=True)
    if sim.success.isna().any():
        die(f"成功の付かない sim 打点が {int(sim.success.isna().sum())}")
    sim["success"] = sim.success.astype(str).str.lower().map({"true": 1.0, "false": 0.0})

    real = pd.read_csv(C6 / "strikes_real.csv").assign(src="real")
    real["success"] = real.success.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    D = pd.concat([real, sim], ignore_index=True)

    # b. 直前の区分（pkg_1105.py item6 と同じ）
    D["prev_bin"] = D.sort_values(["src", "file", "strike_idx"]).groupby(["src", "file"]).bin.shift(1)

    # c. 集計
    rows = []
    for (src, b), x in D.groupby(["src", "bin"]):
        rows.append(dict(src=src, bin=BIN[b], prev_bin="all", n=len(x), th_top_med=x.th_top.median(), A_med=x.A.median(), success=x.success.mean()))
        for pb, y in x.dropna(subset=["prev_bin"]).groupby("prev_bin"):
            rows.append(dict(src=src, bin=BIN[b], prev_bin=BIN[int(pb)], n=len(y), th_top_med=y.th_top.median(), A_med=y.A.median(),
                             success=y.success.mean()))
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "cells.csv", index=False)

    def cell(src, b, pb):
        r = cells[(cells.src == src) & (cells.bin == b) & (cells.prev_bin == pb)]
        return r.iloc[0] if len(r) else None

    # d. 実機の値の確認
    r = cell("real", *REST_CELL)
    ok_real = r is not None and int(r.n) == 123 and abs(r.A_med - 20.7) < 0.05 and abs(r.success - 0.593) < 0.0005
    real_line = (f"実機 200–320 × 直前 ≥320：{int(r.n)} 打点、A {r.A_med:.1f}°、成功率 {r.success:.3f} → "
                 f"{'tab_prevclass（123、20.7°、0.593）と一致' if ok_real else '!!! tab_prevclass（123、20.7°、0.593）と一致しない'}") if r is not None \
        else "!!! 実機 200–320 × 直前 ≥320 の区分がない"

    # e. 判定
    verdict = []
    for b, pb in FAIL_CELLS:
        s = cell("sim", b, pb)
        rr = cell("real", b, pb)
        rtxt = f"（実機 {int(rr.n)} 打点、A {rr.A_med:.1f}°、成功率 {rr.success:.3f}）" if rr is not None else "（実機なし）"
        if s is None:
            v = "sim に打点なし → 判定に使わない"
        elif s.n < MIN_N:
            v = f"sim {int(s.n)} 打点（< {MIN_N}）→ 判定に使わない"
        else:
            base = f"sim {int(s.n)} 打点、A {s.A_med:.1f}°、成功率 {s.success:.3f} → "
            if s.A_med <= 3 and s.success <= 0.3:
                v = base + "**振れない・失敗する**"
            elif s.A_med <= 3:
                v = base + "**振れないが当たる**"
            else:
                v = base + "A > 3°（振れている。どちらにも当たらない）"
        verdict.append(f"- {b} × 直前 {pb}：{v} {rtxt}")
    s = cell("sim", *REST_CELL)
    if s is None:
        verdict.append("- 200–320 × 直前 ≥320：sim に打点なし → 判定に使わない")
    elif s.n < MIN_N:
        verdict.append(f"- 200–320 × 直前 ≥320：sim {int(s.n)} 打点（< {MIN_N}）→ 判定に使わない")
    else:
        verdict.append(f"- 200–320 × 直前 ≥320：sim {int(s.n)} 打点、A {s.A_med:.1f}°、成功率 {s.success:.3f} → "
                       + ("**「休みのあとは振れる」が sim にもある**（A ≥ 10°）" if s.A_med >= 10 else "A < 10°：sim には「休みのあとは振れる」が見えない"))

    # 表
    def fmt(x):
        return "—" if x is None else f"{int(x.n)} | {x.th_top_med:.1f} | {x.A_med:.1f} | {x.success:.3f}"
    tab = ["| 区分 | 直前 | 実機 打点 | 実機 θ_top | 実機 A | 実機 成功率 | sim 打点 | sim θ_top | sim A | sim 成功率 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for b in BIN:
        for pb in ["all"] + BIN:
            rr, ss = cell("real", b, pb), cell("sim", b, pb)
            if rr is None and ss is None:
                continue
            tab.append(f"| {b} | {pb} | {fmt(rr)} | {fmt(ss)} |")

    md = ["## 対応の確認（a）", "",
          "| task | 打点 | onsets と target_time の差の最大（ms） |", "|---|---|---|",
          *[f"| {c['task']} | {c['n']} | {c['max_dt_ms']:.2f} |" for c in chk], "",
          f"sim の成功は r0_favorite_<task>/strikes.csv の success（306 打点、全体の成功率 {sim.success.mean():.3f}）。", "",
          "## 実機の値の確認（d）", "", real_line, "",
          "## 判定（e）", "", *verdict, "",
          "## 区分 × 直前の区分（c）", "",
          "θ_top・A は中央値（°）。直前 = all は直前で分けない行（曲の最初の打撃も含む）。直前で分けた行は曲の最初の打撃を外す。", "",
          *tab, "", f"CSV: {(OUT / 'cells.csv').relative_to(ROOT)}"]
    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(md_path.read_text(encoding="utf-8"))
    if not ok_real:
        sys.exit(2)


if __name__ == "__main__":
    main()
