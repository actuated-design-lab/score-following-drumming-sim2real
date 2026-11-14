"""c6_contact_kind.py — §6.4（論文 §VI-E）の確認：sim の FF の再生で「振れないが当たる」区分の打点は、どんな接触で当たったか（CPU、Isaac 不要）

対象：opt2 r0 favorite の指令を公称・DR なしで再生した sim（data/user0/ff/opt2/diag/r0_favorite_<task>/{trace.npz, strikes.csv}、
diag_opt2.sh の出力）の、次の 3 区分（c6_prevclass_sim.py の FAIL_CELLS と同じ）の打点。
  200–320 × 直前 <120、200–320 × 直前 120–200、120–200 × 直前 120–200
区分・直前の区分は c6_prevclass_sim.py と同じ（ff_lib.ioi_bins を trace の onsets に、曲の最初の打撃は外す）。

接触の定義は採点と同じ（analysis/harness/strike_extract.py、ff_score の値）：
  接触の区間 = 力（k_t × z、5 ms ごと）が 1 N 以上のサンプルが続く区間（mask_long_trains と同じ単位）
  連なり     = 離れ（次の区間の始まり − 前の区間の終わり）が 40 ms 未満でつながった区間の列。100 ms を超える連なりは押し付けとして外す
打点ごとの分類（当たった打点 = strikes.csv の success、採点された接触 = その peak_time を含む区間）：
  a        1 回の短い接触：採点された区間が、ほかの区間と連なっていない（前後 40 ms 未満の離れで隣り合う区間がない）
  b_first  短い接触の列（連なり ≤ 100 ms、区間 2 つ以上）の最初の区間が採点された
  b_later  同じ列の 2 つ目以降の区間が採点された
外れた打点は、目標 ± 1.5 T16 の窓の中に、外された長い連なり（> 100 ms）があるか（b_masked）、1 N 以上の区間がないか（none）、それ以外（other）。
あわせて、採点された区間の前後 150 ms に外された長い連なりがあるか（near_masked）と、区間の数・間隔・ピーク力を出す。

  採点は trace から今の規則（ff_score.score_sim）でやり直す（隣の strikes.csv は、違うときに知らせるだけ）。?long = 100 ms を超える連なりの中の区間が採点された（今の規則では起きないはず）
  python scripts/tools/user0/access/c6_contact_kind.py                         # WS（diag_opt2.sh の出力がある所）
  python scripts/tools/user0/access/c6_contact_kind.py --diag <別の diag のディレクトリ> --prefix <接頭辞>
出力：data/user0/eval/access_1113/c6_contact/{targets.csv, summary.md}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
import ff_lib as FL  # noqa: E402
import ff_score  # noqa: E402

TASKS = ["b_holdout_00_bpm90", "b_holdout_01_bpm96", "double_lead2_bpm160",
         "gmd_legacy_gmd_03_high_recut_bpm138", "gmd_legacy_gmd_04_extreme_recut_bpm170"]
BIN = ["<120", "120-200", "200-320", ">=320"]
CELLS = [("200-320", "<120"), ("200-320", "120-200"), ("120-200", "120-200")]
ON_N, REL_N, GAP_S, MAXC_S, NEAR_S = 1.0, 0.5, 0.040, 0.100, 0.150


def segments(t, f):
    """1 N 以上が続く区間（strike_extract.mask_long_trains が連なりを作るときと同じ単位）。[(i0, i1_excl)]"""
    above = f >= ON_N
    segs, i, n = [], 0, len(f)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def trains(t, segs):
    """離れ < 40 ms でつないだ区間の列。[[seg の番号, ...]]"""
    out = []
    for k, (s0, e0) in enumerate(segs):
        if out and t[s0] - t[segs[out[-1][-1]][1] - 1] < GAP_S:
            out[-1].append(k)
        else:
            out.append([k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="data/user0/ff/opt2/diag")
    ap.add_argument("--prefix", default="r0_favorite_")
    ap.add_argument("--out", default="data/user0/eval/access_1113/c6_contact")
    a = ap.parse_args()
    rows = []
    for task in TASKS:
        d = ROOT / a.diag / f"{a.prefix}{task}"
        if not (d / "trace.npz").exists():
            raise SystemExit(f"trace がない：{d}（diag_opt2.sh の出力の場所を --diag で）")
        z = np.load(d / "trace.npz")
        t, f, on = z["time"], z["force"].astype(float), z["onsets"]
        # 今の採点の規則（ff_score.score_sim：contact_event ＋ 連なり 40 ms）で trace から採点し直す（diag_opt2.py の one と同じ切り方）。
        # 隣の strikes.csv が古い規則のとき（11/04 の nom_* など）でも、本文と同じ規則で数えるため
        keep = t <= on[-1] + 1.3
        ts = pd.DataFrame({"time_s": t, "target_force": z["target"][: len(t)], "force_z": f})[keep]
        st = ff_score.score_sim(ts, float(z["bpm"])).sort_values("target_time").reset_index(drop=True)
        if (d / "strikes.csv").exists():
            old = pd.read_csv(d / "strikes.csv")
            old = old[old.env == 0] if "env" in old else old
            if len(old) == len(st) and (old.success.to_numpy() != st.success.to_numpy()).any():
                print(f"[c6ck] {task}：隣の strikes.csv と今の規則の採点で成功が違う打点 {(old.success.to_numpy() != st.success.to_numpy()).sum()}（今の規則を使う）")
        if len(st) != len(on) or np.max(np.abs(st.target_time.to_numpy() - on)) > 0.020:
            raise SystemExit(f"打点の対応がとれない：{task}")
        b = FL.ioi_bins(on)
        segs = segments(t, f)
        tr = trains(t, segs)
        seg_train = {k: j for j, ks in enumerate(tr) for k in ks}
        tr_len = [t[segs[ks[-1]][1] - 1] - t[segs[ks[0]][0]] for ks in tr]
        masked = [j for j, L in enumerate(tr_len) if L > MAXC_S]
        t16 = 15.0 / float(z["bpm"])
        for i in range(1, len(on)):
            cell = (BIN[b[i]], BIN[b[i - 1]])
            if cell not in CELLS:
                continue
            r = st.iloc[i]
            row = dict(task=task, strike_idx=i, bin=cell[0], prev_bin=cell[1], success=bool(r.success),
                       target_s=float(on[i]), peak_s=r.peak_time, err_ms=r.timing_err_ms, peak_N=r.peak_force)
            if r.success and np.isfinite(r.peak_time):
                kp = int(np.argmin(np.abs(t - r.peak_time)))
                k = next((n for n, (s0, e0) in enumerate(segs) if s0 <= kp < e0), None)
                if k is None:
                    row["kind"] = "?"
                else:
                    ks = tr[seg_train[k]]
                    pos = ks.index(k)
                    long = tr_len[seg_train[k]] > MAXC_S
                    row.update(kind="?long" if long else ("a" if len(ks) == 1 else ("b_first" if pos == 0 else "b_later")),
                               train_segments=len(ks), train_ms=tr_len[seg_train[k]] * 1000,
                               seg_ms=(t[segs[k][1] - 1] - t[segs[k][0]]) * 1000 + float(np.median(np.diff(t))) * 1000,   # サンプル数 × 5 ms
                               seg_peaks_N=";".join(f"{f[s0:e0].max():.1f}" for s0, e0 in (segs[n] for n in ks)),
                               seg_spacing_ms=";".join(f"{(t[segs[n + 1][0]] - t[segs[n][0]]) * 1000:.0f}" for n in ks[:-1]),
                               near_masked=any(abs(t[segs[tr[j][0]][0]] - t[kp]) <= NEAR_S or abs(t[segs[tr[j][-1]][1] - 1] - t[kp]) <= NEAR_S
                                               for j in masked))
            else:
                w0, w1 = on[i] - 1.5 * t16, on[i] + 1.5 * t16
                in_w = [n for n, (s0, e0) in enumerate(segs) if t[s0] <= w1 and t[e0 - 1] >= w0]
                if not in_w:
                    row["kind"] = "none"
                elif any(seg_train[n] in masked for n in in_w):
                    row["kind"] = "b_masked"
                else:
                    row["kind"] = "other"
            rows.append(row)
    D = pd.DataFrame(rows)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "targets.csv", index=False)
    K = ["a", "b_first", "b_later", "b_masked", "none", "other", "?", "?long"]
    T = D.groupby(["bin", "prev_bin"]).kind.value_counts().unstack(fill_value=0).reindex(columns=K, fill_value=0)
    T["n"] = D.groupby(["bin", "prev_bin"]).size()
    T["struck"] = D.groupby(["bin", "prev_bin"]).success.sum()
    hit = D[D.success]
    L = ["# sim の FF の再生で「振れないが当たる」打点の接触（c6_contact_kind.py）", "",
         f"sim：`{a.diag}/{a.prefix}<task>`（公称・DR なし、5 タスク）。力 = trace.npz の force（k_t × z、5 ms ごと）。"
         f"接触の区間 = 1 N 以上が続くサンプル、連なり = 離れ {GAP_S * 1000:.0f} ms 未満、{MAXC_S * 1000:.0f} ms 超は押し付けとして採点から外す（採点と同じ）。", "",
         "a = 1 回の短い接触、b_first / b_later = 短い接触の列（連なり ≤ 100 ms、区間 2 つ以上）の最初 / 2 つ目以降が採点された、"
         "b_masked = 外れ・窓の中に外された長い連なり、none = 外れ・1 N 以上の区間なし、other = その他の外れ", "",
         T.reset_index().to_markdown(index=False) if hasattr(T, "to_markdown") else T.to_string(), "",
         f"当たった打点 {len(hit)}：a {int((hit.kind == 'a').sum())}、b_first {int((hit.kind == 'b_first').sum())}、b_later {int((hit.kind == 'b_later').sum())}。"
         f"採点された区間の前後 {NEAR_S * 1000:.0f} ms に外された長い連なりがある打点 {int(hit.near_masked.fillna(False).astype(bool).sum())}",
         f"採点された区間の長さ（中央値）{hit.seg_ms.median():.0f} ms、ピーク力（中央値）{hit.peak_N.median():.1f} N"]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
