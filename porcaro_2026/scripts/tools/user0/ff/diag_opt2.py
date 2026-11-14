"""diag_opt2.py — 段2（押し付けの判定を直した版、opt2）の解の診断（CPU）：押し付け型の振る舞いが残っていないか

入力：sim_trace.py --nominal --ff <解> で回した実機コアの 5 タスク（diag_opt2.sh、data/user0/ff/opt2/diag/<候補>_<タスク>/）
比べる基準：11/04 の sim（data/user0/eval/access_1104/sim/nom_{E,B,F0}_<タスク>）＝ RL と旧い F0
出す量（5 タスクの平均、DR なし）
  success          新しい判定（contact_event ＋ 連なり 40 ms）
  success_old      旧採点（連なりなし）
  press_train60    押し付けの割合：60 ms の連なりで 100 ms を超えた接触の時間 ÷ 曲の時間（目的関数の罰と同じ式）
  train40_drop     新しい判定で外れた当たり ÷ 旧採点の当たり（連なりが打撃に化けていた分）
  hit_no_lift      当たりのうち、打撃の前の手首の持ち上げが 2° 未満の割合（診断だけ。採点には使わない。c1b_lift.py と同じ式）

  python scripts/tools/user0/ff/diag_opt2.py
出力：data/user0/ff/opt2/diag/diag.md、diag.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
import ff_score  # noqa: E402
from analysis.harness.strike_extract import contact_events, extract_strikes  # noqa: E402

TASKS = ["b_holdout_00_bpm90", "b_holdout_01_bpm96", "double_lead2_bpm160",
         "gmd_legacy_gmd_03_high_recut_bpm138", "gmd_legacy_gmd_04_extreme_recut_bpm170"]
DIAG = ROOT / "data/user0/ff/opt2/diag"
REF = ROOT / "data/user0/eval/access_1104/sim"
LIFT_DEG = 2.0


def to_md(df):
    """tabulate なしで Markdown の表にする（WS の環境に tabulate がない）"""
    df = df.reset_index()
    f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def one(p: Path):
    z = np.load(p / "trace.npz")
    tt, f, w = z["time"], z["force"], z["q"][:, 0]
    keep = tt <= z["onsets"][-1] + 1.3
    ts = pd.DataFrame({"time_s": tt, "target_force": z["target"][: len(tt)], "force_z": f})[keep]
    bpm = float(z["bpm"])
    new = ff_score.score_sim(ts, bpm)
    old = extract_strikes(ts, bpm=bpm, target_ref=ff_score.TARGET, tol_ms=ff_score.TOL_MS, min_strike_frac=ff_score.MIN_FRAC,
                          method=ff_score.METHOD, release_n=ff_score.RELEASE_N, max_contact_s=ff_score.MAX_CONTACT_S, train_gap_s=0.0)
    # 持ち上げ：当たり（新しい判定）のピークごとに、前の接触イベントのピークから立ち上がりまでの手首角の最大 − 立ち上がりの角
    t, ff = tt[keep], f[keep]
    ww = w[keep]
    pk = contact_events(t, ff, 1.0, ff_score.RELEASE_N, ff_score.MAX_CONTACT_S, 0.0, ff_score.TRAIN_GAP_S)
    lift = {}
    for i, k in enumerate(pk):
        s = k
        while s > 0 and ff[s - 1] >= 1.0:
            s -= 1
        a = pk[i - 1] if i else max(0, s - 60)
        lift[round(float(t[k]), 4)] = float(np.max(ww[a:s + 1]) - ww[s]) if s > a else np.nan
    hp = new[new.success].peak_time.to_numpy(float)
    lv = np.array([lift.get(round(x, 4), np.nan) for x in hp])
    return dict(success=new.success.mean(), success_old=old.success.mean(),
                press_train60=ff_score.excess_contact_frac(ts.time_s.to_numpy(), ts.force_z.to_numpy(), train_gap_s=ff_score.PRESS_TRAIN_GAP_S),
                train40_drop=1 - new.success.sum() / max(1, old.success.sum()),
                hit_no_lift=float(np.mean(lv < LIFT_DEG)) if len(lv) else np.nan, n_hit=int(new.success.sum()))


def main():
    rows = []
    for p in sorted(DIAG.glob("*")):
        if not (p / "trace.npz").exists():
            continue
        task = next((t for t in TASKS if p.name.endswith("_" + t)), None)
        if task is None:
            continue
        rows.append(dict(source="opt2", cand=p.name[: -len(task) - 1], task=task, **one(p)))
    for m in ("E", "B", "F0"):
        for t in TASKS:
            p = REF / f"nom_{m}_{t}"
            if (p / "trace.npz").exists():
                rows.append(dict(source="11/04 の sim", cand={"F0": "旧い F0（F0_r2）"}.get(m, m), task=t, **one(p)))
    have_opt2 = any(r["source"] == "opt2" for r in rows)
    if not have_opt2 and (DIAG / "diag.csv").exists():
        D = pd.read_csv(DIAG / "diag.csv")           # opt2 の trace.npz がない（WS 以外）ときは、保存した diag.csv から表だけ作る
        rows = []
    elif not rows:
        raise SystemExit("診断の sim がない（diag_opt2.sh の前？）")
    else:
        D = pd.DataFrame(rows)
    DIAG.mkdir(parents=True, exist_ok=True)
    if rows:
        D.to_csv(DIAG / "diag.csv", index=False)
    cols = ["success", "success_old", "press_train60", "train40_drop", "hit_no_lift"]
    T = D.groupby(["source", "cand"])[cols].mean()
    T["tasks"] = D.groupby(["source", "cand"]).size()
    L = ["# 段2（opt2）の解の診断（diag_opt2.py、DR なし、実機コアの 5 タスクの平均）", "",
         "success = 新しい判定（連なり 40 ms）、success_old = 旧採点、press_train60 = 60 ms の連なりで 100 ms を超えた接触の時間の割合（目的関数の罰と同じ）、"
         "train40_drop = 新しい判定で外れた当たりの割合、hit_no_lift = 当たりのうち打撃の前の手首の持ち上げ 2° 未満の割合（診断だけ）", "",
         to_md(T), "", "## タスクごと", "", to_md(D.set_index(["source", "cand", "task"])[cols])]
    (DIAG / "diag.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
