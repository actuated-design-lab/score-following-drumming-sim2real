"""select_seeds.py — 事前登録した規則でシードを選ぶ（eval_b.py の検証用の列の結果から。GPU なし）

規則（設計ループの決定ドキュメント 2026-11-01 追補）：
  - B と E それぞれ、3 シードから 1 本（F0 は CMA-ES の restart ごとの解から 1 本、同じ規則）。チェックポイントは 1500 iteration（model_1499.pt）で固定
  - F0 の候補は restart ごとの favorite.json（CMA-ES の最後の分布の平均）＝ F0_r<R>。RL の最後のチェックポイントに当たる（2026-11-03 本人の決定）。
    best.json（学習用の列で最良の候補）は F0best_r<R> として並べるだけで、選択の対象にしない（eval_favorite.sh）
  - 基準：(b) の検証用の列を DR 32 回・共通の引きで回したときの成功率（±30 ms・1 N）
  - 同点なら |F − 20 N| の中央値が小さいほう
  - 全シードの sim の成績を、平均と信頼区間で併記する
成功率は「DR の引きごとの成功率（全列の目標をまとめて）」の平均。信頼区間は引きの間のばらつきから（1.96 × SD / √n）。
ペアの差：同じ引きどうしで、選んだシード − ほかのシードの成功率の差の平均と信頼区間。

  python scripts/tools/user0/eval/select_seeds.py            # data/user0/eval/b_eval/val/ を読む
出力：data/user0/eval/b_eval/val/selection.md、selection.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/user0/eval/b_eval/val")
    ap.add_argument("--methods", nargs="+", default=["B", "E", "F0"],
                    help="選ぶ方法（2026-11-04：押し付けの判定を直した版では F0 だけ選び直し、RL の選択は 11/02 のまま＝ --methods F0）")
    a = ap.parse_args()
    d = ROOT / a.dir
    S = {}
    for f in sorted(d.glob("*/summary.json")):
        s = json.load(open(f))
        st = pd.read_csv(f.parent / "strikes.csv")
        s["_per_dr"] = st.groupby("dr").success.mean()
        # 名前はフォルダ名で取る（eval_favorite.sh が best の評価を F0best_r<R> に名前を変えても、summary.json の中の name は
        # F0_r<R> のまま残り、favorite の評価と同じ名前でぶつかって上書きされていた。2026-11-04 修正）
        s["name"] = f.parent.name
        S[s["name"]] = s
    L = ["# シードの選択（select_seeds.py、検証用の列）", ""]
    if not S:
        raise SystemExit(f"!!! {d} に結果がない")
    any_s = next(iter(S.values()))
    L += [f"検証用の列 {any_s['n_seq']} 本 × DR {any_s['n_dr']} 回（dr_seed {any_s['dr_seed']}）。"
          "成功率 = 引きごとの成功率の平均 ± 95 % CI。|ΔF| = 打った打点の |F − 20 N| の中央値。", ""]
    L += ["| run | 観測 | 成功率 | 未打 | 遅い | 早い | 誤差の平均 [ms] | F 中央値 [N] | |ΔF| 中央値 [N] | 目標の数 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in S.items():
        L.append(f"| {n} | {s['obs']} | {s['success_dr_mean']:.3f} ± {s['success_dr_ci95']:.3f} | {s['no_strike']:.3f} | {s['late']:.3f} | "
                 f"{s['early']:.3f} | {s['err_ms_mean']:+.1f} | {s['F_med']:.1f} | {s['dF_abs_med']:.1f} | {s['n_targets']} |")
    pr = {n: s["press_train_frac"] for n, s in S.items() if "press_train_frac" in s}
    if pr:
        L += ["", "押し付けの診断（60 ms の連なりで 100 ms を超えた接触の時間の割合、env の平均。eval_b の press_train_frac）", "",
              "| run | 押し付け |", "|---|---|"] + [f"| {n} | {v:.3f} |" for n, v in pr.items()]
    sel = {}
    pat = {"B": r"B_b_seed\d+", "E": r"E_b_seed\d+", "F0": r"F0_r\d+"}       # F0 = FF の CMA-ES の restart ごとの解（DR あり）
    for m in a.methods:
        c = {n: s for n, s in S.items() if re.fullmatch(pat[m], n)}
        if not c:
            continue
        best = sorted(c.items(), key=lambda kv: (-round(kv[1]["success_dr_mean"], 12), kv[1]["dF_abs_med"]))[0]
        tie = [n for n, s in c.items() if n != best[0] and abs(s["success_dr_mean"] - best[1]["success_dr_mean"]) < 1e-12]
        sel[m] = dict(run=best[0], ckpt=best[1]["ckpt"], success=best[1]["success_dr_mean"], tie_broken_by_dF=bool(tie))
        L += ["", f"## {m}：**{best[0]}** を選ぶ" + ("（同点を |ΔF| で決めた）" if tie else ""), "",
              "| ほかのシード | 成功率の差（選んだ − ほか、同じ引きどうし） |", "|---|---|"]
        for n, s in c.items():
            if n == best[0]:
                continue
            dd = (best[1]["_per_dr"] - s["_per_dr"]).dropna()
            L.append(f"| {n} | {dd.mean():+.3f} ± {1.96 * dd.std() / np.sqrt(len(dd)):.3f} |")
        if len(c) < 3:
            L.append(f"\n!!! {m} のシードが {len(c)} 本しかない")
    L += ["", "保険（観測 v1）は v2 と DR の引きがペアにならない（起動時の乱数の使い方が違う）ので、差は参考。選択には使わない。"]
    out = d / "selection.md"
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(sel, open(d / "selection.json", "w"), indent=1, ensure_ascii=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
