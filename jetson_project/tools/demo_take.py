"""demo_take.py — 動画用の実演テイク（採点対象外）。チェック動作 → 元圧の確認 → 録画開始の合図 → 本番 → その場で採点して表示

実機コアのセッション（core2026_session.csv）にも採点（core2026_score）にも混ざらないように：
  - venue は既定で demo2026（ログは data/<user>/demo2026/ に入る。core_score は access2026 だけを見る）
  - 記録は demo2026_takes.csv に書く（core2026_session.csv には書かない）
チェック動作の判定は実機コアと同じ（core_check.judge、基準は core2026_check_baseline.json = その日の基準）

  python3 tools/demo_take.py --list
  python3 tools/demo_take.py E_double160 --user user0
  python3 tools/demo_take.py F3_holdout0 --user user0 --note "正面・寄り"
  python3 tools/demo_take.py E_holdout0 --user user0 --note "全体"
  python3 tools/demo_take.py E_double160 --user user0 --no_check     # チェック動作を飛ばす（直前のテイクから時間がたっていないとき）

奥井先生の Sing Sing Sing（動画からの採譜、♩ = 200、2026-11-11。songs/sss2026/OKUI.md）：
  ロードセルの上（採点する）：  python3 tools/demo_take.py E_okui_both --venue okui2026 --note "両手"
  ロードセルのないドラム（動画）：
    python3 tools/demo_take.py E_okui_both --venue okui2026_video --no_force                       # 方策をそのまま（接触フラグはずっと 0）
    python3 tools/demo_take.py --replay data/user0/okui2026/gen2b_<日付>/deploy_okui_both_<…>.csv --venue okui2026_video --no_force
                                                                                                      # ロードセルの上のテイクの指令をそのまま再生
  --no_force：チェック動作の「打撃の力が 2 N 未満」の止めを出さない・テイクの採点をしない（力がとれないので）
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import core_check  # noqa: E402
from core_runner import run_cmd  # noqa: E402

TAKES = {
    "E_double160": "python3 src/deploy_policy.py --model GEN2B/E --midi songs/core2026/double_lead2_bpm160.mid --trial 1",
    "F3_holdout0": "python3 tools/run_signal_playback.py core2026/F3_b_holdout_00_bpm90",
    "E_holdout0": "python3 src/deploy_policy.py --model GEN2B/E --midi songs/core2026/b_holdout_00_bpm90.mid --trial 1",
    "F0_double160": "python3 tools/run_signal_playback.py core2026/F0_double_lead2_bpm160",
    "B_double160": "python3 src/deploy_policy.py --model GEN2B/B --midi songs/core2026/double_lead2_bpm160.mid --trial 1",
    # 11/05 のやり直しの FF（opt2 r0、signals/core2026v2）と、比べる E（動画の撮り直し用）
    "F0v2_double160": "python3 tools/run_signal_playback.py core2026v2/F0_double_lead2_bpm160",
    "F0v2_gmd138": "python3 tools/run_signal_playback.py core2026v2/F0_gmd_legacy_gmd_03_high_recut_bpm138",
    "E_gmd138": "python3 src/deploy_policy.py --model GEN2B/E --midi songs/core2026/gmd_legacy_gmd_03_high_recut_bpm138.mid --trial 1",
    # 奥井先生の Sing Sing Sing（動画からの採譜、♩ = 200、2026-11-11）。片手 A 32 打・片手 B 19 打・両手を合わせた 41 打
    "E_okui_A": "python3 src/deploy_policy.py --model GEN2B/E --midi songs/sss2026/okui_handA_bpm200.mid --trial 1",
    "E_okui_B": "python3 src/deploy_policy.py --model GEN2B/E --midi songs/sss2026/okui_handB_bpm200.mid --trial 1",
    "E_okui_both": "python3 src/deploy_policy.py --model GEN2B/E --midi songs/sss2026/okui_both_bpm200.mid --trial 1",
}
REC = os.path.join(ROOT, "demo2026_takes.csv")
FIELDS = ["time", "take", "kind", "log", "verdict", "detail", "success", "n", "no_strike", "F_med", "note", "F_iqr"]


def row(**kw):
    if os.path.exists(REC):          # 列を足したとき（11/05 に F_iqr）、古い記録の見出しを新しい列にそろえて書き直す
        with open(REC, encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
            head = old[0].keys() if old else []
        if old and list(head) != FIELDS:
            with open(REC, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                for r in old:
                    w.writerow({k: r.get(k, "") for k in FIELDS})
    new = not os.path.exists(REC)
    with open(REC, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        kw.setdefault("time", dt.datetime.now().isoformat(timespec="seconds"))
        w.writerow({k: kw.get(k, "") for k in FIELDS})


def check(a, take):
    base = core_check.load_base()
    while True:
        lg = run_cmd("python3 tools/run_signal_playback.py core2026/check_motion", a)
        if lg is None:
            v, fails, info = "ERROR", ["ログなし"], []
        else:
            v, fails, info = core_check.judge(core_check.metrics(lg), base)
        print(f"\n=== チェック動作：{v} ===")
        for s in fails:
            print("   ✗ " + s)
        for s in info:
            print("   · " + s)
        row(take=take, kind="check", log=os.path.relpath(lg, ROOT) if lg else "", verdict=v, detail=" | ".join(fails + info))
        # 打撃の力がほとんど出ていない（ロードセルの配線・打面の位置）。判定は G だけなので PASS になるが、テイクが全部「未打」になる
        #（11/04 の SSS の 2・3 本目：チェックの打撃 0.8 N・0.6 N、テイクの力の最大 1.0 N）。止めて確かめる
        if v in ("PASS", "NO_BASELINE") and lg is not None and not a.no_force:
            m = core_check.metrics(lg)
            if max(m.get("peak1_N", 0), m.get("peak2_N", 0)) < 2.0:
                print(f"\n!!! チェック動作の打撃の力が {m.get('peak1_N', float('nan')):.1f} N・{m.get('peak2_N', float('nan')):.1f} N（ふつうは 8 N 以上）。"
                      "ロードセルの配線と打面の位置を見てください")
                if input(">>> 直したら Enter（もう一度チェック）、このまま進むなら c： ").strip() != "c":
                    continue
        if v in ("PASS", "NO_BASELINE"):
            return
        if input(">>> 通りません。直したら Enter（もう一度）、中断は q： ").strip() == "q":
            sys.exit("中断")


def quick_score(lg):
    import core_score
    info = core_score.classify(lg)
    if info is None:
        return {}
    st, _ = core_score.score(lg, info)
    return core_score.summ(st) if st is not None and len(st) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", nargs="?", choices=sorted(TAKES))
    ap.add_argument("--user", default="user0")
    ap.add_argument("--venue", default="demo2026")
    ap.add_argument("--note", default="")
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no_force", action="store_true", help="ロードセルのないドラム：チェックの力の止めを出さない・採点しない")
    ap.add_argument("--replay", default="", help="deploy_policy のログ：その指令をそのまま再生する（方策なし、tools/replay_signal.py）")
    a = ap.parse_args()
    a.dry = False
    if a.replay:
        import replay_signal
        sig = replay_signal.build(a.replay)
        a.take = "replay:" + os.path.basename(a.replay)
        TAKES[a.take] = f"python3 tools/run_signal_playback.py {sig}"
    if a.list or not a.take:
        for k, v in TAKES.items():
            print(f"{k:14s} {v}")
        return
    if a.venue == "access2026":
        sys.exit("実演テイクを access2026 に入れない（採点に混ざる）")
    if not a.no_check:
        check(a, a.take)
    input(">>> 元圧（レギュレータの表示）を確認したら Enter： ")
    input(">>> カメラの録画を始めてから Enter（3 秒後に流します）： ")
    for s in (3, 2, 1):
        print(f"  {s}…", flush=True)
        time.sleep(1)
    lg = run_cmd(TAKES[a.take], a)
    sc = quick_score(lg) if lg and not a.no_force and not a.replay else {}
    row(take=a.take, kind="take", log=os.path.relpath(lg, ROOT) if lg else "", note=a.note, detail="no_force" if a.no_force else "",
        **{k: (f"{sc[k]:.3f}" if k in sc else "") for k in ("success", "no_strike", "F_med", "F_iqr")}, n=sc.get("n", ""))
    print("\n=== テイク：{}  成功率 {}（{} 打点）  未打 {}  F 中央値 {} N・四分位幅 {} N ===".format(
        a.take, *(("{:.3f}".format(sc[k]) if k != "n" else str(sc[k])) if k in sc else "-" for k in ("success", "n", "no_strike", "F_med", "F_iqr"))))
    print(f"ログ：{lg}\n記録：{REC}（core2026_session.csv には書いていない）")


if __name__ == "__main__":
    main()
