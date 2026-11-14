"""core_runner.py — 実機コア（core2026_runs.csv の 60 本）を、チェック動作とオペレータのゲートをはさんで 1 本ずつ回す

流れ（1 本ごと）
  1) チェック動作（signals/core2026/check_motion.csv、約 15 s）→ core_check.py で基準と比べる
     FAIL：止めて内容を出し、オペレータに「直した → もう一度チェック / 中断」を聞く（PASS するまで本番に進まない）
  2) 本番の run（台本の command に --user / --venue / --no-input を足して実行）
  3) core_qc.py：HARD（通信・ロードセル）→ その run は除外して取り直す。SUSPECT → 次のチェック動作の結果で決める
     （次のチェックが FAIL なら、その run も「チェック FAIL の直前」として除外して取り直す）
  4) ゲート：タスクのかたまり（同じ反復・同じタスクの連続する run）の終わりごとに止め、直近の結果を表示して
     「元圧を目で確認したか・次へ進むか」を聞く（Enter = 次へ / r = 直前の run を取り直す / q = 中断）
記録：core2026_session.csv（1 行 = 1 run または 1 チェック。no・方法・タスク・反復・ログ・チェックの判定と数値・QC・除外の理由・オペレータの入力）
     core_score.py は excluded = 1 の run を採点から外す。途中で止めても --start <no> で続きから回せる

  python3 tools/core_runner.py --user user0 --venue <場所> --phase rl     # B・E の 30 本（FF の指令がまだなくても回せる）
  python3 tools/core_runner.py --user user0 --venue <場所> --phase ff     # F0・F3 の 30 本（F3 まで取り込んだあと）
  python3 tools/core_runner.py --user user0 --venue <場所> --phase rl --start 12   # 途中から（no は台本の通し番号）
  python3 tools/core_runner.py --user user0 --venue <場所> --baseline_only 3     # その日の基準を作る（暖機 1 回 ＋ チェック動作 3 回）
  （--dry：コマンドを表示するだけ。--no_check：チェック動作を挟まない（試験用））
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import re
import shlex
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import core_check  # noqa: E402
import core_qc  # noqa: E402

SESSION = os.path.join(ROOT, "core2026_session.csv")
FIELDS = ["time", "kind", "no", "rep", "task", "method", "log", "verdict", "detail", "qc", "qc_detail", "excluded", "exclude_reason", "operator"]


def log_row(**kw):
    new = not os.path.exists(SESSION)
    with open(SESSION, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        kw.setdefault("time", dt.datetime.now().isoformat(timespec="seconds"))
        w.writerow({k: kw.get(k, "") for k in FIELDS})


def run_cmd(cmd, a):
    """コマンドを実行して、出力を流しながら、保存されたログの csv のパスを拾う"""
    full = cmd + f" --user {a.user} --venue {a.venue}" + (" --no-input" if "deploy_policy" in cmd else "")
    print(f"\n$ {full}")
    if a.dry:
        return None
    t0 = time.time()
    p = subprocess.Popen(shlex.split(full), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    path = None
    for line in p.stdout:
        print(line, end="")
        m = re.search(r"(\S+\.csv)\s*$", line.strip())
        if m and ("[Log]" in line or "->" in line) and os.path.exists(os.path.join(ROOT, m.group(1)) if not os.path.isabs(m.group(1)) else m.group(1)):
            path = m.group(1)
    p.wait()
    if path is None:      # 出力から拾えなければ、いちばん新しい csv
        c = [f for f in glob.glob(os.path.join(ROOT, "data", a.user, a.venue, "**", "*.csv"), recursive=True) if os.path.getmtime(f) >= t0]
        path = max(c, key=os.path.getmtime) if c else None
    return os.path.join(ROOT, path) if path and not os.path.isabs(path) else path


def check(a, base, no=""):
    """PASS するまでチェック動作を繰り返す。戻り値：(最後の判定, 途中で FAIL があったか)"""
    had_fail = False
    while True:
        lg = run_cmd("python3 tools/run_signal_playback.py core2026/check_motion", a)
        if a.dry:
            return "DRY", False
        if lg is None:
            print("!!! チェック動作のログが見つからない")
            v, fails, info, m = "ERROR", ["ログなし"], [], {}
        else:
            m = core_check.metrics(lg)
            v, fails, info = core_check.judge(m, base)
        print(f"\n=== チェック動作：{v} ===")
        for s in fails:
            print("   ✗ " + s)
        for s in info:
            print("   · " + s)
        log_row(kind="check", no=no, log=os.path.relpath(lg, ROOT) if lg else "", verdict=v,
                detail=" | ".join(fails + info))
        if v in ("PASS", "NO_BASELINE"):
            return v, had_fail
        had_fail = True
        ans = input(">>> チェックが通りません。ワイヤ・センサを見て、直したら Enter（もう一度チェック）、中断は q： ").strip()
        log_row(kind="operator", no=no, operator=f"check_fail:{ans or 'retry'}")
        if ans == "q":
            sys.exit("中断（--start で続きから）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--venue", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--stop", type=int, default=10 ** 9)
    ap.add_argument("--phase", choices=["rl", "ff"], default=None, help="台本の phase 列で絞る（rl = B・E、ff = F0・F3）")
    ap.add_argument("--baseline_only", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1, help="基準の前に流して捨てるチェック動作の回数（その日の最初の 1 回は手首が低く出る）")
    ap.add_argument("--runs", default="core2026_runs.csv", help="台本（2026-11-04 のやり直しの FF は core2026_runs_v2.csv）")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no_check", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(core_check.SIG):
        core_check.make_signal()
    if a.baseline_only:
        for i in range(a.warmup):
            lg = run_cmd("python3 tools/run_signal_playback.py core2026/check_motion", a)
            log_row(kind="warmup_check", log=os.path.relpath(lg, ROOT) if lg else "")
        logs = []
        for i in range(a.baseline_only):
            lg = run_cmd("python3 tools/run_signal_playback.py core2026/check_motion", a)
            if lg:
                logs.append(lg)
                log_row(kind="baseline_check", log=os.path.relpath(lg, ROOT))
        if logs:
            subprocess.run([sys.executable, "tools/core_check.py", "make_baseline", *logs], cwd=ROOT)
        return
    base = core_check.load_base()
    if base is None and not a.no_check:
        sys.exit("その日の基準がない：先に --baseline_only 3")
    runs = pd.read_csv(os.path.join(ROOT, a.runs))
    if a.phase:
        if "phase" not in runs.columns:
            sys.exit("台本に phase 列がない（porcaro_2026 を pull して import_real_core.py をもう一度）")
        runs = runs[runs.phase == a.phase]
    runs = runs[(runs.no >= a.start) & (runs.no <= a.stop)]
    if runs.empty:
        sys.exit("回す run がない（--phase / --start / --stop を確認）")
    last_no = int(runs.no.max())
    if runs.command.str.startswith("（").any():
        sys.exit("台本に未準備の run がある（import_real_core.py をもう一度）：\n" + runs[runs.command.str.startswith("（")].to_string())
    tasks = pd.read_csv(os.path.join(ROOT, "songs/core2026/tasks.csv")).set_index("task")
    last = None
    rows = list(runs.itertuples())
    i = 0
    while i < len(rows):
        r = rows[i]
        if not a.no_check:
            v, had_fail = check(a, base, r.no)
            # 規則：run は前後のチェックが通ったときだけ数える。直後のチェックで FAIL が出たら、直前の run は除外して取り直す
            #（外れが起きた時刻は前後のチェックの間としか言えないため。結果を見て決める余地を残さない）
            if had_fail and last is not None and not last.get("redone"):
                log_row(kind="exclude", no=last["no"], log=last["log"], excluded=1,
                        exclude_reason=f"直後のチェック動作が FAIL（QC {last['qc']}）")
                print(f"\n>>> 直前の run（no {last['no']}）を除外して取り直します")
                last["redone"] = True
                i -= 1
                continue
        print(f"\n######## {r.no}（〜{last_no}）  反復 {r.rep}  {r.task}  {r.method} ########")
        lg = run_cmd(r.command, a)
        q = {"qc": "DRY"} if a.dry else core_qc.qc(lg, core_qc.onsets_from_midi(os.path.join(ROOT, "songs/core2026", tasks.loc[r.task, "file"]))) if lg else {"qc": "HARD", "hard": "ログなし"}
        excl = int(q["qc"] == "HARD")
        log_row(kind="run", no=r.no, rep=r.rep, task=r.task, method=r.method, log=os.path.relpath(lg, ROOT) if lg else "",
                qc=q["qc"], qc_detail=(q.get("hard", "") + " " + q.get("suspect", "")).strip(), excluded=excl,
                exclude_reason="QC HARD" if excl else "")
        print(f"\n=== QC：{q['qc']} {q.get('hard', '')} {q.get('suspect', '')} ===")
        last = dict(no=r.no, log=os.path.relpath(lg, ROOT) if lg else "", qc=q["qc"])
        if excl:
            ans = input(">>> この run は除外（QC HARD）。原因を直したら Enter で取り直し、中断は q： ").strip()
            log_row(kind="operator", no=r.no, operator=f"hard:{ans or 'retry'}")
            if ans == "q":
                sys.exit(f"中断（続きは --start {r.no}）")
            continue                                    # 同じ run をもう一度
        block_end = (i + 1 == len(rows)) or (rows[i + 1].task != r.task) or (rows[i + 1].rep != r.rep)
        if block_end or q["qc"] == "SUSPECT":
            ss = pd.read_csv(SESSION, dtype=str).fillna("")
            blk = ss[ss.kind.isin(["check", "run", "exclude"])].tail(14)
            print("\n---- 直近の記録（チェック動作の判定・run の QC・除外）----")
            for x in blk.itertuples():
                if x.kind == "check":
                    print(f"  チェック {x.verdict:5s} {x.detail[:110]}")
                elif x.kind == "run":
                    print(f"  run {x.no:>3s} {x.method:5s} {x.task:10s} QC {x.qc} {x.qc_detail}")
                else:
                    print(f"  除外 run {x.no}：{x.exclude_reason}")
            ans = input(f"\n>>> ゲート（反復 {r.rep} の {r.task} が終わった{'・SUSPECT あり' if q['qc'] == 'SUSPECT' else ''}）："
                        "元圧を確認して、Enter = 次へ / r = 直前の run を取り直す / q = 中断： ").strip()
            log_row(kind="operator", no=r.no, operator=f"gate:{ans or 'next'}")
            if ans == "q":
                sys.exit(f"中断（続きは --start {rows[i + 1].no if i + 1 < len(rows) else r.no}）")
            if ans == "r":
                why = ""
                while not why:
                    why = input(">>> 取り直す理由（機材の事象。演奏の出来は理由にしない）： ").strip()
                log_row(kind="exclude", no=r.no, log=last["log"], excluded=1, exclude_reason=f"オペレータ：{why}")
                continue
        i += 1
    if not a.no_check:
        check(a, base, "end")
    print(f"\n全部終わり。記録：{SESSION}")


if __name__ == "__main__":
    main()
