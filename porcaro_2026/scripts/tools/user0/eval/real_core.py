"""real_core.py — 実機コアの材料をそろえる（Isaac も GPU も使わない）

実機コアの、採点する 5 件のタスクと、時刻のずれの較正用の列を作る。
  タスク：(b) ホールドアウト列 2 本（b_sets の holdout の種の順に先頭から 0・1）、GMD 138・170（規則で切り直したもの、gmd_scan.py）、
          ダブル 160（前の研究と同じ 24 打。最初の打を 2.0 s に置き直した）
  較正：(b) の分布から種 50000+i で 2 本（評価のタスクにも検証用にも使わない）。F3 と RL+Δ の Δ を測るためだけに使う
  方法：B・E（RL、GEN2B）、F0（FF、較正なし）、F3（FF、発火時刻を Δ_FF だけずらす）、RL+Δ（B の目標を Δ_RL だけ前にずらす）
  反復：3 回。反復ごとに、タスクの順番と、タスクの中の方法の順番を並べ替える（種を固定。同日交互）

  python scripts/tools/user0/eval/real_core.py prepare                         # MIDI・較正の列・実行順の表
  python scripts/tools/user0/eval/real_core.py ff --ff data/user0/ff/opt/r1/best.json            # F0 の指令（signals）
  python scripts/tools/user0/eval/real_core.py shift --delta_ff_ms 40 --delta_rl_ms 20 --ff ...  # F3 の指令と RL+Δ の MIDI
  # 2026-11-04：押し付けの判定を直してやり直す FF（v2）。指令は signals_v2/、実行順は run_order_v2.csv（11/04 の FF フェーズと同じ並び）
  python scripts/tools/user0/eval/real_core.py ff --v 2 --ff data/user0/ff/opt2/r<R>/favorite.json
  python scripts/tools/user0/eval/real_core.py shift --v 2 --ff data/user0/ff/opt2/r<R>/favorite.json --delta_ff_ms <Δ>
出力：data/user0/eval/real_core/
  songs/*.mid（タスク）、calib/*.mid（較正）、songs_rl_delta/*.mid（RL+Δ）、signals/F0_*.csv・F3_*.csv（FF の指令、jetson の signals と同じ書式）
  tasks.csv、run_order.csv、README.md
jetson_project には tools/import_real_core.py で取り込む（repo を分けるため、こちらからは書かない）
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import mido
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
sys.path.insert(0, str(ROOT / "scripts/tools/user0/eval"))
import b_sets  # noqa: E402
import ff_lib  # noqa: E402

OUT = ROOT / "data/user0/eval/real_core"
DT = 0.02
METHODS = ["B", "E", "F0", "F3", "RL+Δ"]
N_HOLDOUT, N_CALIB, REPS, ORDER_SEED = 2, 2, 3, 20261103
# 実施しない方法（決めた日と理由）。実行順は 5 方法で並べ替えたものから、該当の行を抜くだけにする（残りの順番は変えない）
# F3 は 11/03 18:50 にいったん外した（#70）が、根拠の Δ_FF = 0 が押し付けの解（F0_r1）の山から出た値だったので戻した（11/03 19:20）。
# 打撃の定義を直した F0 で較正し直してから、外すかどうかを決め直す
PHASE_RL = ["B", "E"]     # 先に回す方法（残りは FF フェーズ）
DROPPED = {"RL+Δ": "2026-11-03：B の較正 2 本で Δ_RL = +8.9 ms（SD 51.1 ms、打点 52）→ 20 ms に丸めて 0 ms。RL+Δ は B と同じ指令になるため、"
                   "本人の決定で実機コアから外した（jetson_project data/user0/access2026/core2026_score/calib.md）"}
CALIB_SEED_BASE = 50000
# 日をまたぐ分の橋渡し（2026-11-04 本人の決定）：RL は 11/03、FF は 11/04 に回るので、FF フェーズの反復 1 の各タスクのかたまりに
# B・E を 1 本ずつ（5 タスク × 2 = 10 本）差し込み、同じ日の RL と 11/03 の RL の差（日の差）を測る。採点の本表には入れず、別の表にする。
# 試行番号は 4（11/03 の 1〜3 と区別）。かたまりの中の位置と B・E の順は種 BRIDGE_SEED で決める（F0・F3 の順番は変えない）
BRIDGE_METHODS, BRIDGE_TRIAL, BRIDGE_SEED = ["B", "E"], 4, 20261104


def write_mid(path: Path, t_s, bpm):
    mid = mido.MidiFile(ticks_per_beat=960)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tempo = mido.bpm2tempo(bpm)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    ticks = np.round(np.asarray(t_s) / mido.tick2second(1, 960, tempo)).astype(int)
    ev = sorted([(k, 1) for k in ticks] + [(k + 30, 0) for k in ticks])
    now = 0
    for k, on in ev:
        tr.append(mido.Message("note_on" if on else "note_off", channel=9, note=38, velocity=100 if on else 0, time=int(k - now)))
        now = k
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))


def onsets_of(path):
    """jetson・sim と同じ 20 ms への切り捨てのあとの打撃時刻 [s]"""
    tgt, bpm = ff_lib.midi_target(path)
    on = ff_lib.onsets_from_target(tgt, bpm)
    return on, bpm, tgt


def prepare():
    songs = OUT / "songs"
    songs.mkdir(parents=True, exist_ok=True)
    H = np.load(ROOT / "data/user0/eval/b_sets/holdout.npz", allow_pickle=True)
    rows = []
    for i in range(N_HOLDOUT):
        src = sorted((ROOT / "data/user0/eval/b_sets/holdout_midi").glob(f"b_holdout_{i:02d}_bpm*.mid"))[0]
        shutil.copy2(src, songs / src.name)
        rows.append(dict(task=f"holdout{i}", file=src.name, role="(b) ホールドアウト（分布内・系列未見）", unseen="i", seed=int(H["seed"][i])))
    for leg, cls in (("gmd_03_high_recut_bpm138", "ii"), ("gmd_04_extreme_recut_bpm170", "iii")):
        src = ROOT / "data/user0/eval/gmd_scan/excerpts" / f"gmd_legacy_{leg}.mid"
        shutil.copy2(src, songs / src.name)
        rows.append(dict(task="gmd138" if "138" in leg else "gmd170", file=src.name,
                         role="GMD レガシー（規則で切り直し）" + ("、テンポ範囲内" if cls == "ii" else "、テンポ範囲外"), unseen=cls, seed=np.nan))
    # ダブル 160：rhythm_generator の double（16 分で 0,1,4,5,8,9,12,13）を 3 小節 = 24 打、最初の打を 2.0 s
    s16 = 15.0 / 160.0
    t = np.concatenate([2.0 + b * 16 * s16 + np.array([0, 1, 4, 5, 8, 9, 12, 13]) * s16 for b in range(3)])
    write_mid(songs / "double_lead2_bpm160.mid", t, 160.0)
    rows.append(dict(task="double160", file="double_lead2_bpm160.mid", role="前論文のパターン（support の端）", unseen="(b) の端", seed=np.nan))
    for r in rows:
        on, bpm, tgt = onsets_of(songs / r["file"])
        ioi = np.diff(on) * 1000
        r.update(bpm=round(bpm, 1), n=len(on), first_s=round(on[0], 3), last_s=round(on[-1], 3), dur_s=round(on[-1] + 1.0, 2),
                 ioi_min_ms=round(ioi.min()), ioi_med_ms=round(np.median(ioi)), frac_ioi_lt120=round(np.mean(ioi < 120), 3))
    tasks = pd.DataFrame(rows)
    tasks.to_csv(OUT / "tasks.csv", index=False)
    # 較正の列
    cal = []
    for i in range(N_CALIB):
        traj, bpm, dur, on = b_sets.one(CALIB_SEED_BASE + i)
        f = f"b_calib_{i:02d}_bpm{int(round(bpm))}.mid"
        write_mid(OUT / "calib" / f, np.asarray(on) * DT + 0.002, bpm)
        cal.append(dict(file=f, seed=CALIB_SEED_BASE + i, bpm=round(bpm, 1), n=len(on)))
    pd.DataFrame(cal).to_csv(OUT / "calib.csv", index=False)
    # 実行順
    rng = np.random.default_rng(ORDER_SEED)
    order = []
    for rep in range(1, REPS + 1):
        for task in rng.permutation(tasks.task.tolist()):
            for m in rng.permutation(METHODS):
                order.append(dict(rep=rep, task=task, method=m))
    od = pd.DataFrame(order)
    od.insert(0, "no_orig", np.arange(1, len(od) + 1))
    od = od[~od.method.isin(DROPPED)].reset_index(drop=True)
    # フェーズ：RL（B・E）を先に全部、FF（F0・F3）をあと（FF の段 2 が終わるのを待たずに RL を回すため。2026-11-03 本人の決定、
    # 方法の順が固定されることによるドリフトの影響は許容）。フェーズの中は元の順番のまま（no_orig の昇順）
    od.insert(0, "phase", np.where(od.method.isin(PHASE_RL), "rl", "ff"))
    od = od.sort_values(["phase", "no_orig"], key=lambda c: c.map({"rl": 0, "ff": 1}) if c.name == "phase" else c,
                        kind="stable").reset_index(drop=True)
    od["trial"], od["bridge"] = od["rep"], 0
    # 橋渡し：FF フェーズの反復 1 の、タスクのかたまりごとに B・E を差し込む
    rng2 = np.random.default_rng(BRIDGE_SEED)
    rows = [r for r in od[od.phase == "rl"].to_dict("records")]
    ff = od[od.phase == "ff"].to_dict("records")
    i = 0
    while i < len(ff):
        j = i
        while j < len(ff) and ff[j]["task"] == ff[i]["task"] and ff[j]["rep"] == ff[i]["rep"]:
            j += 1
        block = ff[i:j]
        if block[0]["rep"] == 1:
            br = [dict(phase="ff", no_orig=np.nan, rep=1, task=block[0]["task"], method=m, trial=BRIDGE_TRIAL, bridge=1)
                  for m in rng2.permutation(BRIDGE_METHODS)]
            n = len(block) + len(br)
            slots = set(rng2.choice(n, len(br), replace=False).tolist())
            bi, oi, merged = iter(br), iter(block), []
            for k in range(n):
                merged.append(next(bi) if k in slots else next(oi))
            block = merged
        rows += block
        i = j
    od = pd.DataFrame(rows)[["phase", "no_orig", "rep", "task", "method", "trial", "bridge"]]
    od["no_orig"] = od["no_orig"].astype("Int64")
    od.insert(0, "no", np.arange(1, len(od) + 1))
    od.to_csv(OUT / "run_order.csv", index=False)
    kept = [m for m in METHODS if m not in DROPPED]
    L = ["# 実機コアの材料（real_core.py）", "",
         f"方法 {kept}、タスク {tasks.task.tolist()}、反復 {REPS} 回 → {len(od)} 本。実行順は 5 方法（{METHODS}）で種 {ORDER_SEED} で並べ替え、"
         f"実施しない方法の行を抜いた（run_order.csv の no_orig が元の番号）。",
         f"実行は 2 フェーズ：phase = rl（{PHASE_RL}、no 1–{int((od.phase == 'rl').sum())}）を先に、phase = ff（残り）をあとに回す。"
         "フェーズの中は元の順番のまま（2026-11-03 本人の決定。FF の段 2 を待たずに RL を回すため。方法の順が固定されるドリフトは許容）。",
         *[f"実施しない方法：{k}（{v}）" for k, v in DROPPED.items()],
         f"日の差の橋渡し（2026-11-04 本人の決定）：FF フェーズの反復 1 の各タスクのかたまりに {BRIDGE_METHODS} を 1 本ずつ差し込んだ"
         f"（bridge = 1、試行番号 {BRIDGE_TRIAL}、{int(od.bridge.sum())} 本）。採点の本表には入れず、11/03 の RL と比べる表にだけ使う。",
         "Sing Sing Sing（アレンジ実証の別枠）は採譜ができてから足す。", "", "## タスク", "",
         "| " + " | ".join(tasks.columns) + " |", "|" + "---|" * len(tasks.columns)]
    for r in tasks.itertuples(index=False):
        L.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    L += ["", "## 較正（F3・RL+Δ の Δ を測る。評価のタスクには使わない）", "",
          f"(b) の分布から種 {CALIB_SEED_BASE}+i で {N_CALIB} 本（calib/）。F0 と B をそれぞれ較正の列で叩き、",
          "打った打点の符号つきタイミング誤差の平均を Δ とする（20 ms に丸める）。F3 = F0 の発火時刻を Δ_FF だけ前に、",
          "RL+Δ = B の目標の MIDI を Δ_RL だけ前にずらす（`real_core.py shift`）。",
          "Δ は方法ごとに自分の較正から出す（Δ_RL は B の較正、Δ_FF は F0 の較正）。RL+Δ は B だけ（E には当てない）。2026-11-03 確定"]
    open(OUT / "README.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def ff_signals(prims_json, tag, delta_ms=0.0, sig_name="signals"):
    spec = json.load(open(prims_json))
    prims = [ff_lib.Prim.from_dict(d) for d in spec["prims"]]
    if delta_ms:
        prims = [ff_lib.replace(p, L=p.L + delta_ms / 1000.0, d_g=p.d_g - delta_ms / 1000.0) for p in prims]   # 発火も G の窓も Δ だけ前に
    sig = OUT / sig_name
    sig.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_csv(OUT / "tasks.csv")
    for r in tasks.itertuples():
        on, bpm, tgt = onsets_of(OUT / "songs" / r.file)
        n = int(np.ceil((on[-1] + 1.0) / DT))
        cmd = ff_lib.synth(on, prims, n)
        ff_lib.to_signal_df(cmd).to_csv(sig / f"{tag}_{Path(r.file).stem}.csv", index=False)
    cal = pd.read_csv(OUT / "calib.csv")
    if tag == "F0":
        for r in cal.itertuples():
            on, bpm, tgt = onsets_of(OUT / "calib" / r.file)
            ff_lib.to_signal_df(ff_lib.synth(on, prims, int(np.ceil((on[-1] + 1.0) / DT)))).to_csv(sig / f"F0_{Path(r.file).stem}.csv", index=False)
    json.dump({"source": str(prims_json), "delta_ms": delta_ms, "prims": [ff_lib.asdict(p) for p in prims]},
              open(sig / f"{tag}_prims.json", "w"), indent=1)
    print(f"→ {sig}（{tag}、Δ {delta_ms} ms）")


def order_v2():
    """押し付けの判定を直してやり直す FF（2026-11-04 本人の決定）の実行順：11/04 の FF フェーズ（F0・F3 と橋渡しの B・E）と
    同じ並びのまま、番号を 1 から振り直す（no_orig・trial・bridge はそのまま。橋渡しは試行番号 4）"""
    od = pd.read_csv(OUT / "run_order.csv")
    v2 = od[od.phase == "ff"].copy().reset_index(drop=True)
    v2["no_v1"] = v2["no"]
    v2["no"] = np.arange(1, len(v2) + 1)
    v2.to_csv(OUT / "run_order_v2.csv", index=False)
    print(f"→ {OUT / 'run_order_v2.csv'}（{len(v2)} 本：{v2.method.value_counts().to_dict()}、橋渡し {int(v2.bridge.sum())} 本）")


def shift_rl(delta_ms):
    d = OUT / "songs_rl_delta"
    d.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_csv(OUT / "tasks.csv")
    for r in tasks.itertuples():
        on, bpm, _ = onsets_of(OUT / "songs" / r.file)
        t = on - delta_ms / 1000.0 + 0.002
        base = re.sub(r"_bpm\d+$", "", Path(r.file).stem)
        write_mid(d / f"{base}_rld{int(round(delta_ms))}_bpm{int(round(bpm))}.mid", t, bpm)
    json.dump({"delta_rl_ms": delta_ms, "note": "採点は元の songs/ の MIDI の打撃時刻で行う（ずらした MIDI の目標では採点しない）"},
              open(d / "delta.json", "w"), ensure_ascii=False)
    print(f"→ {d}（RL+Δ、Δ {delta_ms} ms）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "ff", "shift"])
    ap.add_argument("--ff", default=None, help="F0 のプリミティブ（ff_opt.py の best.json）")
    ap.add_argument("--delta_ff_ms", type=float, default=None)
    ap.add_argument("--delta_rl_ms", type=float, default=None)
    ap.add_argument("--v", type=int, default=1, help="2 = 押し付けの判定を直してやり直す FF（2026-11-04）：指令は signals_v2/、実行順は run_order_v2.csv")
    a = ap.parse_args()
    sig_name = "signals" if a.v == 1 else f"signals_v{a.v}"
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "ff":
        ff_signals(a.ff, "F0", sig_name=sig_name)
        if a.v == 2:
            order_v2()
    else:
        if a.delta_ff_ms is not None:
            ff_signals(a.ff, "F3", a.delta_ff_ms, sig_name=sig_name)
        if a.delta_rl_ms is not None:
            shift_rl(a.delta_rl_ms)


if __name__ == "__main__":
    main()
