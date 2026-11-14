"""b_sets.py — (b) の分布から、検証用の列（シードの選択）と評価用のホールドアウト列を作る（Isaac も GPU も使わない）

シードの選択規則（設計ループの決定ドキュメント 2026-11-01 追補）のとおり、検証用とホールドアウトは生成の種を分け、学習の種（1〜3）とも分ける。
  検証用      ：種 20000 + i（i = 0〜n_val-1）
  ホールドアウト：種 30000 + i
1 列 = 学習の 1 エピソードと同じ作り（Porcaro2026EnvCfg_Gen2_B の値、カリキュラムの最終段 = 全部の音価・テンポ 60〜160）。
  助走 2.0 s ＋ 32 拍 ＋ 余白 0.5 s、目標の波形は学習と同じ（rhythm_generator.reset の畳み込み）。

  python scripts/tools/user0/eval/b_sets.py                    # data/user0/eval/b_sets/{val,holdout}.npz と summary.md
  python scripts/tools/user0/eval/b_sets.py --n_val 16 --n_holdout 16
出力：
  val.npz / holdout.npz：traj (n, T) [N]、bpm (n,)、dur_s (n,)、seed (n,)、onset_steps（object の配列）、params（json の文字列）
  holdout_midi/*.mid  ：実機用（最初の打 = 2.0 s、ノート 38。jetson の MidiRhythmGenerator が 20 ms に切り捨てても同じ tick になるよう +2 ms）
  summary.md          ：列ごとのテンポ・打数・IOI、検証用とホールドアウトと学習の種で同じ列がないことの確認
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import mido
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
RG = ROOT / "source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/user0/rhythm_generator.py"
spec = importlib.util.spec_from_file_location("rhythm_generator", str(RG))
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

DT, MAX_STEPS = 0.02, 5000
# Porcaro2026EnvCfg_Gen2_B と同じ値（eval_b.py が起動時に cfg と突き合わせる）
PARAMS = dict(bpm_range=(60.0, 160.0), note_beats=(0.25, 1.0 / 3.0, 0.5, 0.75, 1.0, 1.5, 2.0),
              note_probs=(0.45, 0.076, 0.183, 0.061, 0.122, 0.061, 0.047), rest_prob=0.15, min_interval_s=0.08,
              lead_in_s=2.0, episode_beats=32, tail_s=0.5, curriculum_level=2, target_force=20.0)
SEED_BASE = {"val": 20000, "holdout": 30000}
TRAIN_SEEDS = (1, 2, 3)


def one(seed):
    g = rg.RhythmGenerator(num_envs=1, device="cpu", dt=DT, max_episode_length=MAX_STEPS,
                           bpm_range=PARAMS["bpm_range"], target_force=PARAMS["target_force"])
    g.set_random_mode(True, bpm_range=PARAMS["bpm_range"], note_beats=PARAMS["note_beats"], note_probs=PARAMS["note_probs"],
                      rest_prob=PARAMS["rest_prob"], min_interval_s=PARAMS["min_interval_s"], curriculum=True,
                      lead_in_s=PARAMS["lead_in_s"], episode_beats=PARAMS["episode_beats"], tail_s=PARAMS["tail_s"])
    g.set_curriculum_level(PARAMS["curriculum_level"])
    g._rng = np.random.default_rng(seed)
    g.reset(torch.tensor([0]))
    traj = g.target_trajectories[0].numpy().astype(np.float32)
    bpm = float(g.current_bpms[0])
    dur = float(g.rand_ep_duration_s[0])
    x = traj
    on = np.nonzero((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= 0.75 * PARAMS["target_force"]))[0] + 1
    return traj, bpm, dur, on


def first_episode_of_training_seed(seed):
    """学習の seed s の乱数で最初に引かれる列（重なりの確認用。学習ではこれ以降の列もあるが、種が違えば流れが別）"""
    return one(seed)[3]


def write_mid(path: Path, onset_steps, bpm):
    mid = mido.MidiFile(ticks_per_beat=960)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tempo = mido.bpm2tempo(bpm)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    t = np.asarray(onset_steps) * DT + 0.002
    ticks = np.round(t / mido.tick2second(1, 960, tempo)).astype(int)
    ev = sorted([(k, 1) for k in ticks] + [(k + 30, 0) for k in ticks])
    now = 0
    for k, on in ev:
        tr.append(mido.Message("note_on" if on else "note_off", channel=9, note=38, velocity=100 if on else 0, time=int(k - now)))
        now = k
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_val", type=int, default=16)
    ap.add_argument("--n_holdout", type=int, default=16)
    ap.add_argument("--out", default="data/user0/eval/b_sets")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    L = ["# (b) の検証用の列とホールドアウト列（b_sets.py）", "",
         f"値：{json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in PARAMS.items()}, ensure_ascii=False)}", "",
         f"種：検証用 {SEED_BASE['val']}+i、ホールドアウト {SEED_BASE['holdout']}+i（学習は {TRAIN_SEEDS}）", ""]
    keys = {}
    for name, n in (("val", a.n_val), ("holdout", a.n_holdout)):
        rows = [one(SEED_BASE[name] + i) for i in range(n)]
        traj = np.stack([r[0] for r in rows])
        bpm = np.array([r[1] for r in rows])
        dur = np.array([r[2] for r in rows])
        ons = np.empty(n, dtype=object)
        for i, r in enumerate(rows):
            ons[i] = r[3]
        np.savez_compressed(out / f"{name}.npz", traj=traj, bpm=bpm, dur_s=dur, seed=np.arange(n) + SEED_BASE[name],
                            onset_steps=ons, params=json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in PARAMS.items()}))
        keys[name] = {tuple(r[3].tolist()) for r in rows}
        L += [f"## {name}（{n} 列）", "", "| i | 種 | BPM | 長さ [s] | 打数 | IOI 最小 [ms] | IOI 中央値 [ms] | IOI < 120 ms |", "|---|---|---|---|---|---|---|---|"]
        all_ioi = []
        for i, r in enumerate(rows):
            ioi = np.diff(r[3]) * DT * 1000
            all_ioi.append(ioi)
            L.append(f"| {i} | {SEED_BASE[name] + i} | {r[1]:.1f} | {r[2]:.2f} | {len(r[3])} | {ioi.min():.0f} | {np.median(ioi):.0f} | {np.mean(ioi < 120):.3f} |")
        A = np.concatenate(all_ioi)
        L += ["", f"まとめ：打数 {sum(len(r[3]) for r in rows)}、IOI < 120 ms の割合 {np.mean(A < 120):.3f}、BPM {bpm.min():.0f}〜{bpm.max():.0f}", ""]
        if name == "holdout":
            for i, r in enumerate(rows):
                write_mid(out / "holdout_midi" / f"b_holdout_{i:02d}_bpm{int(round(r[1]))}.mid", r[3], r[1])
    train = {tuple(first_episode_of_training_seed(s).tolist()) for s in TRAIN_SEEDS}
    ov = len(keys["val"] & keys["holdout"]), len((keys["val"] | keys["holdout"]) & train)
    L += ["## 重なりの確認", "", f"検証用 ∩ ホールドアウト：{ov[0]} 列、（検証用 ∪ ホールドアウト）∩ 学習の各 seed の最初の列：{ov[1]} 列（どちらも 0 であること）"]
    open(out / "summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
