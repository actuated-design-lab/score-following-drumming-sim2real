"""task_descriptors.py — 実機コアの 5 課題の記述子（論文の表 IV 用、2026-11-05）

入力：data/user0/eval/real_core/songs/<課題>.mid（実機と sim に流した MIDI）
  生の時刻：MIDI の note_on の時刻（ff_lib.midi_target と同じ読み方。ファイル名の bpm でテンポを決める）
  評価の時刻：ff_lib.midi_target → onsets_from_target（20 ms の格子に切り捨てて目標波形を作り、そのピーク。採点と FF の指令と同じ）
記述子（評価の時刻で計算。ずれだけ生の時刻と比べる）
  n、span_s（最初から最後の打撃まで）、ioi_min_ms・ioi_med_ms、frac_ioi_lt100/145/200（その値未満の間隔の割合）、
  nPVI = 100/(m−1)·Σ|d_k − d_{k+1}| / ((d_k + d_{k+1})/2)（d は間隔、m は間隔の数）
  rest_frac_16th：16 分音符の格子（課題の BPM）のうち打撃のない位置の割合 = 1 − n / (span / T16 + 1)
  frac_ioi_ge400：直前の間隔が 400 ms 以上の打撃の割合（§6.4 の「休符明け」と同じ境界。曲の最初の打撃は数えない）
  grid_dev_mean_ms・grid_dev_max_ms：生の時刻 − 20 ms 格子に切り捨てた時刻（0〜20 ms）の平均と最大
出力：data/user0/eval/access_1104/pkg/paper/pkg2_task_descriptors.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
import ff_lib as FL  # noqa: E402

SONGS = ROOT / "data/user0/eval/real_core/songs"
TASK = {"b_holdout_00_bpm90": "Held-out 1", "b_holdout_01_bpm96": "Held-out 2", "double_lead2_bpm160": "Double 160",
        "gmd_legacy_gmd_03_high_recut_bpm138": "GMD 138", "gmd_legacy_gmd_04_extreme_recut_bpm170": "GMD 170"}


def raw_onsets(path):
    import mido
    mid = mido.MidiFile(str(path))
    tempo = mido.bpm2tempo(float(re.search(r"bpm(\d+)", path.name).group(1)))
    t, out = 0.0, []
    for msg in mid.merged_track:
        t += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "note_on" and msg.velocity > 0:
            out.append(t)
    return np.array(out)


rows = []
for stem, name in TASK.items():
    p = SONGS / f"{stem}.mid"
    tgt, bpm = FL.midi_target(p)
    on = FL.onsets_from_target(tgt, bpm)
    raw = raw_onsets(p)
    d = np.diff(on) * 1000
    span = on[-1] - on[0]
    t16 = 60.0 / bpm / 4
    dev = (raw - np.floor(raw / FL.CTRL_DT + 1e-9) * FL.CTRL_DT) * 1000
    npvi = 100.0 * np.mean(np.abs(d[:-1] - d[1:]) / ((d[:-1] + d[1:]) / 2))
    rows.append(dict(task=name, file=p.name, bpm=bpm, n=len(on), n_raw=len(raw), span_s=span, ioi_min_ms=d.min(), ioi_med_ms=np.median(d),
                     frac_ioi_lt100=np.mean(d < 100), frac_ioi_lt145=np.mean(d < 145), frac_ioi_lt200=np.mean(d < 200), npvi=npvi,
                     rest_frac_16th=1 - len(on) / (span / t16 + 1), frac_ioi_ge400=np.mean(d >= 400),
                     grid_dev_mean_ms=dev.mean(), grid_dev_max_ms=dev.max()))
D = pd.DataFrame(rows)
out = ROOT / "data/user0/eval/access_1104/pkg/paper"
out.mkdir(parents=True, exist_ok=True)
D.to_csv(out / "pkg2_task_descriptors.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(D.drop(columns="file").round(3).to_string(index=False))
