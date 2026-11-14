"""video_align.py — 実機の動画と sim の動画を「最初の打撃」でそろえて切り出し、横に並べる（CPU、ffmpeg）

sim：video_meta.json の first_strike_s（sim の時刻）から、動画の中の時刻 = first_strike_s − frame0_time_s
実機：--real_t0（動画の中で最初にスティックが打面に当たった時刻 [s]、コマ送りで読む）。
      省略すると音から候補を出す（短い区間の音の大きさが急に上がる時刻、上位 5 つを表示して 1 番目を使う）。
      空気の排気音も拾うので、候補の時刻は必ずコマで確かめ、違えば --real_t0 で指定し直す

  python scripts/tools/user0/access/video_align.py --sim data/user0/eval/access_1104/sim/vid_E_double160 \\
      --real ~/Videos/access2026/E_double160_take2.mp4 --real_t0 3.48 --pre 1.0 --dur 8 --out out/E_double160
出力：<out>_sim.mp4、<out>_real.mp4（どちらも最初の打撃の --pre 秒前から --dur 秒、50 fps・高さ 720）、<out>_sbs.mp4（左 実機・右 sim）
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--sim", required=True, help="sim_trace の出力フォルダ（video_meta.json がある）")
p.add_argument("--real", required=True)
p.add_argument("--real_t0", type=float, default=None)
p.add_argument("--pre", type=float, default=1.0)
p.add_argument("--dur", type=float, default=8.0)
p.add_argument("--height", type=int, default=720)
p.add_argument("--out", required=True)
a = p.parse_args()


def audio_onsets(path, sr=8000, win=0.01, k=5):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(float)
    n = int(sr * win)
    e = np.sqrt(np.mean(x[: len(x) // n * n].reshape(-1, n) ** 2, 1)) + 1e-9
    d = np.diff(np.log(e))
    idx = np.argsort(-d)[: 50]
    picks = []
    for i in idx:                              # 0.2 s 以内の重なりを除いて、時刻の早い順
        if all(abs(i - j) * win > 0.2 for j in picks):
            picks.append(i)
    picks = sorted(picks[: k])
    return [(i + 1) * win for i in picks]


meta = json.load(open(Path(a.sim) / "video_meta.json"))
sim_mp4 = str(sorted((Path(a.sim) / "video").glob("*.mp4"))[0])
t_sim = meta["first_strike_s"] - meta["frame0_time_s"]
if a.real_t0 is None:
    c = audio_onsets(a.real)
    print("[video_align] 音の立ち上がりの候補 [s]:", [round(float(v), 3) for v in c], "→ 1 番目を使う（コマで確かめること）")
    a.real_t0 = c[0]
vf = f"fps=50,scale=-2:{a.height}"
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
for src, t0, tag in [(sim_mp4, t_sim, "sim"), (a.real, a.real_t0, "real")]:
    ss = max(0.0, t0 - a.pre)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.3f}", "-i", src, "-t", f"{a.dur:.3f}", "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", f"{out}_{tag}.mp4"], check=True)
    if t0 - a.pre < 0:
        print(f"[video_align] {tag}：最初の打撃の前が {t0:.2f} s しかない（--pre を縮める）")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f"{out}_real.mp4", "-i", f"{out}_sim.mp4", "-filter_complex", "hstack=inputs=2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", f"{out}_sbs.mp4"], check=True)
print(f"[video_align] sim の最初の打撃 {meta['first_strike_s']:.3f} s（動画の {t_sim:.3f} s）、実機 {a.real_t0:.3f} s → {out}_sbs.mp4")
