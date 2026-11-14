"""sss_task.py — Sing Sing Sing（楽譜版、フロアタムのみ）を実機コアと同じタスク形式の MIDI にする（CPU）

実機・sim のローダ（jetson src/midi_rhythm_generator.py、porcaro ff_lib.midi_target / play_sim_midi の MidiInjector）は
ノートのピッチもベロシティも見ない：velocity > 0 の note_on の時刻だけを打撃にし、目標はどれも 20 N。テンポは set_tempo
（ファイル名に bpm<N> があればそちらが優先）。打撃の時刻は 20 ms（制御周期）に切り捨てて目標波形にする。

ここでは実機コアの MIDI（real_core.write_mid）と同じにする：
  - 打撃の時刻を 20 ms の格子のいちばん近い点に丸め、+2 ms（切り捨てで 1 つ前の格子に落ちないように）
  - ファイル名に bpm217（set_tempo と同じ値。採点の T16 = 15 / 217 s に使う）
  - 元のベロシティ（アクセント 105／弱打 12）は目標に入らないので、横に CSV で残す（あとでアクセントの解析に使う）

  python scripts/tools/user0/eval/sss_task.py --src <sss_a_tomonly_task.mid>
  # メイン（アクセント打 = ベロシティ 105）だけ・ゴースト（弱打 = 12）だけ（2026-11-04）。時刻は全打の版と同じ（あとで動画を重ねるため、ずらさない）
  python scripts/tools/user0/eval/sss_task.py --src <sss_a_main_task.mid> --name sss_a_main      # 研究チャットで分けたもの（全打の版の 105 と一致）
  python scripts/tools/user0/eval/sss_task.py --src <sss_a_ghost_task.mid> --name sss_a_ghost    # 同じく 12 と一致
  python scripts/tools/user0/eval/sss_task.py --src <sss_a_tomonly_task.mid> --velocity 105 --name sss_a_main   # 分けたものがないとき
出力：data/user0/eval/sss/{<name>_bpm217.mid, <name>_onsets.csv, <name>.md}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mido
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/eval"))
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
import ff_lib  # noqa: E402
from real_core import write_mid  # noqa: E402

DT = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/user0/eval/sss")
    ap.add_argument("--name", default="sss_a_tomonly")
    ap.add_argument("--velocity", type=int, default=None, help="このベロシティの打撃だけを残す（メイン 105・ゴースト 12）")
    a = ap.parse_args()
    m = mido.MidiFile(a.src)
    tempo = next((msg.tempo for msg in m if msg.type == "set_tempo"), 500000)
    bpm = mido.tempo2bpm(tempo)
    t, rows = 0.0, []
    for msg in m:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            rows.append(dict(t_src=t, note=msg.note, velocity=msg.velocity))
    D = pd.DataFrame(rows)
    n_all = len(D)
    if a.velocity is not None:
        D = D[D.velocity == a.velocity].reset_index(drop=True)
    D["t_task"] = np.round(D.t_src / DT) * DT + 0.002
    D["shift_ms"] = (D.t_task - 0.002 - D.t_src) * 1000
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    mid = out / f"{a.name}_bpm{int(round(bpm))}.mid"
    write_mid(mid, D.t_task.to_numpy(), round(bpm))
    tg, b2 = ff_lib.midi_target(mid)
    on = ff_lib.onsets_from_target(tg, b2)
    D["t_target"] = on if len(on) == len(D) else np.nan
    D.to_csv(out / f"{a.name}_onsets.csv", index=False)
    ioi = np.diff(D.t_task) * 1000
    L = [f"# {mid.name}（sss_task.py）", "",
         f"元：{Path(a.src).name}（全 {n_all} 打のうち {len(D)} 打" + (f"、ベロシティ {a.velocity} だけ" if a.velocity is not None else "") + f"、ピッチ {sorted(int(x) for x in D.note.unique())}、ベロシティ { {int(k): int(v) for k, v in D.velocity.value_counts().items()} }、♩ = {bpm:.1f}）",
         f"格子に丸めたずれ：最大 {D.shift_ms.abs().max():.1f} ms、平均の絶対値 {D.shift_ms.abs().mean():.1f} ms",
         f"目標として読めた打撃：{len(on)} / {len(D)}（ff_lib.onsets_from_target、BPM {b2:.1f}）",
         f"最初の打撃 {D.t_task.iloc[0]:.3f} s、最後 {D.t_task.iloc[-1]:.3f} s。IOI の最小 {ioi.min():.0f} ms、中央値 {np.median(ioi):.0f} ms、最大 {ioi.max():.0f} ms",
         "ベロシティは目標に入らない（全打 20 N）。元のベロシティは onsets.csv"]
    (out / (f"{a.name}.md" if a.name != "sss_a_tomonly" else "README.md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
