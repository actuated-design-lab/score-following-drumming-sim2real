"""gmd_scan.py — GMD から評価用のスネアの抜粋を、決めた手順で機械的に作る（Isaac も GPU も使わない。mido・numpy・pandas）

手順はプロジェクトの決定ドキュメント「GMD 選定と Sing Sing Sing の扱い」（2026-11-01）の B 節・F 節:
  ・スネア類 = GM ノート番号 {37, 38, 40}（GMD 論文の簡略マッピング）。velocity < 20 は除く
  ・最小 IOI の規則：前から走査し、残した打から 0.08 s 以内の後続を落とす（番号による差は付けない）
  ・母集団は GMD の test split。フィルタ (a) スネア類の系列が取れる → (b) 20〜30 s の抜粋が取れる
    → (c) 抜粋内の最小 IOI ≥ 0.08 s → (d) info.csv の bpm が 60〜160。各段の残数を記録
  ・抜粋の窓：ファイルの最初のオンセット（全楽器）から 27.5 s。ファイルの span（最初〜最後のオンセット）が 20 s 未満なら除外
  ・138 と 170 はレガシー曲として別枠。今のファイル（data/common/midi/）と一致する元のファイルを GMD 全体から探し、
    同じ規則で切り直す

(c) の判定は 2 通り出す（決定ドキュメントで順序が決まっていないため）:
  reject … 0.08 s の規則で落とす前の最小 IOI で判定（近すぎる打を含む曲は外れる）
  drop   … 規則で落としたあとの最小 IOI で判定（実質すべて通る）。落とした打の数を記録

出力（--out）: summary.md、legacy_match.csv、scan_all.csv（全ファイル×各段の判定）、excerpts.csv（通過した抜粋の記述子）、
  excerpts/*.mid（抜粋。ノートは 38 に統一、velocity は元のまま。評価のスクリプトが読むようにファイル名に bpmNNN を入れる）

WS で（GMD はリポジトリの外に置く）:
  mkdir -p ~/datasets && cd ~/datasets
  curl -L -o groove-v1.0.0-midionly.zip https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
  unzip -q groove-v1.0.0-midionly.zip          # → ~/datasets/groove/（info.csv と drummerN/...）
  cd ~/my_isaaclab_projects/porcaro_2026
  python scripts/tools/user0/eval/gmd_scan.py --gmd_dir ~/datasets/groove --out data/user0/eval/gmd_scan
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mido
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SNARE = (37, 38, 40)
VEL_MIN = 20
MIN_IOI = 0.08
WIN_S = 27.5
SPAN_MIN_S = 20.0
BPM_RANGE = (60, 160)
CTRL_DT = 0.02
IOI_REF_MS = (100.0, 120.0, 145.0)    # 短い IOI の割合を出す基準（145 ms は Gen2 の圧力モデルの 0→0.6 MPa の t50。段 1 の値で差し替える）
LEGACY = {"gmd_03_high_bpm138": ROOT / "data/common/midi/gmd_03_high_bpm138.mid",
          "gmd_04_extreme_bpm170": ROOT / "data/common/midi/gmd_04_extreme_bpm170.mid"}


# --------------------------------------------------------------------------- 読み込み
def read_notes(path: Path) -> pd.DataFrame:
    """全ノートオン（velocity > 0）：time_s（ファイルのテンポで秒）、beat（四分音符単位）、note、velocity"""
    mid = mido.MidiFile(str(path))
    rows, t_s, ticks = [], 0.0, 0
    tempo = 500000
    for msg in mido.merge_tracks(mid.tracks):
        ticks += msg.time
        t_s += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            rows.append((t_s, ticks / mid.ticks_per_beat, msg.note, msg.velocity))
    return pd.DataFrame(rows, columns=["time_s", "beat", "note", "velocity"])


def snare_onsets(df: pd.DataFrame, pitches=SNARE) -> pd.DataFrame:
    s = df[df.note.isin(pitches) & (df.velocity >= VEL_MIN)].sort_values("time_s")
    return s.drop_duplicates("time_s").reset_index(drop=True)   # 同時刻の複数ノートは 1 打


def drop_close(t: np.ndarray, min_ioi: float = MIN_IOI) -> np.ndarray:
    """残した打から min_ioi 以内の後続を落とす（前から走査）。残す打の添字"""
    keep, last = [], -np.inf
    for i, x in enumerate(t):
        if x - last >= min_ioi - 1e-9:
            keep.append(i)
            last = x
    return np.array(keep, dtype=int)


# --------------------------------------------------------------------------- 記述子
def npvi(ioi: np.ndarray) -> float:
    if len(ioi) < 2:
        return np.nan
    a, b = ioi[:-1], ioi[1:]
    return float(100.0 / (len(ioi) - 1) * np.sum(np.abs((a - b) / ((a + b) / 2))))


def describe(t: np.ndarray, prefix: str = "") -> dict:
    """打撃時刻 [s] → 記述子。q_ は評価と同じ 20 ms の格子に切り捨てた後（play_sim_midi.py の MidiInjector と同じ）"""
    out = {}
    for tag, x in (("", t), ("q_", np.unique(np.floor(t / CTRL_DT + 1e-9) * CTRL_DT))):
        ioi = np.diff(x) * 1000.0
        out[f"{prefix}{tag}n"] = len(x)
        out[f"{prefix}{tag}ioi_min_ms"] = round(float(ioi.min()), 1) if len(ioi) else np.nan
        out[f"{prefix}{tag}ioi_med_ms"] = round(float(np.median(ioi)), 1) if len(ioi) else np.nan
        if tag == "":
            out[f"{prefix}dur_s"] = round(float(x[-1] - x[0]), 2) if len(x) else np.nan
            for r in IOI_REF_MS:
                out[f"{prefix}frac_ioi_lt{int(r)}"] = round(float(np.mean(ioi < r)), 3) if len(ioi) else np.nan
            out[f"{prefix}npvi"] = round(npvi(ioi), 1)
    return out


# --------------------------------------------------------------------------- レガシー曲の同定
def match_legacy(leg: pd.DataFrame, cand: pd.DataFrame, tol_beat: float = 0.03) -> tuple[float, float]:
    """今のレガシーファイルの打撃（拍の位置）が、候補ファイルのノート 38（velocity ≥ 20）のどこに一致するか。
    テンポの違いに左右されないよう、拍の位置で比べる。(一致した割合, ずらし量 [拍])"""
    L = leg.beat.to_numpy()
    G = cand.beat.to_numpy()
    if len(G) < len(L) * 0.5:
        return 0.0, np.nan
    best = (0.0, np.nan)
    for g0 in G:
        off = g0 - L[0]
        j = np.searchsorted(G, L + off)
        d = np.minimum(np.abs(G[np.clip(j, 0, len(G) - 1)] - (L + off)), np.abs(G[np.clip(j - 1, 0, len(G) - 1)] - (L + off)))
        sc = float(np.mean(d <= tol_beat))
        if sc > best[0]:
            best = (sc, float(off))
            if sc == 1.0:
                break
    return best


# --------------------------------------------------------------------------- 抜粋
def excerpt(df: pd.DataFrame, t0: float, t1: float, pitches=SNARE) -> tuple[np.ndarray, np.ndarray, int, float]:
    """[t0, t1) のスネア類。(残した打の時刻, velocity, 0.08 s で落とした数, 落とす前の最小 IOI [s])"""
    s = snare_onsets(df, pitches)
    s = s[(s.time_s >= t0 - 1e-9) & (s.time_s < t1)]
    t = s.time_s.to_numpy()
    raw_min = float(np.diff(t).min()) if len(t) > 1 else np.nan
    k = drop_close(t)
    return t[k], s.velocity.to_numpy()[k], len(t) - len(k), raw_min


def write_mid(path: Path, t: np.ndarray, vel: np.ndarray, bpm: float, lead_in: float):
    """打撃を最初の打が lead_in [s] に来るように並べ直して書く（ノート 38、480 tpb、テンポは bpm）"""
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tempo = mido.bpm2tempo(bpm)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    ticks = np.round(mido.second2tick(1.0, 480, tempo) * (t - t[0] + lead_in)).astype(int)
    ev = sorted([(k, "on", int(v)) for k, v in zip(ticks, vel)] + [(k + 30, "off", 0) for k in ticks])
    now = 0
    for k, typ, v in ev:
        tr.append(mido.Message("note_on" if typ == "on" else "note_off", channel=9, note=38, velocity=v if typ == "on" else 0,
                               time=int(k - now)))
        now = k
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))


def md_table(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "（なし）"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmd_dir", required=True, help="GMD（midionly）を展開したフォルダ（info.csv がある所）")
    ap.add_argument("--out", default="data/user0/eval/gmd_scan")
    ap.add_argument("--lead_in", type=float, default=2.0, help="抜粋の最初の打の前に置く時間 [s]（決定ドキュメントで未定。仮）")
    a = ap.parse_args()
    gmd, out = Path(a.gmd_dir).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(gmd / "info.csv")
    notes = {}

    def get(fn):
        if fn not in notes:
            notes[fn] = read_notes(gmd / fn)
        return notes[fn]

    # 1) レガシー曲（138・170）の同定：GMD 全体から、ノート 38 の並びが一致するファイルを探す
    lrows, legacy_src = [], {}
    for name, p in LEGACY.items():
        leg = read_notes(p)
        bpm_leg = float(name.split("bpm")[-1])
        best = []
        for r in info.itertuples():
            cand = get(r.midi_filename)
            c38 = cand[(cand.note == 38) & (cand.velocity >= VEL_MIN)].drop_duplicates("beat")
            sc, off = match_legacy(leg, c38)
            if sc > 0:
                best.append((sc, r.midi_filename, r.split, r.bpm, off))
        best.sort(key=lambda x: -x[0])
        for rank, (sc, fn, sp, bpm, off) in enumerate(best[:3]):
            lrows.append(dict(legacy=name, rank=rank + 1, match=round(sc, 3), midi_filename=fn, split=sp, info_bpm=bpm,
                              offset_beat=round(off, 3)))
        if best and best[0][0] >= 0.95:
            sc, fn, sp, bpm, off = best[0]
            legacy_src[name] = dict(fn=fn, split=sp, bpm=bpm, off=off, leg=leg, bpm_leg=bpm_leg)
    L = pd.DataFrame(lrows)
    L.to_csv(out / "legacy_match.csv", index=False)

    # 2) 今のレガシーの範囲で、{38} と {37, 38, 40} を比べる（同じ元ファイル・同じ時間範囲）
    cmp_rows = []
    for name, s in legacy_src.items():
        df = get(s["fn"])
        sec_per_beat = 60.0 / float(s["bpm"])
        # 元ファイルの秒に直す：レガシーの最初と最後の打の拍位置 + ずらし量（元ファイルのテンポで秒にする）
        b0, b1 = s["leg"].beat.iloc[0] + s["off"], s["leg"].beat.iloc[-1] + s["off"]
        beats = df.beat.to_numpy()
        t0 = float(np.interp(b0, beats, df.time_s)) if len(beats) else b0 * sec_per_beat
        t1 = float(np.interp(b1, beats, df.time_s)) + 1e-3
        cur = s["leg"].time_s.to_numpy()
        cmp_rows.append(dict(legacy=name, 系列="今のファイル（38 のみ）", 落とした数=np.nan, **describe(cur)))
        for pitches, lab in (((38,), "同じ範囲・38 のみ（再抽出）"), (SNARE, "同じ範囲・{37,38,40}")):
            t, _, nd, _ = excerpt(df, t0, t1, pitches)
            cmp_rows.append(dict(legacy=name, 系列=lab, 落とした数=nd, **describe(t)))
    C = pd.DataFrame(cmp_rows)

    # 3) test split のスキャン（各段の残数）
    rows = []
    for r in info.itertuples():
        df = get(r.midi_filename)
        rec = dict(midi_filename=r.midi_filename, split=r.split, info_bpm=r.bpm, style=getattr(r, "style", ""),
                   beat_type=getattr(r, "beat_type", ""), drummer=getattr(r, "drummer", ""),
                   a=False, b=False, c_reject=False, c_drop=False, d=bool(BPM_RANGE[0] <= r.bpm <= BPM_RANGE[1]))
        if len(df) == 0:
            rows.append(rec)
            continue
        t_first, t_last = float(df.time_s.min()), float(df.time_s.max())
        span = t_last - t_first
        t, vel, nd, raw_min = excerpt(df, t_first, t_first + WIN_S)
        rec.update(span_s=round(span, 2), n_snare=len(t), dropped_008=nd, raw_min_ioi_ms=round(raw_min * 1000, 1) if np.isfinite(raw_min) else np.nan)
        rec["a"] = len(t) >= 2
        rec["b"] = rec["a"] and span >= SPAN_MIN_S
        rec["c_reject"] = rec["b"] and np.isfinite(raw_min) and raw_min >= MIN_IOI - 1e-9
        rec["c_drop"] = rec["b"]
        rows.append(rec)
    S = pd.DataFrame(rows)
    S.to_csv(out / "scan_all.csv", index=False)
    T = S[S.split == "test"]
    stages = []
    for mode in ("reject", "drop"):
        m = T.a.astype(bool)
        st = {"モード": mode, "test split": len(T), "(a) スネア類": int(m.sum())}
        m = m & T.b.astype(bool)
        st["(b) 20 s 以上"] = int(m.sum())
        m = m & T[f"c_{mode}"].astype(bool)
        st["(c) 最小 IOI"] = int(m.sum())
        m = m & T.d.astype(bool)
        st["(d) bpm 60〜160"] = int(m.sum())
        stages.append(st)
        T = T.assign(**{f"pass_{mode}": m})

    # 4) 通過した抜粋の記述子と MIDI（reject・drop の両方の通過分）、レガシーの切り直し
    exs = []
    passed = T[T.pass_reject | T.pass_drop]
    for r in passed.itertuples():
        df = get(r.midi_filename)
        t_first = float(df.time_s.min())
        t, vel, nd, _ = excerpt(df, t_first, t_first + WIN_S)
        stem = Path(r.midi_filename).stem.replace("-", "_")
        fn = out / "excerpts" / f"gmd_test_{stem}_bpm{int(r.info_bpm)}.mid"
        write_mid(fn, t, vel, float(r.info_bpm), a.lead_in)
        exs.append(dict(区分="test split", midi_filename=r.midi_filename, info_bpm=r.info_bpm, style=r.style,
                        reject通過=bool(r.pass_reject), drop通過=bool(r.pass_drop), 落とした数=nd, excerpt=fn.name, **describe(t)))
    for name, s in legacy_src.items():
        df = get(s["fn"])
        t_first = float(df.time_s.min())
        t, vel, nd, raw_min = excerpt(df, t_first, t_first + WIN_S)
        fn = out / "excerpts" / f"gmd_legacy_{name.split('_bpm')[0]}_recut_bpm{int(s['bpm'])}.mid"
        write_mid(fn, t, vel, float(s["bpm"]), a.lead_in)
        exs.append(dict(区分=f"レガシー（{name}）切り直し", midi_filename=s["fn"], info_bpm=s["bpm"], style="",
                        reject通過=bool(np.isfinite(raw_min) and raw_min >= MIN_IOI), drop通過=True, 落とした数=nd,
                        excerpt=fn.name, **describe(t)))
    E = pd.DataFrame(exs)
    if len(E):
        med = float(np.median(E[E.区分 == "test split"].info_bpm)) if (E.区分 == "test split").any() else np.nan
        E["bpm中央値との差"] = (E.info_bpm - med).abs()
        E = E.sort_values(["区分", "bpm中央値との差", "midi_filename"]).reset_index(drop=True)
    E.to_csv(out / "excerpts.csv", index=False)

    show = ["区分", "midi_filename", "info_bpm", "reject通過", "drop通過", "落とした数", "n", "dur_s", "ioi_min_ms", "ioi_med_ms",
            "q_ioi_min_ms", "frac_ioi_lt100", "frac_ioi_lt120", "frac_ioi_lt145", "npvi", "bpm中央値との差"]
    md = ["# GMD の抜粋のスキャン（gmd_scan.py）", "",
          f"GMD: `{gmd}`（info.csv {len(info)} ファイル）。スネア類 = {list(SNARE)}、velocity ≥ {VEL_MIN}、"
          f"0.08 s の規則（残した打から 0.08 s 以内の後続を落とす）、窓 = ファイルの最初のオンセットから {WIN_S} s、"
          f"span < {SPAN_MIN_S} s は除外、bpm {BPM_RANGE[0]}〜{BPM_RANGE[1]}。抜粋の MIDI は最初の打を {a.lead_in} s に置いた（仮）", "",
          "## 1) レガシー曲の元ファイル（ノート 38 の拍の並びの一致、上位 3）", "", md_table(L), "",
          "一致率 0.95 以上の 1 位を元ファイルとした。" + ("" if len(legacy_src) == len(LEGACY) else
                                                     "**一致するファイルが見つからなかったレガシー曲がある。**"), "",
          "## 2) レガシーの今の範囲で、38 のみと {37,38,40} を比べる", "",
          "同じ元ファイル・同じ時間範囲（今のファイルの最初と最後の打）。q_ は評価と同じ 20 ms の格子に切り捨てた後", "",
          md_table(C[[c for c in ["legacy", "系列", "落とした数", "n", "dur_s", "ioi_min_ms", "ioi_med_ms", "q_n",
                                  "q_ioi_min_ms", "q_ioi_med_ms", "frac_ioi_lt145", "npvi"] if c in C.columns]]) if len(C) else "（同定できず）", "",
          "## 3) test split のフィルタの残数", "",
          "(c) の判定：reject = 0.08 s の規則で落とす前の最小 IOI で判定、drop = 落としたあとで判定（実質すべて通る）", "",
          md_table(pd.DataFrame(stages)), "",
          "## 4) 通過した抜粋とレガシーの切り直し（記述子）", "",
          "frac_ioi_ltNNN = IOI が NNN ms 未満の割合（145 ms は Gen2 の圧力モデルの t50。段 1 の値で差し替える）。"
          "bpm中央値との差 = test split の通過曲の bpm 中央値からの差（sim 拡張の追加曲を選ぶ順。決定ドキュメント B-5）", "",
          md_table(E[[c for c in show if c in E.columns]]) if len(E) else "（なし）", "",
          "全ファイルの判定は `scan_all.csv`、記述子の全列は `excerpts.csv`、抜粋の MIDI は `excerpts/`。"]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
