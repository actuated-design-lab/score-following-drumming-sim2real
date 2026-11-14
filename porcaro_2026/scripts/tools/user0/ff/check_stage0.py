"""check_stage0.py — FF の段 0 の確認（Isaac も GPU も使わない。torch・mido・scipy は要る）

  1) 目標：評価 4 条件と学習分布 (a) の目標時刻と IOI の区分（RL の評価と同じ目標波形から取れているか）
  2) 圧力：Gen2 の圧力チャネル（itv.py の shaped ＋通信の遅れ 8 ms）のステップ応答と、0↔0.6 MPa の交互指令での振れ
  3) 指令：F0 のプリミティブを並べた指令の例（初期値。最適化前）と、その圧力
  4) 採点：jetson_project の実機の採点（strike_metrics.py）が、前の研究の実機ログの集計（summary_1N.csv）を再現するか。
     あわせてそのときの RL の力の誤差（20 N に対して）を、FF と同じ式（ff_score.summarize）で出す

  python scripts/tools/user0/ff/check_stage0.py --jetson ../jetson_project --out data/user0/ff/check
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ff_lib as FL  # noqa: E402
import ff_score as FS  # noqa: E402

ROOT = FL.ROOT
ITV = ROOT / "source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/common/actions/itv.py"
MIDI = {"gmd_03_high_bpm138": ROOT / "data/common/midi/gmd_03_high_bpm138.mid",
        "gmd_04_extreme_bpm170": ROOT / "data/common/midi/gmd_04_extreme_bpm170.mid"}
EVAL_RHYTHM = [("double", 160), ("single_8", 120)]          # eval_gen2.py と同じ
TRAIN_PATTERNS, TRAIN_BPMS = ("single_4", "single_8", "double"), (60, 80, 100, 120, 140, 160)   # rhythm_generator.py
PHYS_DT, CMD_DELAY = 0.005, 0.008


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 1) 目標
def targets_section():
    rows, conds = [], {}
    for pat, bpm in EVAL_RHYTHM:
        tr = FL.rhythm_target(pat, bpm)
        conds[f"{pat}_{bpm}"] = (FL.onsets_from_target(tr, bpm), bpm, len(tr))
    for k, p in MIDI.items():
        tr, bpm = FL.midi_target(p)
        conds[k] = (FL.onsets_from_target(tr, bpm), bpm, len(tr))
    for k, (on, bpm, n) in conds.items():
        ioi = np.diff(on) * 1000
        b = np.bincount(FL.ioi_bins(on), minlength=4)
        rows.append(dict(条件=k, 打撃数=len(on), 最初の打撃_s=round(on[0], 2), IOI最小_ms=round(ioi.min()),
                         IOI中央値_ms=round(np.median(ioi)), 区分0=b[0], 区分1=b[1], 区分2=b[2], 区分3=b[3]))
    tr_rows = []
    for pat in TRAIN_PATTERNS:
        for bpm in TRAIN_BPMS:
            on = FL.onsets_from_target(FL.rhythm_target(pat, bpm), bpm)
            b = np.bincount(FL.ioi_bins(on), minlength=4)
            tr_rows.append(dict(パターン=pat, BPM=bpm, 打撃数=len(on),
                                IOI_ms="/".join(str(int(round(x))) for x in sorted(set(np.round(np.diff(on) * 1000)))),
                                区分0=b[0], 区分1=b[1], 区分2=b[2], 区分3=b[3]))
    return pd.DataFrame(rows), pd.DataFrame(tr_rows), conds


# --------------------------------------------------------------------------- 2) 圧力
def load_itv():
    spec = importlib.util.spec_from_file_location("ff_itv", ITV)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pressure(itv, cmd_ctrl: np.ndarray, scale: float = 1.0, delay: float = CMD_DELAY) -> np.ndarray:
    """指令 (n, C) 50 Hz → 圧力 (n*4, C) 200 Hz。env と同じ順（通信の遅れ → shaped チャネル）"""
    import torch
    cmd_ctrl = np.atleast_2d(cmd_ctrl.T).T
    C = cmd_ctrl.shape[1]
    chs = []
    for _ in range(C):
        ch = itv.ShapedOrificeChannel(PHYS_DT)
        ch.reset(1, "cpu")
        ch.tau[:] *= scale
        ch.L[:] *= scale
        chs.append(ch)
    dl = itv.EnvDelay(PHYS_DT, delay + 0.01)
    dl.reset((1, C), "cpu")
    out = []
    for u in np.repeat(cmd_ctrl, int(round(FL.CTRL_DT / PHYS_DT)), axis=0):
        x = dl.step(torch.tensor(u[None, :], dtype=torch.float32), torch.full((1, C), delay))
        out.append([float(ch.step(x[:, c])) for c, ch in enumerate(chs)])
    return np.array(out)


def pressure_section(itv):
    def tcross(p, t0, lvl, up=True):
        t = np.arange(len(p)) * PHYS_DT
        i = np.where((t >= t0) & ((p >= lvl) if up else (p <= lvl)))[0]
        return round((t[i[0]] - t0) * 1000) if len(i) else np.nan
    rows = []
    for lo, hi in [(0.0, 0.6), (0.0, 0.3), (0.3, 0.6)]:
        for sc in (0.8, 1.0, 1.2):
            c = np.r_[np.full(50, lo), np.full(50, hi), np.full(50, lo)]
            p = pressure(itv, c, sc)[:, 0]
            d = hi - lo
            rows.append(dict(指令=f"{lo}↔{hi}", τL倍率=sc,
                             上げ_t10_t50_t90_ms="/".join(str(tcross(p, 1.0, lo + f * d)) for f in (.1, .5, .9)),
                             下げ_t10_t50_t90_ms="/".join(str(tcross(p, 2.0, hi - f * d, False)) for f in (.1, .5, .9))))
    sq = []
    for half in (2, 3, 4, 5, 8, 10):
        c = np.tile(np.r_[np.full(half, 0.6), np.zeros(half)], 40)
        p = pressure(itv, c)[:, 0]
        seg = p[len(p) // 2:]
        sq.append(dict(半周期_ms=half * 20, 周波数_Hz=round(1 / (2 * half * FL.CTRL_DT), 1),
                       圧力の最小_MPa=round(seg.min(), 3), 圧力の最大_MPa=round(seg.max(), 3)))
    return pd.DataFrame(rows), pd.DataFrame(sq)


# --------------------------------------------------------------------------- 3) 指令の例
def synth_section(itv, conds, out: Path):
    prims = [FL.Prim() for _ in range(len(FL.IOI_EDGES_MS) + 1)]      # 初期値（最適化前）
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=False)
    for ax, key, win in zip(axes, ("double_160", "gmd_03_high_bpm138"), ((1.2, 4.2), (2.0, 5.0))):
        on, bpm, n = conds[key]
        cmd = FL.synth(on, prims, n)
        i0, i1 = int(win[0] / FL.CTRL_DT), int(win[1] / FL.CTRL_DT)
        p = pressure(itv, cmd[i0 - 50:i1])[200:]          # 1 s 先に回して、窓の頭の過渡を落とす
        t = np.arange(i0, i1) * FL.CTRL_DT
        tp = win[0] + np.arange(len(p)) * PHYS_DT
        for c, (name, col) in enumerate((("DF", "#0072B2"), ("F", "#D55E00"), ("G", "#009E73"))):
            ax.step(t, cmd[i0:i1, c], where="post", color=col, lw=1, alpha=0.5, label=f"指令 {name}")
            ax.plot(tp, p[:, c], color=col, lw=2, label=f"圧力 {name}")
        for x in on[(on >= win[0]) & (on <= win[1])]:
            ax.axvline(x, color="k", lw=0.8, ls=":")
        ax.set_title(f"{key}：F0 の指令（初期値、最適化前）と Gen2 の圧力（点線 = 目標時刻）")
        ax.set_ylabel("MPa")
        ax.legend(ncol=6, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("時間 [s]")
    plt.tight_layout()
    f = out / "fig_synth_initial.png"
    plt.savefig(f, dpi=110)
    plt.close()
    return prims, f.name


# --------------------------------------------------------------------------- 4) 採点
def scoring_section(jetson: Path, out: Path):
    sm = jetson / "analysis/strike_metrics.py"
    ral = jetson / "data/user0/ral2026"
    ref = ral / "summary/summary_1N.csv"
    if not (sm.exists() and ref.exists()):
        return None, None, "jetson_project が見つからないので飛ばした"
    sum_csv = out / "_ral_summary_1N.csv"
    subprocess.run([sys.executable, "-I", str(sm), str(ral), "--force_threshold", "1.0", "--summary", str(sum_csv)],
                   check=True, capture_output=True, text=True)
    a, b = pd.read_csv(sum_csv), pd.read_csv(ref)
    a["k"], b["k"] = a.file.str.split("/").str[-1], b.file.str.split("/").str[-1]
    m = a.merge(b, on="k", suffixes=("_new", "_ref"))
    repro = dict(再計算した走行=len(a), 参照の走行=len(b), 突き合わせた走行=len(m),
                 成功率の差の最大=float((m.success_rate_new - m.success_rate_ref).abs().max()))
    # 打点は summary_1N.csv と同じ走行の strikes_1N.csv（打点数の合計が summary_1N の n_target_strikes の合計と一致する）
    st = pd.read_csv(ral / "summary/strikes_1N.csv")
    assert len(st) == int(b.n_target_strikes.sum())
    rows = []
    for (mdl, midi), g in st[st.midi.isin(["test_double_bpm160.mid", "gmd_03_high_bpm138.mid"])
                             & st.model.isin(["B", "E"])].groupby(["model", "midi"]):
        rows.append(dict(model=mdl, 条件=midi.replace(".mid", ""), **FS.fmt_row(FS.summarize(g))))
    sum_csv.unlink(missing_ok=True)
    return repro, pd.DataFrame(rows), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetson", default="../jetson_project")
    ap.add_argument("--out", default="data/user0/ff/check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    itv = load_itv()
    ev, tr, conds = targets_section()
    pr, sq = pressure_section(itv)
    prims, fig = synth_section(itv, conds, out)
    repro, rl_force, note = scoring_section(Path(a.jetson), out)

    L = ["# FF 段 0 の確認（GPU なし）", "",
         f"IOI の区分：区分0 < {FL.IOI_EDGES_MS[0]:.0f} ms ≤ 区分1 < {FL.IOI_EDGES_MS[1]:.0f} ms ≤ 区分2 < "
         f"{FL.IOI_EDGES_MS[2]:.0f} ms ≤ 区分3（直前の打撃との間隔。最初の打撃は区分3）。`ff_lib.py`", "",
         "## 1) 目標時刻（RL の評価と同じ目標波形から）", "",
         "評価 4 条件（リズムは env のテストモード、100 s のエピソード 1 本分。MIDI は MidiInjector と同じ処理）", "",
         md_table(ev), "", "学習分布 (a)（4 小節のうち 2〜4 小節目）", "", md_table(tr), "",
         "## 2) Gen2 の圧力チャネル（itv.py の shaped ＋通信の遅れ 8 ms、物理 5 ms 刻み、指令は 20 ms ごと）", "",
         "ステップ応答（時間は指令を変えてから。τL倍率 = DR の τ・L の倍率）", "", md_table(pr), "",
         "0 と 0.6 MPa を交互に出したときの圧力の振れ（後半の定常）", "", md_table(sq), "",
         "## 3) F0 の指令の例（初期値、最適化前）", "",
         "値（全区分同じ）：" + ", ".join(f"{k} = {v}" for k, v in vars(prims[0]).items()) + "（L・h は s、圧力は MPa）", "",
         f"![]({fig})", "",
         "## 4) 採点", ""]
    if repro is None:
        L.append(note)
    else:
        L += ["jetson_project の `analysis/strike_metrics.py`（1 N）で前の研究の実機ログ（`data/user0/ral2026/`）を採点し直し、"
              "`summary/summary_1N.csv` と突き合わせた", "",
              md_table(pd.DataFrame([repro])), "",
              "前の研究の実機の RL（B・E、`summary_1N.csv` の走行の打点 `summary/strikes_1N.csv`）を、FF と同じ式（`ff_score.summarize`）でまとめたもの。"
              "成功 + 打撃なし + 遅い + 早い = 1。力は打った打点だけ（dF_rmse_all は打っていない目標を 0 N として数える）", "",
              md_table(rl_force)]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
