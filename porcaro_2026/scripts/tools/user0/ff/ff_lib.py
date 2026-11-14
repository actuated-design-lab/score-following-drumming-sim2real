"""ff_lib.py — 遅延補償フィードフォワード（FF）ベースラインの部品（Isaac も GPU も使わない）

FF の定義（F0、設計は claude/ の設計メモ）:
  1 打ごとに同じ形の圧力指令（プリミティブ）を、目標時刻 t_k より L だけ前に出す。
    ・待機（次の打撃の前）：DF = a_up（振り上げ）、F = 0、G = g
    ・発火（t_k − L から h 秒）：DF = 0、F = a_dn（振り下ろし）、G = g
    ・発火の途中で次の打撃の発火時刻が来たら、次の打撃が上書きする
  値 (L, a_dn, h, a_up, g) は、直前の打撃との間隔（IOI）の区分ごとに 1 組持つ。グリップ（G）は打撃中も一定で、
  RL のように 1 打の中で G を動かすことはしない（ベースラインの単純化）。

目標波形は RL の評価と同じものを使う:
  ・リズム（single_4 / single_8 / double）: env と同じ RhythmGenerator のテストモード（rhythm_generator.py をそのまま読む）
  ・MIDI: scripts/rsl_rl/play_sim_midi.py の MidiInjector と同じ処理（音の時刻を 20 ms の格子に切り捨て、幅 35 ms の
    スーパーガウシアンを畳み込む）。play_sim_midi.py は Isaac を読むので import できず、ここに写してある。変えたら両方直す
  ・目標時刻 t_k は、その目標波形から analysis/harness/strike_extract.py の extract_strikes と同じピーク検出で取る
    （採点と同じ t_k を使う）

指令は 50 Hz、列は jetson_project/signals/*.csv と同じ（time, cmd_pressure_DF, cmd_pressure_F, cmd_pressure_G、単位 MPa）。
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[4]
CTRL_DT = 0.02          # 制御周期（50 Hz）
PMAX = 0.6              # 指令の上限 [MPa]
TARGET_N = 20.0         # 目標の打撃力 [N]
MIN_FRAC = 0.05         # 採点の力のしきい値 = 0.05 × 20 N = 1 N（eval_gen2.py と同じ）
IOI_EDGES_MS = (120.0, 200.0, 320.0)   # IOI の区分の境界（4 区分）。2026-10-31 決定
RG_PATH = ROOT / "source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/user0/rhythm_generator.py"


# --------------------------------------------------------------------------- 目標波形
def _rhythm_generator_cls():
    spec = importlib.util.spec_from_file_location("ff_rhythm_generator", RG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RhythmGenerator


def rhythm_target(pattern: str, bpm: float, duration_s: float = 100.0) -> np.ndarray:
    """env のテストモードと同じ目標波形（50 Hz、[N]）。duration_s は env の episode_length_s（既定 100 s）"""
    import torch
    RG = _rhythm_generator_cls()
    n = int(round(duration_s / CTRL_DT))
    g = RG(num_envs=1, device="cpu", dt=CTRL_DT, max_episode_length=n, target_force=TARGET_N)
    g.set_test_mode(enabled=True, bpm=float(bpm), pattern=pattern)
    g.reset(torch.arange(1))
    return g.target_trajectories[0].numpy().astype(float)


def midi_target(path: str | Path, target_force: float = TARGET_N) -> tuple[np.ndarray, float]:
    """play_sim_midi.py の MidiInjector と同じ目標波形（50 Hz、[N]）と BPM"""
    import mido
    path = str(path)
    mid = mido.MidiFile(path)
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
    m = re.search(r"bpm(\d+)", path.lower())
    if m:
        tempo = mido.bpm2tempo(float(m.group(1)))
    bpm = mido.tempo2bpm(tempo)
    t, spikes = 0.0, []
    for msg in mid.merged_track:
        t += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "note_on" and msg.velocity > 0:
            spikes.append(t)
    n = int((t + 2.0) / CTRL_DT) + 100
    sp = np.zeros(n)
    for s in spikes:
        i = int(s / CTRL_DT)
        if i < n:
            sp[i] = 1.0
    width = 0.035
    sigma, radius = width / 2.0, int(width / CTRL_DT)
    tv = np.arange(-radius, radius + 1) * CTRL_DT
    kernel = target_force * np.exp(-0.5 * (tv / sigma) ** 4)
    return np.convolve(sp, kernel, mode="same"), float(bpm)


def onsets_from_target(target: np.ndarray, bpm: float, dt: float = CTRL_DT) -> np.ndarray:
    """目標時刻 t_k [s]。extract_strikes の手順 1（目標波形のピーク）と同じ"""
    t16 = 15.0 / float(bpm)
    idx, _ = find_peaks(target, height=MIN_FRAC * TARGET_N, distance=max(1, int(round(0.3 * t16 / dt))))
    return idx * dt


# --------------------------------------------------------------------------- プリミティブ
@dataclass
class Prim:
    L: float = 0.20      # 発火から目標時刻までの前倒し [s]
    a_dn: float = 0.6    # 振り下ろしの F [MPa]
    h: float = 0.10      # 振り下ろしを保つ時間 [s]
    a_up: float = 0.4    # 待機中の DF [MPa]
    g_up: float = 0.3    # 待機中の G [MPa]
    g_dn: float = 0.3    # 打撃の窓（発火 t_k − L から t_k + d_g まで）の G [MPa]
    d_g: float = 0.0     # G を g_dn から g_up に戻す時刻（目標時刻 t_k からのずれ）[s]
    # 2026-11-03：G を 1 打の中で変えられるようにした（g → g_up・g_dn・d_g）。RL は打つ直前に G を握り、当たる瞬間に抜く
    # （11/03 の実機の B・E のログ）ので、FF にも「握ってから抜く」自由度を持たせる。旧い 5 値の json は from_dict で読める（g_up = g_dn = g）

    def clipped(self) -> "Prim":
        return Prim(L=max(self.L, 0.0), a_dn=float(np.clip(self.a_dn, 0, PMAX)), h=max(self.h, CTRL_DT),
                    a_up=float(np.clip(self.a_up, 0, PMAX)), g_up=float(np.clip(self.g_up, 0, PMAX)),
                    g_dn=float(np.clip(self.g_dn, 0, PMAX)), d_g=float(self.d_g))

    @staticmethod
    def from_dict(d: dict) -> "Prim":
        """json の 1 区分 → Prim。旧い形式（g が 1 つ）は g_up = g_dn = g（1 打の中で一定 = 旧い F0 と同じ指令）"""
        d = dict(d)
        if "g" in d:
            g = d.pop("g")
            d.setdefault("g_up", g)
            d.setdefault("g_dn", g)
            d.setdefault("d_g", d.get("h", 0.10) - d.get("L", 0.20))     # G の窓 = 発火の窓（旧い F0 と同じ）
        return Prim(**d)


PRIM_KEYS = tuple(asdict(Prim()).keys())


def ioi_bins(onsets: np.ndarray, edges_ms=IOI_EDGES_MS) -> np.ndarray:
    """各打撃の区分（直前の打撃との間隔で決める。最初の打撃は一番長い区分）"""
    ioi = np.r_[np.inf, np.diff(onsets)] * 1000.0
    return np.searchsorted(np.asarray(edges_ms, float), ioi, side="right")


def synth(onsets: np.ndarray, prims: list[Prim], n_steps: int, edges_ms=IOI_EDGES_MS,
          rest: tuple[float, float, float] | None = None) -> np.ndarray:
    """指令列 (n_steps, 3) [MPa]、列は DF, F, G。

    rest: 最初の打撃の待機より前の指令（None なら最初の打撃の待機と同じ）。
    発火時刻は 20 ms の格子に丸める（制御周期より細かくは出せない）。
    """
    onsets = np.asarray(onsets, float)
    bins = ioi_bins(onsets, edges_ms)
    P = [p.clipped() for p in prims]
    cmd = np.zeros((n_steps, 3))
    if len(onsets) == 0:
        cmd[:] = rest if rest is not None else (0.0, 0.0, P[-1].g_up)
        return cmd
    fire = np.array([int(round((t - P[b].L) / CTRL_DT)) for t, b in zip(onsets, bins)])
    hold = np.array([max(1, int(round(P[b].h / CTRL_DT))) for b in bins])
    # 待機：次に来る打撃の a_up・g
    k_next = np.searchsorted(fire, np.arange(n_steps), side="right")   # 時刻 i より後で最初に発火する打撃
    for i in range(n_steps):
        k = min(k_next[i], len(onsets) - 1)
        p = P[bins[k]]
        cmd[i] = (p.a_up, 0.0, p.g_up)
    if rest is not None:
        cmd[: max(fire[0], 0)] = rest
    # 発火（後の打撃が前の打撃の残りを上書きする）
    for k in range(len(onsets)):
        a, b = fire[k], fire[k] + hold[k]
        if k + 1 < len(onsets):
            b = min(b, fire[k + 1])
        a, b = max(a, 0), min(b, n_steps)
        if b > a:
            p = P[bins[k]]
            cmd[a:b, 0] = 0.0
            cmd[a:b, 1] = p.a_dn
    # G：待機は次の打撃の g_up（上で入れた）。打撃 k の窓（発火 t_k − L から t_k + d_g まで）は g_dn。
    # 窓は次の打撃の発火で打ち切る（後の打撃が前の打撃の残りを上書きするのは DF・F と同じ）。使う情報は打撃 k の時刻だけ（因果的）
    for k in range(len(onsets)):
        p = P[bins[k]]
        a = fire[k]
        b = int(round((onsets[k] + p.d_g) / CTRL_DT))
        if k + 1 < len(onsets):
            b = min(b, fire[k + 1])
        a, b = max(a, 0), min(b, n_steps)
        if b > a:
            cmd[a:b, 2] = p.g_dn
    return cmd


def to_signal_df(cmd: np.ndarray) -> pd.DataFrame:
    """jetson_project/signals/*.csv と同じ書式"""
    return pd.DataFrame({"time": np.round(np.arange(len(cmd)) * CTRL_DT, 4),
                         "cmd_pressure_DF": cmd[:, 0], "cmd_pressure_F": cmd[:, 1], "cmd_pressure_G": cmd[:, 2]})


def prims_from_vec(x, n_bins: int = len(IOI_EDGES_MS) + 1) -> list[Prim]:
    """最適化のベクトル（区分ごとに PRIM_KEYS の順）→ Prim の並び"""
    x = np.asarray(x, float).reshape(n_bins, len(PRIM_KEYS))
    return [Prim(**dict(zip(PRIM_KEYS, r))).clipped() for r in x]


def prims_to_vec(prims: list[Prim]) -> np.ndarray:
    return np.array([[getattr(p, k) for k in PRIM_KEYS] for p in prims]).ravel()
