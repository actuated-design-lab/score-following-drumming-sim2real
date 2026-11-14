"""okui_onsets.py — 打楽器の演奏動画（部屋録音）から打点列を起こす。近接打点（連打・フラム・ドラッグ）を潰さない版

奥井先生の Sing Sing Sing（片手テイク seg2・seg3、両手テイク seg1）用。処理はすべてローカル（WS / 手元の PC）で行い、
音声・MIDI を外部のサービスに送らない。必要なもの：python3（numpy、scipy、matplotlib）、ffmpeg（動画から音声を取り出すとき）。

検出（take ごと。奥井先生の録音で決めた。2026-11-11）
  1. 音声をモノラル 44.1 kHz にする（動画なら ffmpeg）。--start/--end で元動画から区間を切り出せる
  2. 高域（2 kHz 以上）の包絡（1 ms 平滑）を dB にし、直前 15 ms の最小からの跳ね上がりを見る。打撃は 18〜60 dB、
     打撃の間の減衰のゆらぎは 4〜14 dB（14〜18 dB は空いている）→ 15 dB 以上を打撃、10〜15 dB を「要確認」の候補。山の最小間隔 20 ms
     （初版のスペクトラルフラックスは減衰のゆらぎを拾い、片手 A が 60 打以上になった。実際は 32 打）
  3. 冒頭のカウント（スティックどうしを打つ音）は、低域（50–300 Hz）の山 − 高域の山 が −30〜−50 dB（太鼓は +7〜+28 dB）。
     −20 dB 未満をクリックとして外す。クリックから 40 ms 以内の打撃も、低域 − 高域 が 0 dB 未満ならクリック（2 本のスティックの当たりのずれ）
  4. 演奏の範囲 = クリックでない最初の打撃〜最大の −30 dB 以上の最後の打撃 ＋ 0.4 s。レベルが最大の −45 dB 未満の打撃（候補は −55 dB）は外す
  5. 時刻の精密化：高域の包絡で、山（検出の前 12〜後 8 ms の最大）と谷（山の前 20 ms の最小）の中間を最後に越えた時刻
  6. ベロシティ：高域の山 [dB] を、演奏中の打撃の 95 パーセンタイルを 0 dB として 1–127 に写す（−30 dB → 1）。部屋録音の相対値
  7. 太鼓の胴鳴りのうなりで、低・中域の包絡には打撃でない山が多数ある。弱い 2 打目は、音だけでは確実に分けられないので候補として出す
格子（「合うものだけ寄せる」）
  - テンポは --bpm（既定 200）。拍の周期を ±1.5 % で、位相を 1 ms 刻みで探し、強い打点が 3 連の格子（拍/3 = 100 ms）に
    いちばん多く乗る周期と位相を当てる
  - 寄せる順：3 連の格子（±tol、既定 12 ms）→ 6 連の格子（拍/6 = 50 ms、連打用）→ 寄せない
  - 寄せなかった打点は時刻を生のまま残し、理由を付ける（次の打点の x ms 前 = フラムの装飾音の候補、前の打点から x ms = 連打・
    ドラッグの候補、格子から x ms ずれ）
合成（2 テイク）：後のテイクを、最初の打点が先のテイクの最初の打点にそろうようにずらして重ねる。|Δt| ≤ --unison_ms（既定 40 ms。片手テイクどうしの同じ拍のずれが 24〜33 ms あり、両手テイクにはその間隔の 2 打がないため）
  の 2 打は同時打点として 1 打にまとめ（時刻は強い方、ベロシティは大きい方、両方の元の時刻を記録）、それ以外は両方残す。
  間隔が 80 ms 未満の打点の組（ロボットの片腕の学習・評価の下限より短い）は印を付けるが落とさない
照合：両手テイク（seg1）の打点列と、合成列を、ずらし（±2 s、1 ms 刻み）を探して ±30 ms で 1 対 1 に対応させ、カバー率と差を出す
MIDI：PPQ 480、♩ = 200（--bpm）、ノート 41、チャンネル 10（index 9）、冒頭 2.0 s の空白。時刻は寄せた打点は格子の時刻、
  寄せなかった打点は生の時刻（0.625 ms 単位に丸め）。ノートの長さは 30 tick（次の打点の手前で切る）。
  <name>.mid は信頼度 high・mid だけ、<name>_with_candidates.mid は要確認の候補も入れた版。原点は最初の打撃にいちばん近い 3 連の格子点

  python3 okui_onsets.py take seg2.mp4 --name handA --out out/
  python3 okui_onsets.py take seg3.mp4 --name handB --out out/
  python3 okui_onsets.py take seg1.mp4 --name both --out out/
  python3 okui_onsets.py combine out/handA_onsets.csv out/handB_onsets.csv --name combined --out out/
  python3 okui_onsets.py compare out/combined_onsets.csv out/both_onsets.csv --out out/
  （元動画から切り出すとき：take sss_okui.mp4 --start 15.5 --end 29.9 --name handA）
  （Muse 初版との差：take ... --prev segments_onsets.json --prev_key seg2）
出力：<out>/<name>_onsets.csv、<name>.mid、<name>_spec.png（2 s ごとのスペクトログラムと打点）、<name>_close.png（近接打点の拡大）、
      <name>_report.md、combined_*、compare_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.ndimage import maximum_filter, maximum_filter1d, minimum_filter1d, percentile_filter, uniform_filter1d
from scipy.signal import butter, sosfiltfilt, stft

SR = 44100
NFFT, HOP = 1024, 128
BANDS = {"low": (60, 250, 0.2), "mid": (250, 2000, 0.3), "high": (2000, 8000, 0.5)}
MIN_SEP_S = 0.018
MU = 3
NOTE, CH, PPQ, LEAD_S = 41, 9, 480, 2.0


# ---------------------------------------------------------------- 音声
def load_audio(path, start=None, end=None):
    if path.lower().endswith(".wav") and start is None and end is None:
        sr, x = wavfile.read(path)
    else:
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "a.wav")
            cmd = ["ffmpeg", "-v", "error", "-y"]
            if start is not None:
                cmd += ["-ss", f"{start}"]
            cmd += ["-i", path]
            if end is not None:
                cmd += ["-t", f"{end - (start or 0.0)}"]
            cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", w]
            subprocess.run(cmd, check=True)
            sr, x = wavfile.read(w)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(np.asarray(x).dtype, np.integer) or np.abs(x).max() > 2:
        x = x / 32768.0
    if sr != SR:
        raise SystemExit(f"サンプリング周波数 {sr} Hz（{SR} Hz を想定。ffmpeg 経由で読む）")
    return x


def envelope(x, lo, smooth_s=0.001):
    sos = butter(4, lo, btype="highpass", fs=SR, output="sos")
    y = np.abs(sosfiltfilt(sos, x))
    return uniform_filter1d(y, max(1, int(smooth_s * SR)))


# ---------------------------------------------------------------- 検出
# 奥井先生の録音（部屋、iPhone）の実データで決めた（2026-11-11）：
#  - 打撃は高域（2 kHz 以上）の包絡が 15 ms のうちに 18〜60 dB 跳ね上がる。打撃の間の減衰のゆらぎは 4〜14 dB で、14〜18 dB は空いている
#    → 跳ね上がり 15 dB 以上を打撃、10〜15 dB を「要確認」の候補とする（スペクトラルフラックスでは減衰のゆらぎを拾い、A が 60 打以上になった）
#  - 冒頭のカウント（スティックどうしを打つ音）は、高域は打撃と同じくらい大きいが低域（50–300 Hz）がほとんどない：
#    低域の山 − 高域の山 が −30〜−50 dB（太鼓の打撃は +7〜+28 dB）。−20 dB 未満を「クリック」として打点列から外す
#  - 太鼓の胴鳴りのうなりで、低・中域の包絡には打撃でない山が多数ある。2 打目の検出には使わない
RISE_STROKE, RISE_CAND, RISE_WIN_S, SEP_S = 15.0, 10.0, 0.015, 0.020
FLOOR_STROKE_DB, FLOOR_CAND_DB = -45.0, -55.0      # 取り込む最小のレベル（take の最大に対する高域の山）
CLICK_DB = -20.0                                   # 低域 − 高域 がこれ未満ならクリック


def band_env_db(x, lo, hi, sm=0.003):
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return 20 * np.log10(uniform_filter1d(np.abs(sosfiltfilt(sos, x)), max(1, int(sm * SR))) + 1e-7)


def refine(x, env_hf, env_all, t_peak):
    """高域の包絡で時刻を精密化し、強さと鋭さを出す。
    山 = 検出の時刻の前 12 ms〜後 8 ms の高域包絡の最大。谷 = 山の前 20 ms の最小（前の打点の減衰の上にある弱い打点でも、
    立ち上がりの始まりを取る）。時刻 = 山の前で、包絡が 谷 ＋ 0.5 ×（山 − 谷）を最後に下回った点"""
    rows = []
    n = len(x)
    for tp in t_peak:
        a, b = int((tp - 0.012) * SR), int((tp + 0.008) * SR)
        a, b = max(a, 1), min(b, n - 1)
        if b - a < 10:
            rows.append((tp, np.nan, np.nan, np.nan, np.nan))
            continue
        ip = a + int(np.argmax(env_hf[a:b]))
        c = max(0, ip - int(0.020 * SR))
        ib = c + int(np.argmin(env_hf[c:ip + 1]))
        pk, lo = env_hf[ip], env_hf[ib]
        half = lo + 0.5 * (pk - lo)
        below = np.flatnonzero(env_hf[ib:ip + 1] < half)
        i50 = ib + (int(below[-1]) + 1 if len(below) else 0)
        lv10, lv90 = lo + 0.1 * (pk - lo), lo + 0.9 * (pk - lo)
        seg = env_hf[ib:ip + 1]
        i10 = ib + int(np.argmax(seg >= lv10))
        i90 = ib + int(np.argmax(seg >= lv90))
        post = slice(i50, min(n, i50 + int(0.025 * SR)))
        lvl = 20 * np.log10(max(env_hf[post].max() + env_all[post].max(), 1e-9))
        base = 20 * np.log10(max(lo + env_all[ib], 1e-9))
        rows.append((i50 / SR, lvl, lvl - base, (i90 - i10) / SR * 1000, pk))
    return pd.DataFrame(rows, columns=["t_raw", "level_db", "rise_db", "rise_ms", "env_peak"])


def detect(x, min_level_db=None):
    env = envelope(x, 2000, 0.001)
    edb = uniform_filter1d(20 * np.log10(env + 1e-7), max(1, int(0.0005 * SR)))
    n = int(RISE_WIN_S * SR)
    r = edb - minimum_filter1d(edb, n, origin=n // 2 - 1)        # 直前 15 ms の最小からの跳ね上がり [dB]
    sep = int(SEP_S * SR)
    pk = np.flatnonzero((r == maximum_filter1d(r, 2 * sep + 1)) & (r >= RISE_CAND))
    top = edb.max()
    low = band_env_db(x, 50, 300)
    rows = []
    for i in pk:
        lvl = edb[i:i + int(0.02 * SR)].max() - top
        lmh = low[i:i + int(0.10 * SR)].max() - (edb[i:i + int(0.02 * SR)].max())
        kind = "stroke" if r[i] >= RISE_STROKE else "cand"
        if lvl < (FLOOR_STROKE_DB if kind == "stroke" else FLOOR_CAND_DB):
            continue
        if kind == "stroke" and lmh < CLICK_DB:
            kind = "click"
        rows.append(dict(i=i, rise_db=r[i], level_rel_db=lvl, low_minus_hf_db=lmh, kind=kind))
    D = pd.DataFrame(rows)
    env_all = uniform_filter1d(np.abs(x), int(0.001 * SR))
    R = refine(x, env, env_all, D.i.to_numpy() / SR)
    D = pd.concat([D.reset_index(drop=True), R[["t_raw", "rise_ms", "env_peak"]]], axis=1).drop(columns="i")
    D = D.sort_values("t_raw").reset_index(drop=True)
    # クリックの対（2 本のスティックがわずかにずれて当たる）の片方は、低域が少し出て打撃に見えることがある：
    # クリックから 40 ms 以内で、低域 − 高域 が 0 dB 未満の打撃もクリックにする
    ct = D.t_raw[D.kind == "click"].to_numpy()
    if len(ct):
        near = np.array([np.min(np.abs(ct - t)) < 0.040 for t in D.t_raw])
        D.loc[near & (D.kind == "stroke") & (D.low_minus_hf_db < 0), "kind"] = "click"
    # 弱すぎる打撃（--min_level_db 未満）は「要確認」の候補にする。片手テイクでは、本人の聞き取り（B は 19 打）と合うのが −25 dB
    # （B の −29・−40 dB の 2 打は聞こえない。ほかの打撃は A・B とも −12 dB 以上で、間が空いている）
    if min_level_db is not None:
        D.loc[(D.kind == "stroke") & (D.level_rel_db - D.level_rel_db[D.kind == "stroke"].quantile(0.95) < min_level_db), "kind"] = "cand"
    # 演奏の範囲：クリックでない最初の打撃 〜 −30 dB 以上の最後の打撃 ＋ 0.4 s。外の候補は数えるだけ
    st = D[(D.kind == "stroke")]
    t_on, t_off = (st.t_raw.min(), st[st.level_rel_db >= -30].t_raw.max() + 0.4) if len(st) else (0, 0)
    ck = D.t_raw[D.kind == "click"]
    if len(ck):
        t_on = min(t_on, ck.max() + 0.15)          # カウントのあとの弱い打撃（候補）も範囲に入れる
    D["in_perf"] = (D.t_raw >= t_on - 0.05) & (D.t_raw <= t_off)
    pk_db = D.level_rel_db
    ref = np.percentile(pk_db[(D.kind == "stroke") & D.in_perf], 95) if ((D.kind == "stroke") & D.in_perf).any() else 0.0
    D["rel_db"] = pk_db - ref
    D["velocity"] = np.clip(np.round(127 + D.rel_db * (126 / 30)), 1, 127).astype(int)
    D["conf"] = np.where(D.kind == "stroke", np.where(D.rise_db >= 18, "high", "mid"), "low")
    return D, dict(t=np.arange(len(x))[::32] / SR, edb=edb[::32] - top)


# ---------------------------------------------------------------- 格子
def fit_grid(t, bpm, tol_s, strong):
    """強い打点が 3 連の格子に最も多く乗る（周期, 位相）"""
    beat0 = 60.0 / bpm
    ts = t[strong] if strong.sum() >= 4 else t
    best = (-1, 0.0, beat0, 0.0)
    for beat in np.linspace(beat0 * 0.985, beat0 * 1.015, 61):
        g = beat / 3
        for ph in np.arange(0, g, 0.001):
            r = (ts - ph) / g
            d = np.abs(r - np.round(r)) * g
            n = int((d <= tol_s).sum())
            sc = n - d[d <= tol_s].mean() if n else 0
            if sc > best[0]:
                best = (sc, ph, beat, float(np.median(d)))
    _, ph, beat, med = best
    return beat, ph, med


def snap(D, bpm, tol_ms):
    t = D.t_raw.to_numpy()
    strong = (D.conf == "high").to_numpy() & (D.rel_db > -12).to_numpy()
    beat, ph, _ = fit_grid(t, bpm, tol_ms / 1000, strong)
    out = []
    for i, ti in enumerate(t):
        res = None
        for div, name in ((3, "triplet"), (6, "sextuplet")):
            g = beat / div
            k = np.round((ti - ph) / g)
            tg = ph + k * g
            if abs(ti - tg) * 1000 <= tol_ms:
                res = (name, tg, (ti - tg) * 1000)
                break
        prev = (ti - t[i - 1]) * 1000 if i > 0 else np.nan
        nxt = (t[i + 1] - ti) * 1000 if i + 1 < len(t) else np.nan
        if res is None:
            r3 = (ti - ph) / (beat / 3)
            off = (r3 - np.round(r3)) * beat / 3 * 1000
            if not np.isnan(nxt) and nxt <= 45:
                why = f"次の打点の {nxt:.0f} ms 前（フラムの装飾音の候補）"
            elif not np.isnan(prev) and prev <= 110:
                why = f"前の打点から {prev:.0f} ms（連打・ドラッグの候補）"
            else:
                why = f"3 連の格子から {off:+.0f} ms"
            out.append(dict(grid="none", t_snap=ti, snap_err_ms=np.nan, reason=why))
        else:
            out.append(dict(grid=res[0], t_snap=res[1], snap_err_ms=res[2], reason=""))
        out[-1].update(ioi_prev_ms=prev, ioi_next_ms=nxt)
    S = pd.DataFrame(out)
    D = pd.concat([D.reset_index(drop=True), S], axis=1)
    return D, dict(beat_s=beat, bpm_fit=60 / beat, phase_s=ph)


# ---------------------------------------------------------------- MIDI
def vlq(n):
    b = [n & 0x7F]
    n >>= 7
    while n:
        b.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(b)


def write_midi(path, times, vels, bpm):
    """形式 0、PPQ 480、テンポ bpm、ノート 41、ch 10（index 9）。times は最初の打点を 0 とした秒。冒頭 2.0 s の空白"""
    spt = 60.0 / bpm / PPQ
    ticks = np.round((np.asarray(times) + LEAD_S) / spt).astype(int)
    ev = []
    for j, (tk, v) in enumerate(zip(ticks, vels)):
        nxt = ticks[j + 1] if j + 1 < len(ticks) else tk + 10 ** 6
        dur = max(1, min(30, nxt - tk - 1))
        ev.append((tk, 1, bytes([0x90 | CH, NOTE, int(np.clip(v, 1, 127))])))
        ev.append((tk + dur, 0, bytes([0x80 | CH, NOTE, 0])))
    ev.sort(key=lambda e: (e[0], e[1]))
    tempo = int(round(60e6 / bpm))
    trk = b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big") + b"\x00\xff\x58\x04\x04\x02\x18\x08"
    last = 0
    for tk, _, msg in ev:
        trk += vlq(tk - last) + msg
        last = tk
    # 最後の音のあとに 2 s の空白を置いてからトラックを終える（すぐ終えると、再生ソフトが最後の音の響きを切る）
    trk += vlq(int(round(2.0 / spt))) + b"\xff\x2f\x00"
    with open(path, "wb") as fo:
        fo.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, PPQ) + b"MTrk" + struct.pack(">I", len(trk)) + trk)


# ---------------------------------------------------------------- 図
def plot_take(path, x, D, aux, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, t, Z = stft(x, fs=SR, nperseg=2048, noverlap=2048 - 256)
    fm = f <= 6000
    S = 20 * np.log10(np.abs(Z[fm]) + 1e-7)
    dur = len(x) / SR
    n = int(np.ceil(dur / 3.0))
    fig, axs = plt.subplots(n, 1, figsize=(15, 2.4 * n), squeeze=False)
    col = {"stroke": "#1baf7a", "cand": "#d1495b", "click": "#8a8984"}
    for r in range(n):
        ax = axs[r, 0]
        a, b = 3.0 * r, 3.0 * (r + 1)
        m = (t >= a) & (t <= b)
        ax.pcolormesh(t[m], f[fm], S[:, m], shading="auto", cmap="magma", vmin=S.max() - 70, vmax=S.max())
        ax2 = ax.twinx()
        mm = (aux["t"] >= a) & (aux["t"] <= b)
        ax2.plot(aux["t"][mm], aux["edb"][mm], color="#7fd3ff", lw=0.5)
        ax2.set_ylim(-80, 5)
        ax2.set_yticks([])
        for _, d in D[(D.t_raw >= a) & (D.t_raw <= b)].iterrows():
            ls = "-" if (d.kind == "stroke" and d.in_perf) else ":"
            ax.axvline(d.t_raw, color=col[d.kind], lw=1.0, ls=ls)
            if d.kind != "click":
                ax.text(d.t_raw, 5600, str(int(d.velocity)) if d.kind == "stroke" else "?", fontsize=7, color=col[d.kind], ha="center")
        ax.set_xlim(a, b)
        ax.set_ylabel("Hz")
    axs[0, 0].set_title(f"{title}  緑=打撃（実線=演奏の範囲）　赤=要確認の候補　灰=カウントのクリック　数字=ベロシティ　水色=高域の包絡 dB")
    axs[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_close(path, x, D, title, max_n=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env = envelope(x, 2000)
    sel = D.index[(D.ioi_next_ms <= 120)].tolist()[:max_n]
    if not sel:
        return
    nc = 4
    nr = int(np.ceil(len(sel) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(14, 1.8 * nr), squeeze=False)
    for k, i in enumerate(sel):
        ax = axs[k // nc, k % nc]
        t0 = D.t_raw[i] - 0.08
        a, b = int(max(0, t0) * SR), int(min(len(x) / SR, t0 + 0.30) * SR)
        tt = np.arange(a, b) / SR
        ax.plot(tt, env[a:b], lw=0.6, color="#0b0b0b")
        for _, d in D[(D.t_raw >= tt[0]) & (D.t_raw <= tt[-1])].iterrows():
            ax.axvline(d.t_raw, color="#d1495b" if d.grid == "none" else "#1baf7a", lw=0.8)
        ax.set_title(f"{D.t_raw[i]:.3f} s → +{D.ioi_next_ms[i]:.0f} ms", fontsize=7)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
    for k in range(len(sel), nr * nc):
        axs[k // nc, k % nc].axis("off")
    fig.suptitle(f"{title}：次の打点まで 120 ms 以下の組（高域包絡）", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 前の版との差
def load_prev(path, key):
    js = json.load(open(path, encoding="utf-8"))
    obj = js.get(key, js) if isinstance(js, dict) and key else js

    def find(o):
        if isinstance(o, list) and o and all(isinstance(v, (int, float)) for v in o):
            return o
        if isinstance(o, list) and o and isinstance(o[0], dict):
            for kk in ("t", "time", "t_raw", "onset", "onset_s", "time_s"):
                if kk in o[0]:
                    return [d[kk] for d in o]
        if isinstance(o, dict):
            for kk in ("onsets", "times", "onset_times", "t"):
                if kk in o:
                    return find(o[kk])
        return None
    v = find(obj)
    if v is None:
        raise SystemExit(f"{path}（key {key}）から時刻の列を見つけられない。構造を確かめて --prev_key を指定")
    return np.asarray(v, float)


def match(a, b, tol):
    """a, b（昇順）を ±tol で 1 対 1 に（近い順の貪欲）。返り値：[(i, j)]"""
    pairs = []
    for i, ti in enumerate(a):
        j = np.searchsorted(b, ti)
        for jj in (j - 1, j):
            if 0 <= jj < len(b) and abs(b[jj] - ti) <= tol:
                pairs.append((abs(b[jj] - ti), i, jj))
    pairs.sort()
    ui, uj, out = set(), set(), []
    for _, i, j in pairs:
        if i not in ui and j not in uj:
            ui.add(i)
            uj.add(j)
            out.append((i, j))
    return sorted(out)


# ---------------------------------------------------------------- コマンド
def cmd_take(a):
    x = load_audio(a.input, a.start, a.end)
    D, aux = detect(x, a.min_level_db)
    P = D[D.in_perf & (D.kind != "click")].reset_index(drop=True)
    P, g = snap(P, a.bpm, a.tol_ms)
    # 1 拍目の基準：カウントのクリックがあれば「最後のクリック ＋ 1 拍」（どのテイクも同じカウントのあとに始まるので、
    # 最初の打撃が弱くて外れたテイクでもそろう）。なければ最初の打撃
    clicks = D.t_raw[D.kind == "click"].to_numpy()
    t0 = clicks.max() + g["beat_s"] if len(clicks) else P.t_raw[P.kind == "stroke"].iloc[0]
    # MIDI の原点 = 1 拍目の基準にいちばん近い 3 連の格子点。格子の位相は 3 連（拍/3）の周期でしか決まらないので、
    # 拍ではなく格子点に合わせる。どのテイクも 1 拍目の格子点が 0 になり、合成で格子がそろう
    g3 = g["beat_s"] / 3
    tm0 = g["phase_s"] + np.round((t0 - g["phase_s"]) / g3) * g3
    P.insert(0, "idx", np.arange(len(P)))
    P["t_rel"] = P.t_raw - t0
    P["t_midi"] = np.where(P.grid != "none", P.t_snap, P.t_raw) - tm0
    os.makedirs(a.out, exist_ok=True)
    P.to_csv(os.path.join(a.out, f"{a.name}_onsets.csv"), index=False, float_format="%.4f")
    D.to_csv(os.path.join(a.out, f"{a.name}_all_detections.csv"), index=False, float_format="%.4f")
    M = P[P.kind == "stroke"]
    write_midi(os.path.join(a.out, f"{a.name}.mid"), M.t_midi.to_numpy(), M.velocity.to_numpy(), a.bpm)
    Mc = P[P.t_midi >= -LEAD_S + 0.05]
    write_midi(os.path.join(a.out, f"{a.name}_with_candidates.mid"), Mc.t_midi.to_numpy() - min(0.0, Mc.t_midi.min()),
               Mc.velocity.to_numpy(), a.bpm)
    plot_take(os.path.join(a.out, f"{a.name}_spec.png"), x, D, aux, a.name)
    plot_close(os.path.join(a.out, f"{a.name}_close.png"), x, P, a.name)
    C = D[D.kind == "click"]
    S_ = P[P.kind == "stroke"]
    L = [f"# 打点起こし：{a.name}（{os.path.basename(a.input)}" + (f"、{a.start}–{a.end} s" if a.start is not None else "") + "）", "",
         f"- 打撃 {len(S_)}（演奏の範囲 {S_.t_raw.min():.3f}–{S_.t_raw.max():.3f} s）、要確認の候補 {int((P.kind == 'cand').sum())}",
         f"- 冒頭のカウント（クリック、打点列から外した）：{len(C)} 打（" + ", ".join(f"{v:.3f}" for v in C.t_raw) + " s）",
         f"- 範囲外・レベル不足で外した検出：{int((~D.in_perf & (D.kind != 'click')).sum())}",
         f"- 格子：拍 {g['beat_s'] * 1000:.1f} ms（♩ = {g['bpm_fit']:.1f}）、位相 {g['phase_s'] * 1000:.0f} ms。打撃のうち 3 連に寄せた "
         f"{int(((S_.grid == 'triplet')).sum())}・6 連 {int((S_.grid == 'sextuplet').sum())}・寄せない {int((S_.grid == 'none').sum())}（許容 ±{a.tol_ms:.0f} ms）",
         f"- MIDI：{a.name}.mid は打撃 {len(M)} 打、{a.name}_with_candidates.mid は候補も入れた {len(Mc)} 打",
         f"- 打撃の間隔：最小 {S_.t_raw.diff().min() * 1000:.0f} ms、中央値 {S_.t_raw.diff().median() * 1000:.0f} ms", "",
         "## 打撃の一覧", "", "| # | 時刻 [s] | 先頭から [s] | MIDI の時刻 [s] | 格子 | ずれ [ms] | ベロシティ | 跳ね上がり [dB] | 理由 |", "|---|---|---|---|---|---|---|---|---|"]
    for _, d in S_.iterrows():
        L.append(f"| {d.idx} | {d.t_raw:.3f} | {d.t_rel:.3f} | {d.t_midi:.3f} | {d.grid} | "
                 f"{'' if np.isnan(d.snap_err_ms) else f'{d.snap_err_ms:+.0f}'} | {d.velocity} | {d.rise_db:.0f} | {d.reason} |")
    L += ["", "## 要確認の候補（MIDI の本体には入れていない。聞いて打撃なら入れる）", "",
          "| # | 時刻 [s] | 直前の打撃から [ms] | 次の打撃まで [ms] | レベル（最大比）[dB] | 跳ね上がり [dB] |", "|---|---|---|---|---|---|"]
    st_t = S_.t_raw.to_numpy()
    for _, d in P[P.kind == "cand"].iterrows():
        pv = st_t[st_t < d.t_raw]
        nx = st_t[st_t > d.t_raw]
        L.append(f"| {d.idx} | {d.t_raw:.3f} | {'' if not len(pv) else f'{(d.t_raw - pv[-1]) * 1000:.0f}'} | "
                 f"{'' if not len(nx) else f'{(nx[0] - d.t_raw) * 1000:.0f}'} | {d.level_rel_db:.0f} | {d.rise_db:.1f} |")
    if a.prev:
        Pv = np.sort(load_prev(a.prev, a.prev_key))
        best = None
        for sh in np.arange(-1.0, 1.0, 0.001):
            nn = len(match(S_.t_raw.to_numpy(), Pv + sh, 0.010))
            if best is None or nn > best[0]:
                best = (nn, sh)
        pr = match(S_.t_raw.to_numpy(), Pv + best[1], 0.010)
        L += ["", f"## Muse 初版との差", "", f"初版 {len(Pv)} 打、今回の打撃 {len(S_)} 打、±10 ms で対応 {len(pr)}（初版を {best[1] * 1000:+.0f} ms ずらして）"]
    open(os.path.join(a.out, f"{a.name}_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:8]))


def cmd_combine(a):
    A, B = pd.read_csv(a.a), pd.read_csv(a.b)
    if not a.include_low:
        A, B = A[A.kind == "stroke"].reset_index(drop=True), B[B.kind == "stroke"].reset_index(drop=True)
    # そろえ方：両テイクの最初の信頼度 high の打点を 0 に（take の t_rel と同じ原点）。t_midi（寄せた時刻）で重ねる
    ta, tb = A.t_midi.to_numpy(), B.t_midi.to_numpy()
    A, B = A[ta >= -0.05].reset_index(drop=True), B[tb >= -0.05].reset_index(drop=True)
    ta, tb = A.t_midi.to_numpy(), B.t_midi.to_numpy()
    pairs = match(ta, tb, a.unison_ms / 1000)
    rows = []
    pa, pb = {i for i, _ in pairs}, {j for _, j in pairs}
    for i, j in pairs:
        s = A.iloc[i] if A.level_rel_db[i] >= B.level_rel_db[j] else B.iloc[j]
        rows.append(dict(t=ta[i] if A.level_rel_db[i] >= B.level_rel_db[j] else tb[j], velocity=max(A.velocity[i], B.velocity[j]),
                         src="both", t_a=ta[i], t_b=tb[j], grid=s.grid))
    for i in sorted(set(range(len(A))) - pa):
        rows.append(dict(t=ta[i], velocity=A.velocity[i], src="a", t_a=ta[i], t_b=np.nan, grid=A.grid[i]))
    for j in sorted(set(range(len(B))) - pb):
        rows.append(dict(t=tb[j], velocity=B.velocity[j], src="b", t_a=np.nan, t_b=tb[j], grid=B.grid[j]))
    C = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    C["ioi_prev_ms"] = C.t.diff() * 1000
    C["lt80"] = C.ioi_prev_ms < 80
    os.makedirs(a.out, exist_ok=True)
    C.to_csv(os.path.join(a.out, f"{a.name}_onsets.csv"), index=False, float_format="%.4f")
    write_midi(os.path.join(a.out, f"{a.name}.mid"), C.t.to_numpy(), C.velocity.to_numpy(), a.bpm)
    L = [f"# 合成：{os.path.basename(a.a)} ＋ {os.path.basename(a.b)}", "",
         "そろえ方：両テイクとも、1 拍目（最後のカウント ＋ 1 拍）にいちばん近い 3 連の格子点を 0 にした（先頭でそろえる）。時刻は MIDI と同じ（格子に寄せた打点は格子の時刻、寄せない打点は生の時刻）。"
         + ("要確認の候補（low）も入れた。" if a.include_low else "要確認の候補は入れていない（--include_low で入れる）。"),
         f"同時打点：|Δt| ≤ {a.unison_ms:.0f} ms の組は 1 打にまとめた（時刻は強い方、ベロシティは大きい方）。それ以外は両方を残した。", "",
         f"- A {len(A)} 打、B {len(B)} 打 → 合成 {len(C)} 打（同時打点 {len(pairs)}、A のみ {len(A) - len(pairs)}、B のみ {len(B) - len(pairs)}）",
         f"- 同時打点の割合：A の {len(pairs) / max(1, len(A)):.2f}、B の {len(pairs) / max(1, len(B)):.2f}"
         "（高ければ 2 テイクは同じ打点列のくり返し、低ければ互いに補う別の手）",
         f"- 80 ms 未満の間隔（ロボットの片腕の学習・評価の下限 80 ms より短い）：{int(C.lt80.sum())} 組。落としていない"]
    open(os.path.join(a.out, f"{a.name}_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_compare(a):
    C, S = pd.read_csv(a.combined), pd.read_csv(a.ref)
    S = S[S.kind == "stroke"]
    tc = np.sort(C["t"].to_numpy() if "t" in C else C.t_rel.to_numpy())
    ts = np.sort(S.t_rel.to_numpy())
    best = None
    for sh in np.arange(-2.0, 2.0, 0.001):
        n = len(match(tc + sh, ts, a.tol_ms / 1000))
        if best is None or n > best[0]:
            best = (n, sh)
    pr = match(tc + best[1], ts, a.tol_ms / 1000)
    ic, js = {i for i, _ in pr}, {j for _, j in pr}
    err = np.array([(tc[i] + best[1] - ts[j]) * 1000 for i, j in pr])
    L = [f"# 照合：{os.path.basename(a.combined)} 対 {os.path.basename(a.ref)}", "",
         f"ずらし {best[1] * 1000:+.0f} ms（±2 s を探索）、±{a.tol_ms:.0f} ms で 1 対 1",
         f"- 両手テイクの打点のカバー率（両手テイクのうち合成列にあるもの）：{len(pr)} / {len(ts)} = {len(pr) / max(1, len(ts)):.3f}（両手テイクは打撃だけ。候補とクリックは除く）",
         f"- 合成列のうち両手テイクにあるもの：{len(pr)} / {len(tc)} = {len(pr) / max(1, len(tc)):.3f}",
         f"- 対応した打点の時刻の差（合成 − 両手）：中央値 {np.median(err):+.1f} ms、四分位 {np.percentile(err, 25):+.1f}〜{np.percentile(err, 75):+.1f} ms",
         "", "## 両手テイクにだけある打点", ""]
    L += [f"- {ts[j]:.3f} s" for j in sorted(set(range(len(ts))) - js)]
    L += ["", "## 合成列にだけある打点", ""]
    L += [f"- {tc[i]:.3f} s" for i in sorted(set(range(len(tc))) - ic)]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "compare_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("take")
    p.add_argument("input")
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="out")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--bpm", type=float, default=200.0)
    p.add_argument("--tol_ms", type=float, default=12.0)
    p.add_argument("--min_level_db", type=float, default=None,
                   help="打撃の 95 パーセンタイルよりこれだけ小さい打撃を候補にする（片手テイクは -25。両手テイクは弱打があるので付けない）")
    p.add_argument("--prev", help="Muse 初版の打点（json）")
    p.add_argument("--prev_key")
    p = sp.add_parser("combine")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--name", default="combined")
    p.add_argument("--out", default="out")
    p.add_argument("--bpm", type=float, default=200.0)
    p.add_argument("--unison_ms", type=float, default=40.0)
    p.add_argument("--include_low", action="store_true", help="要確認の候補（信頼度 low）も合成に入れる")
    p = sp.add_parser("compare")
    p.add_argument("combined")
    p.add_argument("ref")
    p.add_argument("--out", default="out")
    p.add_argument("--tol_ms", type=float, default=30.0)
    a = ap.parse_args()
    {"take": cmd_take, "combine": cmd_combine, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
