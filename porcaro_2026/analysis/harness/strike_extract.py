"""Strike-level extraction from simulation_log.csv-shaped data.

target_force in simulation_log.csv is the super-Gaussian target waveform sampled per control step (not a
constant column); its peaks mark the target strike times. The measured strike time and force come from force_z
around each target. pandas / numpy / scipy only (no isaaclab, no torch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def mask_long_trains(t, f, on_n, train_gap_s, max_contact_s=0.100):
    """押し付けの判定（2026-11-04）：on_n 以上の区間を、離れ（次の区間の始まり − 前の区間の終わり）が train_gap_s 未満なら
    1 つの「接触の連なり」にまとめ、連なりの長さ（最初の区間の始まりから最後の区間の終わりまで）が max_contact_s を超えたら、
    その中の力を 0 にした f を返す（中の接触を打撃に数えない）。max_contact_s 以内の連なりは手を付けない（中の打撃は別々に採点する）。

    動機：sim では押し付けたままのスティックが打面の上で跳ね（約 30 ms ごとに約 21 N、間は 0 N）、実機では押し付けの間に
    力が 10〜20 ms だけ 0.5 N を下回ることがあり、どちらも 1 回ずつ打撃に数えられていた（access_1104/c1、c5）。
    分解能：目標どうしの打撃の離れが train_gap_s より短い連打は区別できない（目標の最短の間隔 80 ms − 接触 約 20 ms ≒ 60 ms）
    """
    t = np.asarray(t, float); f = np.asarray(f, float).copy()
    above = f >= on_n
    n = len(f)
    segs, i = [], 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    if not segs:
        return f
    trains = [segs[0][:]]
    for s0, e0 in segs[1:]:
        if t[s0] - t[trains[-1][1] - 1] < train_gap_s:
            trains[-1][1] = e0
        else:
            trains.append([s0, e0])
    for s0, e0 in trains:
        if t[e0 - 1] - t[s0] > max_contact_s:
            f[s0:e0] = 0.0
    return f


def contact_events(t, f, on_n, release_n=0.5, max_contact_s=0.100, min_sep_s=0.0, train_gap_s=0.0):
    """接触のイベント（離れてから当たる打撃）のピークの添字を返す。

    f が on_n 以上になったら接触の始まり、release_n 未満に落ちたら終わり（ヒステリシス）。
    接触が max_contact_s より長いもの（押し付け）は打撃に数えない。ピーク間が min_sep_s 未満なら強い方だけ残す。
    最初から接触していた区間（始まりを見ていない）と、最後まで終わらない区間も数えない。
    train_gap_s > 0 なら、先に mask_long_trains で長い接触の連なり（押し付け）を外す（2026-11-04）。
    """
    t = np.asarray(t, float); f = np.asarray(f, float)
    if train_gap_s > 0:
        f = mask_long_trains(t, f, on_n, train_gap_s, max_contact_s)
    above = f >= on_n
    below = f < release_n
    ev, inside, start, seen_release = [], False, 0, False
    for i in range(len(f)):
        if not inside:
            if below[i]:
                seen_release = True
            elif above[i] and seen_release:
                inside, start = True, i
        elif below[i]:
            inside = False
            if t[i - 1] - t[start] <= max_contact_s:
                ev.append(start + int(np.argmax(f[start:i])))
    if not ev:
        return np.zeros(0, dtype=int)
    ev = np.asarray(ev)
    keep = []
    for k in ev[np.argsort(-f[ev], kind="stable")]:
        if all(abs(t[k] - t[j]) >= min_sep_s for j in keep):
            keep.append(int(k))
    return np.sort(np.asarray(keep, dtype=int))


def extract_strikes(
    ts: pd.DataFrame,
    bpm: float | None = None,
    target_ref: float = 20.0,
    tol_ms: float = 30.0,
    min_strike_frac: float = 0.25,
    method: str = "peak_match",
    release_n: float = 0.5,
    max_contact_s: float = 0.100,
    train_gap_s: float = 0.0,
) -> pd.DataFrame:
    """Recover per-strike timing/force outcomes from a simulation_log-shaped trace.

    Args:
        ts: DataFrame with at least columns ["time_s", "target_force", "force_z"].
            If `bpm` is not given, a "target_bpm" column is used instead.
        bpm: tempo used to derive T16th = 15/bpm (s) for windowing/peak-distance.
        target_ref: reference peak force (N) used for the min_strike_frac gate
            and (implicitly) as the height threshold for detecting target peaks.
        tol_ms: perceptual timing tolerance (ms) for the `success` flag.
        min_strike_frac: minimum fraction of target_ref a measured peak must
            reach to count as a genuine strike (rather than a graze).
        method: "peak_match" (default) detects genuine strikes as local maxima
            of the measured force above min_strike_frac*target_ref and assigns
            each target the nearest unused strike, one-to-one. "argmax" is the
            legacy behaviour (take the argmax of the force inside a
            +-1.5*T16th window around each target) and is kept only to
            reproduce pre-2026-08-06 numbers. With argmax, the windows of
            onsets closer than 3*T16th overlap, so one physical strike can be
            credited to two targets (47 % of adjacent pairs on hardware, 50 %
            in sim on the 160 BPM double stroke). peak_match does not depend
            on the window width (+-45 to +-150 ms give the same success rates).

            "contact_event" (2026-11-03) counts a strike only as a contact event:
            the force rises to >= min_strike_frac*target_ref after having been
            below release_n (the stick left the head), and the event ends when
            the force falls below release_n again. Events longer than
            max_contact_s are presses, not strikes, and are dropped. The event's
            peak gives the strike time and force; events closer than 0.3*T16th
            keep only the stronger one (same rule as find_peaks' distance), then
            the same greedy one-to-one matching as peak_match. Introduced because
            the FF optimiser found a "press the stick on the head and modulate"
            solution that peak_match scored as strikes. 100 ms = the RL reward's
            contact limit (0.4*T16th) at the slowest tempo (60 BPM).

    Returns:
        DataFrame [strike_idx, target_time, peak_time, timing_err_ms,
                   peak_force, success], one row per detected target strike.
    """
    required_cols = {"time_s", "target_force", "force_z"}
    missing = required_cols - set(ts.columns)
    if missing:
        raise ValueError(f"extract_strikes: missing required columns {missing}")

    time_s = ts["time_s"].to_numpy(dtype=float)
    target_force = ts["target_force"].to_numpy(dtype=float)
    force_z = ts["force_z"].to_numpy(dtype=float)

    if len(time_s) < 2:
        raise ValueError("extract_strikes: need at least 2 samples")

    dt = float(np.median(np.diff(time_s)))
    if dt <= 0:
        raise ValueError(f"extract_strikes: non-positive inferred dt={dt}")

    if bpm is None:
        if "target_bpm" not in ts.columns:
            raise ValueError("extract_strikes: bpm not given and no 'target_bpm' column present")
        bpm = float(ts["target_bpm"].median())
    bpm = float(bpm)
    if bpm <= 0:
        raise ValueError(f"extract_strikes: invalid bpm={bpm}")

    t16 = 15.0 / bpm  # seconds, per paper Eq: T16th = 15/BPM

    # 1. Recover target strike times from peaks of target_force.
    distance_samples = max(1, int(round(0.3 * t16 / dt)))
    target_height = min_strike_frac * target_ref
    tgt_peak_idx, _ = find_peaks(target_force, height=target_height, distance=distance_samples)
    target_times = time_s[tgt_peak_idx]

    # 2. For each target time, search a window of +-1.5*T16th for the actual
    #    (measured) force_z peak.
    half_win = 1.5 * t16
    strike_gate = min_strike_frac * target_ref

    if method in ("peak_match", "contact_event"):
        if method == "peak_match":
            # 1. 実打撃そのものを検出する（力のしきい値を超える極大）。
            act_idx, _ = find_peaks(force_z, height=strike_gate, distance=distance_samples)
        else:
            act_idx = contact_events(time_s, force_z, strike_gate, release_n, max_contact_s, 0.3 * t16, train_gap_s)
        act_t, act_f = time_s[act_idx], force_z[act_idx]
        # 2. |誤差| の小さい組から貪欲に1対1で割り当てる。
        cand = []
        for ti, t_target in enumerate(target_times):
            for ai in np.nonzero(np.abs(act_t - t_target) <= half_win)[0]:
                cand.append((abs(float(act_t[ai]) - float(t_target)), ti, int(ai)))
        cand.sort()
        taken_t, taken_a = {}, set()
        for _, ti, ai in cand:
            if ti in taken_t or ai in taken_a:
                continue
            taken_t[ti] = ai
            taken_a.add(ai)
        rows = []
        for i, t_target in enumerate(target_times):
            j = taken_t.get(i)
            if j is None:
                rows.append(dict(strike_idx=i, target_time=float(t_target),
                                 peak_time=np.nan, timing_err_ms=np.nan,
                                 peak_force=0.0, success=False))
                continue
            err = (float(act_t[j]) - float(t_target)) * 1000.0
            rows.append(dict(strike_idx=i, target_time=float(t_target),
                             peak_time=float(act_t[j]), timing_err_ms=err,
                             peak_force=float(act_f[j]),
                             success=bool(abs(err) <= tol_ms)))
        return pd.DataFrame(
            rows,
            columns=["strike_idx", "target_time", "peak_time", "timing_err_ms",
                     "peak_force", "success"],
        )

    if method != "argmax":
        raise ValueError(f"extract_strikes: unknown method {method!r}")

    rows = []
    for i, t_target in enumerate(target_times):
        mask = ((time_s >= t_target - half_win)
                & (time_s <= t_target + half_win))
        if not mask.any():
            rows.append(
                dict(
                    strike_idx=i,
                    target_time=float(t_target),
                    peak_time=np.nan,
                    timing_err_ms=np.nan,
                    peak_force=0.0,
                    success=False,
                )
            )
            continue

        local_t = time_s[mask]
        local_f = force_z[mask]
        j = int(np.argmax(local_f))
        peak_force = float(local_f[j])
        peak_time = float(local_t[j])
        timing_err_ms = (peak_time - t_target) * 1000.0
        success = bool(peak_force >= min_strike_frac * target_ref and abs(timing_err_ms) <= tol_ms)

        rows.append(
            dict(
                strike_idx=i,
                target_time=float(t_target),
                peak_time=peak_time,
                timing_err_ms=timing_err_ms,
                peak_force=peak_force,
                success=success,
            )
        )

    return pd.DataFrame(
        rows,
        columns=["strike_idx", "target_time", "peak_time", "timing_err_ms", "peak_force", "success"],
    )
