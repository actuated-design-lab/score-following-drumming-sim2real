"""check_obs_v2.py — (b) の学習の env（Porcaro-Gen2-B-user0 / -obsv1-）を、投入前に Isaac の上で確かめる（WS 用、数分）

  python scripts/tools/user0/gen2/check_obs_v2.py --headless                         # 本命 v2（B 相当、37 次元）
  python scripts/tools/user0/gen2/check_obs_v2.py --headless --use_frame_stacking    # 本命 v2（E 相当、185 次元）
  python scripts/tools/user0/gen2/check_obs_v2.py --headless --obsv1                 # 保険 v1（35 次元）
  （3 つまとめて回して push するのは run_check_b.sh）

確かめること:
  1) 観測の次元
  2) (b) の分布の値が cfg に入っている（音価の確率・休み・テンポ・助走 2.0 s・32 拍・カリキュラム 300 / 750 iteration）
  3) カリキュラムが iteration 300・750 の境目でちょうど切り替わる
  4) 観測の並び：前回指令・BPM・先読みの位置（v2 は拍位相がない）
  5) v2：圧力の観測 = 1 tick 前の P_out / Pmax（ノイズとずれを 0 にして、全 env・全 tick で一致）
  6) v2：接触のフラグ = 下の参照実装（numpy、仕様そのもの）と全 env・全 tick で一致。接触の回数も出す
  7) v2：ノイズとずれを入れたとき、残差の標準偏差 ≈ obs_pressure_noise_std、ずれ ≈ obs_pressure_bias_range の中
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--use_frame_stacking", action="store_true")
p.add_argument("--obsv1", action="store_true")
p.add_argument("--out_dir", default="data/user0/gen2/env_test")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env import Porcaro2026Env  # noqa: E402
from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env_cfg import (  # noqa: E402
    Porcaro2026EnvCfg_Gen2_B, Porcaro2026EnvCfg_Gen2_B_ObsV1, obs_base_dim)

TAG = ("v1" if args.obsv1 else "v2") + ("_fs5" if args.use_frame_stacking else "")
LINES, NG = [], []


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'NG '}] {name}  {detail}", flush=True)
    LINES.append(f"| {name} | {'OK' if ok else '**NG**'} | {detail} |")
    if not ok:
        NG.append(name)


def latch_reference(c, traj, k_after, D, lead, frac, F):
    """接触のラッチフラグの仕様（実機の deploy_policy も同じ規則）。

    c[t]       : tick t の区間（物理 4 ステップ）に、力が閾値以上になったか（0/1）
    traj[e, k] : 目標波形（N）。k_after[t] は step t のあとの episode_length_buf
    観測に入る接触 = c を D tick 遅らせたもの。それを OR で溜め、その後で、tick (k − D + lead) の目標波形が
    frac × F 以上なら 0 に戻す。
    """
    T, N = c.shape
    out = np.zeros((T, N), np.float32)
    lat = np.zeros(N, np.float32)
    for t in range(T):
        co = c[t - D] if t - D >= 0 else np.zeros(N, np.float32)
        lat = np.maximum(lat, co)
        kk = np.clip(k_after[t] - D + lead, 0, traj.shape[1] - 1)
        on = traj[np.arange(N), kk] >= frac * F
        lat = np.where(on, 0.0, lat)
        out[t] = lat
    return out


def main():
    cfg = Porcaro2026EnvCfg_Gen2_B_ObsV1() if args.obsv1 else Porcaro2026EnvCfg_Gen2_B()
    cfg.scene.num_envs = args.num_envs
    cfg.logging.enabled = False
    cfg.seed = 1
    if args.use_frame_stacking:                       # train.py と同じ上書き
        cfg.use_frame_stacking = True
        cfg.frame_stack_k = 5
    dt_ctrl = cfg.sim.dt * cfg.decimation
    L = int(cfg.lookahead_horizon / dt_ctrl)
    base = obs_base_dim(cfg, L)
    cfg.observation_space = base * cfg.frame_stack_k if cfg.use_frame_stacking else base
    v2 = cfg.obs_version == "v2"
    if v2:                                            # 5)・6) はノイズとずれなしで確かめる
        noise0, bias0 = cfg.obs_pressure_noise_std, cfg.obs_pressure_bias_range
        cfg.obs_pressure_noise_std = 0.0
        cfg.obs_pressure_bias_range = (0.0, 0.0)
    env = Porcaro2026Env(cfg)
    N, dev = env.num_envs, env.device
    obs, _ = env.reset()
    o = obs["policy"]
    want = (37 if v2 else 35) * (5 if args.use_frame_stacking else 1)
    check("1) 観測の次元", o.shape[1] == want, f"{o.shape[1]}（期待 {want}）")

    # 2) 分布の値
    vals = dict(probs=tuple(round(x, 4) for x in cfg.rand_note_probs), rest=cfg.rand_rest_prob, bpm=tuple(cfg.rand_bpm_range),
                lead=cfg.rand_lead_in_s, beats=cfg.rand_episode_beats, ci=tuple(cfg.curriculum_iters), mode=cfg.rhythm_mode,
                once=cfg.rhythm_reset_once, min_iv=cfg.rand_min_interval_s)
    ok = (vals["probs"] == (0.45, 0.076, 0.183, 0.061, 0.122, 0.061, 0.047) and vals["rest"] == 0.15 and vals["bpm"] == (60.0, 160.0)
          and vals["lead"] == 2.0 and vals["beats"] == 32 and vals["ci"] == (300, 750) and vals["mode"] == "random"
          and vals["once"] is True and vals["min_iv"] == 0.08 and abs(sum(cfg.rand_note_probs) - 1.0) < 1e-9)
    check("2) (b) の分布の値", ok, str(vals))

    # 3) カリキュラムの境目
    thr = env.curriculum_thresholds
    exp_thr = [300 * 120 * N, 750 * 120 * N]
    zero = torch.zeros((N, 3), device=dev)
    lv = []
    for th in exp_thr:
        for start in (th - N, th):                     # step で N 足される。足したあとが th を超えたら上がる
            env.total_env_steps = start
            env.rhythm_generator.set_curriculum_level(0 if th == exp_thr[0] else 1)
            env.step(zero)
            lv.append(env.rhythm_generator.curriculum_level)
    check("3) カリキュラムの切り替え", thr == exp_thr and lv == [0, 1, 1, 2],
          f"閾値 {thr}（期待 {exp_thr}）、境目の前後のレベル {lv}（期待 [0, 1, 1, 2]）")
    env.total_env_steps = 0
    env.rhythm_generator.set_curriculum_level(0)

    # 4)〜6) 本番の流れ：reset から、5 tick ごとに変わるランダムな指令で steps tick
    obs, _ = env.reset()
    g = torch.Generator(device=dev).manual_seed(0)
    base_dim = env.base_obs_dim
    O, P, C, K, A = [], [], [], [], []
    a = zero
    for t in range(args.steps):
        if t % 5 == 0:
            a = torch.rand((N, 3), generator=g, device=dev) * 2 - 1
        obs, _, term, trunc, _ = env.step(a)
        if (term | trunc).any():
            print("!!! 途中でリセットが起きた（steps を短くする）")
            break
        o = obs["policy"][:, -base_dim:]               # frame stack は最後のフレーム
        O.append(o.cpu().numpy())
        P.append(env.action_controller.get_last_telemetry()["P_out"].cpu().numpy())
        C.append((env.max_force_z_buffer >= cfg.contact_flag_threshold_n).float().cpu().numpy())
        K.append(env.episode_length_buf.cpu().numpy().copy())
        A.append(env.actions.cpu().numpy())
    O, P, C, K, A = map(np.asarray, (O, P, C, K, A))
    traj = env.rhythm_generator.target_trajectories.cpu().numpy()
    bpm = env.rhythm_generator.current_bpms.cpu().numpy()
    T = len(O)

    # 4) 並び
    i_bpm = 7 if v2 else 9
    e_prev = np.abs(O[1:, :, 4:7] - A[:-1]).max()
    e_bpm = np.abs(O[:, :, i_bpm] - bpm[None] / 180.0).max()
    la = np.stack([traj[np.arange(N)[:, None], np.clip(K[t][:, None] + np.arange(L)[None], 0, traj.shape[1] - 1)] for t in range(T)])
    e_la = np.abs(O[:, :, i_bpm + 1:i_bpm + 1 + L] - la / env.target_hit_force).max()
    check("4) 観測の並び（前回指令・BPM・先読み）", e_prev < 1e-6 and e_bpm < 1e-6 and e_la < 1e-5,
          f"誤差の最大：前回指令 {e_prev:.1e}、BPM {e_bpm:.1e}、先読み {e_la:.1e}（BPM は {i_bpm} 番目）")

    md_extra = ""
    if v2:
        D = env.sensor_delay_steps
        j = i_bpm + 1 + L
        Pm = env.obs_p_scale
        e_p = np.abs(O[D:, :, j:j + 3] * Pm - P[:-D or None]).max() if D > 0 else np.abs(O[:, :, j:j + 3] * Pm - P).max()
        e_p0 = np.abs(O[:D, :, j:j + 3]).max() if D > 0 else 0.0
        check(f"5) 圧力 = {D} tick 前の P_out / Pmax", e_p < 1e-6 and e_p0 < 1e-9,
              f"誤差の最大 {e_p:.1e} MPa、最初の {D} tick の値 {e_p0:.1e}、P_out の範囲 {P.min():.3f}〜{P.max():.3f} MPa")
        ref = latch_reference(C, traj, K, D, cfg.contact_flag_reset_lead_steps, cfg.contact_flag_onset_frac, env.target_hit_force)
        flag = O[:, :, j + 3]
        n_c = int(C.sum())
        n_on = int(sum(((traj[e, 1:] >= 0.75 * 20) & (traj[e, :-1] < 0.75 * 20)).sum() for e in range(N)))
        check("6) 接触のフラグ = 参照実装", bool(np.array_equal(flag, ref)) and n_c > 0,
              f"不一致 {int((flag != ref).sum())} / {flag.size}、接触のある tick {n_c}、フラグ 1 の割合 {flag.mean():.3f}、"
              f"目標打撃 {n_on}（全 env、{T} tick）")
        # 7) ノイズとずれ
        env.cfg.obs_pressure_noise_std = noise0
        env.cfg.obs_pressure_bias_range = bias0
        env.reset()
        R, Pp = [], None
        for t in range(120):
            obs, *_ = env.step(zero if t < 20 else torch.full((N, 3), 0.2, device=dev))
            pnow = env.action_controller.get_last_telemetry()["P_out"].cpu().numpy()
            if Pp is not None and D == 1:
                R.append(obs["policy"][:, -base_dim:][:, j:j + 3].cpu().numpy() * Pm - Pp)
            Pp = pnow
        R = np.asarray(R)
        bias = R.mean(axis=0)
        sd = (R - bias[None]).std()
        lo, hi = bias0
        check("7) ノイズとずれ", abs(sd - noise0) < 0.2 * noise0 + 1e-4 and bias.min() >= lo - 3e-3 and bias.max() <= hi + 3e-3,
              f"残差の標準偏差 {sd:.4f} MPa（設定 {noise0}）、ずれ {bias.min():+.4f}〜{bias.max():+.4f} MPa（設定 {lo}〜{hi}）")
        md_extra = (f"\n接触のフラグの仕様（実機の deploy_policy と同じ規則）：tick の区間に力 ≥ {cfg.contact_flag_threshold_n} N なら接触。"
                    f"観測の接触は {D} tick 遅らせる。それを OR で溜め、そのあと、tick (k − {D} + {cfg.contact_flag_reset_lead_steps}) の目標波形が "
                    f"{cfg.contact_flag_onset_frac} × 20 N 以上なら 0 に戻す（k は step のあとの episode_length_buf）。\n")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"obs_{TAG}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# (b) の env の確認（check_obs_v2.py、{TAG}）\n\n"
                 f"{type(cfg).__name__}、num_envs {N}、{T} tick。結果: **{'すべて OK' if not NG else 'NG あり'}**\n\n"
                 "| 項目 | 結果 | 詳細 |\n|---|---|---|\n" + "\n".join(LINES) + "\n" + md_extra)
    print(f"→ {out}（{'すべて OK' if not NG else 'NG: ' + ', '.join(NG)}）")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        os.write(2, ("\n[check_obs_v2] エラーで止まりました:\n" + traceback.format_exc()).encode("utf-8", "replace"))
        os._exit(1)
    app.close()
