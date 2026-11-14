"""sim_trace.py — 実機コアの 1 タスクを sim で回し、時系列（指令・ITV 出口圧・関節角・力）と、必要なら実時間の動画を残す（WS、Isaac）

eval_b.py と同じ env（Gen2_B、obs v2）・同じ目標波形（ff_lib.midi_target）・同じ採点（ff_score.score_sim）。違いは次の 3 つ：
  1. 物理ステップ（5 ms）ごとに P_cmd・P_out（3 筋）・手首とグリップの角度（実機の向き）・力（k_t を掛けた z 成分）を全 env 分記録する
  2. --nominal：DR を切る（起動時の質量・摩擦・重心・粘性、リセット時の k・τ・L は 1 倍、通信の遅れ 8 ms 固定）。動画と、実機との 1 対 1 の比較用
  3. --video：env 1 本（--nominal 必須）を 50 fps（= 制御周期、実時間）で mp4 に書く。カメラは スティックと打面の中点（高さはスティック）を斜め上から見下ろす（--cam_azim / --cam_elev_deg / --cam_dist、直接指定は --cam_eye / --cam_lookat）。--t_end で頭の数秒だけの試し撮り

方策：--model B|E --ckpt <pt>（RL、閉ループ）か、--signal <jetson の signals/core2026/*.csv>（FF、実機に流したのと同じ指令の開ループ）

  # E・double160：動画（実時間・DR なし）
  python scripts/tools/user0/access/sim_trace.py --headless --enable_cameras --nominal --video \\
      --midi data/user0/eval/real_core/songs/double_lead2_bpm160.mid --model E --ckpt <E の pt> --name E_double160_nom
  # F3・holdout0：動画
  python scripts/tools/user0/access/sim_trace.py --headless --enable_cameras --nominal --video \\
      --midi data/user0/eval/real_core/songs/b_holdout_00_bpm90.mid --signal ../jetson_project/signals/core2026/F3_b_holdout_00_bpm90.csv --name F3_holdout0_nom
  # 時系列だけ（DR 32 本、eval_b と同じ引き）
  python scripts/tools/user0/access/sim_trace.py --headless --midi ... --model E --ckpt ... --n_dr 32 --name E_double160_dr

出力：data/user0/eval/access_1104/sim/<name>/{trace.npz, strikes.csv, summary.json, video/<name>.mp4, video_meta.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--midi", required=True, help="実機コアの MIDI（data/user0/eval/real_core/songs/*.mid）")
p.add_argument("--model", choices=["B", "E"], default=None)
p.add_argument("--ckpt", default=None)
p.add_argument("--signal", default=None, help="FF：実機に流した指令（time, cmd_pressure_DF/F/G、50 Hz、MPa）")
p.add_argument("--ff", default=None, help="FF：プリミティブの json（ff_opt の favorite.json など）。MIDI の打撃時刻から指令を作る（eval_b と同じ ff_lib.synth）")
p.add_argument("--nominal", action="store_true", help="DR を切る（env は 1 本）")
p.add_argument("--n_dr", type=int, default=32)
p.add_argument("--dr_seed", type=int, default=7000)
p.add_argument("--video", action="store_true")
p.add_argument("--cam_azim", choices=["side", "behind", "front"], default="side",
               help="カメラの方向：side = 腕と打面を結ぶ線の真横、behind = ロボットの後ろ（肩越し）、front = 打面の向こう側")
p.add_argument("--cam_elev_deg", type=float, default=40.0, help="見下ろす角度 [deg]（0 = 水平、90 = 真上）")
p.add_argument("--cam_rot_deg", type=float, default=0.0,
               help="cam_azim の向きから注視点のまわりに回す角度 [deg]。+ でカメラが自分から見て左へ回り込む（上から見て時計回り）")
p.add_argument("--cam_dist", type=float, default=1.0, help="カメラと注視点の距離 [m]")
p.add_argument("--cam_side", type=float, default=1.0, help="side のとき、どちら側から見るか（+1 / −1）")
p.add_argument("--t_end", type=float, default=None, help="最初のこの秒数だけ回す（画角の試し撮り用。採点は途中まで）")
p.add_argument("--cam_eye", default=None, help="カメラ位置を直接 'x,y,z'（ワールド）")
p.add_argument("--cam_lookat", default=None, help="注視点を直接 'x,y,z'")
p.add_argument("--name", required=True)
p.add_argument("--out", default="data/user0/eval/access_1104/sim")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.video and not args.nominal:
    raise SystemExit("--video は --nominal と一緒に（env 1 本）")
if sum(x is not None for x in (args.ckpt, args.signal, args.ff)) != 1:
    raise SystemExit("--ckpt（＋ --model）、--signal、--ff のどれか 1 つ")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
import ff_lib  # noqa: E402
import ff_score  # noqa: E402

import porcaro_2026.tasks  # noqa: E402,F401
from porcaro_2026.tasks.direct.porcaro_2026.user0 import porcaro_2026_env_cfg as C  # noqa: E402

TASK = "Porcaro-Gen2-B-user0"


def main():
    tgt, bpm = ff_lib.midi_target(ROOT / args.midi)
    ons = ff_lib.onsets_from_target(tgt, bpm)
    dur = float(ons[-1] + 1.0)
    N = 1 if args.nominal else args.n_dr

    cfg = C.Porcaro2026EnvCfg_Gen2_B()
    cfg.scene.num_envs = N
    cfg.logging.enabled = False
    cfg.seed = args.dr_seed
    if args.nominal:
        cfg.events = None
        c = cfg.controller
        c.contraction_scale_range = (1.0, 1.0)
        c.pressure_tau_scale_range = (1.0, 1.0)
        c.pressure_L_scale_range = (1.0, 1.0)
        c.cmd_delay_range_s = (c.cmd_delay_s, c.cmd_delay_s)
    if args.model == "E":
        cfg.use_frame_stacking = True
        cfg.frame_stack_k = 5
    dt_ctrl = cfg.sim.dt * cfg.decimation
    base = C.obs_base_dim(cfg, int(cfg.lookahead_horizon / dt_ctrl))
    cfg.observation_space = base * cfg.frame_stack_k if cfg.use_frame_stacking else base
    if args.video:
        cfg.sim.render_interval = cfg.decimation

    out = ROOT / args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    n_steps = int(np.ceil(dur / dt_ctrl)) + 2
    if args.t_end:
        n_steps = min(n_steps, int(np.ceil(args.t_end / dt_ctrl)))

    envg = gym.make(TASK, cfg=cfg, render_mode="rgb_array" if args.video else None)
    env = envg.unwrapped
    if args.video:
        envg.metadata["render_fps"] = int(round(1.0 / dt_ctrl))           # 1 フレーム = 1 制御周期 = 実時間
        envg = gym.wrappers.RecordVideo(envg, video_folder=str(out / "video"), name_prefix=args.name,
                                        step_trigger=lambda s: s == 0, video_length=n_steps)
    policy, wrapper = None, None
    if args.ckpt:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner
        entry = "rsl_rl_lstm_cfg_entry_point" if args.model == "B" else "rsl_rl_mlp_cfg_entry_point"
        agent_cfg = load_cfg_from_registry(TASK, entry)
        wrapper = RslRlVecEnvWrapper(envg, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.ckpt)
        policy = runner.get_inference_policy(device=env.device)

    torch.manual_seed(args.dr_seed + 1)           # eval_b と同じ
    torch.cuda.manual_seed_all(args.dr_seed + 1)
    (wrapper if wrapper is not None else envg).reset()
    dev = env.device
    gen = env.rhythm_generator
    T = gen.target_trajectories.shape[1]
    tr = torch.zeros((N, T), device=dev)
    Tn = min(T, len(tgt))
    tr[:, :Tn] = torch.as_tensor(tgt[:Tn], device=dev, dtype=torch.float32)
    gen.target_trajectories = tr
    gen.current_bpms[:] = float(bpm)
    env.episode_duration_steps[:] = env.max_episode_length + 10
    if hasattr(env, "obs_history"):
        env.obs_history[:] = 0.0

    cam = None
    if args.video:
        sid = env.robot.find_bodies("Stick_link")[0][0]
        ps = env.robot.data.body_pos_w[0, sid].cpu().numpy()
        pd_ = env.drum.data.root_pos_w[0].cpu().numpy()
        # 注視点：スティックと打面の水平の中点、高さはスティック（打面の物体の原点は床の高さにあるので、z は使わない。11/05 修正）
        look = np.array([float(v) for v in args.cam_lookat.split(",")]) if args.cam_lookat else \
            np.array([(ps[0] + pd_[0]) / 2, (ps[1] + pd_[1]) / 2, ps[2]])
        rb = env.robot.data.root_pos_w[0].cpu().numpy()
        u = (pd_ - rb)[:2]
        u = u / (np.linalg.norm(u) + 1e-9)                       # ロボットの根元 → 打面（水平）
        hdir = {"side": np.array([-u[1], u[0]]) * args.cam_side, "behind": -u, "front": u}[args.cam_azim]
        r = -np.radians(args.cam_rot_deg)
        hdir = np.array([np.cos(r) * hdir[0] - np.sin(r) * hdir[1], np.sin(r) * hdir[0] + np.cos(r) * hdir[1]])
        el = np.radians(args.cam_elev_deg)
        off = np.r_[args.cam_dist * np.cos(el) * hdir, args.cam_dist * np.sin(el)]
        if args.cam_elev_deg > 85:                                # 真上は向きが決まらないので少しずらす
            off[:2] += 0.02 * u
        eye = np.array([float(v) for v in args.cam_eye.split(",")]) if args.cam_eye else look + off
        cc = getattr(env, "viewport_camera_controller", None)
        if cc is not None:
            cc.update_view_location(eye=tuple(eye), lookat=tuple(look))
        else:
            env.sim.set_camera_view(eye=tuple(eye), target=tuple(look))
        cam = dict(eye=eye.tolist(), lookat=look.tolist(), stick=ps.tolist(), drum=pd_.tolist())
        print(f"[sim_trace] カメラ eye={np.round(eye, 3).tolist()} lookat={np.round(look, 3).tolist()}"
              f"（スティック {np.round(ps, 3).tolist()}、打面 {np.round(pd_, 3).tolist()}）")

    # 物理ステップごとの記録（scene.update = 物理 1 ステップのあと。eval_b の力と同じ時刻）
    ctrl = env.action_controller
    D = cfg.decimation
    rec = torch.zeros((n_steps * D, N, 9), device=dev)
    ptr = [0]
    orig_update = env.scene.update

    def rec_update(*a, **k):
        r = orig_update(*a, **k)
        if ptr[0] < rec.shape[0]:
            tel = ctrl.get_last_telemetry()
            q, _ = env._get_corrected_joint_state()
            f = env.stick_sensor.data.net_forces_w
            f = f[:, 0, :] if f.dim() == 3 else f
            rec[ptr[0], :, 0:3] = tel["P_cmd"].float()
            rec[ptr[0], :, 3:6] = tel["P_out"].float()
            rec[ptr[0], :, 6:8] = torch.rad2deg(q[:, :2]).float()
            rec[ptr[0], :, 8] = f[:, 2] * env.force_scale_sim_to_real
        ptr[0] += 1
        return r
    env.scene.update = rec_update

    A = None
    if args.ff:
        prims = [ff_lib.Prim.from_dict(d) for d in json.load(open(args.ff))["prims"]]
        cmd = ff_lib.synth(np.asarray(ons), prims, n_steps).astype(np.float32)
        A = torch.as_tensor(cmd / cfg.controller.Pmax * 2.0 - 1.0, device=dev).unsqueeze(1).expand(-1, N, -1)
    if args.signal:
        sig = pd.read_csv(args.signal)
        cmd = sig[["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].to_numpy(np.float32)
        cmd = np.vstack([cmd, np.repeat(cmd[-1:], max(0, n_steps - len(cmd)), 0)])[:n_steps]
        A = torch.as_tensor(cmd / cfg.controller.Pmax * 2.0 - 1.0, device=dev).unsqueeze(1).expand(-1, N, -1)

    t0 = time.time()
    with torch.inference_mode():
        if policy is not None:
            o = wrapper.get_observations()
            obs = o[0] if isinstance(o, tuple) else o
            if hasattr(policy, "reset_memory"):
                policy.reset_memory()
        for k in range(n_steps):
            if policy is not None:
                obs, _, _, _ = wrapper.step(policy(obs))
            else:
                envg.step(A[k].contiguous())
    env.scene.update = orig_update
    print(f"[sim_trace] {n_steps} step × {N} env、{time.time() - t0:.0f} s")

    r = rec[: ptr[0]].cpu().numpy()
    tt = (np.arange(r.shape[0]) + 1) * cfg.sim.dt
    tg5 = np.repeat(tgt[:n_steps], D)[: len(tt)] if len(tgt) >= n_steps else np.repeat(np.pad(tgt, (0, n_steps - len(tgt))), D)[: len(tt)]
    rows = []
    for e in range(N):
        ts = pd.DataFrame({"time_s": tt, "target_force": tg5, "force_z": r[:, e, 8]})
        ts = ts[ts.time_s <= dur + 0.3]
        st = ff_score.score_sim(ts, float(bpm))
        st["env"] = e
        rows.append(st)
    st = pd.concat(rows, ignore_index=True)
    st.to_csv(out / "strikes.csv", index=False)
    if N == 1:
        np.savez_compressed(out / "trace.npz", time=tt, P_cmd=r[:, 0, 0:3], P_out=r[:, 0, 3:6], q=r[:, 0, 6:8], force=r[:, 0, 8],
                            target=tg5, onsets=ons, bpm=bpm)
    else:
        # DR の束は env をまたいだ四分位だけ残す（全 env だと 1 本 数十 MB になる）。打点ごとの結果は strikes.csv に全 env 分ある
        qs = np.percentile(r, [25, 50, 75], axis=1).astype(np.float32)           # (3, T, 9)
        np.savez_compressed(out / "trace.npz", time=tt, quantiles=np.array([25, 50, 75]),
                            P_cmd=qs[:, :, 0:3], P_out=qs[:, :, 3:6], q=qs[:, :, 6:8], force=qs[:, :, 8],
                            target=tg5, onsets=ons, bpm=bpm)
    summ = dict(name=args.name, midi=args.midi, model=args.model, ckpt=args.ckpt, signal=args.signal, ff=args.ff, nominal=args.nominal,
                n_env=N, dr_seed=args.dr_seed, n_targets=int(len(st)), **ff_score.summarize(st),
                first_target_s=float(ons[0]), cols=dict(P="DF, F, G [MPa]", q="wrist, grip [deg, 実機の向き]", force="k_t × z [N]"))
    json.dump(summ, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False))

    if args.video:
        envg.close()          # RecordVideo の書き出し
        hits = st[(st.env == 0) & st.peak_time.notna()].sort_values("peak_time")
        first_strike = float(hits.peak_time.iloc[0]) if len(hits) else None
        meta = dict(fps=int(round(1.0 / dt_ctrl)), frame0_time_s=float(dt_ctrl),
                    note="フレーム i（0 始まり）は sim の時刻 (i + 1) × 0.02 s のあとの状態",
                    first_target_s=float(ons[0]), first_strike_s=first_strike, camera=cam,
                    files=sorted(str(f.relative_to(ROOT)) for f in (out / "video").glob("*.mp4")))
        json.dump(meta, open(out / "video_meta.json", "w"), indent=1, ensure_ascii=False)
        print("[sim_trace] 動画:", meta["files"], "最初の打撃（sim）:", first_strike)
    else:
        env.close()


if __name__ == "__main__":
    main()
    app.close()
