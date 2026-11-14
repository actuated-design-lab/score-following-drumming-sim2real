"""ff_sim.py — FF ベースラインを Gen2 の sim で回す（Isaac Lab の python で。WS 用）

モード:
  probe : 段 1。Gen2 の名目値（DR なし）で、env ごとに違う 1 打（と 2 打）の指令を流し、
          指令から接触までの時間 L と打撃力を測る。結果は npz と params.csv（集計は analyze_probe.py）
  （opt・eval は段 2・3 で足す）

打撃力の記録は env のログ（simulation_log.csv の force_z）と同じ量・同じ時点：
  物理ステップごとに、sim.step → scene.update の直後のスティックの接触力 × k_t の z 成分（env の logging_manager と同じ）。
  比べるために大きさ（ノルム）も残す。全 env 分を残す。

  python scripts/tools/user0/ff/ff_sim.py probe --headless --out data/user0/ff/probe
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description="FF ベースラインを Gen2 の sim で回す")
p.add_argument("mode", choices=["probe"])
p.add_argument("--out", default="data/user0/ff/probe")
p.add_argument("--t_settle", type=float, default=3.0, help="発火までの待機 [s]（姿勢を落ち着かせる）")
p.add_argument("--t_after", type=float, default=1.5, help="最後の発火のあとに回す時間 [s]")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

# ここから Isaac Lab のモジュールを import できる
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import torch                  # noqa: E402

from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env import Porcaro2026Env     # noqa: E402
from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env_cfg import Porcaro2026EnvCfg_Gen2  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ff_lib as FL           # noqa: E402

# 段 1 の振り方（初期の当たりを付けるための粗い格子。値は ff_lib.Prim と同じ意味）
GRID_A = dict(a_up=(0.2, 0.4, 0.6), a_dn=(0.3, 0.45, 0.6), h=(0.06, 0.10, 0.16, 0.24), g=(0.0, 0.3, 0.6))
GRID_B_PRIMS = [dict(a_up=a, a_dn=0.6, h=h, g=0.3) for a in (0.4, 0.6) for h in (0.06, 0.10)]
GRID_B_IOI_MS = (80, 100, 120, 140, 160, 200, 240, 280, 320, 400, 600)


def probe_plan():
    """env ごとの (群, 値, 発火時刻の並び)。群 A = 1 打、群 B = 2 打（間隔 IOI）"""
    rows = []
    for vals in itertools.product(*GRID_A.values()):
        prm = dict(zip(GRID_A.keys(), vals))
        rows.append(dict(group="A", ioi_ms=np.nan, **prm, fires=[args.t_settle]))
    for prm in GRID_B_PRIMS:
        for ioi in GRID_B_IOI_MS:
            rows.append(dict(group="B", ioi_ms=float(ioi), **prm, fires=[args.t_settle, args.t_settle + ioi / 1000.0]))
    return rows


def build_commands(plan, n_steps):
    """env ごとの指令 (n_steps, N, 3) [MPa]。発火時刻 = 目標時刻として L = 0 で ff_lib.synth に渡す"""
    cmd = np.zeros((n_steps, len(plan), 3), dtype=np.float32)
    for j, r in enumerate(plan):
        prim = FL.Prim(L=0.0, a_dn=r["a_dn"], h=r["h"], a_up=r["a_up"], g_up=r["g"], g_dn=r["g"], d_g=r["h"])
        cmd[:, j, :] = FL.synth(np.array(r["fires"]), [prim] * (len(FL.IOI_EDGES_MS) + 1), n_steps)
    return cmd


def make_env(num_envs):
    cfg = Porcaro2026EnvCfg_Gen2()
    cfg.scene.num_envs = num_envs
    cfg.events = None                         # 名目値（DR なし）
    cfg.logging.enabled = False
    cfg.episode_length_s = 60.0
    return Porcaro2026Env(cfg), cfg


def run_open_loop(env, cfg, cmd):
    """指令 (n_steps, N, 3) を流し、物理ステップごとの force_z・力の大きさ・手首角を返す（(T, N) ずつ）"""
    N = env.num_envs
    ctrl = env.action_controller
    Pmax = ctrl.Pmax
    rec = {"force_z": [], "force_mag": [], "wrist_deg": []}
    state = {"on": False}
    orig_update = env.scene.update

    def update_and_record(dt):
        orig_update(dt=dt)
        if not state["on"]:
            return
        fv = env.stick_sensor.data.net_forces_w
        fv = (fv[:, 0, :] if fv.dim() == 3 else fv) * env.force_scale_sim_to_real
        q, _ = env._get_corrected_joint_state()
        rec["force_z"].append(fv[:, 2].float().clone())
        rec["force_mag"].append(torch.norm(fv, dim=-1).float())
        rec["wrist_deg"].append(torch.rad2deg(q[:, 0]).float())

    env.scene.update = update_and_record
    env.reset()
    ctrl.reset(N, env.device)                 # replay_open_loop.py と同じ：reset 後にチャネルを初期化し直す
    big = torch.full_like(env.episode_duration_steps, 10 ** 9)
    cmd_t = torch.as_tensor(cmd, device=env.device)
    state["on"] = True
    t0 = time.time()
    for k in range(cmd.shape[0]):
        env.episode_duration_steps[:] = big   # BPM 由来のタイムアウトを無効化
        env.step(cmd_t[k] / Pmax * 2.0 - 1.0)
        if (k + 1) % 100 == 0 or k + 1 == cmd.shape[0]:
            print(f"[ff_sim] {(k + 1) * FL.CTRL_DT:5.1f} / {cmd.shape[0] * FL.CTRL_DT:.1f} s（経過 {time.time() - t0:.0f} s）",
                  flush=True)
    state["on"] = False
    return {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}, cfg.sim.dt


def main():
    os.makedirs(args.out, exist_ok=True)
    plan = probe_plan()
    t_end = max(max(r["fires"]) for r in plan) + args.t_after
    n_steps = int(round(t_end / FL.CTRL_DT))
    cmd = build_commands(plan, n_steps)
    env, cfg = make_env(len(plan))
    rec, dt_phys = run_open_loop(env, cfg, cmd)
    T = rec["force_z"].shape[0]
    np.savez_compressed(os.path.join(args.out, "probe_envs.npz"), time=(np.arange(T) + 1) * dt_phys,
                        cmd=cmd, **rec)
    pd.DataFrame([{k: (json.dumps(v) if k == "fires" else v) for k, v in r.items()} for r in plan]).to_csv(
        os.path.join(args.out, "params.csv"), index_label="env")
    meta = dict(env_cfg="Porcaro2026EnvCfg_Gen2（DR なし）", num_envs=len(plan), dt_phys=dt_phys, ctrl_dt=FL.CTRL_DT,
                impact_force_scale=float(env.force_scale_sim_to_real), t_settle=args.t_settle, t_after=args.t_after,
                grid_A=GRID_A, grid_B_prims=GRID_B_PRIMS, grid_B_ioi_ms=GRID_B_IOI_MS,
                note="time は物理ステップの終わりの時刻（env のログの time_s と同じ数え方）。指令 k は [k·20 ms, (k+1)·20 ms) に効く")
    with open(os.path.join(args.out, "probe_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[ff_sim] 保存: {args.out}/probe_envs.npz・params.csv・probe_meta.json（{len(plan)} env、{T} 物理ステップ）")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # replay_open_loop.py と同じ：例外のときに app.close() で固まらないよう、表示を残して閉じずに終える
        import traceback
        msg = "\n[ff_sim] エラーで止まりました:\n" + traceback.format_exc()
        os.write(2, msg.encode("utf-8", "replace"))
        with open("ff_sim_error.log", "w", encoding="utf-8") as fh:
            fh.write(msg)
        os._exit(1)
    app.close()
