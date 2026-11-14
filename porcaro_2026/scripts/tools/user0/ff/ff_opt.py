"""ff_opt.py — FF 段 2：F0 のプリミティブ（IOI の 4 区分 × L・a_dn・h・a_up・g）を CMA-ES で最適化する（WS、Isaac）

FF 設計メモ §4・§6 のとおり：
  - (b) の分布で調整する。列は b_sets.one() で作る。種は 40000 + 100000 × restart + n_seq × 世代 + i
    （検証用 20000 台・ホールドアウト 30000 台・学習の 1〜3 と重ならない。世代ごとに違う列を使う）
  - 主：DR の分布で最適化（Gen2_B の cfg そのまま）。比較：--nominal で DR を切った名目値だけの版
  - 目的関数（最大化）：目標ごとに、なめらかにした成功 exp(-½(誤差/30 ms)²) × min(F/1 N, 1) の平均 − w_force × |F − 20 N| / 20 N の平均
    − w_press × 押し付けの割合（100 ms を超えた接触の時間 ÷ 全体の時間。RL の contact_limit に当たる。2026-11-03 追加、w_press = 1.0）
    （2026-11-04：押し付けの割合は、離れが 60 ms 未満でつながった接触の連なりを 1 つの接触として数える。採点は 40 ms の連なりで判定。
     sim の跳ねの連続を打撃にも罰の外にもしないため。access_1104/c5、claude/FF再最適化_決定_20261104.md）
    （打っていない目標は F = 0 として |ΔF| = 20 N）。報告する数字は評価と同じ式（ff_score）で、別に eval_b.py で出す
  - 1 世代 = 1 エピソード。env = 候補 × 列 × DR の引き（既定 16 × 8 × 16 = 2048）
    同じ（列、引き）の env どうしで、リセット時の DR（k・θt・τ・L・通信の遅れ）と初期関節角をそろえる（候補の間でペア）。
    起動時の DR（質量・摩擦・重心・手首の粘性と摩擦）は env ごとに違うまま（そろえる手段がない）。
  - restart ごとに初期値を変える（0 = 段 1 で 15〜25 N に入った組を 4 区分に複製、1 以降 = それに乱数を足す）。
    各 restart の最良の解を、あとで eval_b.py の検証用の列で比べて 1 本選ぶ（シードの選択と同じ規則）

  python scripts/tools/user0/ff/ff_opt.py --headless --restart 0 --gens 60
  python scripts/tools/user0/ff/ff_opt.py --headless --restart 0 --gens 60 --nominal
出力：data/user0/ff/opt/<tag>/log.csv（世代ごと）、best.json（eval_b.py の --ff に渡す形）、summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--restart", type=int, default=0)
p.add_argument("--gens", type=int, default=80, help="世代（2026-11-03 夜：28 次元にしたので 60 → 80）")
p.add_argument("--popsize", type=int, default=16)
p.add_argument("--n_seq", type=int, default=8)
p.add_argument("--n_dr", type=int, default=16)
p.add_argument("--sigma0", type=float, default=0.2, help="正規化した探索空間（各軸 0〜1）での初期の幅")
p.add_argument("--w_force", type=float, default=0.25, help="|F − 20 N| / 20 N の項の重み（0.25、2026-11-03 確定。段 2 を回す前に固定）")
p.add_argument("--w_press", type=float, default=1.0,
               help="押し付けの罰の重み：100 ms を超えた接触の時間の割合（ff_score.excess_contact_frac）に掛ける。RL の contact_limit に当たる"
                    "（2026-11-03 本人の決定、段 2 のやり直しの前に固定）")
p.add_argument("--press_train_gap_s", type=float, default=None,
               help="押し付けの罰で、離れがこれ未満の接触の連なりを 1 つの接触として数える（既定 ff_score.PRESS_TRAIN_GAP_S = 60 ms、"
                    "2026-11-04 本人の決定：採点の 40 ms より厳しめにして、残った跳ねに寄らないようにする）")
p.add_argument("--nominal", action="store_true", help="DR を切った名目値だけで最適化（比較用）")
p.add_argument("--out", default="data/user0/ff/opt")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import cma  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/tools/user0/ff"))
sys.path.insert(0, str(ROOT / "scripts/tools/user0/eval"))
import b_sets  # noqa: E402
import ff_lib  # noqa: E402
import ff_score  # noqa: E402

PG = ff_score.PRESS_TRAIN_GAP_S if args.press_train_gap_s is None else args.press_train_gap_s   # 押し付けの罰の連なりの離れ [s]

from porcaro_2026.tasks.direct.porcaro_2026.user0 import porcaro_2026_env_cfg as C  # noqa: E402
from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env import Porcaro2026Env  # noqa: E402

KEYS = ff_lib.PRIM_KEYS                          # L, a_dn, h, a_up, g_up, g_dn, d_g（2026-11-03 夜：G を 1 打の中で変えられるように 5 → 7）
LO = np.array([0.05, 0.0, 0.02, 0.0, 0.0, 0.0, -0.10])
HI = np.array([0.45, 0.6, 0.40, 0.6, 0.6, 0.6, 0.20])     # d_g：G を戻す時刻は目標の 100 ms 前〜200 ms 後
NB = len(ff_lib.IOI_EDGES_MS) + 1
X0_PRIM = dict(L=0.275, a_dn=0.45, h=0.24, a_up=0.4, g_up=0.6, g_dn=0.6, d_g=0.0)   # 段 1 で 15〜25 N に入った組（data/user0/ff/probe/summary.md）。G は 1 打の中で一定（g = 0.6）から始める
TAG = f"r{args.restart}" + ("_nominal" if args.nominal else "")


def to_prims(u):
    u = np.clip(np.asarray(u, float).reshape(NB, len(KEYS)), 0.0, 1.0)
    x = LO + u * (HI - LO)
    return [ff_lib.Prim(**dict(zip(KEYS, r))) for r in x]


def from_prim(d):
    return (np.array([d[k] for k in KEYS]) - LO) / (HI - LO)


def main():
    lam, ns, nd = args.popsize, args.n_seq, args.n_dr
    R = ns * nd
    N = lam * R
    cfg = C.Porcaro2026EnvCfg_Gen2_B()
    cfg.scene.num_envs = N
    cfg.logging.enabled = False
    cfg.seed = 9000 + args.restart
    if args.nominal:
        cfg.events = None
        c = cfg.controller
        c.contraction_scale_range = (1.0, 1.0)
        c.pressure_tau_scale_range = (1.0, 1.0)
        c.pressure_L_scale_range = (1.0, 1.0)
        c.cmd_delay_range_s = None
    env = Porcaro2026Env(cfg)
    dev, D, dt = env.device, cfg.decimation, cfg.sim.dt * cfg.decimation
    ctrl, gen = env.action_controller, env.rhythm_generator
    e_idx = torch.arange(N, device=dev)
    base_of = e_idx % R                         # 候補 0 の、同じ（列、引き）の env
    seq_of = (e_idx % R) % ns
    cand_of = (e_idx // R).cpu().numpy()

    Fz_buf = {}
    orig_update = env.scene.update

    def rec_update(*a, **k):
        r = orig_update(*a, **k)
        f = env.stick_sensor.data.net_forces_w
        f = f[:, 0, :] if f.dim() == 3 else f
        if Fz_buf["ptr"] < Fz_buf["F"].shape[0]:
            Fz_buf["F"][Fz_buf["ptr"]] = f[:, 2] * env.force_scale_sim_to_real
        Fz_buf["ptr"] += 1
        return r

    rng = np.random.default_rng(args.restart)
    u0 = np.tile(from_prim(X0_PRIM), NB)
    if args.restart > 0:
        u0 = np.clip(u0 + rng.normal(0, 0.15, u0.shape), 0.02, 0.98)
    es = cma.CMAEvolutionStrategy(u0, args.sigma0, {"bounds": [0, 1], "popsize": lam, "seed": 1000 + args.restart, "verbose": -9})
    out = ROOT / args.out / TAG
    out.mkdir(parents=True, exist_ok=True)
    log = []
    env_steps = 0
    t_all = time.time()
    best = (-np.inf, None, None)
    for g in range(args.gens):
        t0 = time.time()
        seeds = [40000 + 100000 * args.restart + ns * g + i for i in range(ns)]
        seqs = [b_sets.one(s) for s in seeds]
        n_steps = int(np.ceil(max(s[2] for s in seqs) / dt)) + 2
        U = es.ask()
        # リセット（DR を引き直す）→ 候補の間でそろえる
        torch.manual_seed(100000 * args.restart + g)
        env.reset()
        for t_ in (ctrl.k_contr, ctrl.cmd_delay_L):
            if t_ is not None:
                t_[:] = t_[base_of]
        for k_ in ctrl.theta_t_rad_tensors:
            ctrl.theta_t_rad_tensors[k_][:] = ctrl.theta_t_rad_tensors[k_][base_of]
        for nm in ("ch_DF", "ch_F", "ch_G"):
            ch = getattr(ctrl, nm)
            for a_ in ("tau", "L"):
                v = getattr(ch, a_, None)
                if isinstance(v, torch.Tensor) and v.shape[0] == N:
                    v[:] = v[base_of]
        jp = env.robot.data.joint_pos[base_of].clone()
        jv = env.robot.data.joint_vel[base_of].clone()
        env.robot.write_joint_state_to_sim(jp, jv)
        # 列を入れる
        T = gen.target_trajectories.shape[1]
        tr = torch.zeros((N, T), device=dev)
        trajs = np.stack([s[0][:T] for s in seqs])
        tr[:] = torch.as_tensor(trajs, device=dev)[seq_of]
        gen.target_trajectories = tr
        gen.current_bpms[:] = torch.as_tensor(np.array([s[1] for s in seqs]), device=dev, dtype=torch.float32)[seq_of]
        env.episode_duration_steps[:] = env.max_episode_length + 10
        # 指令
        cmds = np.zeros((lam, ns, n_steps, 3), np.float32)
        for c in range(lam):
            pr = to_prims(U[c])
            for i, s in enumerate(seqs):
                cmds[c, i] = ff_lib.synth(np.asarray(s[3]) * dt, pr, n_steps)
        A = torch.as_tensor(cmds, device=dev)[torch.as_tensor(cand_of, device=dev), seq_of]      # (N, n_steps, 3)
        A = A / cfg.controller.Pmax * 2.0 - 1.0
        Fz_buf["F"] = torch.zeros((n_steps * D, N), device=dev)
        Fz_buf["ptr"] = 0
        env.scene.update = rec_update
        # inference_mode にしない：その中で作られた Isaac の内部バッファ（joint_acc など）を、次の世代の env.reset が
        # inference_mode の外で書き換えて落ちる（2026-11-03）。勾配が要らないだけなので no_grad で足りる
        with torch.no_grad():
            for k in range(n_steps):
                env.step(A[:, k])
        env.scene.update = orig_update
        env_steps += n_steps * N
        Fz = Fz_buf["F"][: Fz_buf["ptr"]].cpu().numpy()
        tt = (np.arange(Fz.shape[0]) + 1) * cfg.sim.dt
        so = seq_of.cpu().numpy()
        J = np.zeros(lam)
        succ = np.zeros(lam)
        dfa = np.zeros(lam)
        per_c = [[] for _ in range(lam)]
        press_c = [[] for _ in range(lam)]
        for e in range(N):
            s = so[e]
            ts = pd.DataFrame({"time_s": tt, "target_force": np.repeat(seqs[s][0][:n_steps], D)[: len(tt)], "force_z": Fz[:, e]})
            ts = ts[ts.time_s <= seqs[s][2] + 0.3]
            per_c[cand_of[e]].append(ff_score.score_sim(ts, seqs[s][1]))
            press_c[cand_of[e]].append(ff_score.excess_contact_frac(ts.time_s.to_numpy(), ts.force_z.to_numpy(), train_gap_s=PG))
        press = np.array([np.mean(x) for x in press_c])
        for c in range(lam):
            st = pd.concat(per_c[c], ignore_index=True)
            F = st.peak_force.fillna(0.0).to_numpy()
            err = st.timing_err_ms.to_numpy()
            hit = np.minimum(F / 1.0, 1.0)
            tim = np.where(np.isfinite(err), np.exp(-0.5 * (np.nan_to_num(err) / ff_score.TOL_MS) ** 2), 0.0)
            dF = np.where(F >= 1.0, np.abs(F - ff_score.TARGET), ff_score.TARGET) / ff_score.TARGET
            J[c] = np.mean(tim * hit) - args.w_force * np.mean(dF) - args.w_press * press[c]
            succ[c] = st.success.mean()
            dfa[c] = np.median(np.abs(F[F >= 1.0] - ff_score.TARGET)) if (F >= 1.0).any() else np.nan
        es.tell(U, list(-J))
        cb = int(np.argmax(J))
        if J[cb] > best[0]:
            best = (J[cb], U[cb], dict(gen=g, success=float(succ[cb]), dF_abs_med=float(dfa[cb]), press=float(press[cb])))
        row = dict(gen=g, J_best=J[cb], J_mean=J.mean(), success_best=succ[cb], success_mean=succ.mean(), dF_med_best=dfa[cb], press_best=press[cb],
                   sigma=es.sigma, n_steps=n_steps, sec=time.time() - t0, env_steps=env_steps)
        log.append(row)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        json.dump({"prims": [ff_lib.asdict(p) for p in to_prims(best[1])], "J": float(best[0]), **best[2], "restart": args.restart,
                   "nominal": args.nominal, "w_force": args.w_force, "w_press": args.w_press, "press_train_gap_s": PG,
                   "score_train_gap_s": ff_score.TRAIN_GAP_S, "env_steps": env_steps},
                  open(out / "best.json", "w"), indent=1)
        print(f"[ff_opt {TAG}] gen {g}  J {J[cb]:+.3f}（平均 {J.mean():+.3f}）成功 {succ[cb]:.3f}  |ΔF| {dfa[cb]:.1f} N  押し付け {100 * press[cb]:.0f} %  σ {es.sigma:.3f}  "
              f"{time.time() - t0:.0f} s", flush=True)
    m = es.result.xfavorite
    L = [f"# FF 段 2：CMA-ES（{TAG}）", "",
         f"世代 {args.gens} × 候補 {lam}、1 世代 = 列 {ns} × DR {nd}（{'名目値' if args.nominal else 'DR あり'}）、w_force {args.w_force}、w_press {args.w_press}（押し付けの連なり {1000 * PG:.0f} ms、採点の連なり {1000 * ff_score.TRAIN_GAP_S:.0f} ms）。"
         f"env-step の総数 {env_steps:.3g}、実時間 {(time.time() - t_all) / 3600:.2f} h", "",
         f"最良（世代 {best[2]['gen']}、その世代の列と引きで）：目的 {best[0]:+.3f}、成功率 {best[2]['success']:.3f}、|ΔF| 中央値 {best[2]['dF_abs_med']:.1f} N", "",
         "| 区分（IOI） | L [s] | a_dn [MPa] | h [s] | a_up [MPa] | g_up [MPa] | g_dn [MPa] | d_g [s] |", "|---|---|---|---|---|---|---|---|"]
    edges = (0,) + tuple(ff_lib.IOI_EDGES_MS) + (np.inf,)
    for b, pr in enumerate(to_prims(best[1])):
        L.append(f"| {edges[b]:.0f}–{edges[b + 1]:.0f} ms | {pr.L:.3f} | {pr.a_dn:.3f} | {pr.h:.3f} | {pr.a_up:.3f} | {pr.g_up:.3f} | {pr.g_dn:.3f} | {pr.d_g:+.3f} |")
    L += ["", "分布の平均（xfavorite）も favorite.json に残した。報告する成績は eval_b.py（検証用・ホールドアウトの列、DR 32 回）で出す。"]
    json.dump({"prims": [ff_lib.asdict(p) for p in to_prims(m)], "restart": args.restart, "nominal": args.nominal}, open(out / "favorite.json", "w"), indent=1)
    open(out / "summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        os.write(2, ("\n[ff_opt] エラーで止まりました:\n" + traceback.format_exc()).encode("utf-8", "replace"))
        os._exit(1)
    app.close()
