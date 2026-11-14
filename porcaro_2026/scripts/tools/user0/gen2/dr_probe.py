"""dr_probe.py — Gen2 の env に実際に入っている DR の値を、env ごとに取り出す（Isaac Lab の中で使う）

replay_open_loop.py（DR の束）と test_gen2_env.py（DR が振れているかの確認）で共通に使う。
返り値は {名前: numpy 配列 (num_envs,)}。
"""
from __future__ import annotations

import numpy as np
import torch

# 抽選のタイミング（GEN2.md の DR の表）。reset = エピソードごと、startup = 起動時だけ
RESET_KEYS = ("k_DF", "k_F", "k_G", "tau_DF_s", "tau_F_s", "tau_G_s", "L_DF_s", "L_F_s", "L_G_s", "cmd_delay_s",
              "q0_wrist_deg", "q0_grip_deg")
STARTUP_KEYS = ("stick_mass_kg", "stick_com_x_m", "stick_com_y_m", "stick_com_z_m", "robot_mass_kg",
                "wrist_damping", "wrist_friction")


def dr_samples(env, ctrl, with_joint_pos: bool = False) -> dict:
    out = {}
    if ctrl.k_contr is not None:
        for i, m in enumerate(("DF", "F", "G")):
            out[f"k_{m}"] = ctrl.k_contr[:, i].cpu().numpy()
    for m, ch in (("DF", ctrl.ch_DF), ("F", ctrl.ch_F), ("G", ctrl.ch_G)):
        for name in ("tau", "L"):
            v = getattr(ch, name, None)
            if isinstance(v, torch.Tensor) and v.dim() == 1:
                out[f"{name}_{m}_s"] = v.cpu().numpy()
    if getattr(ctrl, "cmd_delay_L", None) is not None:
        out["cmd_delay_s"] = ctrl.cmd_delay_L[:, 0].cpu().numpy()
    if getattr(ctrl, "theta_t_rad_tensors", None):
        for m, v in ctrl.theta_t_rad_tensors.items():
            out[f"theta_t_{m}_deg"] = np.degrees(v.cpu().numpy())
    view = env.robot.root_physx_view
    sid = env.robot.find_bodies("Stick_link")[0][0]
    masses = view.get_masses()
    out["stick_mass_kg"] = masses[:, sid].cpu().numpy()
    coms = view.get_coms()
    for i, ax in enumerate("xyz"):
        out[f"stick_com_{ax}_m"] = coms[:, sid, i].cpu().numpy()
    out["robot_mass_kg"] = masses.sum(1).cpu().numpy()
    wj = env.robot.find_joints(".*Wrist_joint")[0][0]
    act = env.robot.actuators.get("wrist")
    if act is not None:
        out["wrist_damping"] = act.damping[:, 0].cpu().numpy()
    fc = getattr(env.robot.data, "joint_friction_coeff", None)
    if fc is not None:
        out["wrist_friction"] = fc[:, wj].cpu().numpy()
    if with_joint_pos:
        q, _ = env._get_corrected_joint_state()          # プロジェクトの向き [rad]
        out["q0_wrist_deg"] = np.degrees(q[:, 0].cpu().numpy())
        out["q0_grip_deg"] = np.degrees(q[:, 1].cpu().numpy())
    return out


def default_refs(env) -> dict:
    """倍率で振るものの既定値（DR をかける前の値）"""
    sid = env.robot.find_bodies("Stick_link")[0][0]
    wj = env.robot.find_joints(".*Wrist_joint")[0][0]
    d = env.robot.data
    ref = {}
    dm = getattr(d, "default_mass", None)
    if dm is not None:
        ref["stick_mass_kg"] = float(dm[0, sid])
    ddmp = getattr(d, "default_joint_damping", None)
    if ddmp is not None:
        ref["wrist_damping"] = float(ddmp[0, wj])
    dfc = getattr(d, "default_joint_friction_coeff", None)
    if dfc is not None:
        ref["wrist_friction"] = float(dfc[0, wj])
    return ref
