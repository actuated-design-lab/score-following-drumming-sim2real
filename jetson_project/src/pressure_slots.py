"""pressure_slots.py — MicroLabBox の圧力の 3 つの枠を、筋（DF / F / G）の名前にそろえる（config/pressure_slots.yaml）"""
from __future__ import annotations

import os

import yaml

_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "pressure_slots.yaml")


def load_slots(path: str = _CFG) -> list[str]:
    with open(path, encoding="utf-8") as f:
        slots = yaml.safe_load(f)["current"]
    if sorted(slots) != ["DF", "F", "G"]:
        raise ValueError(f"{path} の current は DF, F, G を 1 つずつ: {slots}")
    return slots


SLOTS = load_slots()


def named(p0: float, p1: float, p2: float) -> dict:
    """受信の 3 つの枠 → {'meas_pres_DF': .., 'meas_pres_F': .., 'meas_pres_G': ..}"""
    return {f"meas_pres_{m}": v for m, v in zip(SLOTS, (p0, p1, p2))}
