from __future__ import annotations
import abc
import torch
from typing import Protocol

class RobotLike(Protocol):
    def set_joint_velocity_target(self, cmd, joint_ids: list[int] | None = None) -> None: ...
    def set_joint_effort_target(self, cmf) -> None: ...

class ActionController(abc.ABC):
    """行動を関節への指令にする部分のインタフェース。env からはこれだけ呼ぶ。"""
    @abc.abstractmethod
    def apply(self, *,
              actions: torch.Tensor,          # (N, 3)
              q: torch.Tensor,                # (N, DOF)
              joint_ids: tuple[int, int],     # (wrist_id, grip_id)
              robot: RobotLike) -> None:
        ...
