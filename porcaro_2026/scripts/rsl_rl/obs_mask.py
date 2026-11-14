"""Masking of a far-future window of the lookahead observation (used by play_sim_midi.py --mask_mode).

The lookahead buffer is ordered near-to-far (rhythm_generator.get_lookahead), so a [lo_s, hi_s) time
window is a contiguous slice of observation columns after the base state block.
Only the v1 layout is supported (base block = columns [0, 10)); v2 places the lookahead at a different offset.
"""

from __future__ import annotations

import torch

MASK_MODES = ("none", "zero", "noise", "shuffle")
BASE_OBS_DIM = 10  # q(2)+qd(2)+prev_act(3)+sin(1)+cos(1)+bpm(1), fixed regardless of lookahead_horizon


def compute_mask_range(dt_ctrl: float, lookahead_steps: int, mask_lo_s: float, mask_hi_s: float) -> tuple[int, int]:
    """Map a [mask_lo_s, mask_hi_s) time window to [lo_idx, hi_idx) observation columns.

    Only valid for the non-frame-stacked observation layout (obs_single, not
    the frame-stacking reshape) - callers must not use this when
    env_cfg.use_frame_stacking is True.
    """
    lo_step = max(0, int(round(mask_lo_s / dt_ctrl)))
    hi_step = min(int(round(mask_hi_s / dt_ctrl)), lookahead_steps)
    lo_idx = BASE_OBS_DIM + lo_step
    hi_idx = BASE_OBS_DIM + max(lo_step, hi_step)
    return lo_idx, hi_idx


def policy_tensor(obs) -> torch.Tensor:
    """Return the underlying policy observation tensor, whether `obs` is a
    plain torch.Tensor or a TensorDict-like mapping with a "policy" entry."""
    if not isinstance(obs, torch.Tensor) and hasattr(obs, "__getitem__") and "policy" in obs:
        return obs["policy"]
    return obs


def apply_mask(obs, mode: str, lo_idx: int, hi_idx: int):
    """Overwrite the [lo_idx, hi_idx) columns of the policy observation in place.

    `obs` as returned by RslRlVecEnvWrapper (rsl-rl>=3.0's obs_groups support)
    is a TensorDict-like mapping with a "policy" entry, not a plain
    torch.Tensor - obs["policy"][:, lo:hi] must be indexed/assigned rather
    than obs[:, lo:hi] directly (the latter raises IndexError: tuple index
    out of range, since TensorDict.__getitem__ interprets a tuple index as
    multi-dim *batch* indexing, not per-tensor column slicing). Falls back to
    treating `obs` itself as the tensor when it has no "policy" key, so this
    also works for envs/wrappers that return a plain tensor.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"apply_mask: unknown mode {mode!r}, expected one of {MASK_MODES}")
    if mode == "none" or lo_idx >= hi_idx:
        return obs

    is_tensordict = not isinstance(obs, torch.Tensor) and hasattr(obs, "__getitem__") and "policy" in obs
    tensor = obs["policy"] if is_tensordict else obs

    segment = tensor[:, lo_idx:hi_idx]
    if mode == "zero":
        tensor[:, lo_idx:hi_idx] = 0.0
    elif mode == "noise":
        # rhythm_buf is normalized by target_hit_force before concatenation
        # (porcaro_2026_env.py::_get_observations(): rhythm_buf /
        # self.target_hit_force), so values are O(0-1) - uniform [0,1) noise
        # matches that scale without smuggling in real target information.
        tensor[:, lo_idx:hi_idx] = torch.rand_like(segment)
    elif mode == "shuffle":
        # Single permutation shared across the batch: destroys the specific
        # near-to-far timing order within the masked window while preserving
        # the marginal value distribution (unlike "zero"/"noise", which also
        # change the values themselves).
        perm = torch.randperm(segment.shape[1], device=tensor.device)
        tensor[:, lo_idx:hi_idx] = segment[:, perm]

    if is_tensordict:
        obs["policy"] = tensor
    return obs
