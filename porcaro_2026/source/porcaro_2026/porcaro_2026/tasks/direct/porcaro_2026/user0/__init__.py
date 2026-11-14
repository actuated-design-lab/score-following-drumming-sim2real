# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import agents

# 周期的でない打撃列で学習するタスク（2026-11-01）。機構側と DR は Gen2_DR と同じ（porcaro_2026_env_cfg.Porcaro2026EnvCfg_Gen2_B）。
# Gen2 / Gen2_DR の cfg クラスは継承元として残し、タスクとしては登録しない。
#   Porcaro-Gen2-B-user0       観測 v2（拍位相なし、実測圧力 3、接触のラッチフラグ 1）
#   Porcaro-Gen2-B-obsv1-user0 分布は同じ、観測は Gen1 と同じ v1
# 学習ログは porcaro_rslrl_*_gen2/ に入る。run_name の TAG で分ける（scripts/tools/user0/gen2/train_b.sh）
for _id, _cls in (("Porcaro-Gen2-B-user0", "Porcaro2026EnvCfg_Gen2_B"),
                  ("Porcaro-Gen2-B-obsv1-user0", "Porcaro2026EnvCfg_Gen2_B_ObsV1")):
    gym.register(
        id=_id,
        entry_point=f"{__name__}.porcaro_2026_env:Porcaro2026Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.porcaro_2026_env_cfg:{_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_gen2_cfg:PPORunnerCfg",
            "rsl_rl_lstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_gen2_cfg:LstmPPORunnerCfg",
            "rsl_rl_mlp_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_gen2_cfg:MlpPPORunnerCfg",
        },
    )
