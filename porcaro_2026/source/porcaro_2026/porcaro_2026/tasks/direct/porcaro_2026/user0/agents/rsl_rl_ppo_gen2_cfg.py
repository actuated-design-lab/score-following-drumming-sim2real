"""Gen2（user0/GEN2.md）のタスク用の PPO 設定。ネットワークと学習の設定は Gen1 と同じ。

変えるのは experiment_name だけ。Gen1 の学習ログ（porcaro_rslrl_{lstm_modelB_DR,mlp_modelB_DR_lookahead5}/）と
同じフォルダに入れないため。
"""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg as _DefaultRunnerCfg
from .rsl_rl_ppo_lstm_cfg import PPORunnerCfg as _LstmRunnerCfg
from .rsl_rl_ppo_mlp_cfg import PPORunnerCfg as _MlpRunnerCfg


@configclass
class PPORunnerCfg(_DefaultRunnerCfg):
    experiment_name = "porcaro_rslrl_lstm_gen2"


@configclass
class LstmPPORunnerCfg(_LstmRunnerCfg):
    experiment_name = "porcaro_rslrl_lstm_gen2"


@configclass
class MlpPPORunnerCfg(_MlpRunnerCfg):
    experiment_name = "porcaro_rslrl_mlp_gen2"
