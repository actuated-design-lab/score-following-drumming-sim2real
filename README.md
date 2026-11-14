# score-following-drumming-sim2real

Code, model data, trained policies, and figure scripts for the IEEE Access article

> **Score-Following Drumming via Sim-to-Real Transfer on a Pneumatic Musculoskeletal Percussion Robot:
> Feedforward Versus Reinforcement Learning on the Same Identified Model**
> T. Biyajima, R. Oku, T. Takano, and M. Okui

This repository is a snapshot of the code at the time of the study. It contains only what the article uses.
Code comments and most internal reports (`*.md` next to the data) are in Japanese; the commands below are all you need.

## Contents

The repository holds two sibling folders. They must stay side by side under these names, because scripts in one
refer to the other by the relative path `../jetson_project` or `../porcaro_2026`.

| Folder | What it is |
|---|---|
| `porcaro_2026/` | Simulation (Isaac Lab): the training task, the PAM and pressure-channel model with its identified parameters, PPO settings, the two selected policies, the feedforward (FF) optimizer and its adopted parameters, the simulation evaluation, and the research-side scoring and aggregation scripts |
| `jetson_project/` | Hardware deployment (runs on the robot's Jetson): policy runner, observation v2, configuration, the deployed ONNX policies, hardware task files, scoring of hardware logs, and the material of Section VI-H (reproduction of a human performance) |
| `paper/figures/` | The data package behind every data-driven figure and table of the article, and the script that rebuilds them |

Main items and where they are:

| Item | Location |
|---|---|
| Training tasks | `Porcaro-Gen2-B-user0` (observation v2, used for both policies) and `Porcaro-Gen2-B-obsv1-user0` (fallback with observation v1, trained but not deployed), registered in `porcaro_2026/source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/user0/__init__.py`. Environment: `user0/porcaro_2026_env.py`; configuration: `user0/porcaro_2026_env_cfg.py` (`Porcaro2026EnvCfg_Gen2_B`, its parents `_Gen2_DR` and `_Gen2`, and `_Gen2_B_ObsV1`); frozen nominal model: `user0/GEN2.md` |
| Observation v2 | sim: `_get_observations` in `user0/porcaro_2026_env.py`; hardware: `jetson_project/src/obs_v2.py` (`ContactLatch`) and `jetson_project/src/deploy_policy.py` |
| Reward | `user0/rewards/reward.py`, weights in `user0/cfg/rewards_cfg.py` |
| PPO and network settings (Appendix) | `user0/agents/rsl_rl_ppo_gen2_cfg.py` (inherits `rsl_rl_ppo_cfg.py`, `rsl_rl_ppo_lstm_cfg.py`, `rsl_rl_ppo_mlp_cfg.py`); the exact values of each run are in `params/agent.yaml` next to the checkpoint |
| PAM model | `porcaro_2026/source/.../common/actions/pam.py` (force map, hysteresis), `torque.py` (joint torques, randomization), `common/cfg/actuator_cfg.py` (geometry); force maps `tasks/data/pam_force_map_measured_DF.csv` and `pam_force_0_map_measured_DF.csv` |
| Pressure channel and its identified parameters | `common/actions/itv.py`: `DEFAULTS` (flow-limited first-order lag, dead time, small-amplitude correction) |
| Small-amplitude correction table $G(f,A)$ (Appendix) | `common/actions/itv.py`: `TABLE_F` (Hz), `TABLE_A` (MPa), `TABLE_G`; the same table with the fitted channel parameters in `jetson_project/analysis/itv/fit_A.json` (keys `fg`, `ag`, `G`), produced by `jetson_project/analysis/itv/model_A.py` |
| Selected policies (LSTM seed 3, MLP seed 2) | `porcaro_2026/logs/user0/rsl_rl/porcaro_rslrl_lstm_gen2/2026-11-02_07-13-45_B_b_seed3/` and `porcaro_rslrl_mlp_gen2/2026-11-02_03-37-32_E_b_seed2/`: `model_1499.pt`, `params/{agent,env}.yaml`, TensorBoard events. ONNX exports used on hardware: `jetson_project/models/GEN2B/` |
| FF optimizer and adopted parameters | `porcaro_2026/scripts/tools/user0/ff/` (`ff_lib.py` primitives, `ff_opt.py` CMA-ES, `ff_score.py`, `run_ff.sh`); adopted parameters `porcaro_2026/data/user0/ff/opt2/r0/favorite.json` (restart r0, mean of the final CMA-ES distribution); the other restarts in `opt2/r1`, `opt2/r2`, `opt2/r0_nominal` |
| Hardware commands of the FF baselines | `jetson_project/signals/core2026v2/` (FF and FF-cal of the reported session, synthesized from `opt2/r0/favorite.json`), `signals/core2026/` (an earlier session, and the check motion `check_motion.csv`) |
| Task files | sim: `porcaro_2026/data/user0/eval/real_core/songs/` (the five hardware tasks), `data/user0/eval/b_sets/` (validation and evaluation sequences); hardware: `jetson_project/songs/core2026/`, calibration `songs/core2026_calib/` |
| Hardware logs of the main comparison (Sections VI-B to VI-G) | `jetson_project/data/user0/`: `access2026v3/gen2b_20261109/` (RL runs of the reported table, Nov 9) and `access2026v2/playback_20261105/` (FF and FF-cal runs, Nov 5); earlier sessions `access2026/` (Nov 3–4) and the rest of `access2026v2/`; calibration runs `calibe2026/` (Nov 11). Each run is a CSV of the 200 Hz samples and a JSON sidecar |
| Section VI-H material | MIDI transcriptions `jetson_project/songs/sss2026/okui_*` (`OKUI.md` describes them), the two scored runs `jetson_project/data/user0/okui2026/gen2b_20261111/` (MLP policy; part A `…_1794365787`, part B `…_1794365998`), onset extraction `jetson_project/tools/okui_sss/okui_onsets.py` |
| Figures and tables | `paper/figures/data/` (data package; `README.md` there lists every file, its units, and its research-side source), `paper/figures/scripts/make_figures.py` |

Not included in this snapshot: the raw identification and test-motion recordings. Their aggregated values are in
`paper/figures/data/`, and the scripts that produced them are included (see "Research-side pipeline" below).

## Environment

Tested on Ubuntu 24.04, NVIDIA RTX 5090 (driver 580), Python 3.11, with

| Package | Version |
|---|---|
| Isaac Sim | 5.0.0 (pip) |
| Isaac Lab | v2.2.1 (the study used commit `79fc49dce` on `main`, 99 commits after the tag) |
| PyTorch | 2.7.0 + CUDA 12.8 |
| rsl-rl-lib | 3.0.1 |

### 1. Isaac Sim and Isaac Lab

Follow the pip installation of Isaac Lab (https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html). In short:

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.1            # or 79fc49dce for the exact commit of the study
./isaaclab.sh --install rsl_rl # installs isaaclab, isaaclab_tasks, isaaclab_rl, ... and rsl-rl-lib
pip install rsl-rl-lib==3.0.1
cd ..
```

The first start of Isaac Sim asks to accept the NVIDIA EULA (`export OMNI_KIT_ACCEPT_EULA=YES` accepts it non-interactively)
and builds its shader cache, which takes several minutes.

### 2. This repository

```bash
git clone https://github.com/actuated-design-lab/score-following-drumming-sim2real.git
cd score-following-drumming-sim2real

# optional: a venv on top of the Isaac Lab environment, so that nothing is installed into it
python -m venv --system-site-packages .venv
source .venv/bin/activate

pip install -e porcaro_2026/source/porcaro_2026
pip install -r requirements.txt
```

Check that the tasks are registered:

```bash
cd porcaro_2026
python -c "import isaaclab" && python scripts/rsl_rl/train.py --help > /dev/null && echo ok
```

All commands below are run with this environment active. Commands for `porcaro_2026` are run from `porcaro_2026/`,
commands for `jetson_project` from `jetson_project/`, and the figure script from the repository root.
Scripts that start Isaac Sim take `--headless`.

## Rebuild the figures and tables (no GPU, about 1 min)

```bash
python paper/figures/scripts/make_figures.py
git status --short paper/        # no change in tables/*.tex and data/pkg_*.csv = identical to the article
```

The script reads only `paper/figures/data/`. It writes the derived CSV files `data/pkg_*.csv`, the LaTeX tables
`tables/*.tex`, and the figures `fig_hw_tasks.pdf`, `fig_ioi.pdf`, `fig_sss.pdf`, `fig_fidelity.pdf`, and
`fig_rl_curves.pdf`. The TikZ schematic `fig_rl.pdf` is compiled only when `pdflatex` is available. The check of the
numbers in the manuscript text is skipped, because the manuscript sources are not part of this repository.
Fig. 1 (the hardware photograph and diagram) is not generated from data and is not included.

The data of Fig. 6 (Section VI-H) can be recomputed from the two hardware runs with the hardware scoring code:

```bash
python paper/figures/scripts/okui_fig_data.py --jetson jetson_project
git status --short paper/figures/data/   # okui_sss_*.csv unchanged
```

## Simulation

All from `porcaro_2026/`.

### Evaluate the selected policies and the FF baseline

`sim_trace.py` (about 30 s per run on the RTX 5090) runs one hardware task with 32 randomization draws (seed 7000, the draws shared between methods)
and writes `data/user0/eval/access_1104/sim/<name>/{summary.json, strikes.csv, trace.npz}`.
Compare `success` in `summary.json` with the simulation values with randomization of the article
(`paper/figures/data/pkg2_sim_dr.csv`, column `success_mean`).

```bash
LSTM=logs/user0/rsl_rl/porcaro_rslrl_lstm_gen2/2026-11-02_07-13-45_B_b_seed3/model_1499.pt
MLP=logs/user0/rsl_rl/porcaro_rslrl_mlp_gen2/2026-11-02_03-37-32_E_b_seed2/model_1499.pt
TASK=data/user0/eval/real_core/songs/double_lead2_bpm160.mid

python scripts/tools/user0/access/sim_trace.py --headless --midi $TASK --model B --ckpt $LSTM --n_dr 32 --dr_seed 7000 --name dr2_B_double_lead2_bpm160
python scripts/tools/user0/access/sim_trace.py --headless --midi $TASK --model E --ckpt $MLP  --n_dr 32 --dr_seed 7000 --name dr2_E_double_lead2_bpm160
python scripts/tools/user0/access/sim_trace.py --headless --midi $TASK --ff data/user0/ff/opt2/r0/favorite.json --n_dr 32 --dr_seed 7000 --name dr2_F0_double_lead2_bpm160
python scripts/tools/user0/access/sim_dr_summary.py   # table over the dr2_* runs present → overwrites data/user0/eval/access_1104/pkg/sim_dr.{csv,md}
```

`--model B` is the LSTM policy and `--model E` the MLP policy (internal names); `--ff` plays the FF primitives
synthesized from the MIDI onsets, open loop. The five tasks are the files in `data/user0/eval/real_core/songs/`;
`bash scripts/tools/user0/access/run_1105_pkg.sh dr` runs all fifteen method–task pairs.
`--nominal` switches the randomization off (one environment), and `--nominal --video --enable_cameras` records a video.

Seed selection (Appendix, validation sequences: 16 sequences × 32 draws):

```bash
python scripts/tools/user0/eval/eval_b.py --headless --set val --model E --ckpt $MLP  --name E_b_seed2
python scripts/tools/user0/eval/eval_b.py --headless --set val --model B --ckpt $LSTM --name B_b_seed3
python scripts/tools/user0/eval/eval_b.py --headless --set val --ff data/user0/ff/opt2/r0/favorite.json --name F0_r0
```

Each takes about 2 min. Results go to `data/user0/eval/b_eval/val/<name>/summary.json` (`success_dr_mean`, `success_dr_ci95`). Reference
values: `data/user0/eval/b_eval_1110/val/selection.md` (policies) and `data/user0/eval/b_eval2/val/selection.md` (FF).
`bash scripts/tools/user0/eval/run_select.sh` evaluates all trained runs, applies the selection rule
(`select_seeds.py`), and exports the selected policies to ONNX.

### Train

```bash
bash scripts/tools/user0/gen2/train_b.sh                      # the seven runs of the study, one after another
JOBS="B:b:3 E:b:2" bash scripts/tools/user0/gen2/train_b.sh   # only the two selected configurations
```

Each run is `scripts/rsl_rl/train.py --task Porcaro-Gen2-B-user0 --num_envs 2048 --max_iterations 1500 --seed <n>`,
with `--agent rsl_rl_lstm_cfg_entry_point --lookahead_horizon 0.5` for the LSTM policy and
`--agent rsl_rl_mlp_cfg_entry_point --lookahead_horizon 0.5 --use_frame_stacking --frame_stack_k 5` for the MLP policy
(`JOBS` entries are `<B|E>:<b|b_obsv1>:<seed>`; `b_obsv1` is the observation-v1 fallback task). One run took about
3.4 h on the RTX 5090. Logs are written to `logs/user0/rsl_rl/porcaro_rslrl_{lstm,mlp}_gen2/<date>_<run>/`.
Training curves: `python scripts/tools/user0/access/rl_curves.py` (only the two selected runs are included; the other five are reported as missing).

### Export a policy to ONNX

```bash
python scripts/rsl_rl/export_onnx.py --headless --task Porcaro-Gen2-B-user0 --agent rsl_rl_mlp_cfg_entry_point \
  --checkpoint=$MLP --lookahead_horizon 0.5 --use_frame_stacking --frame_stack_k 5 \
  --out_dir data/user0/gen2/onnx --out_name gen2b_E_b_seed2
```

For the LSTM policy, use `--agent rsl_rl_lstm_cfg_entry_point` without the frame-stacking options. Copy the file to
`jetson_project/models/GEN2B/`; the entry in `jetson_project/models/manifest.yaml` is checked against the ONNX
input size at start-up.

### FF baseline

```bash
bash scripts/tools/user0/ff/run_ff.sh check                    # stage 0: target times, pressures, scoring (no GPU)
bash scripts/tools/user0/ff/run_ff.sh probe                    # stage 1: dead time L and strike force
OPT=opt2 EVAL=b_eval2 bash scripts/tools/user0/ff/run_ff.sh opt   # stage 2: CMA-ES, restarts 0-2 and nominal, then validation
```

Stage 2 of the study took about 9.7 h per restart (four in parallel). The adopted solution is
`data/user0/ff/opt2/r0/favorite.json` (7 parameters per interval class: `L`, `h`, `a_dn`, `a_up`, `g_up`, `g_dn`, `d_g`;
times in s, pressures in MPa); `summary.md` next to it records the run. The commands sent to the hardware were
synthesized from it by `scripts/tools/user0/eval/real_core.py` and are in `jetson_project/signals/core2026v2/`.

### Pressure model and fidelity checks

`scripts/tools/user0/jfps2026/replay_open_loop.py` replays a recorded command in the simulator (with the measured
pressure or with the pressure model as input) and `analysis/eval/compare_hys_modes.py` compares the result with the
hardware log (`run_1105_pkg.sh fidelity`, Table I and Fig. 2). `jetson_project/analysis/itv/access_table2.py` computes
the pressure-stage NRMSE. Both need the raw recordings, which are not included.

## Hardware deployment

All from `jetson_project/`, on the robot computer (Jetson Orin Nano, connected to the dSPACE MicroLabBox by serial
at 230400 baud; the MicroLabBox model is not part of this repository). Requirements:
`pip install -r ../requirements-jetson.txt`.

```bash
python3 src/deploy_policy.py --list                                   # deployable policies (models/manifest.yaml)
python3 src/deploy_policy.py --model GEN2B/E --midi songs/core2026/b_holdout_00_bpm90.mid --mock --user user0 --venue test   # dry run without hardware
python3 src/deploy_policy.py --model GEN2B/E --midi songs/core2026/b_holdout_00_bpm90.mid --trial 1 --user user0 --venue access2026
python3 tools/run_signal_playback.py core2026v2/F0_b_holdout_00_bpm90 --user user0 --venue access2026   # FF: play the command file
```

`GEN2B/B` is the LSTM policy, `GEN2B/E` the MLP policy. Logs go to `data/<user>/<venue>/<group>_<date>/` (a CSV of
the 200 Hz samples and a JSON sidecar). The contact flag of observation v2 is computed on the Jetson from the force
received from the MicroLabBox (`ContactLatch` in `src/obs_v2.py`). The pressure channels of the three PAMs are mapped
in `config/pressure_slots.yaml`.

The session of the article (randomized run order, check motions, operator gates) is driven by
`tools/core_runner.py` (run lists `core2026_runs*.csv`, check motion `signals/core2026/check_motion.csv`,
judgement `tools/core_check.py` against `core2026_check_baseline.json`). Scoring:

```bash
python3 tools/core_score.py data/user0/<venue>/            # evaluation runs → <venue>/core2026_score/{runs,strikes}.csv, summary.md
python3 tools/core_score.py data/user0/<venue>/ --calib    # calibration runs
```

The reported table (60 runs: B, E, F0, F3 × five tasks × three repetitions) is scored from the RL runs of Nov 9 and
the FF runs of Nov 5:

```bash
python3 tools/core_score.py data/user0/access2026v3 data/user0/access2026v2/playback_20261105
# → data/user0/access2026v3/core2026_score/{runs,strikes}.csv, summary.md
```

Runs excluded during the session (QC failures, failed check motions, operator retakes) are listed in
`core2026_session.csv` and skipped automatically.

The runs of Section VI-H were taken with `tools/demo_take.py E_okui_A --venue okui2026` (and `E_okui_B`).

## Research-side pipeline (how the data package was made)

`paper/figures/data/README.md` names the script and the inputs of every file in the package. In short:

| Package file | Script (inputs not included are in parentheses) |
|---|---|
| `package_sec4_*.md` §1 (FF parameters) | `data/user0/ff/opt2/r0/favorite.json`, `select_seeds.py` |
| `package_sec4_*.md` §2–5, `pkg2_runs.csv`, `pkg2_timing_force.csv`, `pkg2_ioi_counts.csv` | `jetson_project/tools/core_score.py` → `porcaro_2026/scripts/tools/user0/access/sec4_tables.py`, `pkg_1105.py`, `c6_swing_bins.py`, `ff/diag_opt2.py` (the scored hardware logs, rebuilt with the `core_score.py` command above) |
| `pkg2_sim_dr.csv` | `access/run_1105_pkg.sh dr` → `sim_trace.py` → `sim_dr_summary.py` (runs here) |
| `pkg2_pressure_nrmse.csv`, `pkg2_wrist_rmse.csv`, `fidelity_ts_*.csv` | `jetson_project/analysis/itv/access_table2.py`; `access/run_1105_pkg.sh fidelity` → `fidelity_export.py` (identification and test-motion recordings) |
| `pkg2_task_descriptors.csv` | `access/task_descriptors.py` (runs here) |
| `rl_curves.csv`, `rl_train_dist.csv` | `access/rl_curves.py`, `access/train_dist_return.py` (runs here) |
| `okui_sss_*.csv` | `paper/figures/scripts/okui_fig_data.py` (runs here) |

The automatic `git push` of the original research scripts is disabled in this copy (`NO_PUSH` defaults to 1).

## Notes

- The two hardware GMD tasks are excerpts of the Groove MIDI Dataset (J. Gillick, A. Roberts, J. Engel, D. Eck,
  and D. Bamman, "Learning to Groove with Inverse Sequence Transformations," ICML 2019), distributed under CC BY 4.0;
  see Section V of the article for how they were cut. `scripts/tools/user0/eval/gmd_scan.py` needs a local copy of
  the dataset.
- The PAM channels are DF (dorsiflexion) and F (flexion) on the wrist and G on the grip. Internal method names:
  B = LSTM policy, E = MLP policy, F0 = FF, F3 = FF-cal.
- Removed from the research repositories for this release: the solenoid-valve (binary/discrete) variant of the task
  and the deployment code, models and tasks of earlier studies, and other users' folders. The `params/env.yaml`
  files of the checkpoints still list the then-existing options `use_discrete_action: false` and `discrete_threshold`;
  they had no effect on these runs. Absolute paths in those files were replaced by `<repo>/`.
