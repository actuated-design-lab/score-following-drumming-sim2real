# Data behind the figures and tables

Everything in the paper that is drawn or tabulated from data is rebuilt by one command, run from the
repository root:

```
python3 paper/figures/scripts/make_figures.py
```

The script reads only files in `paper/figures/data`. It writes the derived CSV files below, the LaTeX
tables `paper/figures/tables/*.tex` (included by the sections; do not edit them by hand), the figures
`paper/figures/*.pdf`, and then checks that every three-decimal number in the results text
(sections 03 and 05–09 and the abstract) appears in one of the two data packages.

## Source file (from the research side)

| File | Content | Source |
|---|---|---|
| `package_sec4_20261105.md` | The confirmed data package for Section 4 onward, Sections 1–6 verbatim (the closing list of pending items is not copied). Markdown tables; units are given in the column headers (ms, MPa, deg, deg/s; success rates are fractions). Internal names: B = LSTM policy, E = MLP policy, F0 = feedforward baseline, F3 = hardware-calibrated feedforward baseline. | Project document `claude/Access論文_§4確定データ_20261105.md`. Its tables are outputs of the research-side scripts named at the end of each section there: porcaro_2026 `scripts/tools/user0/access/sec4_tables.py` (→ `data/user0/eval/access_1104/sec4/tables.md`), `scripts/tools/user0/access/c6_swing_bins.py --real_only`, `diag_opt2.py`, the FF optimization outputs `data/user0/ff/opt2/`, and jetson_project `core2026_score/` and `sss2026/sss_table.md`. |

## Source files of data package 2 (from the research side, merged from PR #1)

`package_20261105b.md` documents these files (content, units, the research-side script, and its inputs);
items marked "未" there are pending and their values are not used. The script reads the CSV files as delivered.

| File | Content and units | Source (research side) | Used in |
|---|---|---|---|
| `package_20261105b.md` | Description of data package 2, with the summary tables | Research side, 2026-11-05 evening | — |
| `pkg2_pressure_nrmse.csv` | Pressure-stage NRMSE (%) per recording and channel; strike commands: median over 105 runs with interquartile range, one common delay (ms) | jetson_project `analysis/itv/access_table2.py` | Table 1 (`tab:fidelity`); Sections II-C and VI-A text |
| `pkg2_wrist_rmse.csv` | Wrist-angle RMSE (deg) of the assembled model per motion, input `measured` or `model`; also offset, lag (ms), amplitude ratio, grip RMSE (deg) and grip amplitude ratio | porcaro_2026 `check_gen2.sh` → `compare_hys_modes.py` (`gen2_check/metrics.csv`) | Table 1; Section VI-A text |
| `pkg2_timing_force.csv` | Per method and task (and `all`): onsets, struck onsets, signed timing error median and quartiles (ms), peak force median and quartiles (N), median of \|F − 20 N\| (N) | porcaro_2026 `pkg_1105.py` (item 2) from jetson_project `core2026_score/strikes.csv` | Table 13 (`tab:force`); Sections VI-F and VII text |
| `pkg2_ioi_counts.csv` | Onsets per run (and over three runs, share) in each interval class, per task | `pkg_1105.py` (item 3) | Table 9 (`tab:ioicounts`); Section VI-D text |
| `pkg2_runs.csv` | Success rate of every hardware run (method, task, repetition) under the revised and the original rule | `pkg_1105.py` (item 4a) from jetson_project `core2026_score/runs.csv` | Fig. 4 (`fig:tempo`, points) |
| `pkg2_sss_first_strikes.csv` | First arrangement performance (Krupa score, 123 onsets), one row per target. **Not used since 2026-11-11** (Section VI-H now reports the reproduction of a human performance); kept as delivered | `pkg_1105.py` (item 4b) from jetson_project `sss2026/sss_strikes.csv` | — |
| `pkg2_sss_first_force.csv` | Load-cell force of the same run (s, N; 5 ms steps). **Not used since 2026-11-11**; kept as delivered | `pkg_1105.py` (item 4b) from the run log | — |
| `okui_sss_strikes.csv` | Reproduction of a human performance (2026-11-11): `part` (A, B), `onset`, `target_s` and `strike_s` (s, robot MIDI time with the 2.0 s lead), `timing_err_ms`, `peak_force_N` (empty if not struck), `success` (core_score rule: contact_event + 40 ms chain) | `figures/scripts/okui_fig_data.py --jetson <jetson_project>` from jetson_project `data/user0/okui2026/gen2b_20261111/` (MLP policy GEN2B/E, second run of each part: A 1794365787, B 1794365998) | Fig. 6 (`fig:reproduction`); Section VI-H text |
| `okui_sss_force.csv` | Load-cell force and wrist and grip joint angles of the same two runs, same samples: `part`, `time_s` (rate-corrected deploy time, s), `force_N`, `wrist_deg`, `grip_deg` (encoders, deg), `pres_G_cmd` (PAM-G pressure command of the policy, MPa, held for each 20 ms control step), `pres_G_meas` (PAM-G measured pressure, MPa); 5 ms steps | same | Fig. 6 (force: left axis; wrist and grip angles: right axis) |
| `fidelity_ts_<motion>.csv` (`tm_A_quasistatic`, `tm_B_steps`, `tm_C_sine`, `tm_E_dbl160_seed2`, `tm_E_gmd138_seed2`) | Same command on the hardware (10/05) and in the Gen2 simulation without randomization, aligned by the echo, 5 ms steps: `time_s`; commands `cmd_DF/F/G` and pressures `real_P_*`, `simcfg_P_*` (MPa); angles `real_*`, `simcfg_*` (pressure model), `simmeas_*` (measured pressure as input) for wrist and grip (deg, hardware convention) | porcaro_2026 `run_1105_pkg.sh fidelity` → `fidelity_export.py` → `pkg_1105.py` (item 7, package_20261109b) | Fig. 2 (`fig:fidelity`): (a) `tm_A_quasistatic` 2–42 s, (b) `tm_B_steps` 2–37 s, (c) `tm_C_sine` 47–68 s, (d) `tm_E_dbl160_seed2` 2–11.5 s; `tm_E_gmd138_seed2` not plotted |
| `rl_curves.csv` | Training curve of the two selected runs (LSTM = `B_b_seed3`, MLP = `E_b_seed2`): `iteration`, `mean_reward` (rsl_rl `Train/mean_reward`: mean summed reward of recently finished episodes, stochastic actions), `mean_episode_length` (policy steps) | porcaro_2026 PR #95, `scripts/tools/user0/access/rl_curves.py` → `data/user0/eval/access_1110/rl_curves/curves.csv` (rows of the two runs) | Appendix figure `fig_rl_curves.pdf` |
| `rl_ceiling.csv` | Reward upper bound per curriculum level (iterations 0–299, 300–749, 750–1499): mean notes per episode (and SD), mean episode length (steps), full rest bonus, upper bound = notes × 1.0 + rest bonus; 20,000 episodes per level | Research side, numpy port of `rhythm_generator._random_spikes` with the `Gen2_B` settings (2026-11-10); level 2 checked on the WS (47.50 notes, upper bound 51.42 over 1024 episodes) and the episode lengths against the training log | `fig_rl_curves.pdf` (dashed steps) |
| `rl_train_dist.csv` | Deterministic evaluation of the two runs at the last checkpoint on the final training distribution (Gen2_B with randomization, 1024 episodes): mean and SD of the episode reward, mean upper bound, attainment (mean of per-episode reward ÷ upper bound, 95 % CI) | porcaro_2026 PR #95, `train_dist_return.py` → `access_1110/train_dist/*/summary.json` | `fig_rl_curves.pdf` (points at the right) |
| `pkg2_task_descriptors.csv` | Descriptors of the five hardware tasks: tempo (BPM), onsets, span from first to last onset (s), minimum and median IOI (ms), fractions of IOIs below 100, 145, and 200 ms, nPVI, rest fraction (share of sixteenth-note grid positions without a strike), fraction of IOIs of 400 ms or more (not used), mean and maximum deviation from the 20 ms grid (ms) | porcaro_2026 `task_descriptors.py` (added to branch `data/pkg2-20261105` after PR #1, commit 3b3fba7) | Table 4 (`tab:tasks`, generated); Section V-F text |

## Derived files (written by the script from the package; do not edit)

| File | Content and units | Package section | Used in |
|---|---|---|---|
| `pkg_ff_params.csv` | Feedforward parameters per interval class: `L_ms`, `h_ms`, `delta_ms` (ms); `a_up_MPa`, `a_dn_MPa`, `g_up_MPa`, `g_dn_MPa` (MPa) | 1.2 | Table 19 (`tab:ffparams`, Appendix F); Section IV text |
| `pkg_hw_task.csv` | Hardware result per method and task: `onsets` (pooled over three runs), `success_mean`, `success_sd` (n = 3), `success_old` (original rule), failure fractions `not_struck`, `early`, `late`, `pushed` | 2 | Table 6 (`tab:hw`); Fig. 4 (`fig:tempo`, bars; the script checks that the mean of `pkg2_runs.csv` matches); Section VI text |
| `pkg_hw_mean5.csv` | Five-task means per method, same columns | 2 | Tables 7 and 8 (`tab:hw`, `tab:fail`); abstract, Sections VI–VIII text |
| `pkg_sim_hw.csv` | Success rate in simulation at nominal parameters and on hardware, per task and method | 3 | Table 5 (`tab:simhw`); Section VI-B and VII text |
| `pkg_ioi.csv` | Success rate by interval class (`bin`: <120, 120-200, 200-320, >=320 ms) and method, with pooled `onsets` | 4 | Table 8 (`tab:ioi`); Fig. 5 (`fig:ioi`, `fig_ioi.pdf`); Sections VI–VIII text |
| `pkg_rest.csv` | Success rate after a rest / within a run (`subset`: all tasks or without the double-stroke task) | 4 | Table 10 (`tab:rest`); Section VI-D and VII text |
| `pkg_swing.csv` | Downswing of the feedforward baseline per class: `a_dn_MPa`, `onsets`, `omega_deg_s` (deg/s), `A_deg`, `theta_top_deg` (deg), `PF_meas_MPa` (MPa), `success` | 5 | Table 11 (`tab:swing`); Section VI-E text |
| `pkg_swing_prev.csv` | Same, by class of the preceding onset: `A_median_deg` (deg), `success`, `onsets` | 5 | Table 12 (`tab:prevclass`); Section VI-E text |

Section VI-H (reproduction of a human performance, 2026-11-11) uses `okui_sss_*.csv`; its summary values (success, timing
error, median peak force per part) are recomputed from `okui_sss_strikes.csv`. Section 6 of the first package (arrangement
demonstration) and the `pkg2_sss_first_*` files are no longer used.

## Figures that are not drawn from data

| File | What it is | Source |
|---|---|---|
| `../fig_hardware.pdf` | Fig. 1 (`fig:system`): mechanism and system diagram | Copied unchanged from an earlier manuscript of the authors (`figures/fig1_hardware.pdf`). Checked against the current system: PAM-DF and PAM-F on the wrist, PAM-G alone on the grip. |
| `../fig_rl.pdf` | Fig. 3 (`fig:rl`): learning framework | TikZ, `../fig_rl.tex` with `../tikz_rl.tex`, compiled by the script |
