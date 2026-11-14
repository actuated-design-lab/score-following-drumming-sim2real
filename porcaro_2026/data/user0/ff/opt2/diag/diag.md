# 段2（opt2）の解の診断（diag_opt2.py、DR なし、実機コアの 5 タスクの平均）

success = 新しい判定（連なり 40 ms）、success_old = 旧採点、press_train60 = 60 ms の連なりで 100 ms を超えた接触の時間の割合（目的関数の罰と同じ）、train40_drop = 新しい判定で外れた当たりの割合、hit_no_lift = 当たりのうち打撃の前の手首の持ち上げ 2° 未満の割合（診断だけ）

| source | cand | success | success_old | press_train60 | train40_drop | hit_no_lift | tasks |
|---|---|---|---|---|---|---|---|
| 11/04 の sim | B | 0.562 | 0.596 | 0.004 | 0.064 | 0.333 | 5 |
| 11/04 の sim | E | 0.839 | 0.919 | 0.008 | 0.089 | 0.726 | 5 |
| 11/04 の sim | 旧い F0（F0_r2） | 0.328 | 0.938 | 0.347 | 0.648 | 0.921 | 5 |
| opt2 | r0_best | 0.784 | 0.845 | 0.012 | 0.068 | 0.401 | 5 |
| opt2 | r0_favorite | 0.756 | 0.756 | 0.004 | 0.000 | 0.422 | 5 |
| opt2 | r0_nominal_best | 0.524 | 0.551 | 0.030 | 0.045 | 0.477 | 5 |
| opt2 | r0_nominal_favorite | 0.523 | 0.539 | 0.026 | 0.030 | 0.457 | 5 |
| opt2 | r1_best | 0.650 | 0.689 | 0.012 | 0.052 | 0.323 | 5 |
| opt2 | r1_favorite | 0.666 | 0.680 | 0.011 | 0.020 | 0.313 | 5 |
| opt2 | r2_best | 0.638 | 0.793 | 0.050 | 0.178 | 0.484 | 5 |
| opt2 | r2_favorite | 0.654 | 0.705 | 0.029 | 0.083 | 0.423 | 5 |

## タスクごと

| source | cand | task | success | success_old | press_train60 | train40_drop | hit_no_lift |
|---|---|---|---|---|---|---|---|
| opt2 | r0_best | b_holdout_00_bpm90 | 0.796 | 0.796 | 0.002 | 0.000 | 0.410 |
| opt2 | r0_best | b_holdout_01_bpm96 | 0.800 | 0.800 | 0.002 | 0.000 | 0.318 |
| opt2 | r0_best | double_lead2_bpm160 | 0.667 | 0.917 | 0.048 | 0.273 | 0.625 |
| opt2 | r0_best | gmd_legacy_gmd_03_high_recut_bpm138 | 0.881 | 0.917 | 0.003 | 0.039 | 0.351 |
| opt2 | r0_best | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.777 | 0.798 | 0.003 | 0.027 | 0.301 |
| opt2 | r0_favorite | b_holdout_00_bpm90 | 0.735 | 0.735 | 0.003 | 0.000 | 0.472 |
| opt2 | r0_favorite | b_holdout_01_bpm96 | 0.873 | 0.873 | 0.002 | 0.000 | 0.312 |
| opt2 | r0_favorite | double_lead2_bpm160 | 0.625 | 0.625 | 0.009 | 0.000 | 0.467 |
| opt2 | r0_favorite | gmd_legacy_gmd_03_high_recut_bpm138 | 0.833 | 0.833 | 0.002 | 0.000 | 0.471 |
| opt2 | r0_favorite | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.713 | 0.713 | 0.005 | 0.000 | 0.388 |
| opt2 | r0_nominal_best | b_holdout_00_bpm90 | 0.714 | 0.714 | 0.033 | 0.000 | 0.829 |
| opt2 | r0_nominal_best | b_holdout_01_bpm96 | 0.600 | 0.618 | 0.049 | 0.029 | 0.545 |
| opt2 | r0_nominal_best | double_lead2_bpm160 | 0.250 | 0.250 | 0.012 | 0.000 | 0.167 |
| opt2 | r0_nominal_best | gmd_legacy_gmd_03_high_recut_bpm138 | 0.512 | 0.607 | 0.031 | 0.157 | 0.395 |
| opt2 | r0_nominal_best | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.543 | 0.564 | 0.024 | 0.038 | 0.451 |
| opt2 | r0_nominal_favorite | b_holdout_00_bpm90 | 0.449 | 0.469 | 0.042 | 0.043 | 0.773 |
| opt2 | r0_nominal_favorite | b_holdout_01_bpm96 | 0.509 | 0.545 | 0.024 | 0.067 | 0.571 |
| opt2 | r0_nominal_favorite | double_lead2_bpm160 | 0.542 | 0.542 | 0.006 | 0.000 | 0.077 |
| opt2 | r0_nominal_favorite | gmd_legacy_gmd_03_high_recut_bpm138 | 0.500 | 0.512 | 0.010 | 0.023 | 0.452 |
| opt2 | r0_nominal_favorite | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.617 | 0.628 | 0.047 | 0.017 | 0.414 |
| opt2 | r1_best | b_holdout_00_bpm90 | 0.653 | 0.694 | 0.005 | 0.059 | 0.375 |
| opt2 | r1_best | b_holdout_01_bpm96 | 0.709 | 0.727 | 0.005 | 0.025 | 0.410 |
| opt2 | r1_best | double_lead2_bpm160 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| opt2 | r1_best | gmd_legacy_gmd_03_high_recut_bpm138 | 0.726 | 0.810 | 0.030 | 0.103 | 0.475 |
| opt2 | r1_best | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.660 | 0.713 | 0.020 | 0.075 | 0.355 |
| opt2 | r1_favorite | b_holdout_00_bpm90 | 0.796 | 0.796 | 0.012 | 0.000 | 0.487 |
| opt2 | r1_favorite | b_holdout_01_bpm96 | 0.745 | 0.745 | 0.002 | 0.000 | 0.293 |
| opt2 | r1_favorite | double_lead2_bpm160 | 0.500 | 0.500 | 0.000 | 0.000 | 0.083 |
| opt2 | r1_favorite | gmd_legacy_gmd_03_high_recut_bpm138 | 0.631 | 0.679 | 0.025 | 0.070 | 0.396 |
| opt2 | r1_favorite | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.660 | 0.681 | 0.017 | 0.031 | 0.306 |
| opt2 | r2_best | b_holdout_00_bpm90 | 0.673 | 0.694 | 0.011 | 0.029 | 0.636 |
| opt2 | r2_best | b_holdout_01_bpm96 | 0.636 | 0.673 | 0.024 | 0.054 | 0.514 |
| opt2 | r2_best | double_lead2_bpm160 | 0.375 | 0.917 | 0.125 | 0.591 | 0.222 |
| opt2 | r2_best | gmd_legacy_gmd_03_high_recut_bpm138 | 0.845 | 0.893 | 0.027 | 0.053 | 0.549 |
| opt2 | r2_best | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.660 | 0.787 | 0.061 | 0.162 | 0.500 |
| opt2 | r2_favorite | b_holdout_00_bpm90 | 0.714 | 0.735 | 0.012 | 0.028 | 0.400 |
| opt2 | r2_favorite | b_holdout_01_bpm96 | 0.600 | 0.655 | 0.027 | 0.083 | 0.394 |
| opt2 | r2_favorite | double_lead2_bpm160 | 0.417 | 0.542 | 0.069 | 0.231 | 0.500 |
| opt2 | r2_favorite | gmd_legacy_gmd_03_high_recut_bpm138 | 0.857 | 0.881 | 0.011 | 0.027 | 0.431 |
| opt2 | r2_favorite | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.681 | 0.713 | 0.025 | 0.045 | 0.391 |
| 11/04 の sim | E | b_holdout_00_bpm90 | 0.918 | 0.918 | 0.000 | 0.000 | 0.800 |
| 11/04 の sim | E | b_holdout_01_bpm96 | 0.855 | 0.909 | 0.005 | 0.060 | 0.681 |
| 11/04 の sim | E | double_lead2_bpm160 | 0.667 | 0.833 | 0.021 | 0.200 | 0.625 |
| 11/04 の sim | E | gmd_legacy_gmd_03_high_recut_bpm138 | 0.893 | 0.964 | 0.004 | 0.074 | 0.880 |
| 11/04 の sim | E | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.862 | 0.968 | 0.010 | 0.110 | 0.642 |
| 11/04 の sim | B | b_holdout_00_bpm90 | 0.673 | 0.673 | 0.000 | 0.000 | 0.303 |
| 11/04 の sim | B | b_holdout_01_bpm96 | 0.709 | 0.709 | 0.003 | 0.000 | 0.205 |
| 11/04 の sim | B | double_lead2_bpm160 | 0.458 | 0.542 | 0.006 | 0.154 | 0.273 |
| 11/04 の sim | B | gmd_legacy_gmd_03_high_recut_bpm138 | 0.500 | 0.524 | 0.004 | 0.045 | 0.405 |
| 11/04 の sim | B | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.468 | 0.532 | 0.007 | 0.120 | 0.477 |
| 11/04 の sim | 旧い F0（F0_r2） | b_holdout_00_bpm90 | 0.408 | 0.980 | 0.188 | 0.583 | 0.850 |
| 11/04 の sim | 旧い F0（F0_r2） | b_holdout_01_bpm96 | 0.364 | 0.964 | 0.251 | 0.623 | 0.900 |
| 11/04 の sim | 旧い F0（F0_r2） | double_lead2_bpm160 | 0.083 | 0.958 | 0.583 | 0.913 | 1.000 |
| 11/04 の sim | 旧い F0（F0_r2） | gmd_legacy_gmd_03_high_recut_bpm138 | 0.500 | 0.905 | 0.348 | 0.447 | 0.929 |
| 11/04 の sim | 旧い F0（F0_r2） | gmd_legacy_gmd_04_extreme_recut_bpm170 | 0.287 | 0.883 | 0.367 | 0.675 | 0.926 |
