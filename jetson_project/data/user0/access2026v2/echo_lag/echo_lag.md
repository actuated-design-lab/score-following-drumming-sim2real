# 通信の往復の遅れ（echo_lag.py、DF の指令のエコー）

ログ 77 本、指令の変化 3052 回（|変化| ≥ 0.01 MPa）。往復 = 送信から、エコーが新しい値になった最初のパケットを受け取るまで
（パケットは 5 ms ごとなので、その待ち 0〜5 ms を含む）。転送時間 2.5 ms を引いた値も示す

| | 中央値 [ms] | 平均 | 25 % | 75 % | 5 % | 95 % | 最大 |
|---|---|---|---|---|---|---|---|
| 往復（受信時刻） | 7.0 | 7.3 | 6.0 | 8.4 | 5.0 | 10.0 | 10.7 |
| 往復 − 転送時間 | 4.5 | 4.8 | 3.5 | 5.8 | 2.4 | 7.5 | 8.2 |

## ログごと

| ログ | 回数 | 中央値 [ms] |
|---|---|---|
| data/user0/access2026v2/playback_20261105/data_F0_b_calib_00_bpm82_1793847282.csv | 74 | 9.4 |
| data/user0/access2026v2/playback_20261105/data_F0_b_calib_01_bpm154_1793847300.csv | 38 | 9.0 |
| data/user0/access2026v2/playback_20261105/data_F0_b_holdout_00_bpm90_1793847898.csv | 58 | 8.8 |
| data/user0/access2026v2/playback_20261105/data_F0_b_holdout_00_bpm90_1793849083.csv | 58 | 6.7 |
| data/user0/access2026v2/playback_20261105/data_F0_b_holdout_00_bpm90_1793849307.csv | 58 | 5.8 |
| data/user0/access2026v2/playback_20261105/data_F0_b_holdout_01_bpm96_1793847753.csv | 64 | 8.8 |
| data/user0/access2026v2/playback_20261105/data_F0_b_holdout_01_bpm96_1793848675.csv | 64 | 5.9 |
| data/user0/access2026v2/playback_20261105/data_F0_b_holdout_01_bpm96_1793849472.csv | 64 | 6.4 |
| data/user0/access2026v2/playback_20261105/data_F0_double_lead2_bpm160_1793847618.csv | 24 | 6.9 |
| data/user0/access2026v2/playback_20261105/data_F0_double_lead2_bpm160_1793849119.csv | 24 | 6.8 |
| data/user0/access2026v2/playback_20261105/data_F0_double_lead2_bpm160_1793849401.csv | 24 | 9.5 |
| data/user0/access2026v2/playback_20261105/data_F0_gmd_legacy_gmd_03_high_recut_bpm138_1793848250.csv | 152 | 5.6 |
| data/user0/access2026v2/playback_20261105/data_F0_gmd_legacy_gmd_03_high_recut_bpm138_1793848942.csv | 152 | 8.3 |
| data/user0/access2026v2/playback_20261105/data_F0_gmd_legacy_gmd_03_high_recut_bpm138_1793849253.csv | 152 | 7.7 |
| data/user0/access2026v2/playback_20261105/data_F0_gmd_legacy_gmd_04_extreme_recut_bpm170_1793848411.csv | 132 | 9.4 |
| data/user0/access2026v2/playback_20261105/data_F0_gmd_legacy_gmd_04_extreme_recut_bpm170_1793848840.csv | 132 | 7.8 |
| data/user0/access2026v2/playback_20261105/data_F0_gmd_legacy_gmd_04_extreme_recut_bpm170_1793849634.csv | 132 | 5.9 |
| data/user0/access2026v2/playback_20261105/data_F3_b_holdout_00_bpm90_1793847855.csv | 58 | 8.3 |
| data/user0/access2026v2/playback_20261105/data_F3_b_holdout_00_bpm90_1793849041.csv | 58 | 9.2 |
| data/user0/access2026v2/playback_20261105/data_F3_b_holdout_00_bpm90_1793849349.csv | 58 | 7.8 |
| data/user0/access2026v2/playback_20261105/data_F3_b_holdout_01_bpm96_1793847666.csv | 64 | 5.5 |
| data/user0/access2026v2/playback_20261105/data_F3_b_holdout_01_bpm96_1793848633.csv | 64 | 7.9 |
| data/user0/access2026v2/playback_20261105/data_F3_b_holdout_01_bpm96_1793849513.csv | 64 | 7.1 |
| data/user0/access2026v2/playback_20261105/data_F3_double_lead2_bpm160_1793847530.csv | 24 | 6.9 |
| data/user0/access2026v2/playback_20261105/data_F3_double_lead2_bpm160_1793849144.csv | 24 | 7.5 |
| data/user0/access2026v2/playback_20261105/data_F3_double_lead2_bpm160_1793849427.csv | 24 | 9.3 |
| data/user0/access2026v2/playback_20261105/data_F3_gmd_legacy_gmd_03_high_recut_bpm138_1793848298.csv | 152 | 6.7 |
| data/user0/access2026v2/playback_20261105/data_F3_gmd_legacy_gmd_03_high_recut_bpm138_1793848990.csv | 152 | 5.4 |
| data/user0/access2026v2/playback_20261105/data_F3_gmd_legacy_gmd_03_high_recut_bpm138_1793849205.csv | 152 | 6.9 |
| data/user0/access2026v2/playback_20261105/data_F3_gmd_legacy_gmd_04_extreme_recut_bpm170_1793848460.csv | 132 | 6.7 |
| data/user0/access2026v2/playback_20261105/data_F3_gmd_legacy_gmd_04_extreme_recut_bpm170_1793848889.csv | 132 | 9.4 |
| data/user0/access2026v2/playback_20261105/data_F3_gmd_legacy_gmd_04_extreme_recut_bpm170_1793849682.csv | 132 | 6.0 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847221.csv | 8 | 7.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847236.csv | 8 | 7.0 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847252.csv | 8 | 7.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847520.csv | 8 | 6.8 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847548.csv | 8 | 4.7 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847578.csv | 8 | 6.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847608.csv | 8 | 5.8 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847640.csv | 8 | 7.4 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847681.csv | 8 | 5.3 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847727.csv | 8 | 9.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847768.csv | 8 | 7.7 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847828.csv | 8 | 10.1 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847871.csv | 8 | 8.1 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847913.csv | 8 | 7.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793847960.csv | 8 | 5.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848165.csv | 8 | 5.1 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848218.csv | 8 | 5.7 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848266.csv | 8 | 9.3 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848314.csv | 8 | 5.4 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848378.csv | 8 | 7.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848427.csv | 8 | 7.1 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848475.csv | 8 | 5.7 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848529.csv | 8 | 6.8 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848608.csv | 8 | 7.2 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848649.csv | 8 | 7.0 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848694.csv | 8 | 8.5 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848807.csv | 8 | 7.1 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848855.csv | 8 | 9.0 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848910.csv | 8 | 5.7 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793848958.csv | 8 | 8.2 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849014.csv | 8 | 5.7 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849056.csv | 8 | 5.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849109.csv | 8 | 8.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849134.csv | 8 | 8.0 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849172.csv | 8 | 4.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849220.csv | 8 | 8.6 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849280.csv | 8 | 7.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849322.csv | 8 | 6.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849391.csv | 8 | 6.7 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849417.csv | 8 | 9.0 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849446.csv | 8 | 8.3 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849488.csv | 8 | 7.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849600.csv | 8 | 5.9 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849649.csv | 8 | 7.1 |
| data/user0/access2026v2/playback_20261105/data_check_motion_1793849713.csv | 8 | 7.2 |
