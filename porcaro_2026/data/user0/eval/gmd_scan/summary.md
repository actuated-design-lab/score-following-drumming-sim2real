# GMD の抜粋のスキャン（gmd_scan.py）

GMD: `<GMD>/groove`（Groove MIDI Dataset v1.0.0）（info.csv 1150 ファイル）。スネア類 = [37, 38, 40]、velocity ≥ 20、0.08 s の規則（残した打から 0.08 s 以内の後続を落とす）、窓 = ファイルの最初のオンセットから 27.5 s、span < 20.0 s は除外、bpm 60〜160。抜粋の MIDI は最初の打を 2.0 s に置いた（仮）

## 1) レガシー曲の元ファイル（ノート 38 の拍の並びの一致、上位 3）

| legacy | rank | match | midi_filename | split | info_bpm | offset_beat |
|---|---|---|---|---|---|---|
| gmd_03_high_bpm138 | 1 | 1.0 | drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid | test | 138 | 0.0 |
| gmd_03_high_bpm138 | 2 | 0.654 | drummer2/session3/1_rock-shuffle_85_beat_4-4.mid | train | 85 | 153.315 |
| gmd_03_high_bpm138 | 3 | 0.641 | drummer1/session2/9_jazz-march_120_beat_4-4.mid | train | 120 | 75.742 |
| gmd_04_extreme_bpm170 | 1 | 1.0 | drummer1/session1/80_dance-breakbeat_170_beat_4-4.mid | train | 170 | 0.0 |
| gmd_04_extreme_bpm170 | 2 | 0.544 | drummer1/session2/9_jazz-march_120_beat_4-4.mid | train | 120 | 55.994 |
| gmd_04_extreme_bpm170 | 3 | 0.529 | drummer5/session1/6_latin-brazilian-maracatu_96_beat_4-4.mid | train | 96 | 21.465 |

一致率 0.95 以上の 1 位を元ファイルとした。

## 2) レガシーの今の範囲で、38 のみと {37,38,40} を比べる

同じ元ファイル・同じ時間範囲（今のファイルの最初と最後の打）。q_ は評価と同じ 20 ms の格子に切り捨てた後

| legacy | 系列 | 落とした数 | n | dur_s | ioi_min_ms | ioi_med_ms | q_n | q_ioi_min_ms | q_ioi_med_ms | frac_ioi_lt145 | npvi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gmd_03_high_bpm138 | 今のファイル（38 のみ） |  | 78 | 27.15 | 82.4 | 320.7 | 78 | 80.0 | 320.0 | 0.078 | 54.0 |
| gmd_03_high_bpm138 | 同じ範囲・38 のみ（再抽出） | 14.0 | 78 | 27.15 | 82.4 | 320.7 | 78 | 80.0 | 320.0 | 0.078 | 54.0 |
| gmd_03_high_bpm138 | 同じ範囲・{37,38,40} | 14.0 | 85 | 27.15 | 82.4 | 319.3 | 85 | 80.0 | 320.0 | 0.071 | 46.9 |
| gmd_04_extreme_bpm170 | 今のファイル（38 のみ） |  | 68 | 21.04 | 83.8 | 271.3 | 68 | 80.0 | 280.0 | 0.149 | 55.7 |
| gmd_04_extreme_bpm170 | 同じ範囲・38 のみ（再抽出） | 10.0 | 72 | 21.04 | 83.8 | 268.4 | 72 | 80.0 | 260.0 | 0.155 | 53.9 |
| gmd_04_extreme_bpm170 | 同じ範囲・{37,38,40} | 10.0 | 72 | 21.04 | 83.8 | 268.4 | 72 | 80.0 | 260.0 | 0.155 | 53.9 |

## 3) test split のフィルタの残数

(c) の判定：reject = 0.08 s の規則で落とす前の最小 IOI で判定、drop = 落としたあとで判定（実質すべて通る）

| モード | test split | (a) スネア類 | (b) 20 s 以上 | (c) 最小 IOI | (d) bpm 60〜160 |
|---|---|---|---|---|---|
| reject | 129 | 121 | 70 | 38 | 37 |
| drop | 129 | 121 | 70 | 70 | 69 |

## 4) 通過した抜粋とレガシーの切り直し（記述子）

frac_ioi_ltNNN = IOI が NNN ms 未満の割合（145 ms は Gen2 の圧力モデルの t50。段 1 の値で差し替える）。bpm中央値との差 = test split の通過曲の bpm 中央値からの差（sim 拡張の追加曲を選ぶ順。決定ドキュメント B-5）

| 区分 | midi_filename | info_bpm | reject通過 | drop通過 | 落とした数 | n | dur_s | ioi_min_ms | ioi_med_ms | q_ioi_min_ms | frac_ioi_lt100 | frac_ioi_lt120 | frac_ioi_lt145 | npvi | bpm中央値との差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test split | drummer3/session2/10_rock_100_beat_4-4.mid | 100 | False | True | 1 | 19 | 18.0 | 88.8 | 1192.5 | 100.0 | 0.056 | 0.056 | 0.056 | 32.0 | 0.0 |
| test split | drummer7/session1/16_hiphop_100_beat_4-4.mid | 100 | False | True | 5 | 32 | 24.16 | 80.0 | 171.3 | 80.0 | 0.097 | 0.097 | 0.161 | 114.7 | 0.0 |
| test split | drummer9/session1/3_rock_100_beat_4-4.mid | 100 | False | True | 7 | 46 | 26.41 | 80.0 | 450.0 | 80.0 | 0.111 | 0.178 | 0.289 | 96.0 | 0.0 |
| test split | drummer9/session1/7_rock_100_beat_4-4.mid | 100 | True | True | 0 | 30 | 23.22 | 105.0 | 761.3 | 120.0 | 0.0 | 0.034 | 0.034 | 35.5 | 0.0 |
| test split | drummer1/eval_session/10_soul-groove10_102_beat_4-4.mid | 102 | False | True | 2 | 33 | 26.38 | 84.6 | 756.1 | 80.0 | 0.031 | 0.031 | 0.031 | 55.7 | 2.0 |
| test split | drummer5/eval_session/10_soul-groove10_102_beat_4-4.mid | 102 | False | True | 3 | 26 | 25.88 | 80.9 | 1170.3 | 80.0 | 0.12 | 0.12 | 0.12 | 44.4 | 2.0 |
| test split | drummer7/eval_session/10_soul-groove10_102_beat_4-4.mid | 102 | False | True | 6 | 40 | 26.63 | 84.6 | 719.4 | 100.0 | 0.026 | 0.077 | 0.077 | 64.6 | 2.0 |
| test split | drummer8/eval_session/10_soul-groove10_102_beat_4-4.mid | 102 | False | True | 14 | 43 | 26.66 | 80.9 | 463.2 | 80.0 | 0.071 | 0.071 | 0.071 | 79.2 | 2.0 |
| test split | drummer5/session1/12_latin-brazilian-sambareggae_96_beat_4-4.mid | 96 | True | True | 0 | 67 | 26.49 | 134.1 | 278.0 | 140.0 | 0.0 | 0.0 | 0.121 | 62.4 | 4.0 |
| test split | drummer5/session1/7_latin-brazilian-maracatu_96_beat_4-4.mid | 96 | False | True | 61 | 165 | 27.41 | 82.0 | 152.3 | 80.0 | 0.073 | 0.201 | 0.36 | 31.2 | 4.0 |
| test split | drummer7/session2/112_jazz-fusion_96_beat_4-4.mid | 96 | True | True | 0 | 22 | 21.57 | 145.8 | 635.4 | 160.0 | 0.0 | 0.0 | 0.0 | 147.7 | 4.0 |
| test split | drummer8/session1/8_funk_96_beat_4-4.mid | 96 | True | True | 0 | 11 | 25.0 | 2471.4 | 2501.3 | 2480.0 | 0.0 | 0.0 | 0.0 | 0.8 | 4.0 |
| test split | drummer8/session2/23_rock_96_beat_4-4.mid | 96 | False | True | 1 | 31 | 26.23 | 82.0 | 1226.6 | 80.0 | 0.067 | 0.067 | 0.067 | 26.0 | 4.0 |
| test split | drummer8/session2/40_rock-indie_104_beat_4-4.mid | 104 | False | True | 8 | 58 | 26.8 | 93.7 | 300.5 | 80.0 | 0.018 | 0.035 | 0.175 | 62.8 | 4.0 |
| test split | drummer1/eval_session/2_funk-groove2_105_beat_4-4.mid | 105 | False | True | 19 | 54 | 26.8 | 81.0 | 406.0 | 80.0 | 0.075 | 0.113 | 0.17 | 76.5 | 5.0 |
| test split | drummer1/eval_session/9_soul-groove9_105_beat_4-4.mid | 105 | True | True | 0 | 24 | 26.3 | 1108.3 | 1140.5 | 1100.0 | 0.0 | 0.0 | 0.0 | 2.9 | 5.0 |
| test split | drummer5/eval_session/2_funk-groove2_105_beat_4-4.mid | 105 | False | True | 1 | 40 | 26.23 | 106.0 | 698.8 | 100.0 | 0.0 | 0.051 | 0.128 | 78.9 | 5.0 |
| test split | drummer5/eval_session/9_soul-groove9_105_beat_4-4.mid | 105 | True | True | 0 | 26 | 26.27 | 107.1 | 1140.5 | 120.0 | 0.0 | 0.04 | 0.04 | 23.3 | 5.0 |
| test split | drummer7/eval_session/2_funk-groove2_105_beat_4-4.mid | 105 | True | True | 0 | 24 | 26.29 | 1103.6 | 1146.4 | 1100.0 | 0.0 | 0.0 | 0.0 | 2.3 | 5.0 |
| test split | drummer7/eval_session/9_soul-groove9_105_beat_4-4.mid | 105 | True | True | 0 | 25 | 26.28 | 231.0 | 1141.7 | 240.0 | 0.0 | 0.0 | 0.0 | 12.8 | 5.0 |
| test split | drummer7/session3/109_rock_95_beat_4-4.mid | 95 | False | True | 4 | 38 | 25.24 | 118.4 | 619.7 | 120.0 | 0.0 | 0.027 | 0.108 | 56.6 | 5.0 |
| test split | drummer7/session3/117_rock_95_beat_4-4.mid | 95 | False | True | 4 | 35 | 21.15 | 90.8 | 313.8 | 100.0 | 0.029 | 0.059 | 0.118 | 48.7 | 5.0 |
| test split | drummer7/session3/132_soul_105_beat_4-4.mid | 105 | False | True | 6 | 60 | 26.42 | 86.9 | 400.0 | 100.0 | 0.017 | 0.051 | 0.085 | 56.3 | 5.0 |
| test split | drummer7/session3/133_soul_105_beat_4-4.mid | 105 | True | True | 0 | 71 | 26.82 | 88.1 | 402.4 | 100.0 | 0.029 | 0.043 | 0.043 | 47.8 | 5.0 |
| test split | drummer8/eval_session/2_funk-groove2_105_beat_4-4.mid | 105 | True | True | 0 | 24 | 26.28 | 1119.0 | 1140.5 | 1120.0 | 0.0 | 0.0 | 0.0 | 2.0 | 5.0 |
| test split | drummer8/eval_session/9_soul-groove9_105_beat_4-4.mid | 105 | True | True | 0 | 24 | 26.24 | 1126.2 | 1141.7 | 1120.0 | 0.0 | 0.0 | 0.0 | 0.9 | 5.0 |
| test split | drummer8/session2/16_funk_95_beat_4-4.mid | 95 | False | True | 4 | 36 | 26.51 | 128.9 | 775.0 | 120.0 | 0.0 | 0.0 | 0.057 | 44.5 | 5.0 |
| test split | drummer8/session1/19_funk_108_beat_4-4.mid | 108 | False | True | 27 | 98 | 26.92 | 83.3 | 237.3 | 100.0 | 0.052 | 0.103 | 0.175 | 55.9 | 8.0 |
| test split | drummer9/session1/24_rock_90_beat_4-4.mid | 90 | False | True | 2 | 35 | 25.82 | 109.7 | 604.9 | 100.0 | 0.0 | 0.029 | 0.059 | 65.9 | 10.0 |
| test split | drummer1/eval_session/6_hiphop-groove6_87_beat_4-4.mid | 87 | True | True | 0 | 18 | 26.19 | 1364.9 | 1385.1 | 1360.0 | 0.0 | 0.0 | 0.0 | 16.9 | 13.0 |
| test split | drummer5/eval_session/6_hiphop-groove6_87_beat_4-4.mid | 87 | True | True | 0 | 16 | 24.95 | 113.5 | 1379.3 | 120.0 | 0.0 | 0.067 | 0.067 | 51.7 | 13.0 |
| test split | drummer7/eval_session/6_hiphop-groove6_87_beat_4-4.mid | 87 | False | True | 2 | 16 | 24.77 | 497.1 | 1385.1 | 500.0 | 0.0 | 0.0 | 0.0 | 52.4 | 13.0 |
| test split | drummer8/eval_session/6_hiphop-groove6_87_beat_4-4.mid | 87 | True | True | 0 | 15 | 24.83 | 1377.9 | 1395.1 | 1360.0 | 0.0 | 0.0 | 0.0 | 40.5 | 13.0 |
| test split | drummer1/eval_session/3_soul-groove3_86_beat_4-4.mid | 86 | True | True | 0 | 61 | 26.48 | 84.3 | 467.3 | 80.0 | 0.017 | 0.017 | 0.017 | 47.8 | 14.0 |
| test split | drummer5/eval_session/3_soul-groove3_86_beat_4-4.mid | 86 | True | True | 0 | 23 | 26.5 | 106.1 | 1388.1 | 100.0 | 0.0 | 0.091 | 0.136 | 50.2 | 14.0 |
| test split | drummer7/eval_session/3_soul-groove3_86_beat_4-4.mid | 86 | True | True | 0 | 30 | 26.48 | 486.9 | 873.5 | 500.0 | 0.0 | 0.0 | 0.0 | 62.6 | 14.0 |
| test split | drummer8/eval_session/3_soul-groove3_86_beat_4-4.mid | 86 | True | True | 0 | 30 | 26.49 | 524.7 | 863.4 | 520.0 | 0.0 | 0.0 | 0.0 | 59.8 | 14.0 |
| test split | drummer8/session2/17_afrobeat_85_beat_4-4.mid | 85 | False | True | 1 | 38 | 26.26 | 277.9 | 555.9 | 260.0 | 0.0 | 0.0 | 0.0 | 38.6 | 15.0 |
| test split | drummer1/eval_session/5_funk-groove5_84_beat_4-4.mid | 84 | False | True | 5 | 55 | 27.3 | 168.2 | 529.0 | 160.0 | 0.0 | 0.0 | 0.0 | 45.6 | 16.0 |
| test split | drummer1/session1/28_latin-samba_116_beat_4-4.mid | 116 | True | True | 0 | 67 | 26.94 | 145.5 | 359.4 | 140.0 | 0.0 | 0.0 | 0.0 | 71.1 | 16.0 |
| test split | drummer1/session1/4_jazz-funk_116_beat_4-4.mid | 116 | True | True | 0 | 94 | 25.87 | 109.9 | 261.9 | 100.0 | 0.0 | 0.054 | 0.312 | 59.1 | 16.0 |
| test split | drummer5/eval_session/5_funk-groove5_84_beat_4-4.mid | 84 | False | True | 16 | 57 | 26.49 | 83.3 | 174.1 | 80.0 | 0.143 | 0.232 | 0.304 | 89.6 | 16.0 |
| test split | drummer7/eval_session/5_funk-groove5_84_beat_4-4.mid | 84 | True | True | 0 | 46 | 26.93 | 156.3 | 544.6 | 160.0 | 0.0 | 0.0 | 0.0 | 42.6 | 16.0 |
| test split | drummer8/eval_session/5_funk-groove5_84_beat_4-4.mid | 84 | True | True | 0 | 47 | 26.97 | 153.3 | 546.9 | 140.0 | 0.0 | 0.0 | 0.0 | 49.7 | 16.0 |
| test split | drummer1/eval_session/4_soul-groove4_80_beat_4-4.mid | 80 | True | True | 0 | 18 | 25.47 | 1459.4 | 1495.3 | 1460.0 | 0.0 | 0.0 | 0.0 | 3.0 | 20.0 |
| test split | drummer3/session1/11_rock_120_beat_4-4.mid | 120 | True | True | 0 | 35 | 23.01 | 96.9 | 631.2 | 100.0 | 0.029 | 0.029 | 0.088 | 52.0 | 20.0 |
| test split | drummer3/session1/14_rock_120_beat_4-4.mid | 120 | False | True | 3 | 10 | 19.98 | 81.2 | 232.3 | 80.0 | 0.111 | 0.111 | 0.222 | 125.5 | 20.0 |
| test split | drummer5/eval_session/4_soul-groove4_80_beat_4-4.mid | 80 | True | True | 0 | 20 | 25.49 | 351.6 | 1492.2 | 360.0 | 0.0 | 0.0 | 0.0 | 29.6 | 20.0 |
| test split | drummer7/eval_session/4_soul-groove4_80_beat_4-4.mid | 80 | True | True | 0 | 18 | 25.51 | 1475.0 | 1501.6 | 1480.0 | 0.0 | 0.0 | 0.0 | 1.2 | 20.0 |
| test split | drummer8/eval_session/4_soul-groove4_80_beat_4-4.mid | 80 | True | True | 0 | 18 | 25.5 | 1482.8 | 1498.4 | 1480.0 | 0.0 | 0.0 | 0.0 | 1.0 | 20.0 |
| test split | drummer7/session2/93_hiphop_75_beat_4-4.mid | 75 | True | True | 0 | 30 | 24.63 | 88.3 | 776.7 | 80.0 | 0.069 | 0.172 | 0.207 | 83.2 | 25.0 |
| test split | drummer8/session2/41_highlife_126_beat_4-4.mid | 126 | False | True | 20 | 101 | 27.36 | 87.3 | 255.5 | 80.0 | 0.01 | 0.04 | 0.18 | 44.7 | 26.0 |
| test split | drummer1/session2/35_punk_128_beat_4-4.mid | 128 | True | True | 0 | 87 | 27.16 | 100.6 | 322.8 | 100.0 | 0.0 | 0.081 | 0.233 | 41.9 | 28.0 |
| test split | drummer1/session1/239_funk-purdieshuffle_130_beat_4-4.mid | 130 | False | True | 5 | 42 | 27.02 | 100.0 | 578.8 | 100.0 | 0.024 | 0.024 | 0.049 | 71.5 | 30.0 |
| test split | drummer7/session3/24_hiphop_67_beat_4-4.mid | 67 | False | True | 1 | 32 | 25.11 | 85.8 | 442.2 | 80.0 | 0.129 | 0.355 | 0.452 | 51.6 | 33.0 |
| test split | drummer1/eval_session/8_rock-groove8_65_beat_4-4.mid | 65 | True | True | 0 | 18 | 25.84 | 228.8 | 1838.5 | 220.0 | 0.0 | 0.0 | 0.0 | 60.7 | 35.0 |
| test split | drummer5/eval_session/8_rock-groove8_65_beat_4-4.mid | 65 | True | True | 0 | 24 | 25.84 | 82.7 | 1603.8 | 80.0 | 0.087 | 0.217 | 0.217 | 117.0 | 35.0 |
| test split | drummer7/eval_session/8_rock-groove8_65_beat_4-4.mid | 65 | True | True | 0 | 19 | 25.83 | 223.1 | 1797.1 | 220.0 | 0.0 | 0.0 | 0.0 | 70.6 | 35.0 |
| test split | drummer7/session2/53_rock_135_beat_4-4.mid | 135 | False | True | 4 | 76 | 27.13 | 84.3 | 345.4 | 80.0 | 0.067 | 0.213 | 0.267 | 39.7 | 35.0 |
| test split | drummer8/eval_session/8_rock-groove8_65_beat_4-4.mid | 65 | True | True | 0 | 18 | 25.88 | 219.2 | 1826.9 | 220.0 | 0.0 | 0.0 | 0.0 | 60.6 | 35.0 |
| test split | drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid | 138 | False | True | 14 | 84 | 26.61 | 82.4 | 318.8 | 80.0 | 0.06 | 0.06 | 0.072 | 46.4 | 38.0 |
| test split | drummer1/eval_session/7_pop-groove7_138_beat_4-4.mid | 138 | True | True | 0 | 33 | 26.96 | 84.2 | 870.5 | 80.0 | 0.031 | 0.031 | 0.031 | 12.4 | 38.0 |
| test split | drummer5/eval_session/1_funk-groove1_138_beat_4-4.mid | 138 | False | True | 2 | 96 | 26.95 | 82.4 | 303.4 | 80.0 | 0.084 | 0.105 | 0.105 | 46.3 | 38.0 |
| test split | drummer5/eval_session/7_pop-groove7_138_beat_4-4.mid | 138 | False | True | 1 | 41 | 26.98 | 87.0 | 860.5 | 80.0 | 0.1 | 0.2 | 0.225 | 74.0 | 38.0 |
| test split | drummer7/eval_session/1_funk-groove1_138_beat_4-4.mid | 138 | True | True | 0 | 71 | 26.61 | 199.3 | 329.7 | 200.0 | 0.0 | 0.0 | 0.0 | 40.9 | 38.0 |
| test split | drummer7/eval_session/7_pop-groove7_138_beat_4-4.mid | 138 | True | True | 0 | 32 | 26.96 | 842.4 | 866.8 | 840.0 | 0.0 | 0.0 | 0.0 | 1.7 | 38.0 |
| test split | drummer8/eval_session/1_funk-groove1_138_beat_4-4.mid | 138 | False | True | 12 | 100 | 26.63 | 108.7 | 207.4 | 100.0 | 0.0 | 0.03 | 0.222 | 59.2 | 38.0 |
| test split | drummer8/eval_session/7_pop-groove7_138_beat_4-4.mid | 138 | True | True | 0 | 32 | 26.99 | 847.8 | 870.5 | 840.0 | 0.0 | 0.0 | 0.0 | 1.6 | 38.0 |
| test split | drummer5/session2/4_reggae_141_beat_4-4.mid | 141 | False | True | 10 | 31 | 25.81 | 94.0 | 394.9 | 80.0 | 0.033 | 0.067 | 0.167 | 73.7 | 41.0 |
| レガシー（gmd_03_high_bpm138）切り直し | drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid | 138 | False | True | 14 | 84 | 26.61 | 82.4 | 318.8 | 80.0 | 0.06 | 0.06 | 0.072 | 46.4 | 38.0 |
| レガシー（gmd_04_extreme_bpm170）切り直し | drummer1/session1/80_dance-breakbeat_170_beat_4-4.mid | 170 | False | True | 12 | 94 | 27.42 | 83.8 | 267.6 | 80.0 | 0.118 | 0.14 | 0.151 | 50.1 | 70.0 |

全ファイルの判定は `scan_all.csv`、記述子の全列は `excerpts.csv`、抜粋の MIDI は `excerpts/`。
