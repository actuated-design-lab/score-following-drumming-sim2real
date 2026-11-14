#!/usr/bin/env bash
# check_gen2.sh — Gen2 の設定クラスで再生し，凍結版（t0k1g，--ctrl で上書きしていたもの）と同じ数字が出るかを確かめる
#
# 1) 実測圧力（measured）で tm_B / tm_C / tm_E dbl160 seed2（実機 10/05）を Gen2 で再生 → freeze の t0k1g と比べる（同じになるはず）
# 2) 同じ 3 本を Gen2 の圧力モデルそのまま（--pmodel cfg = shaped＋通信の遅れ 8 ms）で再生（指令→角度の通し）
# 出力: data/user0/jfps2026/replay/gen2_check/（終わったら commit / push。NO_PUSH=1 で止める）
#
#   PY=python bash scripts/tools/user0/jfps2026/check_gen2.sh
set -u
cd "$(dirname "$0")/../../../.."
PY=${PY:-python}
J=${JETSON:-../jetson_project}
D=$J/data/user0/jfps2026/playback_20261005
OUT=data/user0/jfps2026/replay/gen2_check
mkdir -p "$OUT"
declare -A LOG=([tm_B_steps]=data_tm_B_steps_1791183490.csv [tm_C_sine]=data_tm_C_sine_1791183590.csv
                [tm_E_dbl160_seed2]=data_tm_E_dbl160_seed2_1791183686.csv)
for m in tm_B_steps tm_C_sine tm_E_dbl160_seed2; do
  for pm in measured cfg; do
    tag=gen2_${pm}
    o=$OUT/sim_${m}__${tag}.csv
    [ -f "$o" ] && continue
    extra="--real_log $D/${LOG[$m]}"     # cfg でも比べる相手として json に残す（圧力には使わない）
    [ $pm = measured ] && extra="$extra --meas_lpf_ms 0 --meas_play_kpa 0"
    echo "=== $m / $tag ($(date +%H:%M)) ==="
    $PY scripts/tools/user0/jfps2026/replay_open_loop.py --env_cfg Gen2 --pmodel $pm --signal "$J/signals/$m.csv" \
      $extra --no_drum --headless --out "$o" || echo "!!! $m / $tag で失敗（replay_error.log）"
  done
done
$PY analysis/eval/compare_hys_modes.py --dir "$OUT" --jetson "$J" --plot gen2_measured,gen2_cfg > "$OUT/compare.log" 2>&1
grep -A6 "^## RMSE" "$OUT/summary.md"
echo "比べる相手（freeze の t0k1g）: tm_B 3.34 / tm_C 5.89 / tm_E dbl160 s2 6.44"
if [ "${NO_PUSH:-1}" != "1" ]; then
  git add -f "$OUT"/metrics.csv "$OUT"/summary.md "$OUT"/compare.log "$OUT"/*.json "$OUT"/fig_*.png 2>/dev/null
  git commit -q -m "Gen2 の設定クラスで再生して確認（check_gen2.sh）" && git pull -q --no-rebase --no-edit && git push -q \
    && echo pushed || echo "!!! push に失敗（手で git push）"
fi
