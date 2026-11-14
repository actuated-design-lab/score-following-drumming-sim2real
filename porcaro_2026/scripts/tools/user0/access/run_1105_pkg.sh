#!/usr/bin/env bash
# run_1105_pkg.sh — IEEE Access の図表の残り（11/05 夜の依頼）のうち、WS（Isaac）が要るものを回す。porcaro の環境で。済んだものは飛ばす
#   bash scripts/tools/user0/access/run_1105_pkg.sh fidelity  # 1・7：Gen2（DR なし）で 10/05 のテスト動作を再生（実測圧力 / Gen2 の圧力モデル）→ 手首角 RMSE と時系列
#   bash scripts/tools/user0/access/run_1105_pkg.sh dr        # 5：F0（opt2 r0 favorite）・E・B を実機コアの 5 タスクで DR 32 回（dr_seed 7000、新しい判定）
#   bash scripts/tools/user0/access/run_1105_pkg.sh c6        # 6：c6 の sim 側（同じ F0 の指令の公称 sim、diag_opt2.sh の trace）
#   bash scripts/tools/user0/access/run_1105_pkg.sh all       # 上の 3 つ → 結果をこのブランチに commit・push
# 目安：fidelity 6 本 × 2 で 10〜20 分、dr 15 本で 20〜40 分、c6 は数秒
set -u
cd "$(dirname "$0")/../../../.."
python -c "import isaaclab" 2>/dev/null || { echo "porcaro の環境が有効になっていません"; exit 1; }
PY=${PY:-python}
J=${JETSON:-../jetson_project}
STEP=${1:-all}
LOG=data/user0/eval/access_1104/pkg/logs; mkdir -p "$LOG"
G=data/user0/jfps2026/replay/gen2_check
D5=$J/data/user0/jfps2026/playback_20261005
declare -A RL5=([tm_A_quasistatic]=data_tm_A_quasistatic_1791183406.csv [tm_B_steps]=data_tm_B_steps_1791183490.csv
                [tm_C_sine]=data_tm_C_sine_1791183590.csv [tm_D_antagonist]=data_tm_D_antagonist_1791183665.csv
                [tm_E_dbl160_seed2]=data_tm_E_dbl160_seed2_1791183686.csv [tm_E_dbl160_seed3]=data_tm_E_dbl160_seed3_1791183707.csv
                [tm_E_gmd138_seed2]=data_tm_E_gmd138_seed2_1791183750.csv [tm_E_gmd138_seed3]=data_tm_E_gmd138_seed3_1791183793.csv)

step_fidelity() {
  # check_gen2.sh と同じ設定（--env_cfg Gen2、measured は生の実測圧力、cfg は Gen2 の圧力モデル＋通信の遅れ 8 ms）。既にある tm_B・tm_C・dbl160 s2 は飛ばす
  for m in tm_A_quasistatic tm_B_steps tm_C_sine tm_D_antagonist tm_E_dbl160_seed2 tm_E_dbl160_seed3 tm_E_gmd138_seed2 tm_E_gmd138_seed3; do
    for pm in measured cfg; do
      o=$G/sim_${m}__gen2_${pm}.csv
      [ -f "$o" ] && { echo "[skip] $o"; continue; }
      extra="--real_log $D5/${RL5[$m]}"
      [ $pm = measured ] && extra="$extra --meas_lpf_ms 0 --meas_play_kpa 0"
      echo "=== $m / gen2_$pm ($(date +%H:%M)) ==="
      $PY scripts/tools/user0/jfps2026/replay_open_loop.py --env_cfg Gen2 --pmodel $pm --signal "$J/signals/$m.csv" \
        $extra --no_drum --headless --out "$o" > "$LOG/fid_${m}_${pm}.log" 2>&1 || { echo "!!! $m / $pm（$LOG/fid_${m}_${pm}.log）"; tail -3 "$LOG/fid_${m}_${pm}.log"; }
    done
  done
  $PY analysis/eval/compare_hys_modes.py --dir "$G" --jetson "$J" --plot gen2_measured,gen2_cfg > "$G/compare.log" 2>&1
  grep -A12 "^## RMSE" "$G/summary.md"
  $PY scripts/tools/user0/access/fidelity_export.py --jetson "$J"
}

step_dr() {
  local S=data/user0/eval/real_core/songs O=data/user0/eval/access_1104/sim
  local E B
  E=$($PY -c "import json;print(json.load(open('data/user0/eval/b_eval/val/selection.json'))['E']['ckpt'])")
  B=$($PY -c "import json;print(json.load(open('data/user0/eval/b_eval/val/selection.json'))['B']['ckpt'])")
  for t in b_holdout_00_bpm90 b_holdout_01_bpm96 double_lead2_bpm160 gmd_legacy_gmd_03_high_recut_bpm138 gmd_legacy_gmd_04_extreme_recut_bpm170; do
    for M in F0 E B; do
      n=dr2_${M}_$t
      [ -f "$O/$n/summary.json" ] && { echo "[skip] $n"; continue; }
      case $M in
        F0) pol="--ff data/user0/ff/opt2/r0/favorite.json" ;;
        E)  pol="--model E --ckpt $E" ;;
        B)  pol="--model B --ckpt $B" ;;
      esac
      echo "=== $n ($(date +%H:%M)) ==="
      $PY scripts/tools/user0/access/sim_trace.py --headless --midi $S/$t.mid $pol --n_dr 32 --dr_seed 7000 --name $n \
        > "$LOG/$n.log" 2>&1 || { echo "!!! $n（$LOG/$n.log）"; tail -3 "$LOG/$n.log"; }
    done
  done
  $PY scripts/tools/user0/access/sim_dr_summary.py
}

step_c6() {
  ls data/user0/ff/opt2/diag/r0_favorite_*/trace.npz > /dev/null 2>&1 || { echo "!!! data/user0/ff/opt2/diag/r0_favorite_*/trace.npz がない（diag_opt2.sh の出力）"; return; }
  git -C "$J" pull -q
  $PY scripts/tools/user0/access/c6_swing_bins.py --jetson "$J"
}

case $STEP in
  fidelity) step_fidelity ;;
  dr) step_dr ;;
  c6) step_c6 ;;
  all) step_fidelity; step_dr; step_c6 ;;
  *) echo "fidelity / dr / c6 / all"; exit 1 ;;
esac

# 結果を今のブランチ（user0/pkg-1105）に入れる。sim の再生の csv は大きいので入れず、そろえた時系列（pkg/）と表だけ
if [ "${NO_PUSH:-1}" != "1" ]; then
  git add -f $G/metrics.csv $G/summary.md $G/compare.log $G/*.json $G/fig_*.png 2>/dev/null
  git add -f data/user0/eval/access_1104/pkg/*.csv data/user0/eval/access_1104/pkg/*.md 2>/dev/null
  git add -f data/user0/eval/access_1104/sim/dr2_*/summary.json data/user0/eval/access_1104/sim/dr2_*/strikes.csv 2>/dev/null
  git add -f data/user0/eval/access_1104/c6/summary.md data/user0/eval/access_1104/c6/strikes.csv 2>/dev/null
  git commit -q -m "run_1105_pkg.sh $STEP（$(hostname)）：Gen2 の再生（忠実度）、DR 32 回の sim、c6 の sim 側" \
    && git push -q && echo "[pushed]" || echo "!!! commit / push に失敗（git status を見て、手で push）"
fi
