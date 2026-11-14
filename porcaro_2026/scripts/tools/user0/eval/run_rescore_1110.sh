#!/usr/bin/env bash
# run_rescore_1110.sh — 検証列（val）を新判定（ff_score.py：METHOD = contact_event、TRAIN_GAP_S = 0.040）で評価し直し、シードを選び直す
# 7 本 × 3〜5 分（GPU）。tmux の中で回す：
#   tmux new -d -s rescore 'source ~/venvs/porcaro/bin/activate && bash scripts/tools/user0/eval/run_rescore_1110.sh 2>&1 | tee /tmp/rescore_1110.log'
# 出力: data/user0/eval/b_eval_1110/val/<run>/summary.json、selection.{md,json}（既存の b_eval/val・val_peakmatch_1103 は触らない）
set -u
cd ~/my_isaaclab_projects/porcaro_2026
OUT=data/user0/eval/b_eval_1110; mkdir -p data/user0/eval/logs
for J in B:B_b_seed1:v2 B:B_b_seed2:v2 B:B_b_seed3:v2 E:E_b_seed1:v2 E:E_b_seed2:v2 E:E_b_seed3:v2 B:B_b_obsv1_seed1:v1; do
  IFS=: read M RN OBS <<< "$J"
  EXP=$([ "$M" = B ] && echo porcaro_rslrl_lstm_gen2 || echo porcaro_rslrl_mlp_gen2)
  D=$(ls -td logs/user0/rsl_rl/$EXP/*_"$RN" | head -1)
  [ -f "$OUT/val/$RN/summary.json" ] && { echo "[skip] $RN"; continue; }
  echo "=== $RN $(date +%H:%M:%S) $D"
  python scripts/tools/user0/eval/eval_b.py --headless --set val --model "$M" --obs "$OBS" \
    --ckpt "$D/model_1499.pt" --name "$RN" --out "$OUT" > "data/user0/eval/logs/eval_b_1110_$RN.log" 2>&1 \
    || echo "!!! $RN failed"
done
python scripts/tools/user0/eval/select_seeds.py --dir "$OUT/val" --methods B E
echo "=== DONE $(date +%H:%M:%S)"
