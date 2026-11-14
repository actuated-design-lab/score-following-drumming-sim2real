#!/usr/bin/env bash
# run_select.sh — (b) の学習のあと：検証用の列で全 run を評価 → 規則でシードを選ぶ → 選んだ B・E と保険を ONNX に書き出す（WS、tmux の中で）
#
#   bash scripts/tools/user0/eval/run_select.sh            # 全部（1 run 数分 × 7）
#   STEP=eval bash ...   / STEP=select / STEP=export        # 一部だけ
#
# 1) b_sets.py：検証用・ホールドアウトの列（GPU なし、数秒）
# 2) eval_b.py：7 run（B・E の seed 1〜3、保険）を検証用の列 × DR 32 回で評価（Porcaro-Gen2-B-user0 / -obsv1-）
# 3) select_seeds.py：事前登録の規則で B・E を 1 本ずつ選ぶ → data/user0/eval/b_eval/val/selection.md
# 4) export_onnx.py：選んだ B・E と保険を data/user0/gen2/onnx/ に書き出す（jetson_project には自分で入れる。repo を分けるため）
# 結果の md と json を push する（NO_PUSH=1 で止める）。ONNX は push しない
set -u
cd "$(dirname "$0")/../../../.."
PY=${PY:-python}
STEP=${STEP:-all}
T=scripts/tools/user0/eval
LOGD=data/user0/eval/logs; mkdir -p "$LOGD"
LR=logs/user0/rsl_rl

run_dir() {   # run_dir <B|E> <run_name>  → 最新のフォルダ
  local exp; [ "$1" = B ] && exp=porcaro_rslrl_lstm_gen2 || exp=porcaro_rslrl_mlp_gen2
  ls -td $LR/$exp/*_"$2" 2>/dev/null | head -1
}

if [ "$STEP" = all ] || [ "$STEP" = eval ]; then
  $PY $T/b_sets.py || exit 1
  for J in B:B_b_seed1 B:B_b_seed2 B:B_b_seed3 E:E_b_seed1 E:E_b_seed2 E:E_b_seed3 B:B_b_obsv1_seed1; do
    M=${J%%:*}; RN=${J#*:}
    D=$(run_dir "$M" "$RN")
    if [ -z "$D" ] || [ ! -f "$D/model_1499.pt" ]; then echo "!!! $RN の model_1499.pt がない（$D）"; continue; fi
    if [ -f data/user0/eval/b_eval/val/$RN/summary.json ]; then echo "[skip] $RN（評価済み）"; continue; fi
    OBS=v2; case $RN in *obsv1*) OBS=v1 ;; esac
    echo "=== $RN（$D、観測 $OBS）$(date +%H:%M) ==="
    $PY $T/eval_b.py --headless --set val --model "$M" --obs $OBS --ckpt "$D/model_1499.pt" --name "$RN" \
      > "$LOGD/eval_b_val_$RN.log" 2>&1 || { echo "!!! $RN で失敗（$LOGD/eval_b_val_$RN.log）"; tail -5 "$LOGD/eval_b_val_$RN.log"; }
    grep -o '"success_dr_mean": [0-9.]*' data/user0/eval/b_eval/val/$RN/summary.json 2>/dev/null
  done
fi

if [ "$STEP" = all ] || [ "$STEP" = select ]; then
  $PY $T/select_seeds.py || exit 1
fi

if [ "$STEP" = all ] || [ "$STEP" = export ]; then
  O=data/user0/gen2/onnx; mkdir -p "$O"
  SEL=data/user0/eval/b_eval/val/selection.json
  for M in B E; do
    RN=$($PY -c "import json;print(json.load(open('$SEL'))['$M']['run'])")
    D=$(run_dir "$M" "$RN")
    A="--agent rsl_rl_lstm_cfg_entry_point"; X=""
    [ "$M" = E ] && { A="--agent rsl_rl_mlp_cfg_entry_point"; X="--use_frame_stacking --frame_stack_k 5"; }
    echo "=== export $RN ==="
    $PY scripts/rsl_rl/export_onnx.py --task Porcaro-Gen2-B-user0 $A --checkpoint="$D/model_1499.pt" --lookahead_horizon 0.5 $X \
      --out_dir "$O" --out_name "gen2b_${RN}" --headless > "$LOGD/export_$RN.log" 2>&1 || { echo "!!! export $RN で失敗"; tail -5 "$LOGD/export_$RN.log"; }
  done
  D=$(run_dir B B_b_obsv1_seed1)
  echo "=== export B_b_obsv1_seed1（保険） ==="
  $PY scripts/rsl_rl/export_onnx.py --task Porcaro-Gen2-B-obsv1-user0 --agent rsl_rl_lstm_cfg_entry_point --checkpoint="$D/model_1499.pt" \
    --lookahead_horizon 0.5 --out_dir "$O" --out_name gen2b_B_b_obsv1_seed1 --headless > "$LOGD/export_obsv1.log" 2>&1 \
    || { echo "!!! export 保険で失敗"; tail -5 "$LOGD/export_obsv1.log"; }
  ls -la "$O"
fi

if [ "${NO_PUSH:-1}" != "1" ]; then
  git add -f data/user0/eval/b_sets/summary.md data/user0/eval/b_sets/*.npz data/user0/eval/b_sets/holdout_midi/*.mid \
    data/user0/eval/b_eval/val/*/summary.json data/user0/eval/b_eval/val/selection.* 2>/dev/null
  git commit -q -m "(b) のシードの選択（run_select.sh、$(hostname)）" && git pull -q --no-rebase --no-edit && git push -q \
    && echo "[pushed]" || echo "!!! push に失敗（手で git push）"
fi
