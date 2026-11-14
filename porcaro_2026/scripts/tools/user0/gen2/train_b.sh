#!/usr/bin/env bash
# train_b.sh — 学習を決めた順番で 1 本ずつ回す（2026-11-01）
#
#   順番:
#     1. B seed 1（Porcaro-Gen2-B-user0、観測 v2）
#     2. E seed 1
#     3. 予備 B seed 1（Porcaro-Gen2-B-obsv1-user0、分布は同じ・観測は v1）
#     4〜7. B・E の seed 2、seed 3（選択規則：B・E それぞれ 3 シードから検証用の列で 1 本）
#   1 本ずつ回す：2 本並列は GPU の取り合いで 1 本ずつより遅い（並列で 1 本 9 h、1 本だけなら 3.4 h）
#
# 使い方（tmux の中で）:
#   bash scripts/tools/user0/gen2/train_b.sh                 # 上の 7 本
#   WAIT_GPU=1 bash scripts/tools/user0/gen2/train_b.sh      # GPU で他の学習（(a) など）が動いていれば、終わるまで待ってから始める
#   JOBS="B:b:1 E:b:1" bash ...                              # 一部だけ（<モデル>:<b か b_obsv1>:<seed> を空白区切り）
#   INS=E bash ...                                           # 保険を E にする（既定 B）
#
# ログ：logs/user0/rsl_rl/porcaro_rslrl_{lstm,mlp}_gen2/<日時>_<run_name>/（run_name = <モデル>_<b|b_obsv1>_seed<n>）
# 投入時・1 本終わるごとに data/user0/gen2/train/runs_<日時>_b.md を push する（NO_PUSH=1 で止める）
# 1 本終わるごとに、その run の params/env.yaml から obs_version・音価の確率・カリキュラムを md に写す（取り違えの確認用）
set -u
cd "$(dirname "$0")/../../../.."
PY=${PY:-python}
NUM_ENVS=${NUM_ENVS:-2048}
MAX_IT=${MAX_IT:-1500}
INS=${INS:-B}
JOBS=${JOBS:-"B:b:1 E:b:1 ${INS}:b_obsv1:1 B:b:2 E:b:2 B:b:3 E:b:3"}
O=data/user0/gen2/train; mkdir -p "$O" data/user0/gen2/logs
MD=$O/runs_$(date +%Y%m%d_%H%M)_b.md
REV=$(git rev-parse --short HEAD)

push() {
  [ "${NO_PUSH:-1}" = "1" ] && return
  git add -f "$MD" && git commit -q -m "$1" && git pull -q --no-rebase --no-edit && git push -q && echo "[pushed] $1" \
    || echo "!!! push に失敗（手で git push）"
}

task_for() {
  case $1 in
    b) echo Porcaro-Gen2-B-user0 ;;
    b_obsv1) echo Porcaro-Gen2-B-obsv1-user0 ;;
    *) echo "!!! 2 つめは b か b_obsv1: $1" >&2; exit 1 ;;
  esac
}
args_for() {
  case $1 in
    B) echo "--agent rsl_rl_lstm_cfg_entry_point --lookahead_horizon 0.5" ;;
    E) echo "--agent rsl_rl_mlp_cfg_entry_point --lookahead_horizon 0.5 --use_frame_stacking --frame_stack_k 5" ;;
    *) echo "!!! モデルは B か E: $1" >&2; exit 1 ;;
  esac
}
exp_for() { [ "$1" = B ] && echo porcaro_rslrl_lstm_gen2 || echo porcaro_rslrl_mlp_gen2; }

# 起動前の確認：書式だけ先に全部通す（途中で落ちないように）
for J in $JOBS; do IFS=: read -r M K S <<< "$J"; task_for "$K" > /dev/null; args_for "$M" > /dev/null; done

if [ "${WAIT_GPU:-0}" = "1" ]; then
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
    echo "[wait] GPU で他のプロセスが動いている（$(date +%H:%M)）。5 分後にもう一度見る"; sleep 300
  done
fi

{
  echo "# (b) の学習（$(date '+%Y-%m-%d %H:%M') 投入、$(hostname)）"
  echo ""
  echo "- コミット: $REV（$(git log -1 --format=%s | head -c 80)）"
  echo "- num_envs $NUM_ENVS、$MAX_IT iteration、1 本ずつ"
  echo "- 順番: $JOBS（<モデル>:<b=本命 v2 / b_obsv1=保険 v1>:<seed>）"
  echo "- 中身：user0/porcaro_2026_env_cfg.py の Porcaro2026EnvCfg_Gen2_B / _Gen2_B_ObsV1。各 run の設定の全部は params/env.yaml"
  echo ""
  echo "| run_name | タスク | 開始 | 終了 | 終了コード | obs_version | 音価の確率 | カリキュラム（iteration） | ログ |"
  echo "|---|---|---|---|---|---|---|---|---|"
} > "$MD"
push "(b) の学習を投入（train_b.sh、$JOBS）"

for J in $JOBS; do
  IFS=: read -r M K S <<< "$J"
  TASK=$(task_for "$K")
  RN=${M}_${K}_seed${S}
  L=data/user0/gen2/logs/train_${RN}_$(date +%m%d%H%M).log
  T0=$(date +%H:%M)
  echo "=== $RN（$TASK）開始 $T0 ==="
  $PY scripts/rsl_rl/train.py --task "$TASK" $(args_for "$M") --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_IT" --seed "$S" --run_name "$RN" --headless > "$L" 2>&1
  C=$?
  Y=$(ls -td logs/user0/rsl_rl/$(exp_for "$M")/*_"$RN" 2>/dev/null | head -1)/params/env.yaml
  OV=$(grep -m1 "^obs_version:" "$Y" 2>/dev/null | awk '{print $2}')
  NP=$(grep -A7 "^rand_note_probs:" "$Y" 2>/dev/null | grep -E "^- " | awk '{print $2}' | paste -sd, -)
  CI=$(grep -A2 "^curriculum_iters:" "$Y" 2>/dev/null | grep -E "^- " | awk '{print $2}' | paste -sd, -)
  echo "| $RN | $TASK | $T0 | $(date +%H:%M) | $C | ${OV:-?} | ${NP:-?} | ${CI:-?} | $L |" >> "$MD"
  echo "=== $RN 終了（コード $C、obs_version ${OV:-?}）==="
  push "(b) の学習：$RN 終了（コード $C）"
done
echo "" >> "$MD"; echo "全部終了: $(date '+%Y-%m-%d %H:%M')" >> "$MD"
push "(b) の学習が全部終了（train_b.sh）"
cat "$MD"
