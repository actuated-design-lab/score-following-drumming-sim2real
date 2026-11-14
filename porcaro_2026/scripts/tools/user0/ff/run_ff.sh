#!/usr/bin/env bash
# run_ff.sh — FF ベースラインの段を回す（WS で。長いものは tmux の中で）
#
#   porcaro                                   # 自分の venv に入る
#   cd ~/my_isaaclab_projects/porcaro_2026 && git pull
#   bash scripts/tools/user0/ff/run_ff.sh check    # 段 0（GPU なし、1 分ほど）。目標時刻・圧力・採点の確認
#   bash scripts/tools/user0/ff/run_ff.sh probe    # 段 1（Isaac、152 env × 4.5 s）。L と打撃力
#   bash scripts/tools/user0/ff/run_ff.sh opt      # 段 2（Isaac、1 restart 2〜3 h）。CMA-ES（DR あり restart 0〜2 ＋ 名目値 r0）→
#                                                  検証用の列で eval_b.py → 規則で 1 本選ぶ（RESTARTS・GENS で変える）
#   OPT=opt2 EVAL=b_eval2 bash .../run_ff.sh opt   # 2026-11-04：押し付けの判定を直した版（連なり）。出力の置き場所を分ける
#   restart を並列にするとき：窓ごとに RESTARTS=0 NOMINAL=0 …、RESTARTS=1 NOMINAL=0 …、RESTARTS=2 NOMINAL=0 …、RESTARTS=none NOMINAL=1 …
#   （それぞれ終わったところまで評価と選択をする。最後に終わった窓の選択が全部をそろえたもの）
#
# 各段の結果は data/user0/ff/<段>/summary.md。終わるたびに commit → pull --no-rebase → push（NO_PUSH=1 で止める）。
# npz と大きい csv は push しない。
set -u
cd "$(dirname "$0")/../../../.."
PY=${PY:-python}
J=${JETSON:-../jetson_project}
T=scripts/tools/user0/ff
OUTROOT=data/user0/ff
OPT=${OPT:-opt}                       # 段 2 の出力：data/user0/ff/$OPT
EVAL=${EVAL:-b_eval}                  # 検証用の評価：data/user0/eval/$EVAL/val
GENS=${GENS:-80}                      # ff_opt.py の既定と同じ（28 次元、2026-11-03）
STEP=${1:-}
LOG=$OUTROOT/logs; mkdir -p "$LOG"

push() {   # push <メッセージ> <パス...>
  local msg=$1; shift
  [ "${NO_PUSH:-1}" = "1" ] && { echo "[NO_PUSH] $msg"; return; }
  local f files=()
  for f in "$@"; do [ -f "$f" ] && files+=("$f"); done
  [ ${#files[@]} -eq 0 ] && { echo "[push] 入れるファイルなし（$msg）"; return; }
  git add -f "${files[@]}"
  git commit -q -m "$msg" -m "run_ff.sh $STEP（$(hostname)）" && git pull -q --no-rebase --no-edit && git push -q \
    && echo "[pushed] $msg" || echo "!!! push に失敗（git status を見て、手で git push）"
}

step_check() {
  local O=$OUTROOT/check
  $PY $T/check_stage0.py --jetson "$J" --out "$O" 2>&1 | tee "$LOG/check.log" | tail -5
  push "FF 段 0 の確認（run_ff.sh check）" "$O"/summary.md "$O"/*.png
}

step_probe() {
  local O=$OUTROOT/probe; mkdir -p "$O"
  echo "=== probe（$(date +%H:%M)）==="
  $PY $T/ff_sim.py probe --headless --out "$O" > "$LOG/probe.log" 2>&1 || { echo "!!! probe で失敗（$LOG/probe.log の最後）"; tail -20 "$LOG/probe.log"; return 1; }
  $PY $T/analyze_probe.py --dir "$O" 2>&1 | tee "$LOG/probe_analyze.log" | tail -8
  push "FF 段 1：L と打撃力（run_ff.sh probe）" "$O"/summary.md "$O"/probe_table.csv "$O"/params.csv "$O"/probe_meta.json "$O"/*.png
}

step_opt() {
  local O=$OUTROOT/$OPT; mkdir -p "$O"
  local V=data/user0/eval/$EVAL/val
  local R RS=${RESTARTS:-0 1 2}
  [ "$RS" = "none" ] && RS=""
  for R in $RS; do
    if [ -f "$O/r$R/summary.md" ]; then echo "[skip] r$R（済み）"; continue; fi
    echo "=== opt r$R（$(date +%H:%M)）==="
    $PY $T/ff_opt.py --headless --restart "$R" --gens "$GENS" --out "$O" > "$LOG/${OPT}_r$R.log" 2>&1 || { echo "!!! opt r$R で失敗"; tail -20 "$LOG/${OPT}_r$R.log"; }
    grep "^\[ff_opt" "$LOG/${OPT}_r$R.log" | tail -2
    push "FF 段 2（$OPT）：CMA-ES r$R" "$O/r$R/summary.md" "$O/r$R/log.csv" "$O/r$R/best.json" "$O/r$R/favorite.json"
  done
  if [ "${NOMINAL:-1}" = "1" ] && [ ! -f "$O/r0_nominal/summary.md" ]; then
    echo "=== opt r0 名目値（$(date +%H:%M)）==="
    $PY $T/ff_opt.py --headless --restart 0 --gens "$GENS" --nominal --out "$O" > "$LOG/${OPT}_r0_nominal.log" 2>&1 || { echo "!!! opt 名目値で失敗"; tail -20 "$LOG/${OPT}_r0_nominal.log"; }
    push "FF 段 2（$OPT）：CMA-ES 名目値" "$O/r0_nominal/summary.md" "$O/r0_nominal/log.csv" "$O/r0_nominal/best.json" "$O/r0_nominal/favorite.json"
  fi
  # 検証用の列で評価（RL と同じ eval_b.py、DR 32 回のペア）。favorite（最後の分布の平均）を主 = F0_r<R>、best を副 = F0best_r<R>（2026-11-03 の決定）
  ev() {  # ev <json> <名前>
    [ -f "$1" ] || return 0
    [ -f "$V/$2/summary.json" ] && return 0
    $PY scripts/tools/user0/eval/eval_b.py --headless --set val --ff "$1" --name "$2" --out "data/user0/eval/$EVAL" > "$LOG/${EVAL}_$2.log" 2>&1 \
      || { echo "!!! eval $2 で失敗"; tail -5 "$LOG/${EVAL}_$2.log"; }
  }
  for R in $RS; do ev "$O/r$R/favorite.json" "F0_r$R"; ev "$O/r$R/best.json" "F0best_r$R"; done
  if [ "${NOMINAL:-1}" = "1" ]; then ev "$O/r0_nominal/favorite.json" "F0nom_r0"; ev "$O/r0_nominal/best.json" "F0nombest_r0"; fi
  [ -d "$V" ] && $PY scripts/tools/user0/eval/select_seeds.py --dir "data/user0/eval/$EVAL/val" --methods ${SELECT_METHODS:-B E F0} | tail -25
  push "FF 段 2（$OPT）：検証用の列での評価と選択" "$V"/*/summary.json "$V"/selection.md "$V"/selection.json
}

case "$STEP" in
  check) step_check ;;
  probe) step_probe ;;
  opt) step_opt ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
