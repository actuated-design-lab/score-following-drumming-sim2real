#!/usr/bin/env bash
# eval_favorite.sh — FF の候補を favorite.json（CMA-ES の最後の分布の平均）を主にして選び直す（2026-11-03 本人の決定）
#
# RL の候補は最後のチェックポイント（model_1499）で固定。FF でそれに当たるのは最後の分布の平均（favorite.json）で、
# best.json（60 世代 × 16 候補のうち、その世代の学習用の列・引きで J が最大のもの）は学習用のデータで多く選んだ分だけ有利になる。
# そこで、検証用の列での選択は favorite を主（F0_r<R>）、best を副（F0best_r<R>、選択の対象外）にする。
#
# run_ff.sh opt（段 2 → best の評価 → 選択）が終わるのを待ってから：
#   val/F0_r<R>（best の評価）→ val/F0best_r<R> に名前を変え、favorite を F0_r<R> として評価、名目値も同じ（F0nom_r0 / F0nombest_r0）
#   → select_seeds.py で選び直す → push
#   bash scripts/tools/user0/ff/eval_favorite.sh          # tmux の別の窓で（porcaro の環境）
set -u
cd "$(dirname "$0")/../../../.."
PY=${PY:-python}
V=data/user0/eval/b_eval/val
O=data/user0/ff/opt
LOG=data/user0/ff/logs; mkdir -p "$LOG"
echo "[eval_favorite] run_ff.sh の終わりを待つ（$(date +%H:%M)）"
while [ ! -f "$V/selection.md" ] || pgrep -f "run_ff.sh|ff_opt.py|eval_b.py" > /dev/null; do sleep 300; done
echo "[eval_favorite] 開始（$(date +%H:%M)）"
one() {   # one <favorite.json> <主の名前> <副の名前>
  local F=$1 N=$2 NB=$3
  [ -f "$F" ] || { echo "[skip] $F がない"; return; }
  # すでに favorite で評価済みなら何もしない（もう一度流しても同じ結果）
  if [ -f "$V/$N/summary.json" ] && grep -q "\"ff\": \"$F\"" "$V/$N/summary.json"; then echo "[skip] $N（favorite で評価済み）"; return; fi
  if [ -d "$V/$N" ] && [ ! -d "$V/$NB" ]; then mv "$V/$N" "$V/$NB"; echo "[mv] $N → $NB（best の評価）"; fi
  $PY scripts/tools/user0/eval/eval_b.py --headless --set val --ff "$F" --name "$N" > "$LOG/eval_fav_$N.log" 2>&1 \
    || { echo "!!! $N で失敗"; tail -5 "$LOG/eval_fav_$N.log"; }
  grep -o '"success_dr_mean": [0-9.]*' "$V/$N/summary.json" 2>/dev/null
}
for R in 0 1 2; do one "$O/r$R/favorite.json" "F0_r$R" "F0best_r$R"; done
one "$O/r0_nominal/favorite.json" "F0nom_r0" "F0nombest_r0"
$PY scripts/tools/user0/eval/select_seeds.py | tail -25
if [ "${NO_PUSH:-1}" != "1" ]; then
  git add -f "$V"/*/summary.json "$V"/selection.md "$V"/selection.json "$O"/*/favorite.json 2>/dev/null
  git commit -q -m "FF 段 2：favorite を主にした検証用の評価と選択（eval_favorite.sh）" && git pull -q --no-rebase --no-edit && git push -q \
    && echo "[pushed]" || echo "!!! push に失敗（手で git push）"
fi
echo "[eval_favorite] 終わり（$(date +%H:%M)）"
