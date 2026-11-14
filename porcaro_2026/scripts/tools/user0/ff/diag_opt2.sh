#!/usr/bin/env bash
# diag_opt2.sh — 段2（opt2）の候補（restart ごとの favorite と best、名目値）を、実機コアの 5 タスクで DR なしの sim に流して診断する（WS、porcaro の環境で）
# 段2の 4 本が全部終わってから回す。済んだものは飛ばす。そのあと diag_opt2.py で表にする
#   bash scripts/tools/user0/ff/diag_opt2.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."
python -c "import isaaclab" 2>/dev/null || { echo "porcaro の環境が有効になっていません"; exit 1; }
O=data/user0/ff/opt2; D=$O/diag; S=data/user0/eval/real_core/songs
TASKS="b_holdout_00_bpm90 b_holdout_01_bpm96 double_lead2_bpm160 gmd_legacy_gmd_03_high_recut_bpm138 gmd_legacy_gmd_04_extreme_recut_bpm170"
for R in r0 r1 r2 r0_nominal; do
  for K in favorite best; do
    J=$O/$R/$K.json; [ -f "$J" ] || { echo "[--] $J がない"; continue; }
    for t in $TASKS; do
      n=${R}_${K}_$t
      [ -f "$D/$n/summary.json" ] && { echo "[skip] $n"; continue; }
      python scripts/tools/user0/access/sim_trace.py --headless --nominal --midi $S/$t.mid --ff "$J" --name "$n" --out "$D" > /tmp/diag_$n.log 2>&1 \
        && echo "[done] $n" || { echo "!!! $n（/tmp/diag_$n.log）"; tail -5 /tmp/diag_$n.log; }
    done
  done
done
python scripts/tools/user0/ff/diag_opt2.py
