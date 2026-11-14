#!/usr/bin/env bash
# run_1104.sh — 11/04 の動画と解析用の sim（WS、porcaro の環境で）。済んだものは飛ばすので、止まっても同じコマンドで続きから回る
#   bash scripts/tools/user0/access/run_1104.sh video     # 1. 動画 3 本（実時間・DR なし）
#   bash scripts/tools/user0/access/run_1104.sh replay    # 2. C1：F0 の指令の再生（sim の圧力モデル 5 本 ＋ 実測圧を入れる 15 本）
#   bash scripts/tools/user0/access/run_1104.sh trace     # 3. C3：5 タスク × B・E・F0・F3 の時系列（DR なし ＋ DR 32 本）
#   bash scripts/tools/user0/access/run_1104.sh all
# 出力：data/user0/eval/access_1104/{sim,replay}/（replay の CSV は gzip）。動画（*.mp4）は git に入れない（access_1104/.gitignore）
set -euo pipefail
cd "$(dirname "$0")/../../../.."
python -c "import isaaclab" 2>/dev/null || { echo "porcaro の環境が有効になっていません"; exit 1; }
J=../jetson_project
SONGS=data/user0/eval/real_core/songs
OUT=data/user0/eval/access_1104
TR=scripts/tools/user0/access/sim_trace.py
RP=scripts/tools/user0/jfps2026/replay_open_loop.py
sel() { python -c "import json;print(json.load(open('data/user0/eval/b_eval/val/selection.json'))['$1']['ckpt'])"; }
E=$(sel E); B=$(sel B)
TASKS="b_holdout_00_bpm90 b_holdout_01_bpm96 double_lead2_bpm160 gmd_legacy_gmd_03_high_recut_bpm138 gmd_legacy_gmd_04_extreme_recut_bpm170"
done_() { [ -f "$1" ] && echo "[skip] $1" ; }
ck() { [ "$1" = E ] && echo "$E" || echo "$B"; }

video() {
  local V="--headless --enable_cameras --nominal --video"
  done_ $OUT/sim/vid_E_double160/video_meta.json || python $TR $V --midi $SONGS/double_lead2_bpm160.mid --model E --ckpt "$E" --name vid_E_double160
  done_ $OUT/sim/vid_F3_holdout0/video_meta.json || python $TR $V --midi $SONGS/b_holdout_00_bpm90.mid --signal $J/signals/core2026/F3_b_holdout_00_bpm90.csv --name vid_F3_holdout0
  done_ $OUT/sim/vid_E_holdout0/video_meta.json || python $TR $V --midi $SONGS/b_holdout_00_bpm90.mid --model E --ckpt "$E" --name vid_E_holdout0
}

replay() {
  mkdir -p $OUT/replay
  for t in $TASKS; do
    done_ $OUT/replay/F0_${t}_cfg.csv.gz || { python $RP --headless --env_cfg Gen2 --pmodel cfg \
        --signal $J/signals/core2026/F0_${t}.csv --out $OUT/replay/F0_${t}_cfg.csv && gzip -f $OUT/replay/F0_${t}_cfg.csv; }
  done
  # 実機の F0 の本表の run（core2026_score/runs.csv）の実測圧を入れる
  python - <<'EOF' > /tmp/f0_logs.txt
import pandas as pd
r = pd.read_csv("../jetson_project/data/user0/access2026/core2026_score/runs.csv")
r = r[(r.method == "F0") & (~r.bridge.astype(bool))]
for _, x in r.iterrows():
    print(x.task, x.file)
EOF
  while read -r t f; do
    u=$(basename "$f" .csv | sed 's/.*_//')
    done_ $OUT/replay/F0_${t}_meas_${u}.csv.gz || { python $RP --headless --env_cfg Gen2 --pmodel measured \
        --signal $J/signals/core2026/F0_${t}.csv --real_log $J/$f --out $OUT/replay/F0_${t}_meas_${u}.csv && gzip -f $OUT/replay/F0_${t}_meas_${u}.csv; }
  done < /tmp/f0_logs.txt
}

trace() {
  for t in $TASKS; do
    for m in E B; do
      done_ $OUT/sim/nom_${m}_${t}/summary.json || python $TR --headless --nominal --midi $SONGS/$t.mid --model $m --ckpt "$(ck $m)" --name nom_${m}_${t}
      done_ $OUT/sim/dr_${m}_${t}/summary.json  || python $TR --headless --n_dr 32 --midi $SONGS/$t.mid --model $m --ckpt "$(ck $m)" --name dr_${m}_${t}
    done
    for f in F0 F3; do
      done_ $OUT/sim/nom_${f}_${t}/summary.json || python $TR --headless --nominal --midi $SONGS/$t.mid --signal $J/signals/core2026/${f}_${t}.csv --name nom_${f}_${t}
      done_ $OUT/sim/dr_${f}_${t}/summary.json  || python $TR --headless --n_dr 32 --midi $SONGS/$t.mid --signal $J/signals/core2026/${f}_${t}.csv --name dr_${f}_${t}
    done
  done
}

case "${1:-all}" in
  video) video ;; replay) replay ;; trace) trace ;;
  all) video; replay; trace ;;
  *) echo "video | replay | trace | all"; exit 1 ;;
esac
echo "[run_1104] $1 おわり"
