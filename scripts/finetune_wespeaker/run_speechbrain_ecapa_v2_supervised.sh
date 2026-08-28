#!/usr/bin/env bash
# Supervisor cho speechbrain_ecapa_v2 (VoxVietnam+VoxCeleb1+CommonVoice) -- giong het
# run_speechbrain_ecapa_v1_supervised.sh. GPU 4GB dung chung nen thinh thoang gap OOM/loi
# cuDNN co the lam hong CUDA context -> phai khoi dong lai process.
#
# Chay: bash run_speechbrain_ecapa_v2_supervised.sh

cd "$(dirname "$0")" || exit 1
LOG_DIR="C:/Users/Adm/AppData/Local/Temp/claude/c--Lily-voiceKYC/ec6650ff-739a-45dc-8b26-b9c3aa0b2a75/scratchpad"
MAX_RETRY=80
SLEEP_SEC=45

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 "$MAX_RETRY"); do
  echo "=== [supervisor] lan chay $i/$MAX_RETRY : $(date '+%H:%M:%S') ===" \
    | tee -a "$LOG_DIR/speechbrain_ecapa_v2_supervisor.log"
  ../../.venv/Scripts/python.exe -u finetune_speechbrain_ecapa_v2.py \
    >> "$LOG_DIR/speechbrain_ecapa_v2_supervisor.log" 2>&1
  code=$?
  echo "=== [supervisor] ket thuc voi ma $code ===" | tee -a "$LOG_DIR/speechbrain_ecapa_v2_supervisor.log"

  if [ "$code" -eq 0 ]; then
    echo "[supervisor] train hoan tat binh thuong." | tee -a "$LOG_DIR/speechbrain_ecapa_v2_supervisor.log"
    break
  fi
  if grep -q "Early stopping" "$LOG_DIR/speechbrain_ecapa_v2_supervisor.log" 2>/dev/null; then
    echo "[supervisor] da early-stop, khong chay lai." | tee -a "$LOG_DIR/speechbrain_ecapa_v2_supervisor.log"
    break
  fi
  echo "[supervisor] loi (ma $code) -> nghi ${SLEEP_SEC}s roi resume tu last.pt" \
    | tee -a "$LOG_DIR/speechbrain_ecapa_v2_supervisor.log"
  sleep "$SLEEP_SEC"
done
