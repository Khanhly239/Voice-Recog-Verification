#!/usr/bin/env bash
# Supervisor cho v10: GPU 4GB dung chung voi desktop (Chrome/Slack/VSCode) nen thinh thoang
# gap OOM hoac "cuDNN CUDNN_STATUS_EXECUTION_FAILED". Loi cuDNN co the lam hong CUDA context
# -> phai khoi dong lai process, khong the chi bo qua batch.
#
# Script luu last.pt MOI epoch va tu resume, nen relaunch khong mat tien do (toi da mat 1 epoch).
#
# Chay: bash run_v10_supervised.sh

cd "$(dirname "$0")" || exit 1
LOG_DIR="C:/Users/Adm/AppData/Local/Temp/claude/c--Lily-voiceKYC/ac1798fb-5bfa-4a5d-b89f-e1d42d416d4b/scratchpad"
MAX_RETRY=40
SLEEP_SEC=45

# Giam phan manh bo nho -- giup tranh OOM khi VRAM bi chia se
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 "$MAX_RETRY"); do
  echo "=== [supervisor] lan chay $i/$MAX_RETRY : $(date '+%H:%M:%S') ===" \
    | tee -a "$LOG_DIR/v10_supervisor.log"
  python -u finetune_v10.py >> "$LOG_DIR/v10_supervisor.log" 2>&1
  code=$?
  echo "=== [supervisor] ket thuc voi ma $code ===" | tee -a "$LOG_DIR/v10_supervisor.log"

  if [ "$code" -eq 0 ]; then
    echo "[supervisor] train hoan tat binh thuong." | tee -a "$LOG_DIR/v10_supervisor.log"
    break
  fi
  if grep -q "Early stopping" "$LOG_DIR/v10_supervisor.log" 2>/dev/null; then
    echo "[supervisor] da early-stop, khong chay lai." | tee -a "$LOG_DIR/v10_supervisor.log"
    break
  fi
  echo "[supervisor] loi (ma $code) -> nghi ${SLEEP_SEC}s roi resume tu last.pt" \
    | tee -a "$LOG_DIR/v10_supervisor.log"
  sleep "$SLEEP_SEC"
done
