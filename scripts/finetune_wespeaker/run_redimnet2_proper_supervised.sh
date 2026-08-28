#!/usr/bin/env bash
# Supervisor cho finetune_redimnet2_proper.py -- giong cac supervisor script khac trong thu
# muc nay. Script nay KHONG tu bat OOM/cuDNN error trong tung batch (khac finetune_v11.py),
# nen mot crash giua epoch se mat tien do CA epoch do (khong chi 1 batch) -- van resume duoc
# tu last.pt (luu cuoi moi epoch) nhung co the lap lai vai phut tinh toan.
#
# Chay: bash run_redimnet2_proper_supervised.sh

cd "$(dirname "$0")" || exit 1
LOG_DIR="C:/Users/Adm/AppData/Local/Temp/claude/c--Lily-voiceKYC/ec6650ff-739a-45dc-8b26-b9c3aa0b2a75/scratchpad"
MAX_RETRY=60
SLEEP_SEC=45

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 "$MAX_RETRY"); do
  echo "=== [supervisor] lan chay $i/$MAX_RETRY : $(date '+%H:%M:%S') ===" \
    | tee -a "$LOG_DIR/redimnet2_proper_supervisor.log"
  ../../.venv/Scripts/python.exe -u finetune_redimnet2_proper.py \
    >> "$LOG_DIR/redimnet2_proper_supervisor.log" 2>&1
  code=$?
  echo "=== [supervisor] ket thuc voi ma $code ===" | tee -a "$LOG_DIR/redimnet2_proper_supervisor.log"

  if [ "$code" -eq 0 ]; then
    echo "[supervisor] train hoan tat binh thuong." | tee -a "$LOG_DIR/redimnet2_proper_supervisor.log"
    break
  fi
  if grep -q "Early stopping" "$LOG_DIR/redimnet2_proper_supervisor.log" 2>/dev/null; then
    echo "[supervisor] da early-stop, khong chay lai." | tee -a "$LOG_DIR/redimnet2_proper_supervisor.log"
    break
  fi
  echo "[supervisor] loi (ma $code) -> nghi ${SLEEP_SEC}s roi resume tu last.pt" \
    | tee -a "$LOG_DIR/redimnet2_proper_supervisor.log"
  sleep "$SLEEP_SEC"
done
