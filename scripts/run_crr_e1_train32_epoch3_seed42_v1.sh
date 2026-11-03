#!/usr/bin/env bash
set -euo pipefail

root=/root/work/data/New_MOT_v2/motip_paradigm_reset
repo=${root}/source/MOTIP_crr_entity_state_v1
python_env=/root/work/New_MOT/venv
output=${root}/crr_e1_entity_state_train32_epoch3_seed42_v1
bootstrap=${root}/crr_pointer_train32_epoch10_v3/checkpoint_9.pth
detector=/root/work/data/New_MOT_v2/pretrains/r50_deformable_detr_coco_dancetrack.pth
config=${repo}/configs/crr_e1_entity_state_train32_epoch3_seed42_v1.yaml
status=${output}/run_status.json

test -s "${bootstrap}"
test -s "${detector}"
test -s "${config}"
test ! -e "${output}"
mkdir -p "${output}"
cp "${config}" "${output}/frozen_config.yaml"
cp "$0" "${output}/frozen_launcher.sh"
export PATH="${python_env}/bin:${PATH}"
export PYTHONUNBUFFERED=1

"${python_env}/bin/python" - "${status}" <<'PY'
import json,socket,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status":"running", "stage":"crr_e1_entity_state_train32_epoch3",
    "started_at":datetime.now(timezone.utc).isoformat(), "hostname":socket.gethostname(),
    "source":"MOTIP_crr_entity_state_v1", "bootstrap":"P0 checkpoint_9",
    "train_split":"crr_pointer_train32_v1", "epochs":3,
    "trainable":"CRR-E1 only", "num_workers":4, "prefetch_factor":2,
    "locked_validation_used":False, "gt_runtime":False
}, indent=2))
PY
finish(){ code="$?"; "${python_env}/bin/python" - "${status}" "$code" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); c=int(sys.argv[2]); x.update(status="completed" if c==0 else "failed",exit_code=c,ended_at=datetime.now(timezone.utc).isoformat()); p.write_text(json.dumps(x,indent=2))
PY
}
trap finish EXIT
cd "${repo}"
"${python_env}/bin/accelerate" launch --num_processes=1 train.py \
  --data-root /root/work/data/New_MOT_v2/datasets --outputs-dir "${output}" \
  --exp-name crr_e1_entity_state_train32_epoch3_seed42_v1 --config-path "${config}" \
  --detr-pretrain "${detector}" --resume-model "${bootstrap}" \
  --resume-optimizer False --resume-scheduler False --use-wandb False
