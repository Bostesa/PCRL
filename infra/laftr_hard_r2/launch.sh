#!/bin/bash
# One-button launcher for the NeurIPS rebuttal LAFTR-hard-R² pilot.
#
# Stage 1 (this script): sed-substitute user-data, run aws ec2 run-instances.
# Stage 2 (on-instance): user_data.sh handles deps, data sync, smoke,
#   full 60-cell pilot (3 datasets × 3 seeds × 200 epochs, 6h timeout each),
#   S3 sync, shutdown.
#
# Estimated wall-clock: ~8-10 GPU-h on g4dn.xlarge (LAFTR adds ~1.3× overhead
# over V2's ~7.5h baseline for the same 60 cells). Hard cap: 20 h.
# Estimated cost: ~$5-10 at $0.526/h × 10h typical, $11 worst case at 20h cap.

set -euo pipefail

# ── KNOBS ────────────────────────────────────────────────────────────────
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
AMI_ID="${AMI_ID:-ami-012ba162b9cd2729c}"   # DL Ubuntu 22.04 (GPU) — same as erase_pilot 2026-05-17
KEY_NAME="${KEY_NAME:-pcrl-gpu-key}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-0f93a621e295c9ab9}"
SUBNET_ID="${SUBNET_ID:-}"
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-pcrl-bios-s3-writer}"
REGION="${REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-pcrl-bios-overnight-20260504}"
S3_PREFIX_PATH="laftr_hard_r2"
S3_PREFIX="s3://${S3_BUCKET}/${S3_PREFIX_PATH}"
GIT_REF="${GIT_REF:-laftr-hard-r2-2026-05-17}"
TAG_NAME="${TAG_NAME:-LAFTR_HARD_R2}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"

# ── Pre-flight 1: AWS creds ─────────────────────────────────────────────
echo "[pre-flight 1] aws sts get-caller-identity"
aws sts get-caller-identity --output json | sed 's/^/  /'

# ── Pre-flight 2: git ref pushed ────────────────────────────────────────
echo "[pre-flight 2] git ref ${GIT_REF} reachable on origin"
LOCAL_HEAD="$(git rev-parse HEAD 2>/dev/null || echo none)"
REMOTE_HEAD="$(git ls-remote origin "${GIT_REF}" | awk '{print $1}')"
if [ -z "${REMOTE_HEAD}" ]; then
  echo "ERROR: ${GIT_REF} not found on origin. Push first:" >&2
  echo "  git push origin ${GIT_REF}" >&2
  exit 2
fi
echo "  local HEAD: ${LOCAL_HEAD}"
echo "  origin/${GIT_REF}: ${REMOTE_HEAD}"
if [ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ] && [ "${LOCAL_HEAD}" != "none" ]; then
  echo "  WARNING: local HEAD != origin/${GIT_REF}. The instance will clone from origin/${GIT_REF}."
fi

# ── Pre-flight 3: no existing LAFTR_HARD_R2 instance ────────────────────
EXISTING="$(aws ec2 describe-instances \
  --region "${REGION}" \
  --filters "Name=tag:Name,Values=${TAG_NAME}" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
if [ -n "${EXISTING}" ]; then
  echo "ERROR: instance(s) already tagged ${TAG_NAME} are running: ${EXISTING}" >&2
  echo "  Terminate them or change TAG_NAME before launching." >&2
  exit 2
fi

# ── Pre-flight 4: S3 bucket reachable ───────────────────────────────────
echo "[pre-flight 4] s3 bucket ${S3_BUCKET} reachable"
aws s3 ls "s3://${S3_BUCKET}/" --region "${REGION}" >/dev/null
echo "  bucket reachable"

# ── Pre-flight 5: running G-family instances (do not touch) ─────────────
echo "[pre-flight 5] running G-family + Standard-family instances (do not touch)"
aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].[InstanceType,Tags[?Key==`Name`]|[0].Value,InstanceId]' \
  --output text | sed 's/^/  /'

# ── Render user-data ─────────────────────────────────────────────────────
USER_DATA_RENDERED="$(mktemp)"
sed \
  -e "s|__S3_PREFIX__|${S3_PREFIX}|g" \
  -e "s|__S3_BUCKET__|${S3_BUCKET}|g" \
  -e "s|__GIT_REF__|${GIT_REF}|g" \
  "${REPO_ROOT}/infra/laftr_hard_r2/user_data.sh" > "${USER_DATA_RENDERED}"
echo "[render] user-data rendered to ${USER_DATA_RENDERED}"
echo "  size: $(wc -c < ${USER_DATA_RENDERED}) bytes (limit 16384)"
if [ "$(wc -c < ${USER_DATA_RENDERED})" -gt 16000 ]; then
  echo "  WARNING: user-data near 16 KB limit; trim before launching." >&2
fi

# ── Compose the launch command ──────────────────────────────────────────
LAUNCH_ARGS=(
  --image-id "${AMI_ID}"
  --instance-type "${INSTANCE_TYPE}"
  --key-name "${KEY_NAME}"
  --security-group-ids "${SECURITY_GROUP_ID}"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${TAG_NAME}}]"
  --instance-initiated-shutdown-behavior terminate
  --user-data "file://${USER_DATA_RENDERED}"
  --region "${REGION}"
  --output json
)
if [ -n "${IAM_INSTANCE_PROFILE}" ]; then
  LAUNCH_ARGS+=(--iam-instance-profile "Name=${IAM_INSTANCE_PROFILE}")
fi
if [ -n "${SUBNET_ID}" ]; then
  LAUNCH_ARGS+=(--subnet-id "${SUBNET_ID}")
fi

echo
echo "[launch command]"
printf '  aws ec2 run-instances'
for arg in "${LAUNCH_ARGS[@]}"; do
  printf ' \\\n    %q' "${arg}"
done
echo

if [ "${DRY_RUN}" = "1" ]; then
  echo
  echo "DRY_RUN=1 — not launching. Unset DRY_RUN to actually launch."
  exit 0
fi

INSTANCE_JSON="$(aws ec2 run-instances "${LAUNCH_ARGS[@]}")"
INSTANCE_ID="$(echo "${INSTANCE_JSON}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["Instances"][0]["InstanceId"])')"
echo "launched: ${INSTANCE_ID}"
echo
echo "Live monitoring:"
echo "  aws s3 ls ${S3_PREFIX}/live_logs/ --region ${REGION}"
echo "  aws s3 cp ${S3_PREFIX}/live_logs/laftr_hard_r2.log - | tail -100"
echo "  aws s3 cp ${S3_PREFIX}/smoke/smoke.log              - | tail -200"
echo
echo "Stage gates (touched as each completes):"
echo "  aws s3 ls ${S3_PREFIX}/ --region ${REGION} | grep STAGE_"
echo
echo "Per-dataset summaries land at:"
echo "  ${S3_PREFIX}/per_dataset_summary/{adult,hmda,diabetes}_summary.json"
echo
echo "Final results land at:"
echo "  ${S3_PREFIX}/results_final/"
echo
echo "To pull final to local:"
echo "  aws s3 sync ${S3_PREFIX}/results_final/ results/"
