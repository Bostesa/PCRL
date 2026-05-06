#!/bin/bash
# One-button launcher for SPLICE_BENCHMARK on a c5.4xlarge.
#
# Pre-flight:
#   - aws sts get-caller-identity passes
#   - target git ref pushed to origin
#   - PCRL warmstart S3 prefix exists (we read from it; we don't push to it)
#   - DO NOT TOUCH PCRL_VARCONSTRAINT or BIOS_LAYER12 instances
#
# Stage 1 (this script): sed-substitute user-data, run aws ec2 run-instances,
#   print the instance-id.
# Stage 2 (on-instance): user_data.sh handles smoke + full 60-cell run +
#   aggregation + shutdown.

set -euo pipefail

# ── KNOBS ────────────────────────────────────────────────────────────────
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"
AMI_ID="${AMI_ID:-ami-05603a42e5254c4bb}"
KEY_NAME="${KEY_NAME:-pcrl-gpu-key}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"
SUBNET_ID="${SUBNET_ID:-}"
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-pcrl-bios-s3-writer}"
REGION="${REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-pcrl-bios-overnight-20260504}"
S3_PREFIX_PATH="splice_benchmark"
S3_PREFIX="s3://${S3_BUCKET}/${S3_PREFIX_PATH}"
S3_PCRL_PREFIX="s3://${S3_BUCKET}/pcrl_varconstraint"
GIT_REF="${GIT_REF:-bios-pcrl-layer12-2026-05-05}"
TAG_NAME="${TAG_NAME:-SPLICE_BENCHMARK}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"

# ── Pre-flight 0: required env ──────────────────────────────────────────
if [ -z "${SECURITY_GROUP_ID}" ]; then
  echo "ERROR: SECURITY_GROUP_ID env var must be set" >&2
  exit 2
fi

# ── Pre-flight 1: AWS creds ─────────────────────────────────────────────
echo "[pre-flight 1] aws sts get-caller-identity"
aws sts get-caller-identity --output json | sed 's/^/  /'

# ── Pre-flight 2: git ref pushed ────────────────────────────────────────
echo "[pre-flight 2] git ref ${GIT_REF} reachable on origin"
LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git ls-remote origin "${GIT_REF}" | awk '{print $1}')"
if [ -z "${REMOTE_HEAD}" ]; then
  echo "ERROR: ${GIT_REF} not found on origin. Push first." >&2
  exit 2
fi
echo "  local HEAD: ${LOCAL_HEAD}"
echo "  origin/${GIT_REF}: ${REMOTE_HEAD}"
if [ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]; then
  echo "  WARNING: local HEAD != origin/${GIT_REF}. The instance will clone from origin/${GIT_REF}."
fi

# ── Pre-flight 3: PCRL S3 warmstart exists (we'll read from it) ─────────
echo "[pre-flight 3] PCRL warmstart S3 prefix ${S3_PCRL_PREFIX}/warmstart/"
aws s3 ls "${S3_PCRL_PREFIX}/warmstart/" --region "${REGION}" >/dev/null
echo "  warmstart prefix reachable"

# ── Pre-flight 4: confirm PCRL_VARCONSTRAINT + BIOS_LAYER12 are running ─
echo "[pre-flight 4] confirming PCRL_VARCONSTRAINT and BIOS_LAYER12 are running (DO NOT TOUCH)"
PCRL_VC_INSTANCES="$(aws ec2 describe-instances \
  --region "${REGION}" \
  --filters "Name=tag:Name,Values=PCRL_VARCONSTRAINT" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
BIOS_INSTANCES="$(aws ec2 describe-instances \
  --region "${REGION}" \
  --filters "Name=tag:Name,Values=BIOS_LAYER12" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
echo "  PCRL_VARCONSTRAINT: ${PCRL_VC_INSTANCES:-(none)}"
echo "  BIOS_LAYER12:       ${BIOS_INSTANCES:-(none)}"
echo "  → this launcher will NOT touch those instances."

# ── Pre-flight 5: G + Standard family quotas ────────────────────────────
echo "[pre-flight 5] running G-family + Standard-family vCPU usage"
aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].[InstanceType,Tags[?Key==`Name`]|[0].Value,InstanceId]' \
  --output text | sed 's/^/  /'

# ── Pre-flight 6: S3 bucket reachable ───────────────────────────────────
echo "[pre-flight 6] s3 bucket ${S3_BUCKET}"
aws s3 ls "s3://${S3_BUCKET}/" --region "${REGION}" >/dev/null
echo "  bucket reachable"

# ── Render user-data ─────────────────────────────────────────────────────
USER_DATA_RENDERED="$(mktemp)"
sed \
  -e "s|__S3_PREFIX__|${S3_PREFIX}|g" \
  -e "s|__S3_PCRL_PREFIX__|${S3_PCRL_PREFIX}|g" \
  -e "s|__GIT_REF__|${GIT_REF}|g" \
  "${REPO_ROOT}/infra/splince/user_data.sh" > "${USER_DATA_RENDERED}"
echo "[render] user-data rendered to ${USER_DATA_RENDERED}"
echo "  size: $(wc -c < ${USER_DATA_RENDERED}) bytes (limit 16384)"
if [ "$(wc -c < ${USER_DATA_RENDERED})" -gt 16000 ]; then
  echo "  WARNING: user-data near 16 KB limit; consider trimming."
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
  echo "DRY_RUN=1 — not launching. To actually launch: unset DRY_RUN."
  exit 0
fi

INSTANCE_JSON="$(aws ec2 run-instances "${LAUNCH_ARGS[@]}")"
INSTANCE_ID="$(echo "${INSTANCE_JSON}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["Instances"][0]["InstanceId"])')"
echo "launched: ${INSTANCE_ID}"
echo
echo "Live monitoring:"
echo "  aws s3 ls ${S3_PREFIX}/live_logs/ --region ${REGION}"
echo "  aws s3 cp ${S3_PREFIX}/live_logs/splince_train.log - | tail -100"
echo "  aws s3 cp ${S3_PREFIX}/smoke/smoke.log - | tail -200"
echo
echo "Final results land at:"
echo "  ${S3_PREFIX}/results_final/"
echo
echo "To pull final to local:"
echo "  aws s3 sync ${S3_PREFIX}/results_final/ results/splince_benchmark/"
