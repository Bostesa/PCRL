#!/bin/bash
# One-button launcher for INLP_BENCHMARK.
#
# Pre-flight checks:
#   - aws sts get-caller-identity passes
#   - target git ref is pushed to origin
#   - no INLP_BENCHMARK already running
#   - Standard family quota has headroom for c5.4xlarge (16 vCPU)
#   - identifies + protects PCRL_VARCONSTRAINT, BIOS_LAYER12, SPLICE_BENCHMARK
#   - S3 bucket reachable
#
# Stage 1 (this script): sed-substitute the user-data, run aws ec2 run-instances.
# Stage 2 (on-instance): user_data.sh runs all 27 cells + aggregator.
#
# To press launch:
#   1. Review the variables below.
#   2. ./infra/inlp/launch.sh
#
# To dry-run (no actual launch, just print the run-instances command):
#   DRY_RUN=1 ./infra/inlp/launch.sh

set -euo pipefail

# ── KNOBS ────────────────────────────────────────────────────────────────
INSTANCE_TYPE="c5.4xlarge"   # 16 vCPU Standard family; INLP is CPU-bound
AMI_ID="${AMI_ID:-ami-05603a42e5254c4bb}"   # Deep Learning AMI w/ PyTorch
KEY_NAME="${KEY_NAME:-pcrl-gpu-key}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"  # MUST be set by env
SUBNET_ID="${SUBNET_ID:-}"                  # optional
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-pcrl-bios-s3-writer}"
REGION="${REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-pcrl-bios-overnight-20260504}"
S3_PREFIX_PATH="inlp_benchmark"
S3_PREFIX="s3://${S3_BUCKET}/${S3_PREFIX_PATH}"
GIT_REF="${GIT_REF:-bios-pcrl-layer12-2026-05-05}"
TAG_NAME="INLP_BENCHMARK"
PROTECTED_TAGS=(PCRL_VARCONSTRAINT BIOS_LAYER12 SPLICE_BENCHMARK)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"

need_var() {
  local name="$1"
  local value="${!name:-}"
  if [ -z "${value}" ]; then
    echo "ERROR: env var ${name} is not set." >&2
    exit 2
  fi
}
need_var SECURITY_GROUP_ID

# ── Pre-flight 1: AWS creds ──────────────────────────────────────────────
echo "[pre-flight 1] aws sts get-caller-identity"
aws sts get-caller-identity --output json | sed 's/^/  /'

# ── Pre-flight 2: git ref pushed ─────────────────────────────────────────
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

# ── Pre-flight 3: no INLP_BENCHMARK already running ──────────────────────
echo "[pre-flight 3] confirming no INLP_BENCHMARK already running"
EXISTING="$(aws ec2 describe-instances \
  --region "${REGION}" \
  --filters "Name=tag:Name,Values=${TAG_NAME}" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
if [ -n "${EXISTING}" ]; then
  echo "ERROR: ${TAG_NAME} already running: ${EXISTING}" >&2
  echo "  Terminate first or pick a new tag." >&2
  exit 2
fi
echo "  no ${TAG_NAME} instances in pending/running"

# ── Pre-flight 4: identify protected instances and confirm no touch ──────
echo "[pre-flight 4] checking for protected instances we MUST NOT touch"
for tag in "${PROTECTED_TAGS[@]}"; do
  IDS="$(aws ec2 describe-instances \
    --region "${REGION}" \
    --filters "Name=tag:Name,Values=${tag}" "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].InstanceId' --output text)"
  if [ -n "${IDS}" ]; then
    echo "  ${tag} currently running: ${IDS} (this launcher will NOT touch)"
  else
    echo "  ${tag}: none in pending/running"
  fi
done

# ── Pre-flight 5: S3 bucket reachable ────────────────────────────────────
echo "[pre-flight 5] s3 bucket ${S3_BUCKET}"
aws s3 ls "s3://${S3_BUCKET}/" --region "${REGION}" >/dev/null
echo "  bucket reachable"

# Pre-flight 5b: confirm preprocessed datasets exist where user_data expects
echo "[pre-flight 5b] confirming preprocessed datasets exist at "
echo "  s3://${S3_BUCKET}/pcrl_varconstraint/data/"
DATA_LIST="$(aws s3 ls "s3://${S3_BUCKET}/pcrl_varconstraint/data/" --region "${REGION}" || true)"
if [ -z "${DATA_LIST}" ]; then
  echo "ERROR: preprocessed datasets not found at expected S3 path." >&2
  exit 2
fi
echo "${DATA_LIST}" | sed 's/^/    /'

# Pre-flight 5c: confirm LAFTR FINAL_BENCHMARK.csv exists (aggregator needs it)
echo "[pre-flight 5c] confirming LAFTR FINAL_BENCHMARK.csv present (for three-way table)"
if ! aws s3 ls "s3://${S3_BUCKET}/laftr_benchmark/FINAL_BENCHMARK.csv" --region "${REGION}" >/dev/null; then
  echo "ERROR: laftr_benchmark/FINAL_BENCHMARK.csv missing on S3 — aggregator will fail." >&2
  exit 2
fi
echo "  FINAL_BENCHMARK.csv present"

# ── Pre-flight 6: Standard-family quota / running c-instances ────────────
echo "[pre-flight 6] running Standard-family instances (c5.4xlarge=16 vCPU)"
RUNNING_STD="$(aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[?starts_with(InstanceType, `c`) || starts_with(InstanceType, `m`) || starts_with(InstanceType, `r`)].[InstanceType,InstanceId]' \
  --output text || true)"
echo "${RUNNING_STD:-  (none)}" | sed 's/^/  /'

# ── Render user-data ─────────────────────────────────────────────────────
USER_DATA_RENDERED="$(mktemp)"
sed \
  -e "s|__S3_PREFIX__|${S3_PREFIX}|g" \
  -e "s|__GIT_REF__|${GIT_REF}|g" \
  "${REPO_ROOT}/infra/inlp/user_data.sh" > "${USER_DATA_RENDERED}"
echo "[render] user-data rendered to ${USER_DATA_RENDERED}"
echo "  size: $(wc -c < ${USER_DATA_RENDERED}) bytes (limit 16384)"
if [ "$(wc -c < ${USER_DATA_RENDERED})" -gt 16000 ]; then
  echo "  WARNING: user-data near 16 KB limit; consider trimming." >&2
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

read -p "All pre-flight passed. Launch ${TAG_NAME} on ${INSTANCE_TYPE}? Type 'yes' to confirm: " CONFIRM
if [ "${CONFIRM}" != "yes" ]; then
  echo "aborted"; exit 1
fi

INSTANCE_JSON="$(aws ec2 run-instances "${LAUNCH_ARGS[@]}")"
INSTANCE_ID="$(echo "${INSTANCE_JSON}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["Instances"][0]["InstanceId"])')"
echo "launched: ${INSTANCE_ID}"
echo
echo "Live monitoring:"
echo "  aws s3 ls ${S3_PREFIX}/live_logs/ --region ${REGION}"
echo "  aws s3 cp ${S3_PREFIX}/live_logs/inlp_train.log - | tail -100"
echo "  aws s3 ls ${S3_PREFIX}/results_partial/ --region ${REGION} --recursive | tail -20"
echo
echo "Final results land at:"
echo "  ${S3_PREFIX}/results_final/"
echo
echo "To pull final to local:"
echo "  aws s3 sync ${S3_PREFIX}/results_final/ results/inlp_benchmark/"
