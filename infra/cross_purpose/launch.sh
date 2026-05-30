#!/bin/bash
# One-button launcher for the NeurIPS rebuttal cross-purpose Phase D pilot.
#
# Two-instance plan (mirrors the laftr_hard_r2 hardening pattern at c23e113):
#   ./infra/cross_purpose/launch.sh CROSS_PURPOSE_AB        # Adult + HMDA
#   ./infra/cross_purpose/launch.sh CROSS_PURPOSE_DIABETES  # Diabetes
#
# Each instance: g4dn.xlarge, 1× T4, 200 epochs × 3 seeds, erase-layer +
# linear-R² constraint on h_concat at τ_concat=0.10 per dataset.
# Hard wall cap: 20 hours. Expected wall: ~4-5 h.
# Estimated cost (both instances): ~$5-10 at $0.526/h.
#
# Stage 6 emits STATUS.txt (durable headline) to ARCHIVE_DEST; Stage 7 syncs
# results_final to ARCHIVE_DEST (lifecycle-exempt). See memory
# reference_s3_bucket_lifecycle for why the archive prefix matters.

set -euo pipefail

if [ "$#" -lt 1 ] || ! [[ "$1" =~ ^CROSS_PURPOSE_(AB|DIABETES)$ ]]; then
  echo "usage: $0 <CROSS_PURPOSE_AB|CROSS_PURPOSE_DIABETES>" >&2
  exit 2
fi
INSTANCE_TAG="$1"

case "${INSTANCE_TAG}" in
  CROSS_PURPOSE_AB)        DATASETS_DEFAULT="adult hmda";      S3_LEAF="cross_purpose_ab" ;;
  CROSS_PURPOSE_DIABETES)  DATASETS_DEFAULT="diabetes";        S3_LEAF="cross_purpose_diabetes" ;;
esac

# ── KNOBS ────────────────────────────────────────────────────────────────
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
AMI_ID="${AMI_ID:-ami-012ba162b9cd2729c}"
KEY_NAME="${KEY_NAME:-pcrl-gpu-key}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-0f93a621e295c9ab9}"
SUBNET_ID="${SUBNET_ID:-}"
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-pcrl-bios-s3-writer}"
REGION="${REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-pcrl-bios-overnight-20260504}"
S3_PREFIX="s3://${S3_BUCKET}/${S3_LEAF}"
ARCHIVE_DEST="${ARCHIVE_DEST:-s3://${S3_BUCKET}/archive/${S3_LEAF}}"
GIT_REF="${GIT_REF:-cross-purpose-rebuttal-2026-05-18}"
TAG_NAME="${TAG_NAME:-${INSTANCE_TAG}}"
DATASETS="${DATASETS:-${DATASETS_DEFAULT}}"

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
  echo "  WARNING: local HEAD != origin/${GIT_REF}. The instance clones from origin/${GIT_REF}."
fi

# ── Pre-flight 3: no existing same-tag instance ─────────────────────────
EXISTING="$(aws ec2 describe-instances \
  --region "${REGION}" \
  --filters "Name=tag:Name,Values=${TAG_NAME}" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
if [ -n "${EXISTING}" ]; then
  echo "ERROR: instance(s) already tagged ${TAG_NAME} are running: ${EXISTING}" >&2
  exit 2
fi

# ── Pre-flight 4: S3 bucket reachable ───────────────────────────────────
echo "[pre-flight 4] s3 bucket ${S3_BUCKET} reachable"
aws s3 ls "s3://${S3_BUCKET}/" --region "${REGION}" >/dev/null
echo "  bucket reachable"

# ── Pre-flight 5: running instances (visibility only) ───────────────────
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
  -e "s|__ARCHIVE_DEST__|${ARCHIVE_DEST}|g" \
  -e "s|__GIT_REF__|${GIT_REF}|g" \
  -e "s|__INSTANCE_TAG__|${INSTANCE_TAG}|g" \
  -e "s|__DATASETS__|${DATASETS}|g" \
  "${REPO_ROOT}/infra/cross_purpose/user_data.sh" > "${USER_DATA_RENDERED}"
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
echo "launched: ${INSTANCE_ID}  (tag=${TAG_NAME}, datasets=${DATASETS})"
echo
echo "Live monitoring:"
echo "  aws s3 ls ${S3_PREFIX}/ --region ${REGION} | grep STAGE_"
echo "  aws s3 cp ${S3_PREFIX}/live_logs/cross_purpose.log - | tail -100"
echo
echo "Durable headline (survives lifecycle expiry):"
echo "  aws s3 cp ${ARCHIVE_DEST}/STATUS.txt -"
echo
echo "Pull final to local:"
echo "  aws s3 sync ${ARCHIVE_DEST}/results_final/ results/"
