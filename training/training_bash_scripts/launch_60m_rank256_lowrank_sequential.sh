#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash training/training_bash_scripts/launch_60m_rank256_lowrank_sequential.sh [GPU_ID]
# Example:
#   bash training/training_bash_scripts/launch_60m_rank256_lowrank_sequential.sh 0

GPU_ID="4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCRIPTS=(
  # "training/training_bash_scripts/2_galore/train_60m_rank256.sh"
  # "training/training_bash_scripts/3_fira/train_60m_rank256.sh"
  "training/training_bash_scripts/4_cola/train_60m_rank256.sh"
  "training/training_bash_scripts/5_sltrain/train_60m_rank256.sh"
  "training/training_bash_scripts/6_relora/train_60m_rank256.sh"
)

run_one() {
  local script_path="$1"
  local abs_script="${ROOT_DIR}/${script_path}"

  if [[ ! -f "${abs_script}" ]]; then
    echo "[ERROR] Missing script: ${abs_script}" >&2
    exit 1
  fi

  echo "=============================================================="
  echo "Starting: ${script_path}"
  echo "GPU: ${GPU_ID}"
  echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

  # Create a temporary copy that uses the selected GPU id, then execute it.
  local tmp_script
  tmp_script="$(mktemp)"
  sed -E "s/^CUDA_VISIBLE_DEVICES=[^ ]+/CUDA_VISIBLE_DEVICES=${GPU_ID}/" "${abs_script}" > "${tmp_script}"
  chmod +x "${tmp_script}"

  bash "${tmp_script}"
  rm -f "${tmp_script}"

  echo "Finished: ${script_path}"
  echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
}

for script in "${SCRIPTS[@]}"; do
  run_one "${script}"
  echo
  echo "Completed one job. Launching next..."
  echo
done

echo "All 60M rank-256 low-rank trainings completed successfully."
