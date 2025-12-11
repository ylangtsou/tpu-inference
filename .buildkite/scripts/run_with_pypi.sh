#!/bin/bash
# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

# Build vllm-tpu with nightly tpu-inference from PyPI (using docker/Dockerfile.pypi instead of docker/Dockerfile).
export RUN_WITH_PYPI="true"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/run_in_docker.sh"
