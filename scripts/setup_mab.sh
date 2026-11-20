#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# The GitHub code-only publication intentionally excludes companion configs/locks/patches.
for b_required_file in requirements-runtime-20261120.lock patches/mpctools-upstream-commit.txt patches/mpctools-python312.patch; do
  if [ ! -f "$b_required_file" ]; then
    echo "Missing companion file: $b_required_file. Obtain the private B delivery before setup." >&2
    exit 1
  fi
done
python3.12 -m venv .venv-mab
.venv-mab/bin/python -m pip install -r requirements-runtime-20261120.lock timeout-decorator==0.5.0 setuptools==84.2.1
mkdir -p work
if [ -e work/mpc-tools-casadi ]; then
  echo 'work/mpc-tools-casadi already exists; use a clean checkout to preserve evidence.' >&2
  exit 1
fi
git clone https://github.com/smpl-env/mpc-tools-casadi.git work/mpc-tools-casadi
git -C work/mpc-tools-casadi checkout "$(cat patches/mpctools-upstream-commit.txt)"
git -C work/mpc-tools-casadi apply ../../patches/mpctools-python312.patch
.venv-mab/bin/python -m pip install --no-build-isolation ./work/mpc-tools-casadi
.venv-mab/bin/python -m pip check
