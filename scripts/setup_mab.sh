#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Restore previously authorized public companion files if absent.
if [ ! -f requirements-runtime-20261120.lock ]; then
  python3.12 scripts/restore_b_companions.py
fi
if [ -e .venv-mab ] || [ -e work/mpc-tools-casadi ]; then
  echo "Use a clean checkout; preserve existing environment and MPCTools source." >&2
  exit 1
fi
for b_required_file in requirements-runtime-20261120.lock patches/mpctools-upstream-commit.txt patches/mpctools-python312.patch; do
  if [ ! -f "$b_required_file" ]; then
    echo "Missing companion file: $b_required_file. Run scripts/restore_b_companions.py before setup." >&2
    exit 1
  fi
done
python3.12 -m venv .venv-mab
.venv-mab/bin/python -m pip install -r requirements-runtime-20261120.lock timeout-decorator==0.5.0 setuptools==84.2.1 pyzmq==27.3.0
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
