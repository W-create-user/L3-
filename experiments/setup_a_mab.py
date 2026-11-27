"""Install the pinned public MPCTools source and Python compatibility patch."""

import hashlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
MPC_REPO = "https://github.com/smpl-env/mpc-tools-casadi.git"
MPC_COMMIT = "ac683dc8199d3e11bf4f3971a3b8f613c83a162b"
PATCH_COMMIT = "dd76ceafb1f1b331660d3a171cbcd087fd0b7559"
PATCH_OBJECT = f"{PATCH_COMMIT}:patches/mpctools-python312.patch"
PATCH_SHA256 = "7d626760a4990030817a34ba0e610d8af061d7ac2a339454c38557973df3ab95"


def main():
    with TemporaryDirectory(prefix="l3-mpctools-") as directory:
        temporary = Path(directory)
        patch_source = subprocess.run(
            ["git", "show", PATCH_OBJECT], cwd=ROOT, capture_output=True, check=False
        )
        if patch_source.returncode:
            # Looking up the historical artifact in a temporary repo leaves the user's Git refs intact.
            patch_repo = temporary / "patch-source"
            subprocess.run(["git", "init", "-q", str(patch_repo)], check=True)
            subprocess.run(
                ["git", "fetch", "-q", "--depth=1", "https://github.com/W-create-user/L3-.git", PATCH_COMMIT],
                cwd=patch_repo, check=True,
            )
            patch = subprocess.check_output(["git", "show", PATCH_OBJECT], cwd=patch_repo)
        else:
            patch = patch_source.stdout
        if hashlib.sha256(patch).hexdigest() != PATCH_SHA256:
            raise RuntimeError("The MPCTools compatibility patch failed its SHA-256 check")
        patch_file = temporary / "python312.patch"
        patch_file.write_bytes(patch)

        source = temporary / "mpc-tools"
        subprocess.run(["git", "clone", "-q", MPC_REPO, str(source)], check=True)
        subprocess.run(["git", "checkout", "-q", "--detach", MPC_COMMIT], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch_file)], cwd=source, check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", str(source)], check=True
        )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
