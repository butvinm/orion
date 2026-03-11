"""Build script for the orion-evaluator Python package.

Builds the Go bridge shared library into the orion_evaluator/ package directory.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def build(setup_kwargs=None):
    """Build the Go shared library for the evaluator bridge."""
    print("=== Building orion-evaluator bridge shared library ===")

    if platform.system() == "Windows":
        output_file = "orion-evaluator-windows.dll"
    elif platform.system() == "Darwin":
        if platform.machine().lower() in ("arm64", "aarch64"):
            output_file = "orion-evaluator-mac-arm64.dylib"
        else:
            output_file = "orion-evaluator-mac.dylib"
    elif platform.system() == "Linux":
        output_file = "orion-evaluator-linux.so"
    else:
        raise RuntimeError("Unsupported platform")

    root_dir = Path(__file__).parent
    bridge_dir = root_dir / "bridge"
    output_dir = root_dir / "orion_evaluator"
    output_path = output_dir / output_file

    env = os.environ.copy()
    env["CGO_ENABLED"] = "1"

    if platform.system() == "Darwin":
        if platform.machine().lower() in ("arm64", "aarch64"):
            env["GOARCH"] = "arm64"
        else:
            env["GOARCH"] = "amd64"

    build_cmd = [
        "go",
        "build",
        "-buildmode=c-shared",
        "-buildvcs=false",
        "-o",
        str(output_path),
        str(bridge_dir),
    ]

    try:
        print(f"Running: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, cwd=str(bridge_dir), env=env, check=True)
        print(f"Successfully built {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Go build failed with exit code {e.returncode}")
        sys.exit(1)

    return setup_kwargs or {}


if __name__ == "__main__":
    build()
