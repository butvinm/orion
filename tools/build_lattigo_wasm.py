"""Build the Go WASM binary for the js/lattigo bridge.

Mirrors tools/build_lattigo.py but targets GOOS=js GOARCH=wasm instead of CGO.
Outputs:
  js/lattigo/wasm/lattigo.wasm  -- the WASM binary
  js/lattigo/wasm/wasm_exec.js  -- Go runtime glue (copied from GOROOT)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

MAX_WASM_SIZE = 10 * 1024 * 1024  # 10 MB


def build(setup_kwargs=None):
    """Build the Go WASM binary for the Lattigo JS bridge."""
    print("=== Building Go WASM binary ===")

    root_dir = Path(__file__).parent.parent
    bridge_dir = root_dir / "js" / "lattigo" / "bridge"
    wasm_dir = root_dir / "js" / "lattigo" / "wasm"
    output_path = wasm_dir / "lattigo.wasm"

    wasm_dir.mkdir(parents=True, exist_ok=True)

    # Resolve GOROOT for wasm_exec.js
    try:
        goroot = subprocess.check_output(
            ["go", "env", "GOROOT"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: 'go' not found in PATH")
        sys.exit(1)

    # wasm_exec.js is in lib/wasm/ since Go 1.21, misc/wasm/ before that
    wasm_exec_candidates = [
        Path(goroot) / "lib" / "wasm" / "wasm_exec.js",
        Path(goroot) / "misc" / "wasm" / "wasm_exec.js",
    ]
    wasm_exec_src = next(
        (p for p in wasm_exec_candidates if p.exists()), None
    )
    if wasm_exec_src is None:
        print(f"ERROR: wasm_exec.js not found in GOROOT ({goroot})")
        sys.exit(1)

    # Build the WASM binary
    env = os.environ.copy()
    env["GOOS"] = "js"
    env["GOARCH"] = "wasm"

    build_cmd = [
        "go", "build",
        "-buildvcs=false",
        "-o", str(output_path),
        ".",
    ]

    print(f"Running: {' '.join(build_cmd)}")
    try:
        subprocess.run(
            build_cmd, cwd=str(bridge_dir), env=env, check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Go WASM build failed with exit code {e.returncode}")
        sys.exit(1)

    # Copy wasm_exec.js
    dest_wasm_exec = wasm_dir / "wasm_exec.js"
    shutil.copy2(wasm_exec_src, dest_wasm_exec)
    print(f"Copied {wasm_exec_src} → {dest_wasm_exec}")

    # Report binary size
    size = output_path.stat().st_size
    size_mb = size / 1024 / 1024
    print(f"Built {output_path} ({size_mb:.2f} MB)")

    if size > MAX_WASM_SIZE:
        print(
            f"ERROR: WASM binary exceeds 10 MB target ({size_mb:.2f} MB)"
        )
        sys.exit(1)

    print("=== WASM build complete ===")
    return setup_kwargs or {}


if __name__ == "__main__":
    build()
