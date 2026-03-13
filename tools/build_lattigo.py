"""Build Go CGO shared libraries for lattigo and orion-evaluator bridges."""

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def _platform_suffix() -> str:
    system = platform.system()
    if system == "Windows":
        return "windows.dll"
    elif system == "Darwin":
        machine = platform.machine().lower()
        if machine in ("arm64", "aarch64"):
            return "mac-arm64.dylib"
        return "mac.dylib"
    elif system == "Linux":
        return "linux.so"
    raise RuntimeError(f"Unsupported platform: {system}")


def _go_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CGO_ENABLED"] = "1"
    if platform.system() == "Darwin":
        machine = platform.machine().lower()
        env["GOARCH"] = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    return env


def _build_bridge(name: str, bridge_dir: Path, output_path: Path) -> None:
    cmd = [
        "go", "build",
        "-buildmode=c-shared",
        "-buildvcs=false",
        "-o", str(output_path),
        str(bridge_dir),
    ]
    print(f"=== Building {name} ===")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(bridge_dir), env=_go_env(), check=True)
    print(f"Built {output_path.name}")


def build() -> None:
    suffix = _platform_suffix()

    bridges = [
        (
            "lattigo bridge",
            ROOT_DIR / "python" / "lattigo" / "bridge",
            ROOT_DIR / "python" / "lattigo" / "lattigo" / f"orionclient-{suffix}",
        ),
        (
            "orion-evaluator bridge",
            ROOT_DIR / "python" / "orion-evaluator" / "bridge",
            ROOT_DIR / "python" / "orion-evaluator" / "orion_evaluator" / f"orion-evaluator-{suffix}",
        ),
    ]

    for name, bridge_dir, output_path in bridges:
        try:
            _build_bridge(name, bridge_dir, output_path)
        except subprocess.CalledProcessError as e:
            print(f"{name} build failed with exit code {e.returncode}")
            sys.exit(1)

    print("\n=== All bridges built successfully ===")


if __name__ == "__main__":
    build()
