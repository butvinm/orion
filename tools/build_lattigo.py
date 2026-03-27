"""Build Go CGO shared libraries for lattigo and orion-evaluator bridges."""

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# (system, machine) → (GOOS, GOARCH, file_suffix)
_PLATFORM_MAP: dict[tuple[str, str], tuple[str, str, str]] = {
    ("Linux", "x86_64"): ("linux", "amd64", "linux-amd64.so"),
    ("Darwin", "x86_64"): ("darwin", "amd64", "darwin-amd64.dylib"),
    ("Darwin", "arm64"): ("darwin", "arm64", "darwin-arm64.dylib"),
}


def _detect_platform() -> tuple[str, str, str]:
    """Detect platform and return (GOOS, GOARCH, file_suffix)."""
    key = (platform.system(), platform.machine())
    result = _PLATFORM_MAP.get(key)
    if result is None:
        supported = ", ".join(f"{s}/{m}" for s, m in _PLATFORM_MAP)
        raise RuntimeError(f"Unsupported platform: {key[0]}/{key[1]}. Supported: {supported}")
    return result


def _go_env(goos: str, goarch: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CGO_ENABLED"] = "1"
    env["GOOS"] = goos
    env["GOARCH"] = goarch
    return env


def _build_bridge(name: str, bridge_dir: Path, output_path: Path, env: dict[str, str]) -> None:
    cmd = [
        "go",
        "build",
        "-buildmode=c-shared",
        "-buildvcs=false",
        "-o",
        str(output_path),
        str(bridge_dir),
    ]
    print(f"=== Building {name} ===")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(bridge_dir), env=env, check=True)
    print(f"Built {output_path.name}")


def build() -> None:
    goos, goarch, suffix = _detect_platform()
    env = _go_env(goos, goarch)

    bridges = [
        (
            "lattigo bridge",
            ROOT_DIR / "python" / "lattigo" / "bridge",
            ROOT_DIR / "python" / "lattigo" / "lattigo" / f"orionclient-{suffix}",
        ),
        (
            "orion-evaluator bridge",
            ROOT_DIR / "python" / "orion-evaluator" / "bridge",
            ROOT_DIR
            / "python"
            / "orion-evaluator"
            / "orion_evaluator"
            / f"orion-evaluator-{suffix}",
        ),
    ]

    for name, bridge_dir, output_path in bridges:
        try:
            _build_bridge(name, bridge_dir, output_path, env)
        except subprocess.CalledProcessError as e:
            print(f"{name} build failed with exit code {e.returncode}")
            sys.exit(1)

    print("\n=== All bridges built successfully ===")


if __name__ == "__main__":
    build()
