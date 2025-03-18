from setuptools import setup
from setuptools.command.build_py import build_py
import sys
import os
import platform
import subprocess
from pathlib import Path

class BuildLattigo(build_py):
    """Custom build command to build the Go shared library for Lattigo."""
    def run(self):
        build_py.run(self)
        
        # Build our Go shared library
        print("=== Building Go shared library ===")
        
        # Determine the output filename based on platform
        if platform.system() == "Windows":
            output_file = "lattigo-windows.dll"
        elif platform.system() == "Darwin":  # macOS
            if platform.machine().lower() in ("arm64", "aarch64"):
                output_file = "lattigo-mac-arm64.dylib"
            else:
                output_file = "lattigo-mac.dylib"
        elif platform.system() == "Linux":
            output_file = "lattigo-linux.so"
        else:
            raise RuntimeError("Unsupported platform")
        
        # Set up paths
        root_dir = Path(__file__).parent
        backend_dir = root_dir / "orion" / "backend" / "lattigo"
        output_path = backend_dir / output_file
        
        # Set up CGO for Go build
        env = os.environ.copy()
        env["CGO_ENABLED"] = "1"
        
        # Set architecture for macOS
        if platform.system() == "Darwin":
            if platform.machine().lower() in ("arm64", "aarch64"):
                env["GOARCH"] = "arm64"
            else:
                env["GOARCH"] = "amd64"
        
        # Build command
        build_cmd = [
            "go", "build", 
            "-buildmode=c-shared",
            "-o", str(output_path),
            str(backend_dir)
        ]
        
        # Run the build command with the configured environment
        try:
            print(f"Running: {' '.join(build_cmd)}")
            subprocess.run(build_cmd, cwd=str(backend_dir), env=env, check=True)
            print(f"Successfully built {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Go build failed with exit code {e.returncode}")
            sys.exit(1)

setup(
    cmdclass={'build_py': BuildLattigo},
)