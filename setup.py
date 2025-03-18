from setuptools import setup
from setuptools.command.build_py import build_py
import platform
import subprocess
from pathlib import Path
import os

class BuildLattigo(build_py):
    """Custom build command to build the Go shared library for Lattigo."""
    def run(self):
        build_py.run(self)
        
        print("=== Building Go shared library ===")
        
        # Determine target platform
        go_os = platform.system().lower()
        
        # Architecture -> Go environment variables
        arch_map = {
            "x86_64": "amd64", 
            "amd64": "amd64", 
            "arm64": "arm64", 
            "aarch64": "arm64", 
            "i386": "386", 
            "i686": "386"
        }
        
        curr_arch = platform.machine().lower()
        go_arch = arch_map.get(curr_arch, None)
        
        # Determine output filename
        extensions = {
            "windows": "dll", 
            "darwin": "dylib", 
            "linux": "so"
        }

        if go_os not in extensions or go_arch is None:
            raise RuntimeError(f"Unsupported platform: {go_os} {go_arch}")
            
        platform_name = "mac" if go_os == "darwin" else go_os
        output_file = f"lattigo-{platform_name}-{go_arch}.{extensions[go_os]}"
        
        # Set up paths
        root_dir = Path(__file__).parent
        backend_dir = root_dir / "orion" / "backend" / "lattigo"
        output_path = backend_dir / output_file
        
        # Set up environment and build command
        env = os.environ.copy()
        env.update({"CGO_ENABLED": "1", "GOOS": go_os, "GOARCH": go_arch})
                
        # Run the build command
        build_cmd = [
            "go", "build", 
            "-buildmode=c-shared",
            "-o", str(output_path),
            str(backend_dir)
        ]

        try:
            print(f"Running: {' '.join(build_cmd)}")
            print(f"Environment: GOOS={env['GOOS']} GOARCH={env['GOARCH']}")
            subprocess.run(build_cmd, cwd=str(backend_dir), env=env, check=True)
            print(f"Successfully built {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Go build failed with exit code {e.returncode}")
            raise

setup(
    cmdclass={'build_py': BuildLattigo},
    options={'bdist_wheel': {'universal': False}},
)