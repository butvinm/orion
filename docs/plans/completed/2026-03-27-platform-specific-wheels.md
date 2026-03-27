# Platform-Specific Wheel Support

## Overview

Orion's Python packages (`orion-v2-lattigo`, `orion-v2-evaluator`) ship Go CGO shared libraries loaded via `ctypes.CDLL`. Currently wheels are tagged `py3-none-any` but contain Linux-only `.so` files, which means they install on any platform but crash at import on macOS.

This plan adds proper platform-specific wheels for:

- Linux x86_64 (`manylinux_2_17_x86_64`)
- macOS x86_64 (`macosx_11_0_x86_64`)
- macOS ARM64 (`macosx_11_0_arm64`)

No Windows support for now (CGO on Windows needs MinGW, lower priority).

## Context

**Files involved:**

- `tools/build_lattigo.py` — Go CGO build script (Linux-only, lines 12-18)
- `python/lattigo/lattigo/gohandle.py` — FFI loader (Linux-only, lines 20-34)
- `python/orion-evaluator/orion_evaluator/ffi.py` — FFI loader (Linux-only, lines 25-39)
- `python/lattigo/pyproject.toml` — hatchling config, artifacts glob `*.so` only
- `python/orion-evaluator/pyproject.toml` — same
- `.github/workflows/release-lattigo-python.yml` — single ubuntu-latest job
- `.github/workflows/release.yml` — single ubuntu-latest job

**Key findings:**

- The `.so` files only link `libc` and `libresolv` (verified via `ldd`). No libgmp/libssl dynamic deps. Cross-platform should be clean.
- Go statically links its own runtime into the `.so`/`.dylib`.
- Hatchling needs a `hatch_build.py` hook with `build_data["tag"]` set from env to produce platform-tagged wheels.
- `orion-v2-compiler` is pure Python — no changes needed.
- Build uses native GitHub Actions runners per platform (no cross-compilation).

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- Run tests after each change

## Implementation Steps

### Task 1: Multi-platform `build_lattigo.py`

Update the build script to detect platform and architecture, set correct Go env vars, and produce correctly-named outputs.

- [ ] Replace `_platform_suffix()` with platform/arch detection: `(system, machine)` → `(GOOS, GOARCH, file_suffix)`
  - Linux x86_64 → `linux-amd64.so`
  - Darwin x86_64 → `darwin-amd64.dylib`
  - Darwin arm64 → `darwin-arm64.dylib`
- [ ] Update `_go_env()` to set `GOOS` and `GOARCH` explicitly (not just `CGO_ENABLED`)
- [ ] Update bridge output paths in `build()` to use new suffixes
- [ ] Test locally: `python tools/build_lattigo.py` still produces working `.so` on Linux
- [ ] Verify on macOS if available, or defer to CI

### Task 2: Platform-detecting FFI loaders

Update both loaders to select the correct library name based on runtime platform.

- [ ] Create shared platform detection logic: `(system, machine)` → library filename
  - `("Linux", "x86_64")` → `orionclient-linux-amd64.so`
  - `("Darwin", "x86_64")` → `orionclient-darwin-amd64.dylib`
  - `("Darwin", "arm64")` → `orionclient-darwin-arm64.dylib`
  - Raise descriptive error for unsupported platforms
- [ ] Update `_load_library()` in `python/lattigo/lattigo/gohandle.py`
- [ ] Update `_load_library()` in `python/orion-evaluator/orion_evaluator/ffi.py`
- [ ] Update tests: verify `_load_library()` finds correct file on current platform
- [ ] Run `pytest python/tests/` — must pass

### Task 3: Hatchling build hooks for platform tagging

Add `hatch_build.py` to both native packages so wheels get correct platform tags.

- [ ] Create `python/lattigo/hatch_build.py`:
  - Read `ORION_WHEEL_PLAT` env var (e.g., `manylinux_2_17_x86_64`, `macosx_11_0_arm64`)
  - If set: `build_data["tag"] = f"py3-none-{plat}"`
  - If not set: `build_data["pure_python"] = False; build_data["infer_tag"] = True` (local dev builds)
- [ ] Create `python/orion-evaluator/hatch_build.py` (identical logic)
- [ ] Register hooks in both `pyproject.toml`: `[tool.hatch.build.targets.wheel.hooks.custom]`
- [ ] Update `artifacts` globs to include `*.dylib`: `["lattigo/*.so", "lattigo/*.dylib", "lattigo/*.h"]`
- [ ] Test locally: `python -m build python/lattigo/` produces platform-tagged wheel
- [ ] Test with env var: `ORION_WHEEL_PLAT=manylinux_2_17_x86_64 python -m build python/lattigo/` produces correctly tagged wheel

### Task 4: Update release workflow for lattigo (`release-lattigo-python.yml`)

Replace single ubuntu job with matrix across 3 platforms.

- [ ] Add matrix strategy:
  - `ubuntu-latest` / `linux-amd64.so` / `ORION_WHEEL_PLAT=manylinux_2_17_x86_64`
  - `macos-13` / `darwin-amd64.dylib` / `ORION_WHEEL_PLAT=macosx_11_0_x86_64`
  - `macos-14` / `darwin-arm64.dylib` / `ORION_WHEEL_PLAT=macosx_11_0_arm64`
- [ ] Add platform-specific system dependency installation (Linux: `apt-get`, macOS: nothing needed since no libgmp/libssl linked)
- [ ] Set `ORION_WHEEL_PLAT` env var per matrix entry
- [ ] Update artifact upload to use matrix-specific names
- [ ] Update publish job to download all platform artifacts and publish together
- [ ] Keep GitHub release job working with all wheels

### Task 5: Update release workflow for evaluator/compiler (`release.yml`)

Same matrix treatment for evaluator. Compiler stays pure Python.

- [ ] Add matrix for evaluator build (same 3 platforms as Task 4)
- [ ] Keep compiler build as single ubuntu job (pure Python, `py3-none-any` is correct)
- [ ] Split into separate jobs: `build-compiler` (single) and `build-evaluator` (matrix)
- [ ] Update publish job to collect all artifacts
- [ ] Keep GitHub release job working

### Task 6: Rename existing `.so` files for consistency

Current naming (`orionclient-linux.so`) doesn't include architecture. Rename to include arch for consistency with new naming.

- [ ] Rename `orionclient-linux.so` → `orionclient-linux-amd64.so`
- [ ] Rename `orion-evaluator-linux.so` → `orion-evaluator-linux-amd64.so`
- [ ] Update `.gitignore` if it references old names
- [ ] Run full test suite to verify nothing breaks

### Task 7: Verify acceptance criteria

- [ ] `python tools/build_lattigo.py` works on Linux, produces `*-linux-amd64.so`
- [ ] FFI loaders detect platform correctly and load right library
- [ ] `python -m build python/lattigo/` produces platform-tagged wheel (not `none-any`)
- [ ] `ORION_WHEEL_PLAT=macosx_11_0_arm64 python -m build python/lattigo/` produces macOS wheel tag
- [ ] Run full test suite: `pytest python/tests/`
- [ ] Run linter: `ruff check python/`
- [ ] Run type checker: `mypy python/lattigo/ python/orion-compiler/ python/orion-evaluator/`

### Task 8: Update documentation

- [ ] Update README.md installation section: mention macOS support
- [ ] Update CLAUDE.md if build commands changed

## Technical Details

### Library naming convention

```
orionclient-{os}-{arch}.{ext}
orion-evaluator-{os}-{arch}.{ext}
```

| Platform     | os     | arch  | ext    |
| ------------ | ------ | ----- | ------ |
| Linux x86_64 | linux  | amd64 | .so    |
| macOS x86_64 | darwin | amd64 | .dylib |
| macOS ARM64  | darwin | arm64 | .dylib |

### Wheel platform tags

| Platform     | Tag                     |
| ------------ | ----------------------- |
| Linux x86_64 | `manylinux_2_17_x86_64` |
| macOS x86_64 | `macosx_11_0_x86_64`    |
| macOS ARM64  | `macosx_11_0_arm64`     |

### CI matrix

```yaml
strategy:
  matrix:
    include:
      - os: ubuntu-latest
        wheel_plat: manylinux_2_17_x86_64
      - os: macos-13
        wheel_plat: macosx_11_0_x86_64
      - os: macos-14
        wheel_plat: macosx_11_0_arm64
```

### hatch_build.py hook pattern

```python
import os
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        plat = os.environ.get("ORION_WHEEL_PLAT")
        if plat:
            build_data["tag"] = f"py3-none-{plat}"
        else:
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
```

## Post-Completion

**Manual verification:**

- Test wheel installation on macOS x86_64 and ARM64 machines
- Verify `import lattigo` works after `pip install` from the platform-specific wheel
- Trigger a test release to TestPyPI to verify multi-platform publishing

**Future work:**

- Windows support (requires MinGW CGO toolchain in CI)
- Linux ARM64 / aarch64 support
- `manylinux` compliance verification (current .so only links libc/libresolv, should be fine)
