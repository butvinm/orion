# Python Tooling and API Polish

## Overview

Bring the three Python packages (`lattigo`, `orion-compiler`, `orion-evaluator`) and the JS/TS package (`js/lattigo`) up to library-grade quality:

- Context manager support for all handle-owning Python classes
- Library-specific exception hierarchies per package
- Replace `print()` with `logging` in library code
- Rename JS `free()` → `close()` for cross-language consistency
- Add ruff (linting + formatting) with shared root config
- Add mypy (moderate strictness) with type annotations completed
- Mark all Python packages with `py.typed` (PEP 561)

## Context

- 15 Python classes have `close()` but none implement `__enter__`/`__exit__`
- All errors are generic `RuntimeError`/`ValueError` — no programmatic catching
- ~20 `print()` calls in `orion-compiler` core (progress, diagnostics, graphviz fallbacks)
- JS uses `free()`, Python uses `close()` — JS package not published yet, hard rename safe
- Zero linting/typing infrastructure: no ruff, no mypy, no `py.typed`, no dev deps
- Type annotation coverage: ~85% lattigo/evaluator, ~50-70% compiler core
- Root `pyproject.toml` already has `[tool.pytest]`, will add ruff/mypy there
- All three `pyproject.toml` declare `python = ">=3.9,<3.13"` — will bump to `>=3.11`
- Code already uses `X | None` and `list[int]` syntax (3.10+) without `from __future__ import annotations` in most files — bumping to 3.11 makes this officially correct

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests**
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix

## Implementation Steps

### Task 1: Bump Python minimum to 3.11 and add ruff

- [ ] Update `python = ">=3.9,<3.13"` to `">=3.11,<3.13"` in all three `pyproject.toml` files (`python/lattigo/`, `python/orion-compiler/`, `python/orion-evaluator/`)
- [ ] Remove unnecessary `from __future__ import annotations` imports (in `ckks.py`, `rlwe.py`, `compiler_backend.py`) — no longer needed with 3.11+
- [ ] Add `[tool.ruff]` and `[tool.ruff.lint]` to root `pyproject.toml` — target Python 3.11, enable `E`, `F`, `W`, `I` (isort), `UP` (pyupgrade), `B` (bugbear), `SIM`, `RUF` rule sets
- [ ] Add `[tool.ruff.format]` section (quote-style double, line-length 99)
- [ ] Run `ruff check --fix python/` to auto-fix lint issues
- [ ] Run `ruff format python/` to format all Python files
- [ ] Review and manually fix any remaining lint errors
- [ ] Run `pytest python/tests/` — must pass before next task

### Task 2: Add library-specific exceptions

- [ ] Create `python/lattigo/lattigo/errors.py` with `LatticeError(Exception)`, `HandleClosedError(LatticeError)`, `FFIError(LatticeError)`
- [ ] Create `python/orion-compiler/orion_compiler/errors.py` with `CompilerError(Exception)`, `CompilationError(CompilerError)`, `ValidationError(CompilerError)`
- [ ] Create `python/orion-evaluator/orion_evaluator/errors.py` with `EvaluatorError(Exception)`, `ModelLoadError(EvaluatorError)`
- [ ] Replace `RuntimeError("Use of closed handle")` in `gohandle.py` with `HandleClosedError`
- [ ] Replace `RuntimeError` in lattigo FFI/binding code with `FFIError` or `LatticeError`
- [ ] Replace `RuntimeError` in evaluator model/evaluator code with `ModelLoadError` / `EvaluatorError`
- [ ] Replace `ValueError` in `params.py` validation with `ValidationError`
- [ ] Replace `RuntimeError` in `compiler.py` with `CompilationError`
- [ ] Keep `RuntimeError` in `core/utils.py` for dataset download/extraction failures (lines 58, 137, 186) — these are IO errors, not compilation errors
- [ ] Export new exceptions from each package's `__init__.py`
- [ ] Update `python/tests/test_error_propagation.py` to assert on new exception types (`FFIError` instead of `RuntimeError`)
- [ ] Update `python/tests/test_gohandle.py` line 134 (`pytest.raises(RuntimeError)` → `HandleClosedError`) and line 491 (`RuntimeError` → `FFIError`)
- [ ] Update `python/tests/test_orion_evaluator.py` lines 97, 103, 178 (`pytest.raises(RuntimeError)` → `EvaluatorError` / `ModelLoadError`)
- [ ] Add tests for each new exception type (construct, catch by base class, catch by specific class)
- [ ] Run `pytest python/tests/` — must pass before next task

### Task 3: Add context manager support to Python classes

- [ ] Add `__enter__`/`__exit__` to `GoHandle` in `gohandle.py` (`__exit__` calls `self.close()`)
- [ ] Add `__enter__`/`__exit__` to all lattigo classes in `ckks.py` (Parameters, Encoder)
- [ ] Add `__enter__`/`__exit__` to all lattigo classes in `rlwe.py` (SecretKey, PublicKey, RelinearizationKey, GaloisKey, Ciphertext, Plaintext, KeyGenerator, Encryptor, Decryptor, MemEvaluationKeySet)
- [ ] Add `__enter__`/`__exit__` to `Compiler`, `CompilerBackend`, `PlainTensor` in orion-compiler
- [ ] Add `__enter__`/`__exit__` to `Model` and `Evaluator` in orion-evaluator
- [ ] Write tests: `with Parameters(...) as p:` works, `p` is closed after block
- [ ] Write tests: nested `with` statements work correctly
- [ ] Write tests: `__exit__` is called even on exception
- [ ] Run `pytest python/tests/` — must pass before next task

### Task 4: Replace print() with logging in orion-compiler

- [ ] Add `logger = logging.getLogger(__name__)` to `compiler.py`, use `logger.info()` for progress output
- [ ] Add logger to `core/packing.py`, replace diagnostic `print()` with `logger.debug()`
- [ ] Add logger to `core/auto_bootstrap.py`, `core/level_dag.py`, `core/network_dag.py` — replace graphviz fallback `print()` with `logger.warning()`
- [ ] Add logger to `utils.py` — replace download/training progress `print()` with `logger.info()` (keep `tqdm` if already used)
- [ ] Verify no `print()` remains in library code (exclude tests, examples)
- [ ] Write test: verify logger name matches module name
- [ ] Run `pytest python/tests/` — must pass before next task

### Task 5: Rename JS/TS `free()` to `close()`

- [ ] Rename `free()` → `close()` in `js/lattigo/src/ckks.ts` (CKKSParameters)
- [ ] Rename `free()` → `close()` in `js/lattigo/src/rlwe.ts` (all key/ciphertext/plaintext classes)
- [ ] Rename `free()` → `close()` in `js/lattigo/src/encoder.ts` (Encoder, Encryptor, Decryptor)
- [ ] Update `FinalizationRegistry` callbacks if they reference `free`
- [ ] Update all test files in `js/lattigo/tests/` — replace `.free()` with `.close()`
- [ ] Update `js/lattigo/src/index.ts` exports if needed
- [ ] Update `examples/wasm-demo/client/client.ts` — rename all `.free()` to `.close()` (~16 occurrences)
- [ ] Rebuild wasm-demo client: `cd examples/wasm-demo/client && npm run build:ts`
- [ ] Run `cd js/lattigo && npm run lint` — fix any issues
- [ ] Run `cd js/lattigo && npm run typecheck` — must pass
- [ ] Run `cd js/lattigo && npm test` — must pass before next task

### Task 6: Add mypy configuration and fix type annotations

- [ ] Add `[tool.mypy]` to root `pyproject.toml` — `python_version = "3.11"`, `disallow_untyped_defs = true`, `warn_return_any = true`, `check_untyped_defs = true`, `warn_unused_configs = true`, `warn_redundant_casts = true`, `warn_unused_ignores = true`
- [ ] Add `[[tool.mypy.overrides]]` for third-party libs without stubs (torch, networkx, tqdm, scipy, matplotlib) — `ignore_missing_imports = true`
- [ ] Add `[[tool.mypy.overrides]]` for tests and examples — `disallow_untyped_defs = false` (relaxed strictness for test code)
- [ ] Add missing type annotations to `lattigo/ffi.py` (FFI function signatures)
- [ ] Add missing type annotations to `lattigo/ckks.py` and `lattigo/rlwe.py`
- [ ] Add missing type annotations to `orion_compiler/compiler.py` (internal methods)
- [ ] Add missing type annotations to `orion_compiler/core/compiler_backend.py`
- [ ] Add missing type annotations to `orion_compiler/nn/` modules
- [ ] Add missing type annotations to `orion_evaluator/model.py` and `evaluator.py`
- [ ] Run `mypy python/lattigo/ python/orion-compiler/ python/orion-evaluator/` — fix all errors
- [ ] Run `pytest python/tests/` — must pass before next task

### Task 7: Add py.typed markers

- [ ] Create `python/lattigo/lattigo/py.typed` (empty file)
- [ ] Create `python/orion-compiler/orion_compiler/py.typed` (empty file)
- [ ] Create `python/orion-evaluator/orion_evaluator/py.typed` (empty file)
- [ ] Add `py.typed` to package data in each `pyproject.toml` if needed
- [ ] Verify `mypy` resolves types from installed packages with `py.typed`
- [ ] Run `pytest python/tests/` — must pass before next task

### Task 8: Verify acceptance criteria

- [ ] `ruff check python/` passes with zero warnings
- [ ] `ruff format --check python/` passes (all formatted)
- [ ] `mypy python/lattigo/ python/orion-compiler/ python/orion-evaluator/` passes
- [ ] All Python classes with `close()` also support `with` statement
- [ ] No `print()` in library code (only in tests/examples)
- [ ] All exceptions are library-specific, not generic `RuntimeError`
- [ ] JS `close()` works, no references to `free()` remain
- [ ] All `py.typed` markers present
- [ ] Run full `pytest python/tests/`
- [ ] Run `cd js/lattigo && npm test`
- [ ] Run linter — all issues fixed

### Task 9: [Final] Update documentation

- [ ] Update CLAUDE.md with new exception types, context manager usage, and tool commands (ruff, mypy)
- [ ] Update ARCH.md if relevant sections reference error handling or API patterns

## Technical Details

**Exception hierarchy:**

```
LatticeError (lattigo)
├── HandleClosedError
└── FFIError

CompilerError (orion-compiler)
├── CompilationError
└── ValidationError

EvaluatorError (orion-evaluator)
└── ModelLoadError
```

**Ruff config (root pyproject.toml):**

```toml
[tool.ruff]
target-version = "py311"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]

[tool.ruff.format]
quote-style = "double"
```

**Mypy config (root pyproject.toml):**

```toml
[tool.mypy]
python_version = "3.11"
disallow_untyped_defs = true
check_untyped_defs = true
warn_return_any = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = ["torch.*", "networkx.*", "tqdm.*", "scipy.*", "matplotlib.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["tests.*", "examples.*"]
disallow_untyped_defs = false
```

## Post-Completion

**Manual verification:**

- Confirm `pip install -e .` still works for all three packages after pyproject.toml changes
- Verify CGO bridge build (`python tools/build_lattigo.py`) still succeeds
- Check that mypy type resolution works from a consuming project that imports the packages
