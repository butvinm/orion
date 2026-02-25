# Post-Refactor Findings Cleanup (F4, F5, F8, F11, F12)

## Overview
Address the remaining five findings from the GoHandle refactor retrospective. F1–F3 are already implemented and committed. Each task below corresponds to one finding — small, isolated, low-risk changes.

## Context
- Branch: `go-client-evaluator-refactor-v2`
- Source document: `FINDINGS.md`
- F1 commit: `e9bf28f` — FFI wrappers for keygen blob functions
- F2 commit: `f53a851` — error-path handle cleanup tests
- F3 commit: `f325d46` — BatchNorm and Bootstrap handle tracking tests

## Development Approach
- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**
- Run tests after each change
- Maintain backward compatibility

## Progress Tracking
- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix

## Implementation Steps

### Task 1: F4 — Add `__repr__` to GoHandle
- [x] Add `tag` parameter to `GoHandle.__init__` in `orion/backend/orionclient/ffi.py` (default `""`)
- [x] Add `__repr__` method: `GoHandle(<raw> <tag>)` or `GoHandle(closed <tag>)`
- [x] Pass tag strings from FFI wrapper functions that create GoHandles (e.g. `tag="Client"`, `tag="Evaluator"`, `tag="Ciphertext"`, etc.)
- [x] Write tests for `__repr__` output (open handle, closed handle, with tag, without tag)
- [x] Run tests — must pass before next task

### Task 2: F5 — Standardize `__del__` to one pattern; add `close()` to Ciphertext/PlainText
- [x] Add `close()` method to `Ciphertext` in `orion/ciphertext.py` (delegates to `self._handle.close()`, idempotent)
- [x] Add `close()` method to `PlainText` in `orion/ciphertext.py` (same pattern)
- [x] Replace all 8 `__del__` methods with the canonical pattern: `try: self.close() except Exception: pass`
  - `GoHandle` (`ffi.py`) — already close to canonical, remove `sys.modules` guard
  - `Ciphertext` (`ciphertext.py`) — replace `"sys" in globals()` guard
  - `PlainText` (`ciphertext.py`) — replace `"sys" in globals()` guard
  - `Client` (`client.py`) — replace `hasattr` guard, fix bare `except:` → `except Exception:`
  - `_MultiPlainText` (`client.py`) — replace local `_sys` import guard
  - `Evaluator` (`evaluator.py`) — replace `hasattr` guard
  - `Compiler` (`compiler.py`) — replace nested guard
  - `PlainTensor` (`compiler_backend.py`) — replace `"sys" in globals()` guard
- [x] Write tests for `Ciphertext.close()` and `PlainText.close()` (idempotent, double-close safe)
- [x] Run tests — must pass before next task

### Task 3: F8 — Add `errOut` to `ClientModuliChain` and `ClientAuxModuliChain`
- [x] Add `errOut **C.char` parameter and `defer catchPanic(errOut)` to `ClientModuliChain` in `orionclient/bridge/client.go`
- [x] Add `errOut **C.char` parameter and `defer catchPanic(errOut)` to `ClientAuxModuliChain` in `orionclient/bridge/client.go`
- [x] Update Python FFI prototypes in `ffi.py` — add `ctypes.POINTER(ctypes.c_char_p)` to argtypes
- [x] Update Python wrapper functions `client_moduli_chain()` and `client_aux_moduli_chain()` in `ffi.py` to pass errOut and call `_check_err()`
- [x] Rebuild shared library (`pip install -e .`)
- [x] Write tests verifying the functions still work correctly after signature change
- [x] Run tests — must pass before next task

### Task 4: F11 — Replace duplicated `compile()` with `super().compile(context)`
- [x] In `Linear.compile()` (`orion/nn/linear.py:103-107`), replace duplicated lines with `super().compile(context)`
- [x] In `Conv2d.compile()` (`orion/nn/linear.py:192-196`), replace duplicated lines with `super().compile(context)`
- [x] Run existing compile/model tests to verify behavior unchanged
- [x] Run tests — must pass before next task

### Task 5: F12 — Remove dead legacy fallback in `evaluate_transforms`
- [x] Remove the `hasattr(self, 'transform_handles')` guard in `LinearTransform.evaluate_transforms()` (`orion/nn/linear.py:52-55`)
- [x] Keep only the `ctx.evaluate_transforms(self, x)` path
- [x] Run existing model/evaluator tests to verify no regression
- [x] Run tests — must pass before next task

### Task 6: Verify acceptance criteria
- [ ] Verify all 5 findings addressed (F4, F5, F8, F11, F12)
- [ ] Run full test suite (`pytest tests/`)
- [ ] Verify no private `ffi._*` access remains in `client.py` (F1 follow-up sanity check)
- [ ] Verify test coverage of new code

### Task 7: [Final] Update documentation
- [ ] Update FINDINGS.md — mark F4, F5, F8, F11, F12 as resolved with commit refs

## Technical Details

**F4 tag propagation:** Every function in `ffi.py` that returns `GoHandle(r)` should become `GoHandle(r, tag="<Type>")`. Roughly 15 call sites. The tag is for debugging only — no behavioral change.

**F5 canonical `__del__`:**
```python
def __del__(self):
    try:
        self.close()
    except Exception:
        pass
```
Works because `close()` is idempotent everywhere, and if ctypes is torn down during interpreter shutdown, the exception is caught.

**F8 Go signature change:**
```go
// Before
func ClientModuliChain(clientH C.uintptr_t, outLen *C.int) *C.ulonglong
// After
func ClientModuliChain(clientH C.uintptr_t, outLen *C.int, errOut **C.char) *C.ulonglong
```

**F11/F12:** Pure refactoring — no behavioral change expected. Existing tests serve as regression tests.

## Post-Completion

**Manual verification:**
- Spot-check `__repr__` output in a debugger session with live GoHandles
- Verify `__del__` doesn't produce warnings during interpreter shutdown
