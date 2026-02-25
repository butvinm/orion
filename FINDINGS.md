# Post-Refactor Findings

Findings from the retrospective on the GoHandle refactor (v1 + v2, branches `go-client-and-evaluator-refactor` and `go-client-evaluator-refactor-v2`).

## Action items

Findings we agreed to fix:

| Finding | What                                                                                               | Effort  | Status               |
| ------- | -------------------------------------------------------------------------------------------------- | ------- | -------------------- |
| F1      | Add 3 FFI wrappers for keygen blob functions, remove private access from `client.py`               | Small   | Resolved (`e9bf28f`) |
| F2      | Add error-path tests for encrypt, evaluate_transforms, and Evaluator init                          | Medium  | Resolved (`f53a851`) |
| F3      | Add model tests with BatchNorm and Bootstrap to exercise handle tracking                           | Medium  | Resolved (`f325d46`) |
| F4      | Add `__repr__` to GoHandle                                                                         | Trivial | Resolved (`7b95789`) |
| F5      | Standardize all 8 `__del__` methods to one pattern; add `close()` to Ciphertext/PlainText          | Small   | Resolved (`571233e`) |
| F8      | Add `errOut` parameter to `ClientModuliChain` and `ClientAuxModuliChain` in Go bridge + Python FFI | Small   | Resolved (`18e8f57`) |
| F11     | Replace duplicated `compile()` bodies in Linear/Conv2d with `super().compile(context)`             | Trivial | Resolved (`e5d8216`) |
| F12     | Remove dead legacy fallback in `LinearTransform.evaluate_transforms`                               | Trivial | Resolved (`9b5cf94`) |

## Declined / deferred

| Finding | What                                       | Decision                                                                                                                                                   |
| ------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F6      | `_EvalContext` self-aliasing               | Low priority. Solves a real polymorphism problem. Proper fix comes when FX graph lands in CompiledModel and the compile/inference duality gets redesigned. |
| F7      | Copy-paste FFI wrappers                    | Tolerate. FFI bindings are supposed to be boring and greppable. A factory with `getattr` hurts debuggability.                                              |
| F9      | `cBytesToGoSlice` C.ulong→C.int truncation | Theoretical. No payload approaches 2GB. Not worth fixing now.                                                                                              |
| F10     | CompilerBackend CamelCase methods          | Tolerate. Internal adapter class. CamelCase signals 1:1 Go mapping.                                                                                        |

---

## F1: `client.py:generate_keys()` bypasses FFI encapsulation — RESOLVED

**Resolved in:** `e9bf28f` — Added 3 FFI wrappers for keygen blob functions, removed all private `ffi._*` access from `client.py`.

**Severity:** Code smell (broken abstraction boundary)

**Location:** `orion/client.py:56-101`

**Problem:**

`generate_keys()` is the only place in the codebase that reaches into `ffi.py` private internals: `ffi._get_lib()`, `ffi._make_errout()`, `ffi._check_err()`, `ffi._uintptr()`. Every other consumer (ciphertext.py, evaluator.py, compiler_backend.py) uses public wrapper functions exclusively.

The reason: the three Go bridge functions called here (`ClientGenerateRLK`, `ClientGenerateGaloisKey`, `ClientGenerateBootstrapKeys`) return byte blobs (pointer + length + FreeCArray), not handles. This pattern differs from the other 27 FFI functions that return `GoHandle`. Nobody wrote proper wrappers for them — v1 wrote the raw calls, v2 only patched in `.raw` access.

```python
# Current: 12 private ffi accesses
err = ffi._make_errout()
lib = ffi._get_lib()
ptr = lib.ClientGenerateRLK(
    ffi._uintptr(self._handle.raw),
    ffi.ctypes.byref(out_len),
    ffi.ctypes.byref(err),
)
ffi._check_err(err)
keys.rlk_data = ffi.ctypes.string_at(ptr, out_len.value)
lib.FreeCArray(ptr)
```

**Fix:**

Add three public functions to `ffi.py` that encapsulate the blob-returning pattern:

```python
def client_generate_rlk(h: GoHandle) -> bytes:
    lib = _get_lib()
    err = _make_errout()
    out_len = ctypes.c_ulong(0)
    ptr = lib.ClientGenerateRLK(_uintptr(h.raw), ctypes.byref(out_len), ctypes.byref(err))
    _check_err(err)
    data = ctypes.string_at(ptr, out_len.value)
    lib.FreeCArray(ptr)
    return data

def client_generate_galois_key(h: GoHandle, gal_el: int) -> bytes: ...
def client_generate_bootstrap_keys(h: GoHandle, slot_count: int, logp: list[int]) -> bytes: ...
```

Then `generate_keys()` becomes ~15 lines with zero private access:

```python
if manifest.needs_rlk:
    keys.rlk_data = ffi.client_generate_rlk(self._handle)
for gal_el in sorted(manifest.galois_elements):
    keys.galois_keys[gal_el] = ffi.client_generate_galois_key(self._handle, gal_el)
# ...
```

## F2: Error-path handle cleanup is untested — RESOLVED

**Resolved in:** `f53a851` — Added error-path handle cleanup tests for encrypt, evaluate_transforms, and Evaluator init.

**Severity:** Test gap (code is correct, but no CI coverage)

**Location:** Multiple files

**Problem:**

The GoHandle refactor added try/except cleanup blocks in several places. All of them are written correctly, but none are exercised by any test:

| Error path                                  | Location               | What it cleans up                                         |
| ------------------------------------------- | ---------------------- | --------------------------------------------------------- |
| `client.encrypt()` partial failure          | `client.py:157-164`    | Individual ct_handles if `client_encrypt` fails mid-loop  |
| `evaluate_transforms` mid-operation failure | `evaluator.py:140`     | `res_h` intermediate if `eval_add` raises                 |
| `Evaluator.__init__` partial construction   | `evaluator.py:235-237` | All `_tracked_handles` allocated before the failure point |

The `test_evaluator_partial_init_failure` test covers init failure by corrupting an LT blob, but only tests one failure point. The encrypt and evaluate_transforms error paths are never triggered.

**Risk:** If someone refactors these blocks and breaks the cleanup, no test catches the regression. Handle leaks in error paths are the exact class of bug that motivated this entire refactor.

**Fix:** Add tests that mock FFI failures at each stage and verify handle cleanup. Specifically:

- Force `ffi.client_encrypt` to fail on the 2nd call in a multi-plaintext encrypt, verify all ct_handles are closed
- Force `ffi.eval_add` to fail mid-`evaluate_transforms`, verify intermediate handles are closed
- Force failure at each stage of `Evaluator.__init__` (key bundle, evaluator creation, each reconstruct phase)

## F3: BatchNorm and Bootstrap handle tracking is untested — RESOLVED

**Resolved in:** `f325d46` — Added model tests with BatchNorm and Bootstrap to exercise handle tracking.

**Severity:** Test gap

**Location:** `orion/evaluator.py:335-346` (BatchNorm), `orion/evaluator.py:370-375` (Bootstrap)

**Problem:**

`_reconstruct_batchnorm` tracks PlainText handles for `on_running_mean_ptxt`, `on_running_var_ptxt`, `on_weight_ptxt`, `on_bias_ptxt` via hard-coded attribute names. `_reconstruct_bootstrap_hooks` tracks `prescale_ptxt._handle`.

Neither code path is exercised by any test. The existing model tests use MLP (no BatchNorm) and a Chebyshev model (no Bootstrap). If these tracking calls silently break, handles leak with no CI signal.

Additionally, the hard-coded attribute list in `_reconstruct_batchnorm` is fragile — if a module adds a new PlainText attribute, it won't be tracked unless someone remembers to update the list.

**Fix:**

- Add a model test that includes `BatchNorm1d` or `BatchNorm2d` and verify `_tracked_handles` includes the expected count of BN PlainText handles after construction.
- Add a model test with bootstrap enabled (requires params with enough levels to trigger auto-bootstrap) and verify bootstrap prescale handles are tracked.
- Consider replacing the hard-coded attribute list with explicit registration during `module.compile()`.

## F4: No `__repr__` on GoHandle — RESOLVED

**Resolved in:** `7b95789` — Added `tag` parameter and `__repr__` to GoHandle; all FFI wrapper functions now pass descriptive tags.

**Severity:** Developer experience

**Location:** `orion/backend/orionclient/ffi.py:355-383`

**Problem:**

When a developer hits `RuntimeError("Use of closed handle")`, there's no context about which handle it was, what Go object it wrapped, or where it was created. Every GoHandle looks the same in tracebacks.

**Fix:**

```python
def __repr__(self):
    if self._raw:
        return f"GoHandle({self._raw})"
    return "GoHandle(closed)"
```

Optionally, accept a `tag` parameter for debugging:

```python
def __init__(self, raw: int, tag: str = ""):
    self._raw = raw
    self._tag = tag

def __repr__(self):
    label = f" {self._tag}" if self._tag else ""
    if self._raw:
        return f"GoHandle({self._raw}{label})"
    return f"GoHandle(closed{label})"
```

Then FFI functions can tag: `GoHandle(r, tag="Client")`, `GoHandle(r, tag="Ciphertext")`, etc.

## F5: Eight `__del__` methods, six different patterns — RESOLVED

**Resolved in:** `571233e` — Standardized all 8 `__del__` methods to canonical `try: self.close() except Exception: pass` pattern. Added `close()` to Ciphertext and PlainText.

**Severity:** Code smell

**Location:** All handle-owning classes

**Problem:**

Every class that wraps a Go handle has a `__del__` method. They all try to do the same thing (close the handle safely during interpreter shutdown), but each one invented its own incantation:

| Class                                   | Guard                                                                                 | Cleanup                       | Catch              |
| --------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------- | ------------------ |
| `GoHandle` (ffi.py:374)                 | `self._raw and sys and sys.modules`                                                   | `self.close()`                | `except Exception` |
| `Ciphertext` (ciphertext.py:49)         | `"sys" in globals() and sys.modules and self._handle`                                 | `self._handle.close()`        | `except Exception` |
| `PlainText` (ciphertext.py:263)         | `"sys" in globals() and sys.modules and self._handle`                                 | `self._handle.close()`        | `except Exception` |
| `Client` (client.py:203)                | `hasattr(self, "_handle") and self._handle`                                           | `self.close()`                | bare `except:`     |
| `_MultiPlainText` (client.py:225)       | `import sys as _sys; _sys and _sys.modules`                                           | loop `h.close()`              | `except Exception` |
| `Evaluator` (evaluator.py:417)          | `hasattr(self, '_eval_handle') and self._eval_handle`                                 | `self.close()`                | `except Exception` |
| `Compiler` (compiler.py:548)            | `hasattr(self, "backend") and self.backend` then `"sys" in globals() and sys.modules` | `self.backend.DeleteScheme()` | `except Exception` |
| `PlainTensor` (compiler_backend.py:314) | `"sys" in globals() and sys.modules and self.context`                                 | loop `h.close()`              | `except Exception` |

Three different shutdown guard styles (`sys and sys.modules`, `"sys" in globals()`, `hasattr`). One bare `except:` that catches `KeyboardInterrupt` and `SystemExit`. One class reimports sys as a local (`_MultiPlainText`). Client and Evaluator delegate to `self.close()`, while Ciphertext/PlainText call `self._handle.close()` directly. The Compiler nests two guard conditions.

None of this is _wrong_ — GoHandle's idempotent close prevents double-frees regardless. But it's the kind of inconsistency that breeds copy-paste bugs when someone adds a new handle-owning class.

**Fix:**

Pick one pattern and enforce it everywhere:

```python
def __del__(self):
    try:
        self.close()
    except Exception:
        pass
```

This works because: (1) `close()` is already idempotent on every class, (2) `close()` calls `GoHandle.close()` which is idempotent, (3) if ctypes is torn down, the exception is caught and swallowed. The `sys.modules` guards are cargo cult — if `close()` touches a dead module, the try/except catches it. For classes without `close()` (Ciphertext, PlainText), add one.

The try/except suppression in `__del__` is standard practice — psycopg2, h5py, pyzmq all do the same. The alternative (`weakref.finalize`) is cleaner in theory, but no major FFI library uses it for handle cleanup in practice.

## F6: `_EvalContext` self-aliasing — one object pretending to be six

**Severity:** Code smell (architectural) — **deferred**

**Location:** `orion/evaluator.py:38-48`

**Problem:**

```python
self.encoder = self
self.poly_evaluator = self
self.lt_evaluator = self
self.params = self
self.evaluator = self
self.bootstrapper = self
```

`_EvalContext` assigns itself to six different attribute names so that nn modules can call `context.encoder.encode(...)`, `context.poly_evaluator.evaluate_polynomial(...)`, etc. — all dispatching to the same object.

This exists because the compile-time context passes _separate_ objects for each role (via `SimpleNamespace` in `compiler.py:92-106`), and the nn modules were written to call `context.encoder.encode(...)` not `context.encode(...)`. Rather than changing 15+ module classes, the evaluator made one object pretend to be all of them.

**Decision:** Low priority. It's a duck-typing adapter solving a real polymorphism problem (nn modules must work identically at compile time and inference time). The proper fix comes when the FX graph lands in CompiledModel and the compile/inference duality gets redesigned.

## F7: Copy-paste FFI wrappers — 10 identical functions

**Severity:** Code smell — **tolerate**

**Location:** `orion/backend/orionclient/ffi.py:548-627`

**Problem:**

`eval_add`, `eval_sub`, `eval_mul`, their plaintext variants, and scalar variants are ~80 lines of copy-paste following two templates (ct-ct binary op, ct-scalar op).

**Decision:** Tolerate. FFI bindings are supposed to be boring and mechanical. A factory with `getattr(lib, op_name)` saves lines but hurts greppability, debuggability, and stack trace readability. Every FFI binding generator (SWIG, cffi, pybind11) produces this kind of boilerplate.

## F8: Go bridge — `ClientModuliChain` and `ClientAuxModuliChain` missing panic recovery — RESOLVED

**Resolved in:** `18e8f57` — Added `errOut` parameter and `defer catchPanic(errOut)` to both functions. Updated Python FFI prototypes and wrappers.

**Severity:** Bug (process crash)

**Location:** `orionclient/bridge/client.go:244`, `orionclient/bridge/client.go:263`

**Problem:**

Every other bridge function that dereferences a `cgo.Handle` has either `defer catchPanic(errOut)` or `defer func() { recover() }()`. These two have neither:

```go
//export ClientModuliChain
func ClientModuliChain(clientH C.uintptr_t, outLen *C.int) *C.ulonglong {
    client := cgo.Handle(clientH).Value().(*orionclient.Client)  // UNGUARDED
    chain := client.ModuliChain()
    // ...
}
```

If `clientH` is invalid (deleted handle, wrong type), the type assertion panics and crashes the entire process.

**Fix:**

Add `errOut` parameter and use `catchPanic(errOut)`. This matches the pattern used by all mutating bridge functions and gives Python a real error message instead of a silent zero return. Update the Python FFI prototypes and wrappers to match.

## F9: Go bridge — `cBytesToGoSlice` truncates length from `C.ulong` to `C.int`

**Severity:** Latent bug — **deferred**

**Location:** `orionclient/bridge/main.go:53-55`

**Problem:**

`C.GoBytes` takes `C.int` (32-bit signed). `dataLen` is `C.ulong` (64-bit unsigned). Silent truncation on >2GB payloads.

**Decision:** Theoretical. No FHE payload approaches 2GB with current parameter sets. Not worth fixing now.

## F10: `CompilerBackend` uses CamelCase methods

**Severity:** Code smell (style) — **tolerate**

**Location:** `orion/core/compiler_backend.py:221-268`

**Problem:**

`Encode()`, `Decode()`, `DeletePlaintext()`, `GetMaxSlots()`, etc. — Go naming convention in a Python class.

**Decision:** Tolerate. Internal adapter class only used by the compiler. CamelCase serves as a visual signal that this is the FFI boundary layer, mapping 1:1 to Go functions. Renaming gains PEP 8 compliance on an internal class that nobody outside the compiler touches.

## F11: `Linear.compile()` duplicates parent instead of extending it — RESOLVED

**Resolved in:** `e5d8216` — Replaced duplicated `compile()` bodies in Linear and Conv2d with `super().compile(context)`.

**Severity:** Code smell

**Location:** `orion/nn/linear.py:103-107` vs `orion/nn/linear.py:43-45`

**Problem:**

```python
# LinearTransform.compile (parent)
def compile(self, context):
    self._lt_evaluator = context.lt_evaluator
    self.transform_ids = context.lt_evaluator.generate_transforms(self)

# Linear.compile (child) — DUPLICATES parent body
def compile(self, context):
    bias = packing.construct_linear_bias(self)
    self.on_bias_ptxt = context.encoder.encode(bias, self.level-self.depth)
    self._lt_evaluator = context.lt_evaluator                          # duplicated
    self.transform_ids = context.lt_evaluator.generate_transforms(self) # duplicated
```

`Conv2d.compile()` (`linear.py:192-196`) has the same duplication. If someone changes the parent's `compile()` (e.g., adds handle tracking), Linear and Conv2d won't inherit it.

**Fix:**

```python
def compile(self, context):
    bias = packing.construct_linear_bias(self)
    self.on_bias_ptxt = context.encoder.encode(bias, self.level - self.depth)
    super().compile(context)
```

## F12: `LinearTransform.evaluate_transforms` has a dead legacy fallback — RESOLVED

**Resolved in:** `9b5cf94` — Removed the `hasattr` guard and dead else branch. Only the `ctx.evaluate_transforms(self, x)` path remains.

**Severity:** Dead code

**Location:** `orion/nn/linear.py:48-55`

**Problem:**

```python
if hasattr(self, 'transform_handles') and self.transform_handles:
    out = ctx.evaluate_transforms(self, x)
else:
    out = ctx.lt_evaluator.evaluate_transforms(self, x)
```

After v2, `transform_handles` is always populated by `Evaluator._reconstruct_linear`. The `hasattr` guard is a dead artifact from the transition period. `transform_handles` is initialized to `{}` in `__init__`, which is falsy — but at inference time it's always populated. The else branch is unreachable.

**Fix:**

Remove the guard and the else branch:

```python
def evaluate_transforms(self, x):
    ctx = x.context
    out = ctx.evaluate_transforms(self, x)
    # ...
```
