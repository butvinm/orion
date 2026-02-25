# Go Client & Evaluator Refactor v2 — GoHandle Ownership Layer

## Context

v1 (branch `go-client-and-evaluator-refactor`, now merged to `main`) delivered the full Go library, FFI bridge, and Python wrappers (+17k/-4k lines). Three review rounds found 11 bugs — 6 were handle lifecycle issues (leaks, double-frees, silent drops). Root cause: raw `uintptr_t` handles with no ownership tracking. Python code passed bare ints around, cleanup was ad-hoc `__del__` methods, and there was no single owner for reconstruction handles.

**What v1 delivered (already in codebase, no changes needed):**

- `orionclient/` Go package: `Client`, `Evaluator`, `Ciphertext`, `LinearTransform`, `Polynomial`, `Bootstrapper` — all instance-based, tested
- `orionclient/bridge/`: 73 C-exported functions via `cgo.Handle`. `Close != Delete` enforced. `catchPanic` on all exports.
- `orion/backend/orionclient/ffi.py`: ctypes wrappers for all 73 bridge functions
- `orion/ciphertext.py`: unified `Ciphertext`/`PlainText` (replaces old `CipherText`/`CipherTensor`)
- `orion/client.py`: `Client` with keygen, encode, encrypt, decrypt
- `orion/evaluator.py`: `Evaluator` with module reconstruction, FHE inference
- `orion/core/compiler_backend.py`: `CompilerBackend` using FFI
- `demo/wasm-fhe-demo/wasm/`: already imports `orionclient`
- Old `orion/backend/python/` and `orion/backend/lattigo/` already deleted

**What v2 does:** Introduce `GoHandle` RAII wrapper and thread it through the Python layer. ~2500 lines of Python to update across 6 files. No Go changes.

**Branch:** `go-client-evaluator-refactor-v2` (forked from `main` with v1 merged)

## Memory Ownership Convention

### Rule 1: GoHandle wraps every Go object in Python

No raw `uintptr_t` values outside `ffi.py`. Every FFI function returning a handle returns `GoHandle`.

```python
class GoHandle:
    """RAII wrapper for a cgo.Handle (opaque uintptr_t). Idempotent close."""
    __slots__ = ('_raw',)

    def __init__(self, raw: int):
        self._raw = raw

    @property
    def raw(self) -> int:
        if not self._raw:
            raise RuntimeError("Use of closed handle")
        return self._raw

    def close(self):
        """Release the Go object. Idempotent — second call is a no-op."""
        if self._raw:
            _get_lib().DeleteHandle(_uintptr(self._raw))
            self._raw = 0

    def __del__(self):
        if self._raw and sys and sys.modules:
            try: self.close()
            except: pass

    def __bool__(self): return bool(self._raw)
```

### Rule 2: Bridge functions BORROW input handles, never consume

- `ClientClose(h)` calls `client.Close()` only (zeros secret key). Does NOT call `h.Delete()`.
- `EvaluatorClose(h)` calls `evaluator.Close()` only. Does NOT call `h.Delete()`.
- `DeleteHandle(h)` is the ONLY bridge function that calls `cgo.Handle.Delete()`.

Python two-step close:

```python
# Client.close()
ffi.client_close(self._handle)   # step 1: resource cleanup (zeros SK)
self._handle.close()             # step 2: handle table cleanup (DeleteHandle)
```

### Rule 3: Evaluator owns all reconstruction handles

`Evaluator._tracked_handles: list[GoHandle]` collects every handle created during `_reconstruct_modules`:

- LinearTransform handles (stored as GoHandle in `module.transform_handles`)
- Polynomial handles (`module.poly`)
- Bias PlainText handles (`module.on_bias_ptxt._handle`)

`Evaluator.close()` iterates the list, calls `h.close()` on each, then closes the evaluator handle. Module `__del__` does NOT clean up Go handles — Evaluator owns them.

### Rule 4: FFI functions accept GoHandle, return GoHandle

```python
def eval_add(eval_h: GoHandle, ct1_h: GoHandle, ct2_h: GoHandle) -> GoHandle:
    err = _make_errout()
    r = lib.EvalAdd(c_size_t(eval_h.raw), c_size_t(ct1_h.raw), c_size_t(ct2_h.raw), byref(err))
    _check_err(err)
    return GoHandle(r)
```

### Rule 5: Intermediate handles freed immediately, error-safe

```python
combined = ffi.eval_add(eval_h, ct_out_h, res_h)
ct_out_h.close()  # free intermediate — idempotent via GoHandle
res_h.close()
ct_out_h = combined
```

Multi-step handle sequences use try/finally:

```python
handles = []
try:
    for blob in blobs:
        h = ffi.linear_transform_unmarshal(blob)
        handles.append(h)
    # ... use handles ...
except:
    for h in handles:
        h.close()
    raise
```

## Serialization Ownership

```
Go orionclient owns:         Python compiler owns:
- Ciphertext wire format     - CompiledModel (params + module metadata
- EvalKeys                     + topology + LT/bias blob container)
- LinearTransform blobs      - Module graph (forward() connections)
- Secret key

Handle ownership after reconstruction:
  Evaluator._tracked_handles: [GoHandle, GoHandle, ...]  (LT + poly + bias)
  Evaluator._eval_handle:     GoHandle  (the Go Evaluator)
  Client._handle:             GoHandle  (the Go Client)
  Ciphertext._handle:         GoHandle  (inference-time, created by ops)
```

## Bug Prevention Matrix

| Bug | Description                                  | Design fix                                     | Task |
| --- | -------------------------------------------- | ---------------------------------------------- | ---- |
| #1  | ClientClose/EvaluatorClose leaked Go handles | Rule 2: Close != Delete                        | 1    |
| #2  | CompilerBackend double-free                  | Rule 2: two-step close                         | 6    |
| #3  | Client.encrypt() dropped multi-ct handles    | GoHandle + combine + try/finally               | 3    |
| #4  | EvalKeyBundle handle leaked after NewEval    | Explicit close after NewEvaluator              | 4    |
| #5  | Multi-row LT leaked intermediate handles     | GoHandle wrapping intermediates                | 4    |
| #6  | LinearTransform.**del** double-freed         | Rule 3: Evaluator owns, module **del** removed | 5    |
| #7  | No panic recovery in bridge functions        | Already fixed in v1 (catchPanic on all)        | —    |
| #8  | math.Log instead of math.Log2                | Already fixed in v1                            | —    |
| #9  | DefaultScale lossy float64                   | Already fixed in v1 (uint64)                   | —    |
| #10 | Examples used deleted CipherText API         | Already fixed in v1                            | —    |
| #11 | Redundant key generation                     | Single generate_keys path                      | 3    |

## Task 1: GoHandle class + FFI layer update

Introduce `GoHandle` into `ffi.py`. Update all wrapper functions to accept and return `GoHandle` instead of raw ints.

**File:** `orion/backend/orionclient/ffi.py` (908 lines)

### Handle-returning functions (must return `GoHandle`)

These are every FFI wrapper whose corresponding C function has `restype = _uintptr` and returns an opaque Go object:

- **Client lifecycle:** `new_client`, `new_client_from_secret_key`
- **Client operations:** `client_encode`, `client_encrypt`
- **Client keygen:** `client_generate_keys`
- **Evaluator lifecycle:** `new_evaluator`
- **Evaluator encoding:** `eval_encode`
- **Evaluator arithmetic (ct-ct):** `eval_add`, `eval_sub`, `eval_mul`
- **Evaluator arithmetic (ct-pt):** `eval_add_plaintext`, `eval_sub_plaintext`, `eval_mul_plaintext`
- **Evaluator arithmetic (scalar):** `eval_add_scalar`, `eval_mul_scalar`
- **Evaluator unary:** `eval_negate`
- **Evaluator rotation/rescale:** `eval_rotate`, `eval_rescale`
- **Evaluator polynomial:** `eval_poly`
- **Evaluator linear transform:** `eval_linear_transform`
- **Evaluator bootstrap:** `eval_bootstrap`
- **Ciphertext:** `ciphertext_unmarshal`
- **EvalKeyBundle:** `new_eval_key_bundle`
- **LinearTransform:** `linear_transform_unmarshal`, `generate_linear_transform`
- **Polynomial:** `generate_polynomial_monomial`, `generate_polynomial_chebyshev`

**Total: 27 functions** that must return `GoHandle(result)`.

### Functions that accept handles but do NOT return handles (pass `h.raw`)

These accept GoHandle parameters but return data (bytes, ints, floats, lists), not handles:

- `client_close`, `client_secret_key`, `client_decode`, `client_max_slots`, `client_default_scale`, `client_galois_element`, `client_moduli_chain`, `client_aux_moduli_chain`
- `evaluator_close`, `eval_max_slots`, `eval_galois_element`, `eval_moduli_chain`, `eval_default_scale`
- `ciphertext_marshal`, `ciphertext_level`, `ciphertext_scale`, `ciphertext_set_scale`, `ciphertext_slots`, `ciphertext_degree`, `ciphertext_shape`, `ciphertext_num_ciphertexts`
- `plaintext_level`, `plaintext_scale`, `plaintext_set_scale`, `plaintext_slots`
- `eval_key_bundle_set_rlk`, `eval_key_bundle_add_galois_key`, `eval_key_bundle_add_bootstrap_key`, `eval_key_bundle_set_boot_logp`
- `linear_transform_marshal`, `linear_transform_required_galois_elements`

These change from `_uintptr(h)` to `_uintptr(h.raw)` (or `h.raw` if already using `c_size_t`).

### Functions that don't touch handles at all (no changes)

- `free_c_array`, `generate_minimax_sign_coeffs`

### Special case: `combine_single_ciphertexts`

Currently takes `handles: list[int]`. Change to `handles: list[GoHandle]`:

```python
def combine_single_ciphertexts(handles: list[GoHandle], shape):
    h_arr = (_uintptr * n)(*[_uintptr(h.raw) for h in handles])
    ...
    return GoHandle(r)
```

### Special case: `client_decrypt`

Currently returns `list[int]` (list of plaintext handles). Change to `list[GoHandle]`:

```python
def client_decrypt(h, ct_h):
    ...
    handles = [GoHandle(ptr[i]) for i in range(n)]
    ...
    return handles
```

### Special case: `delete_handle`

Keep the function but it becomes internal-only — only called by `GoHandle.close()`. No external callers should exist after migration.

### Steps

- [x] Add `GoHandle` class to `ffi.py` (see Rule 1 above), placed after `_get_lib()` and before wrapper functions
- [x] Update the 27 handle-returning functions to return `GoHandle(result)`
- [x] Update all handle-accepting functions to use `h.raw` instead of bare `h`
- [x] Update `combine_single_ciphertexts` to accept `list[GoHandle]`
- [x] Update `client_decrypt` to return `list[GoHandle]`
- [x] `client_close(h: GoHandle)` calls `lib.ClientClose(_uintptr(h.raw))` only — does NOT call DeleteHandle **(prevents bug #1)**
- [x] `evaluator_close(h: GoHandle)` same pattern **(prevents bug #1)**

### Acceptance criteria

- `GoHandle(h).raw` returns the raw value; `.close()` is idempotent; `.raw` after close raises `RuntimeError`
- All 27 handle-returning FFI functions return `GoHandle` — verified by `isinstance(ffi.new_client(...), GoHandle)`
- `ffi.client_close(h)` + `h.raw` still works (close doesn't delete the handle — only `h.close()` does)
- Existing test suite passes (`pytest tests/`)

## Task 2: Unified Ciphertext and PlainText with GoHandle

Update `Ciphertext` and `PlainText` to store `GoHandle` instead of raw int.

**File:** `orion/ciphertext.py` (271 lines)

### Ciphertext changes

- `__init__(self, handle, ...)`: `self._handle` is now `GoHandle`. Callers already pass the return value of FFI functions (which are now `GoHandle` after Task 1).
- `__del__`: replace `ffi.delete_handle(self._handle)` with `self._handle.close()`. The GoHandle `__del__` is the safety net; explicit `close()` is preferred.
- `handle` property: returns `self._handle` (a GoHandle). Callers that pass it to FFI (e.g. `ffi.ciphertext_level(self._handle)`) — these now pass a GoHandle. The FFI functions (after Task 1) expect GoHandle and call `.raw` internally.
- `_wrap(self, handle)`: `handle` parameter is GoHandle. No change to call sites.

### Arithmetic methods — `in_place` semantics

Current code (e.g. `add`):

```python
r = ffi.eval_add(self._eval_h(), self._handle, other._handle)  # r is GoHandle after Task 1
if in_place:
    ffi.delete_handle(self._handle)  # OLD: raw int
    self._handle = r
    return self
return self._wrap(r)
```

Change to:

```python
r = ffi.eval_add(self._eval_h(), self._handle, other._handle)  # GoHandle
if in_place:
    self._handle.close()  # NEW: idempotent close on old handle
    self._handle = r
    return self
return self._wrap(r)
```

The `in_place=True` path always allocates a NEW Go ciphertext (the Go bridge always returns a new handle, never mutates). So the old handle must always be closed. `in_place=False` path: old handle stays alive (the caller still holds the original Ciphertext object).

Apply the same pattern to: `add`, `sub`, `mul`, `roll`. The `mul` method has an additional intermediate (`r` before rescale → `r2` after rescale → `ffi.delete_handle(r)` → change to `r.close()`).

### PlainText changes

Same as Ciphertext: `_handle` becomes GoHandle, `__del__` calls `self._handle.close()`.

`PlainText.mul(self, other)` has the same intermediate pattern as `Ciphertext.mul` — change `ffi.delete_handle(r)` to `r.close()`, and `ffi.delete_handle(other._handle)` to `other._handle.close()`.

### Steps

- [x] `Ciphertext.__init__`: `self._handle` stores GoHandle (no type conversion needed — FFI already returns GoHandle after Task 1)
- [x] `Ciphertext.__del__`: replace `ffi.delete_handle(self._handle)` → `self._handle.close()`
- [x] `Ciphertext.add/sub/mul/roll` (4 methods): replace `ffi.delete_handle(self._handle)` → `self._handle.close()` in the `in_place` branch
- [x] `Ciphertext.mul`: additionally replace `ffi.delete_handle(r)` → `r.close()` for the pre-rescale intermediate
- [x] `PlainText.__del__`: replace `ffi.delete_handle(self._handle)` → `self._handle.close()`
- [x] `PlainText.mul`: replace `ffi.delete_handle(r)` → `r.close()` and `ffi.delete_handle(other._handle)` → `other._handle.close()`
- [x] Remove `from orion.backend.orionclient import ffi` usage of `ffi.delete_handle` from this file (all replaced by `.close()`)

### Acceptance criteria

- `isinstance(ct._handle, GoHandle)` and `isinstance(pt._handle, GoHandle)` for any Ciphertext/PlainText
- `ct.add(other, in_place=True)` produces correct result and old handle is closed (`_raw == 0`)
- `ct.add(other, in_place=False)` produces correct result and original `ct` is still usable
- `del ct` on a live Ciphertext does not crash
- `Ciphertext.from_bytes(ct.to_bytes())` decrypts to same values as original

## Task 3: Client with GoHandle + two-step close

Update `Client` to use `GoHandle` and proper two-step close.

**File:** `orion/client.py` (219 lines)

### `__init__` changes

`ffi.new_client()` and `ffi.new_client_from_secret_key()` now return GoHandle (after Task 1). `self._handle` is GoHandle. No explicit conversion needed.

### `close()` — two-step pattern

Current v1 code:

```python
def close(self):
    if hasattr(self, "_handle") and self._handle:
        ffi.client_close(self._handle)
        self._handle = None
```

v1 bug: `client_close` calls Go `Client.Close()` (zeros SK) but the cgo handle is never deleted — **leak**. Change to:

```python
def close(self):
    if hasattr(self, "_handle") and self._handle:
        ffi.client_close(self._handle)   # step 1: zeros SK in Go
        self._handle.close()             # step 2: DeleteHandle (frees cgo slot)
        self._handle = None              # prevent double-close
```

Note: `self._handle = None` after close means `__del__` and second `close()` are no-ops (the `if self._handle` check catches None).

### `generate_keys()` — bypasses FFI wrappers, needs `.raw` fixup

Current v1 code (lines 56–101) calls `lib.ClientGenerateRLK`, `lib.ClientGenerateGaloisKey`, `lib.ClientGenerateBootstrapKeys` directly via `ffi._get_lib()`, bypassing the FFI wrapper layer. These raw calls return `c_void_p` (byte pointers, not handles) that are freed via `lib.FreeCArray(ptr)`. The byte extraction logic is correct and does not need GoHandle.

**However**, every raw call passes `ffi._uintptr(self._handle)` to convert the client handle to a C integer. After Task 1, `self._handle` is a GoHandle — so `ffi._uintptr(self._handle)` becomes `ffi._uintptr(GoHandle)` which is wrong. Every occurrence must change to `ffi._uintptr(self._handle.raw)`.

Affected lines (6 occurrences):

```python
# Line 60 (RLK)
ptr = lib.ClientGenerateRLK(ffi._uintptr(self._handle), ...)
# Line 73 (Galois)
ptr = lib.ClientGenerateGaloisKey(ffi._uintptr(self._handle), ...)
# Line 90 (Bootstrap)
ptr = lib.ClientGenerateBootstrapKeys(ffi._uintptr(self._handle), ...)
```

Change each to `ffi._uintptr(self._handle.raw)`.

No logic changes, no ownership changes. Pure mechanical `.raw` insertion.

### `encrypt()` — multi-ciphertext try/finally

Current v1 code:

```python
ct_handles = []
for pt_h in plaintext.handles:
    ct_h = ffi.client_encrypt(self._handle, pt_h)
    ct_handles.append(ct_h)
# ...
combined_h = ffi.combine_single_ciphertexts(ct_handles, list(plaintext.shape))
for h in ct_handles:
    ffi.delete_handle(h)
return Ciphertext(combined_h, shape=plaintext.shape)
```

After Task 1, `ct_handles` is `list[GoHandle]`. Change to:

```python
ct_handles = []
try:
    for pt_h in plaintext.handles:
        ct_h = ffi.client_encrypt(self._handle, pt_h)
        ct_handles.append(ct_h)
    if len(ct_handles) == 1:
        return Ciphertext(ct_handles[0], shape=plaintext.shape)
    combined_h = ffi.combine_single_ciphertexts(ct_handles, list(plaintext.shape))
finally:
    # Close individual handles — combine copied the data into Go
    # If combine raised, this cleans up partial handles
    for h in ct_handles:
        h.close()
return Ciphertext(combined_h, shape=plaintext.shape)
```

Wait — if `len(ct_handles) == 1`, we return `ct_handles[0]` directly but the `finally` will close it. Fix: only close individual handles when we combined:

```python
ct_handles = []
for pt_h in plaintext.handles:
    ct_h = ffi.client_encrypt(self._handle, pt_h)
    ct_handles.append(ct_h)
if len(ct_handles) == 1:
    return Ciphertext(ct_handles[0], shape=plaintext.shape)
try:
    combined_h = ffi.combine_single_ciphertexts(ct_handles, list(plaintext.shape))
except:
    for h in ct_handles:
        h.close()
    raise
# combine succeeded — close individuals, they're now inside combined
for h in ct_handles:
    h.close()
return Ciphertext(combined_h, shape=plaintext.shape)
```

### `_MultiPlainText` — handles become GoHandle

`_MultiPlainText.handles` is `list[GoHandle]`. `__del__` changes from `ffi.delete_handle(h)` to `h.close()`.

### `__del__` — simplify

Current v1 reimports `sys`. With GoHandle, just call `self.close()` which is already idempotent:

```python
def __del__(self):
    if hasattr(self, "_handle") and self._handle:
        try:
            self.close()
        except:
            pass
```

### Steps

- [x] `self._handle` is GoHandle (automatic after Task 1, no explicit change)
- [x] `close()`: add `self._handle.close()` after `ffi.client_close()`, set `self._handle = None`
- [x] `encrypt()`: restructure multi-ciphertext path as shown above — close individuals after combine, close-on-error if combine fails
- [x] `_MultiPlainText.__del__`: replace `ffi.delete_handle(h)` → `h.close()`
- [x] `__del__`: simplify to `self.close()` with try/except
- [x] `generate_keys()`: change 6 occurrences of `ffi._uintptr(self._handle)` → `ffi._uintptr(self._handle.raw)`. No logic changes.

### Acceptance criteria

- `client.close(); client.close()` — no error, no crash
- After `client.close()`, `client._handle` is None (fully released)
- Encode → encrypt → decrypt → decode round-trip produces correct values
- `with Client(params) as c: c.encode(...)` — client is usable inside, closed after
- Multi-ciphertext encrypt with valid input produces correct combined Ciphertext

## Task 4: Evaluator with handle tracking

Add `_tracked_handles` ownership list. Cleanup partial construction on failure.

**File:** `orion/evaluator.py` (397 lines)

### `__init__` — initialization order and error recovery

Current v1 `__init__` (lines 197–235):

1. `keys_handle = ffi.new_eval_key_bundle()` → GoHandle
2. Populate bundle (set_rlk, add_galois_key, add_bootstrap_key, set_boot_logp)
3. `self._eval_handle = ffi.new_evaluator(pj, keys_handle)` → GoHandle
4. `ffi.delete_handle(keys_handle)` → change to `keys_handle.close()`
5. Build `_EvalContext`
6. Call `_reconstruct_modules`

Change to:

```python
def __init__(self, net, compiled, keys):
    self.net = net
    self.compiled = compiled
    self.ckks_params = compiled.params
    self._eval_handle = None      # set before try, so close() can check it
    self._tracked_handles = []

    try:
        # Build EvalKeyBundle
        keys_handle = ffi.new_eval_key_bundle()
        # ... populate bundle ...
        self._eval_handle = ffi.new_evaluator(pj, keys_handle)
        keys_handle.close()  # bundle consumed by Go, free the handle

        self._context = _EvalContext(self._eval_handle, compiled.params, compiled)
        self._reconstruct_modules(compiled)
    except:
        self.close()
        raise
```

`close()` must handle partially-initialized state: `_eval_handle` may be None.

### `_reconstruct_linear` — what handles get tracked

Current code (lines 283–303): For each `LinearTransform` module, iterates `meta["transform_blobs"]` (a dict of `"row,col" -> blob_index`). Each blob yields one LT handle via `ffi.linear_transform_unmarshal(blob_data)`.

Number of LT handles per module: **one per (row, col) pair in the diagonal packing**. For a simple Linear layer, typically 1 pair `(0,0)`. For Conv2d, may have multiple pairs.

After Task 1, `ffi.linear_transform_unmarshal` returns GoHandle. Track each:

```python
def _reconstruct_linear(self, module, meta, blobs):
    transform_handles = {}
    for key_str, blob_idx in meta["transform_blobs"].items():
        row, col = map(int, key_str.split(","))
        lt_h = ffi.linear_transform_unmarshal(blobs[blob_idx])  # GoHandle
        transform_handles[(row, col)] = lt_h
        self._tracked_handles.append(lt_h)  # Evaluator owns this

    module.transform_handles = transform_handles
    module.bsgs_ratio = meta["bsgs_ratio"]
    module.output_rotations = meta["output_rotations"]

    # Re-encode bias
    bias_blob = blobs[meta["bias_blob"]]
    bias_vec = np.frombuffer(bias_blob, dtype=np.float64)
    bias_tensor = torch.tensor(bias_vec, dtype=torch.float64)
    bias_pt = self._context.encode(bias_tensor, level=module.level - module.depth)
    module.on_bias_ptxt = bias_pt
    self._tracked_handles.append(bias_pt._handle)  # Evaluator owns bias handle too
    module._eval_context = self._context
```

### `_reconstruct_chebyshev` — polynomial handle tracked

Current code (lines 305–316): calls `module.compile(self._context)`. Inside `compile()` (activation.py line 107): `self.poly = context.poly_evaluator.generate_chebyshev(self.coeffs)`.

`context.poly_evaluator` is `self._context` (via self-aliasing). `generate_chebyshev` (evaluator.py line 95) calls `ffi.generate_polynomial_chebyshev(coeffs)` which now returns GoHandle.

So `module.poly` becomes a GoHandle. Track it:

```python
def _reconstruct_chebyshev(self, module, meta):
    # ... set coeffs, prescale, etc. ...
    module.compile(self._context)  # sets module.poly = GoHandle
    self._tracked_handles.append(module.poly)
```

### `_reconstruct_activation` — same pattern

Current code (lines 318–323): calls `module.compile(self._context)`. Inside `compile()` (activation.py line 29): `self.poly = context.poly_evaluator.generate_monomial(self.coeffs)` → GoHandle.

```python
def _reconstruct_activation(self, module, meta):
    # ... set coeffs, output_scale ...
    module.compile(self._context)  # sets module.poly = GoHandle
    self._tracked_handles.append(module.poly)
```

### `_reconstruct_bootstrap_hooks` — prescale_ptxt handle tracked

Current code (lines 336–367): creates `Bootstrap` objects, calls `bootstrapper.compile(self._context)`. Inside `Bootstrap.compile()` (operations.py line 50–59): `self.prescale_ptxt = context.encoder.encode(prescale_vec, level=..., scale=ql)` → PlainText with GoHandle.

Track the PlainText's handle:

```python
def _reconstruct_bootstrap_hooks(self, net, meta, module_map):
    for mod_name, mod_meta in meta.items():
        if mod_meta["type"] != "Bootstrap" or "hook_target" not in mod_meta:
            continue
        # ... create bootstrapper, set fields ...
        bootstrapper.compile(self._context)  # sets bootstrapper.prescale_ptxt = PlainText(GoHandle)
        self._tracked_handles.append(bootstrapper.prescale_ptxt._handle)
        # ... register hook ...
```

### `_EvalContext.evaluate_transforms` — intermediate handle cleanup

Current code (lines 110–168). The loop structure:

```
for each row i:
    ct_out_h = None
    for each col j:
        lt_h = transform_handles.get((i, j))   # GoHandle (from tracked list, NOT closed here)
        res_h = ffi.eval_linear_transform(...)  # NEW GoHandle (intermediate)
        if ct_out_h is None:
            ct_out_h = res_h                    # first col: no intermediate to free
        else:
            combined = ffi.eval_add(ct_out_h, res_h)  # NEW GoHandle
            ct_out_h.close()    # free old accumulator
            res_h.close()       # free column result
            ct_out_h = combined
    rescaled_h = ffi.eval_rescale(ct_out_h)  # NEW GoHandle
    ct_out_h.close()  # free pre-rescale
    cts_out_handles.append(rescaled_h)
```

All intermediate handles (`res_h`, old `ct_out_h`, pre-rescale `ct_out_h`) are created and destroyed within the loop. They are NOT tracked in `_tracked_handles`. The `lt_h` handles (from `transform_handles`) are tracked and must NOT be closed here.

The multi-row tail (lines 159–168) closes unused rows — change `ffi.delete_handle(h)` to `h.close()`.

### `close()` — full cleanup

```python
def close(self):
    # Close all tracked reconstruction handles
    for h in getattr(self, '_tracked_handles', []):
        try:
            h.close()
        except:
            pass
    self._tracked_handles = []

    # Close the Go Evaluator
    if hasattr(self, '_eval_handle') and self._eval_handle:
        ffi.evaluator_close(self._eval_handle)  # step 1: resource cleanup
        self._eval_handle.close()                # step 2: delete handle
        self._eval_handle = None
```

Using `getattr` with default protects against `close()` being called before `_tracked_handles` is initialized (partial init failure).

### Steps

- [x] Add `self._eval_handle = None` and `self._tracked_handles = []` at top of `__init__`, before try block
- [x] Wrap `__init__` body in try/except — on failure, call `self.close()`, then re-raise
- [x] `keys_handle.close()` instead of `ffi.delete_handle(keys_handle)` after `new_evaluator`
- [x] `_reconstruct_linear`: append each LT GoHandle and bias PlainText GoHandle to `_tracked_handles`
- [x] `_reconstruct_chebyshev`: append `module.poly` (GoHandle) to `_tracked_handles`
- [x] `_reconstruct_activation`: append `module.poly` (GoHandle) to `_tracked_handles`
- [x] `_reconstruct_bootstrap_hooks`: append `bootstrapper.prescale_ptxt._handle` to `_tracked_handles`
- [x] `evaluate_transforms`: replace `ffi.delete_handle(...)` with `.close()` on intermediates
- [x] `close()`: iterate `_tracked_handles` → close each → `ffi.evaluator_close` → `_eval_handle.close()`; use `getattr` for safety
- [x] `__del__`: call `self.close()` with shutdown guard

### Acceptance criteria

- `evaluator.close()` leaves all `_tracked_handles` with `_raw == 0`
- `evaluator.close(); evaluator.close()` — no error, no crash
- Evaluator constructed with corrupt LT blob raises and does not leak (all partial handles freed)
- Full inference round-trip (compile → keys → Evaluator → run → decrypt) produces correct values
- `len(evaluator._tracked_handles)` equals total LT handles + poly handles + bias handles across all modules

## Task 5: Remove module `__del__` handle cleanup

Evaluator owns all reconstruction handles. Modules must not compete for cleanup.

**File:** `orion/nn/linear.py` (lines 24–32)

### What to remove

Current `LinearTransform.__del__` (lines 24–32):

```python
def __del__(self):
    if 'sys' in globals() and sys.modules:
        try:
            if hasattr(self, '_lt_evaluator') and self._lt_evaluator:
                self._lt_evaluator.delete_transforms(self.transform_ids)
            if hasattr(self, '_eval_context') and self._eval_context:
                self._eval_context.delete_transforms(self.transform_handles)
        except Exception:
            pass
```

This attempts to clean up both compile-time (`transform_ids`) and inference-time (`transform_handles`) handles. After v2:

- **Inference-time handles** (`transform_handles`): owned by `Evaluator._tracked_handles`. Module must not touch them.
- **Compile-time handles** (`transform_ids`): currently leaked. The compiler (`compiler.py` lines 267–274) serializes them but never frees them. `TransformEncoder.delete_transforms()` exists but nobody calls it — the only caller was this `__del__` method. Removing `__del__` turns a hidden GC dependency into a real leak.

**Fix:** Add explicit cleanup in the compiler after serializing each module's LT blobs. In `compiler.py`, after line 274 (end of the serialization loop for a LinearTransform module):

```python
# Free compile-time LT handles (serialized into blobs, no longer needed)
self._lt_evaluator.delete_transforms(module.transform_ids)
```

Then delete the entire `__del__` method from `LinearTransform`.

### No changes needed in activation.py or operations.py

- `activation.py`: `module.poly` is a GoHandle stored on the module, but no `__del__` method exists there — `poly` is just an attribute. Evaluator tracks it.
- `operations.py`: `Bootstrap.prescale_ptxt` is a PlainText with GoHandle. No `__del__` on Bootstrap that touches handles. Evaluator tracks the handle.

### Steps

- [x] Delete `LinearTransform.__del__` method entirely (lines 24–32 in `orion/nn/linear.py`)
- [x] Add compile-time LT cleanup in `orion/compiler.py`: after serializing each LinearTransform module's blobs (after line 274), call `self._lt_evaluator.delete_transforms(module.transform_ids)` to free compile-time handles immediately
- [x] Verify `orion/nn/activation.py` has no `__del__` methods that touch handles
- [x] Verify `orion/nn/operations.py` has no `__del__` methods that touch handles

### Acceptance criteria

- `del module` on a reconstructed `nn.Linear` does not crash or call `DeleteHandle`
- Compile → serialize round-trip works: LT blobs produced correctly, compiler does not crash after freeing compile-time handles
- Full end-to-end test (compile + evaluate) passes — proving compile-time cleanup doesn't break inference

## Task 6: CompilerBackend with GoHandle

Update compile-time FFI usage to GoHandle.

**File:** `orion/core/compiler_backend.py` (480 lines)

### CompilerBackend — two-step close

Current `DeleteScheme()` (line 299):

```python
def DeleteScheme(self):
    if self._client_h:
        ffi.client_close(self._client_h)
        self._client_h = None
```

Same leak as Client.close() v1 — `client_close` zeros SK but doesn't delete the cgo handle. Change to:

```python
def DeleteScheme(self):
    if self._client_h:
        ffi.client_close(self._client_h)   # step 1: zeros SK
        self._client_h.close()              # step 2: DeleteHandle
        self._client_h = None
```

`self._client_h` becomes GoHandle after Task 1 (`ffi.new_client` returns GoHandle).

### CompilerBackend methods that pass handles

- `Encode(values, level, scale)` → calls `ffi.client_encode(self._client_h, ...)`. After Task 1, `self._client_h` is GoHandle, `ffi.client_encode` accepts GoHandle, returns GoHandle. **No code changes** — types flow automatically.
- `Decode(pt_h)` → calls `ffi.client_decode(self._client_h, pt_h)`. `pt_h` comes from `Encode()`, is now GoHandle. FFI accepts GoHandle. **No code changes.**
- `DeletePlaintext(pt_h)` → calls `ffi.delete_handle(pt_h)`. Change to `pt_h.close()`.
- `GetMaxSlots()`, `GetGaloisElement()`, `GetModuliChain()`, `GetAuxModuliChain()` → all pass `self._client_h` to FFI. **No code changes** — GoHandle flows through.
- `GenerateLinearTransform(...)` → calls `ffi.generate_linear_transform(...)`, returns GoHandle. **No code changes** — callers receive GoHandle.
- `SerializeLinearTransform(lt_h)` → calls `ffi.linear_transform_marshal(lt_h)`. `lt_h` is GoHandle. **No code changes.**
- `GetLinearTransformRotationKeys(lt_h)` → same. **No code changes.**
- `DeleteLinearTransform(lt_h)` → calls `ffi.delete_handle(lt_h)`. Change to `lt_h.close()`.
- `GenerateMonomial(coeffs)`, `GenerateChebyshev(coeffs)` → return GoHandle after Task 1. **No code changes.**

### PlainTensor — owns compile-time plaintext handles

`PlainTensor.ids` stores handle values. After Task 1, `ffi.client_encode` returns GoHandle, so `ids` becomes `list[GoHandle]`.

`PlainTensor.__init__` (line 318) has a type guard: `self.ids = [ptxt_ids] if isinstance(ptxt_ids, int) else ptxt_ids`. After Task 1, single handles are GoHandle, not int — so this guard is dead. Update to `isinstance(ptxt_ids, GoHandle)`. (In practice `NewEncoder.encode()` always passes a list, but the guard should be correct.)

`PlainTensor.__del__` (line 322): calls `self.backend.DeletePlaintext(idx)` for each idx. After above change, `DeletePlaintext` calls `pt_h.close()`. So `__del__` becomes:

```python
def __del__(self):
    if "sys" in globals() and sys.modules and self.context:
        for h in self.ids:
            try:
                h.close()
            except:
                pass
```

**Ownership clarification:** PlainTensor is a **compile-time** object created by `NewEncoder.encode()` during `compiler.fit()`. It has no relationship to `Evaluator._tracked_handles` (that's inference-time). PlainTensor owns its own handles and cleans up in `__del__`. This does NOT violate Rule 3, which applies only to inference-time reconstruction handles.

### TransformEncoder — compile-time LT handles

`TransformEncoder.generate_transforms()` returns `dict[(row,col) -> GoHandle]`. These compile-time LT handles are stored in `module.transform_ids`. They are cleaned up by `TransformEncoder.delete_transforms(transform_ids)` which calls `self.backend.DeleteLinearTransform(tid)` → `tid.close()`.

After Task 5 adds the explicit `self._lt_evaluator.delete_transforms(module.transform_ids)` call in the compiler, this method becomes the cleanup path for compile-time LT handles. No changes to `TransformEncoder` itself — only its caller (the compiler) changes.

### Steps

- [x] `DeleteScheme()`: add `self._client_h.close()` after `ffi.client_close()`
- [x] `DeletePlaintext(pt_h)`: replace `ffi.delete_handle(pt_h)` → `pt_h.close()`
- [x] `DeleteLinearTransform(lt_h)`: replace `ffi.delete_handle(lt_h)` → `lt_h.close()`
- [x] `PlainTensor.__init__`: update type guard `isinstance(ptxt_ids, int)` → `isinstance(ptxt_ids, GoHandle)`
- [x] `PlainTensor.__del__`: replace `self.backend.DeletePlaintext(idx)` → `h.close()` directly
- [x] No changes needed to `Encode`, `Decode`, `GenerateLinearTransform`, `SerializeLinearTransform`, `GenerateMonomial`, `GenerateChebyshev` — GoHandle flows through from Task 1

### Acceptance criteria

- `DeleteScheme(); DeleteScheme()` — no error, no crash
- After `DeleteScheme()`, `self._client_h` is None
- `PlainTensor` created via `NewEncoder.encode()` — `del` on it does not crash
- Full compilation (`compiler.compile()`) succeeds end-to-end with GoHandle-based backend

## Task 7: Tests + Validation

Run existing suites first, then write new tests for GoHandle lifecycle. All BEFORE cleanup (Task 8).

### Run existing test suites

- [ ] Full Python test suite: `pytest tests/ -v`
- [ ] Full Go test suite: `cd orionclient && go test ./... -v`
- [ ] WASM demo builds

### Write new tests: `tests/test_gohandle.py`

New test file covering GoHandle lifecycle — the behavior that caused all 6 handle bugs. Each item below is a test function:

- [ ] `test_gohandle_raw_returns_value`: `GoHandle(42).raw == 42`
- [ ] `test_gohandle_close_is_idempotent`: `h.close(); h.close()` — no error on second call
- [ ] `test_gohandle_raw_after_close_raises`: `h.close(); h.raw` → `RuntimeError("Use of closed handle")`
- [ ] `test_gohandle_bool_false_after_close`: `h.close(); assert not h`
- [ ] `test_client_two_step_close`: `client.close()` zeros SK then deletes handle; `client.close()` again is no-op
- [ ] `test_client_context_manager`: `with Client(params) as c: ...` calls `close()` on exit
- [ ] `test_evaluator_close_clears_tracked_handles`: after `evaluator.close()`, all `_tracked_handles` have `_raw == 0`
- [ ] `test_evaluator_double_close`: `evaluator.close(); evaluator.close()` — no error
- [ ] `test_evaluator_partial_init_failure`: mock a corrupt LT blob, verify `__init__` raises and all already-allocated handles are freed
- [ ] `test_multi_instance_independence`: two Clients with different params coexist, operate independently
- [ ] `test_ciphertext_wire_format_roundtrip`: `Ciphertext.from_bytes(ct.to_bytes())` produces same decrypted values
- [ ] `test_error_propagation`: trigger a Go panic (e.g. invalid params), verify Python gets `RuntimeError`, not process crash

### Acceptance criteria

- `pytest tests/` — zero failures (including new `test_gohandle.py`)
- `go test ./...` in `orionclient/` — zero failures
- All 12 new test functions pass

## Task 8: Cleanup

### Steps

- [ ] Remove `_EvalContext.delete_transforms()` (evaluator.py lines 170–173) — dead code after Task 5 removed its only caller (`LinearTransform.__del__`)
- [ ] Update `CLAUDE.md` architecture section to document GoHandle convention
- [ ] Search for stale imports (`from orion.backend.python`, `from orion.backend.lattigo`) and remove if found
- [ ] Search for raw `ffi.delete_handle` calls outside `GoHandle.close()` and remove if found

### Acceptance criteria

- `pytest tests/` still passes after cleanup
- `CLAUDE.md` documents GoHandle, Rules 1–5, and two-step close pattern

## Out of Scope

- Go library changes (correct as-is)
- FFI bridge changes (correct as-is)
- Compiler refactoring (stays in Python)
- ONNX export (future)
- Eliminating skeleton network (requires FX graph in CompiledModel)
- `_EvalContext` self-aliasing cleanup (works, avoid scope creep)
- Triple-redundant parameter types CKKSParams/CKKSParameters/Params (architectural debt, not a bug)
- Shape serialized twice in ciphertext wire format (low impact)
- Global minimax cache (low risk, read-only after init)
