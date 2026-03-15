# Orion v2 Library Issues

Issues encountered while building the C3AE demo as a library consumer.

## 1. No high-level Python class for bootstrap key generation

Bootstrap requires raw FFI calls and private handle access:

```python
from lattigo.ffi import new_bootstrap_params, bootstrap_params_gen_eval_keys, bootstrap_eval_keys_marshal
btp_params_h = new_bootstrap_params(params._h, logn=..., logp=..., h=192, log_slots=...)
_evk_h, btp_evk_h = bootstrap_params_gen_eval_keys(btp_params_h, sk._handle)
```

Every other operation (keygen, encode, encrypt, decrypt) has a clean class-based API.

**Fix:** Add `BootstrapParameters` class to `lattigo.ckks` wrapping these FFI calls.

## 2. Inconsistent handle access across Python classes

- `Parameters` exposes `._h` (a `@property`)
- `SecretKey` exposes `._handle` (a plain attribute)

Both are needed for bootstrap key generation. Accessor should be consistent.

**Fix:** Standardize on a single public accessor (e.g. `.handle` property on all classes).

## 3. `params_dict` from evaluator doesn't match `Parameters` constructor

`Model.client_params()` returns `logscale` but `Parameters.__init__` expects `log_default_scale`. `ring_type` is sometimes missing.

Every example has this boilerplate:

```python
pd = dict(params_dict)
if "logscale" in pd:
    pd["log_default_scale"] = pd.pop("logscale")
pd.setdefault("ring_type", "conjugate_invariant")
```

**Fix:** Either rename in the evaluator output, or add `Parameters.from_dict(params_dict)`.

## 4. TypeScript `.close()` methods missing from type declarations

All handle-owning JS classes (`CKKSParameters`, `SecretKey`, `Encoder`, etc.) have `.close()` at runtime but `tsc --noEmit` fails because `.d.ts` files don't declare it.

Affects both `wasm-demo` and `c3ae-demo` — `typecheck` npm script is broken for any consumer.

**Fix:** Add `close(): void` to all class declarations in `js/lattigo/src/*.ts`.

## 5. esbuild browser bundling broken — `url` not externalized

`js/lattigo/dist/index.js` contains `await import("url")` (Node.js built-in). Browser bundling fails unless `--external:url` is added to esbuild.

The existing `wasm-demo/client/package.json` also lacks this flag — broken on current main.

**Fix:** Either externalize `url` in the package's own build, or document the required esbuild flag.

## 6. No `Parameters.from_dict()` convenience

The evaluator returns a dict, and every consumer must manually construct `Parameters` with field renames. A classmethod would eliminate repeated boilerplate.

## 7. Compiler holds all diagonals in memory — OOMs on real models

The v2 compiler stores all weight diagonals in memory to produce the `.orion` binary. For C3AE (a small CNN with ~31K params), this uses **125 GB peak RSS** during compilation. The 64 GB VPS was OOM-killed.

The old Orion streamed diagonals to HDF5 on disk. The v2 format requires them all in memory.

**Measured:** C3AE compilation produced 46,304+ diagonals, each 8192 float64 values. The `.orion` file is 5.5 GB. Compilation alone needs 128 GB RAM.

**Impact:** Cannot compile + keygen + infer in a single process. Must compile separately, then load the `.orion` file for inference. Even compilation itself requires a beefy machine.

**Fix:** Stream diagonals to disk during compilation instead of accumulating in memory, or support chunked `.orion` format.

## 8. Go evaluator deserializes all diagonals into memory — 23x blowup

Loading a 5.5 GB `.orion` file into the Go evaluator allocates **125 GB of Go heap**. The 23x memory blowup comes from deserializing packed float64 diagonal data into Lattigo's internal ring polynomial structures.

**Measured:** `evaluator.LoadModel()` on the C3AE model takes 8m48s and 125 GB heap. Adding keygen + eval keys on top would exceed 128 GB.

**Impact:** The C3AE model (a small CNN with ~31K params) cannot run FHE inference on a 128 GB machine. The old Orion loaded diagonals on-demand from HDF5.

**Fix:** Lazy-load diagonals during inference (memory-map from `.orion` file) instead of deserializing all at once. Or keep diagonals as raw float64 arrays and convert to ring polynomials only when needed per-operation.
