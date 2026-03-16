# Orion v2 Library Issues

Issues encountered while building the C3AE demo as a library consumer.

## 1. Compiler holds all diagonals in memory — OOMs on real models

The v2 compiler stores all weight diagonals in memory to produce the `.orion` binary. For C3AE (a small CNN with ~31K params), this uses **103 GB peak RSS** during compilation. The 64 GB VPS was OOM-killed.

The old Orion streamed diagonals to HDF5 on disk. The v2 format requires them all in memory.

**Measured:** C3AE compilation produced 46,304+ diagonals, each 8192 float64 values. The `.orion` file is 5.5 GB. Compilation alone needs 128 GB RAM.

**Impact:** Cannot compile + keygen + infer in a single process. Must compile separately, then load the `.orion` file for inference. Even compilation itself requires a beefy machine.

**Fix:** Stream diagonals to disk during compilation instead of accumulating in memory, or support chunked `.orion` format.

## 2. Go evaluator deserializes all diagonals into memory — 23x blowup

Loading a 5.5 GB `.orion` file into the Go evaluator allocates **125 GB of Go heap**. The 23x memory blowup comes from deserializing packed float64 diagonal data into Lattigo's internal ring polynomial structures.

**Measured:** `evaluator.LoadModel()` on the C3AE model takes 8m48s and 125 GB heap. Adding keygen + eval keys on top would exceed 128 GB.

**Impact:** The C3AE model (a small CNN with ~31K params) cannot run FHE inference on a 128 GB machine. The old Orion loaded diagonals on-demand from HDF5.

**Fix:** Lazy-load diagonals during inference (memory-map from `.orion` file) instead of deserializing all at once. Or keep diagonals as raw float64 arrays and convert to ring polynomials only when needed per-operation.

## Resolved

- ~~No high-level Python class for bootstrap key generation~~ — Fixed: `BootstrapParams` class added to `lattigo.rlwe`.
- ~~Inconsistent handle access across Python classes~~ — Fixed: all classes now use `._handle` consistently.
- ~~`params_dict` naming mismatch~~ — Fixed: `CKKSParams` now uses `log_default_scale`, `Parameters.from_dict()` added.
- ~~TypeScript `.close()` missing from declarations~~ — Fixed: `.d.ts` now includes `close(): void` on all handle-owning classes.
- ~~No `Parameters.from_dict()` convenience~~ — Fixed: `Parameters.from_dict(params_dict)` added.
- ~~esbuild browser bundling broken (`url`/`fs`/`path` not externalized)~~ — Fixed: loader uses computed dynamic import for `fs` in Node.js branch, invisible to bundlers. No `--external` flags needed.
