# Experiment 03: Library Modifications

All changes below are experimental and should be reverted before Phase 2 production implementation.

## Go Backend Changes

### `orion/backend/lattigo/scheme.go`

- **Added `GetGaloisElement(rotation)`**: New Go export that calls `scheme.Params.GaloisElement(int(rotation))` and returns the Galois element (uint64) for a given rotation amount. Needed so Python can compute Galois elements for power-of-2 rotations and hybrid output rotations without reimplementing Lattigo's ring-specific formula.

- **Added `GetMaxSlots()`**: New Go export that returns `scheme.Params.MaxSlots()`. Needed to compute the range of power-of-2 rotation keys without hardcoding slot counts.

## Python Backend Changes

### `orion/backend/lattigo/bindings.py`

- **Added `GetGaloisElement` binding**: FFI binding for the new Go `GetGaloisElement` export. Takes `c_int` rotation, returns `c_ulong` galois element.

- **Added `GetMaxSlots` binding**: FFI binding for the new Go `GetMaxSlots` export. Takes no args, returns `c_int`.

### `orion/backend/python/lt_evaluator.py`

- **Added `keyless` parameter to `NewEvaluator.__init__()`**: When `keyless=True`, skips `NewLinearTransformEvaluator()` (which requires EvalKeys/sk) and skips accessing `scheme.evaluator` (which doesn't exist in keyless mode).

- **Added `required_galois_elements` set**: Accumulates all Galois elements needed by linear transforms during keyless compilation.

- **Modified `generate_rotation_keys()`**: In keyless mode, appends Galois elements to `required_galois_elements` set instead of calling Go keygen functions. In normal mode, behavior is unchanged.

## Core Changes

### `orion/core/orion.py` (Scheme class)

- **Added `keyless` attribute**: Boolean flag on `Scheme.__init__()`, defaults to `False`.

- **Added `init_params_only()` method**: Alternative to `init_scheme()` that creates only `backend` + `encoder` + `lt_evaluator(keyless=True)`. Skips: keygen, encryptor, evaluator, poly_evaluator, bootstrapper. Sets `self.keyless = True`.

- **Modified `compile()` method**: When `self.keyless is True`:
  - Skips `bootstrapper.generate_bootstrapper()` calls, prints a message recording bootstrap slot counts instead.
  - After module compilation, calls `_build_key_manifest()` to collect all key requirements.
  - Returns `(input_level, manifest)` tuple instead of just `input_level`.

- **Added `_build_key_manifest()` method**: Collects all required Galois elements from three sources:
  1. Linear transform rotation keys (from `lt_evaluator.required_galois_elements`)
  2. Power-of-2 rotation keys (computed via `GetGaloisElement` for rotations 1, 2, 4, ..., MaxSlots/2)
  3. Hybrid method output rotations (from each `LinearTransform.output_rotations`, converted via `GetGaloisElement`)

- **Added `_get_po2_galois_elements()` method**: Computes power-of-2 Galois elements via Go backend.

- **Added `_rotation_to_galois_element()` method**: Delegates to Go `GetGaloisElement` for correct ring-specific computation.

### `orion/core/__init__.py`

- **Added `init_params_only` export**: Exposes `scheme.init_params_only` at the `orion.core` level.

### `orion/__init__.py`

- **Added `init_params_only` import**: Exposes at the top-level `orion` package for flat API access.
