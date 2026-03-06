# Phase 6: Stream evaluation keys in WASM demo

## Overview

The WASM demo currently assembles all Galois keys into a single `MemEvaluationKeySet`, marshals the entire blob, and uploads it in one HTTP POST. For models with many Galois elements (e.g., LeNet's 178 keys ~ 1.3 GB), this creates a large single transfer and high peak memory in the browser.

Change the WASM demo to upload keys individually in a generate-marshal-upload-free loop, so only one key (~7 MB for logN=14) is in browser memory at a time.

**Problem solved:** Reduces browser peak memory from O(all keys) to O(1 key). Enables larger models in the browser demo.

## Context (from discovery)

- **WASM bridge** (`js/lattigo/bridge/serialize.go`, `js/lattigo/src/rlwe.ts`): Already exposes `GaloisKey.marshalBinary()`, `RelinearizationKey.marshalBinary()`, and their unmarshal counterparts. Section 6.1 is already complete.
- **Server** (`examples/wasm-demo/server/main.go`): Currently has `POST /session` (accepts full `MemEvaluationKeySet` blob), `POST /session/{id}/infer`, `GET /params`. Sessions store per-session `Evaluator` instances with mutex for forward calls.
- **Client** (`examples/wasm-demo/client/client.ts`): Generates all Galois keys in a loop (lines 204-221 with per-key progress), then assembles into `MemEvaluationKeySet`, marshals entire blob, POSTs to `/session`.
- **Key manifest**: `Model.ClientParams()` returns `galois_elements: number[]`, `needs_rlk: boolean`, `bootstrap_slots: number[]`.
- **Dependencies**: No external dependencies needed. All Lattigo primitives already available.

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**

## Testing Strategy

- **Unit tests**: Go tests for server session management, key accumulation, finalize validation
- **Integration tests**: Happy path end-to-end, missing key on finalize, duplicate key upload, session timeout
- **Manual verification**: Browser demo works end-to-end for MLP inference

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with + prefix
- Document issues/blockers with ! prefix
- Update plan if implementation deviates from original scope

## Implementation Steps

### Task 1: Server session state machine and key storage

Refactor `examples/wasm-demo/server/main.go` to support incremental key upload with session states.

- [x] Cache key manifest on `Server` struct at construction time (like `ckksParams` is already cached) — needed for finalize validation
- [x] Add `maxSingleKeyBytes` constant (~16 MB) for individual key upload endpoints (separate from `maxKeySetBytes`)
- [x] Add session state enum: `pending` (keys being uploaded), `ready` (evaluator created)
- [x] Add per-session key storage: `rlk *rlwe.RelinearizationKey`, `galoisKeys map[uint64]*rlwe.GaloisKey`, `lastActivity time.Time`
- [x] Change `POST /session` to accept no body — creates a pending session, returns `{"session_id": "..."}`
- [x] Add `POST /session/{id}/keys/relin` — reads body bytes (limit `maxSingleKeyBytes`), unmarshals `RelinearizationKey`, stores in session. Reject with 409 if session is already `ready`.
- [x] Add `POST /session/{id}/keys/galois/{element}` — parse `{element}` as uint64, reads body bytes, unmarshals `GaloisKey`, stores by element (idempotent: re-upload overwrites). Reject with 409 if session is already `ready`.
- [x] Add `POST /session/{id}/keys/finalize` — validates all required Galois elements present (against cached manifest), RLK present if `needs_rlk`. Assembles `rlwe.NewMemEvaluationKeySet(rlk, galKeys...)`, creates `Evaluator`, transitions to `ready`. Returns 400 with missing elements list if incomplete.
- [x] Register all new routes in `Handler()` method
- [x] Update `POST /session/{id}/infer` to reject requests unless session is `ready`
- [x] Manually test with `curl`: session creation, key upload to ready session (expect 409), finalize with missing keys (expect 400)
- [x] Run `go build ./examples/wasm-demo/server/... && go vet ./examples/wasm-demo/server/...` — must pass

### Task 2: Session cleanup goroutine

- [x] Add cleanup goroutine that sweeps pending sessions with no key upload for 5 minutes (check `lastActivity`)
- [x] Update `lastActivity` on every key upload and finalize call
- [x] Ready sessions are NOT cleaned up (persist until server shutdown, same as current behavior)
- [x] Write Go test for session timeout cleanup (use short timeout for test)
- [x] Run `go test ./examples/wasm-demo/server/...` — must pass

### Task 3: Client streaming loop

Update `examples/wasm-demo/client/client.ts` to use generate-marshal-upload-free loop.

- [x] After `POST /session` (no body), store `session_id`
- [x] If `manifest.needs_rlk`, generate RLK, marshal, POST to `/session/{id}/keys/relin`, free RLK handle
- [x] Replace Galois key batch with streaming loop: for each element in `manifest.galois_elements`, generate key, marshal, POST to `/session/{id}/keys/galois/{element}`, free key handle, update progress. Abort on any upload error (show error message, re-enable init button).
- [x] Call `POST /session/{id}/keys/finalize` after all keys uploaded
- [x] Remove `MemEvaluationKeySet` assembly for standard keys (keep import — needed for future bootstrap support)
- [x] Verify per-key progress display works (count, percentage, elapsed time)
- [x] Note in code comment: bootstrap path (lines 224-277) is temporarily non-functional since the old single-upload `POST /session` is removed. No code change needed — the block never executes (bootstrap_slots always empty for current models).
- [x] Run `cd js/lattigo && npm run typecheck && npm run lint` — must pass

### Task 4: Verify acceptance criteria

- [ ] Verify WASM bridge exposes `GaloisKey.marshalBinary()` and `RelinearizationKey.marshalBinary()` (already done)
- [ ] Verify `POST /session` creates pending session
- [ ] Verify `POST /session/{id}/keys/relin` accepts and stores RLK
- [ ] Verify `POST /session/{id}/keys/galois/{element}` accepts and stores individual Galois keys (idempotent)
- [ ] Verify `POST /session/{id}/keys/finalize` validates completeness, returns 400 with missing elements if incomplete
- [ ] Verify client uses generate-marshal-upload-free loop (one key in memory at a time)
- [ ] Verify client shows per-key upload progress
- [ ] Verify pending sessions cleaned up after 5-minute inactivity timeout
- [ ] Verify `POST /session/{id}/infer` rejects non-finalized sessions
- [ ] Run full Go test suite: `go test ./...`
- [ ] Run JS lint/typecheck: `cd js/lattigo && npm run typecheck && npm run lint`

### Task 5: [Final] Update documentation

- [ ] Document bootstrap key limitation (6.4): current demo uses no bootstrap, Phase 7 will need blob upload endpoint
- [ ] Update `examples/wasm-demo/README.md` if it documents the key upload flow

## Technical Details

**Session state machine:**

- `pending` -> (keys uploaded) -> `POST /keys/finalize` -> `ready`
- `pending` -> (5 min inactivity) -> deleted
- `ready` -> persists until server shutdown

**New server endpoints:**

| Method | Path                                  | Body             | Response                     |
| ------ | ------------------------------------- | ---------------- | ---------------------------- |
| POST   | `/session`                            | (none)           | `{"session_id": "..."}`      |
| POST   | `/session/{id}/keys/relin`            | RLK bytes        | 200 OK                       |
| POST   | `/session/{id}/keys/galois/{element}` | GaloisKey bytes  | 200 OK                       |
| POST   | `/session/{id}/keys/finalize`         | (none)           | 200 OK or 400 + missing list |
| POST   | `/session/{id}/infer`                 | ciphertext bytes | result ciphertext bytes      |

**Memory profile (browser):**

- Before: All Galois keys + MemEvaluationKeySet blob in memory simultaneously
- After: At most 1 GaloisKey (~7 MB for logN=14) in memory at any time

## Post-Completion

**Manual verification:**

- Run WASM demo end-to-end in browser with MLP model
- Verify browser memory stays low during key upload (DevTools Memory tab)
- Test with LeNet model (178 Galois keys) if available

**Future work (Phase 7):**

- Bootstrap evaluation keys are generated as a single `MemEvaluationKeySet` — will need additional endpoint or blob upload path
