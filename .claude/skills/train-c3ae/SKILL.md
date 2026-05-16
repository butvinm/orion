# Train C3AE on GPU VPS

Rent a GPU VPS, train both C3AE variants (`relu` and `fhe`) on UTKFace, run
cleartext FPR/FNR evaluation, capture artifacts locally, and tear down.

Reference: `docs/plans/completed/2026-05-09-c3ae-vps-runs.md` (Phase 1).

**Arguments:** `$ARGUMENTS`

Optional:

- `--flavor FLAVOR` — VPS flavor override (default `rtx4090-1.8.16.40`).
- `--keep` — skip the teardown step (leaves VPS billing — only for debugging).

## Configuration

- VPS name: `orion-c3ae-train` (hardcoded — project-prefixed per `/vps` skill rule).
- Default flavor: `rtx4090-1.8.16.40` (1×RTX 4090, 8 vCPU, 16 GB RAM, 40 GB disk).
- Image: auto-selected by `/vps create` for `rtx*` flavors (`Ubuntu 22.04 CUDA 13.2 (Apr 2026) [BIOS]`).
- Provisioning script: `docs/plans/2026-05-09-c3ae-vps-runs/setup-train.sh` (already committed; do not duplicate into the skill dir).
- Expected wall clock: ~5 min bootstrap + ~7 min train+eval + ~30 s capture = **~13 min total**.

## Preflight — fail fast before spending money

Run all of these before `vps create`. If any fails, stop and surface the failure.

1. **OpenStack auth healthy** — `openstack --os-cloud immers token issue` must return a token. HTTP 401 means the immers.cloud balance is empty; tell the user to top up.
2. **Local weight collision** — refuse to start if any of these exist locally:
   - `/home/butvinm/Dev/orion/examples/c3ae-demo/out/weights_relu.pth`
   - `/home/butvinm/Dev/orion/examples/c3ae-demo/out/weights_fhe.pth`
   - `/home/butvinm/Dev/orion/examples/c3ae-demo/results/cleartext.csv`

   These would be overwritten by the capture step. Ask the user to move or delete them. Do **not** auto-rename.

3. **Branch pushed to origin** — `setup-train.sh` clones from `https://github.com/butvinm/orion.git` and checks out `experiments`. Verify the current branch is pushed (`git rev-parse @{u}` must succeed and `git status -sb` must not show "ahead"). If not, tell the user to `git push -u origin <branch>` first. If the working branch is something other than `experiments`, edit `setup-train.sh`'s `git checkout` line **on the VPS only**, never locally.
4. **Existing VPS check** — if `openstack --os-cloud immers server show orion-c3ae-train` succeeds, the VPS already exists. Show the user and ask what to do (resume, delete-and-recreate, abort).

## Steps

### 1. Rent

```sh
openstack --os-cloud immers server create \
    --flavor "${FLAVOR:-rtx4090-1.8.16.40}" \
    --image "Ubuntu 22.04 CUDA 13.2 (Apr 2026) [BIOS]" \
    --network immers --key-name butvinm --wait \
    orion-c3ae-train -f json
```

Capture the IP from `.addresses.immers[0]`. Record rent-start ISO timestamp (cost log).

### 2. Wait for SSH

Cloud-init still running for ~30–60 s after `--wait` returns. Poll:

```sh
IP=<from step 1>
until ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=5 ubuntu@$IP true 2>/dev/null; do sleep 5; done
```

### 3. Bootstrap

```sh
scp /home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-train.sh \
    ubuntu@$IP:~/
ssh -o ServerAliveInterval=60 ubuntu@$IP 'bash setup-train.sh'
```

Expected: ~5 min. Final line must be `PROVISIONING DONE`. If `setup-train.sh` is non-idempotent in some edge case, re-run it — most steps short-circuit on existing artifacts.

**Verify:**

```sh
ssh ubuntu@$IP 'cd ~/orion && source .venv/bin/activate && \
    python -c "import torch, orion_compiler; print(torch.__version__, torch.cuda.is_available())" && \
    ls examples/c3ae-demo/data/UTKFace/ | head -3 && \
    go version'
```

Expected: torch `2.10.0+cu128` (or current), `cuda.is_available() == True`, three jpg filenames, `go version go1.24.0 linux/amd64`.

### 4. Train + eval (single ssh, blocks ~7 min)

The consolidated `scripts/run_cleartext.sh` trains both variants and runs `models.eval` in one shot. Block on it — train is too short to justify nohup/polling.

```sh
ssh -o ServerAliveInterval=60 ubuntu@$IP \
    'cd ~/orion/examples/c3ae-demo && source ../../.venv/bin/activate && \
     bash scripts/run_cleartext.sh'
```

If SSH disconnects mid-run (rare in ~7 min), fall back to the FHE skill's nohup-and-poll pattern. See `fhe-bench-c3ae/SKILL.md` step 5 for the polling block.

**Verify on the VPS:**

```sh
ssh ubuntu@$IP 'cd ~/orion/examples/c3ae-demo && \
    ls -la out/weights_relu.pth out/weights_fhe.pth && \
    wc -l results/cleartext.csv && head -5 results/cleartext.csv'
```

Expected: both weights ~136 kB, `cleartext.csv` has 5 lines (header + 4 rows: relu/overall, relu/boundary, fhe/overall, fhe/boundary).

### 5. Capture

```sh
mkdir -p /home/butvinm/Dev/orion/examples/c3ae-demo/out \
         /home/butvinm/Dev/orion/examples/c3ae-demo/results
scp ubuntu@$IP:'~/orion/examples/c3ae-demo/out/weights_relu.pth' \
       ubuntu@$IP:'~/orion/examples/c3ae-demo/out/weights_fhe.pth' \
    /home/butvinm/Dev/orion/examples/c3ae-demo/out/
scp ubuntu@$IP:'~/orion/examples/c3ae-demo/results/cleartext.csv' \
    /home/butvinm/Dev/orion/examples/c3ae-demo/results/
```

**Verify local:**

```sh
ls -la /home/butvinm/Dev/orion/examples/c3ae-demo/out/weights_*.pth
wc -l /home/butvinm/Dev/orion/examples/c3ae-demo/results/cleartext.csv
```

Both weights present (~136 kB), CSV has 5 lines.

### 6. Teardown (skipped if `--keep`)

```sh
openstack --os-cloud immers server delete orion-c3ae-train --wait
openstack --os-cloud immers server list | grep orion-c3ae-train  # must return nothing
```

Record rent-end ISO timestamp. Compute billed duration: `rent_end - rent_start`.

### 7. Report

Print a summary to the user:

- Billed duration (minutes), flavor used.
- `cleartext.csv` contents (cat it — 4 metric rows).
- Local artifact paths: `examples/c3ae-demo/out/weights_{relu,fhe}.pth`, `examples/c3ae-demo/results/cleartext.csv`.
- Next step suggestion: run `/fhe-bench-c3ae logn15` (or `logn16`) to consume `weights_fhe.pth`.

## Failure handling

- **Any ssh/scp failure**: **do not** auto-teardown. Print `ssh ubuntu@$IP` and the failing command, hand control to the user. They run `openstack --os-cloud immers server delete orion-c3ae-train --wait` when done.
- **`run_cleartext.sh` fails before producing weights**: skip step 5 (nothing to capture), keep VPS for inspection.
- **OpenStack 401 mid-run**: balance ran out. Tell the user; manual `openstack server delete` once balance is topped.
- **Cloud-init never finishes**: usually `apt` lock from unattended-upgrades. SSH in, `sudo cloud-init status --wait`, re-run setup-train.sh.

## Cost notes

- Plan baseline: `rtx4090-1.8.16.40` rented 2 h 14 min for training (much longer than the ~13 min strict need — left running for human inspection).
- Strict-minimum billed should be ~15 min if no manual intervention. Pricing comes from the immers.cloud console — record at rent time.
