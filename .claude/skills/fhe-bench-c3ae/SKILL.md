# FHE Bench C3AE on CPU VPS

Rent a CPU VPS, run the C3AE FHE benchmark for a single CKKS configuration
(`logn15` or `logn16`), capture metrics + correctness check locally, and tear
down.

Reference: `docs/plans/completed/2026-05-09-c3ae-vps-runs.md` (Phases 2 and 3).

**Arguments:** `$ARGUMENTS`

Required:

- `<config>` — first positional arg, one of `logn15` | `logn16`.

Optional:

- `--flavor FLAVOR` — VPS flavor override (default `cpu.16.128.240`).
- `--keep` — skip the teardown step (leaves VPS billing — only for debugging).

## Configuration

- VPS name: `orion-c3ae-fhe-<config>` (e.g. `orion-c3ae-fhe-logn15`).
- Default flavor: `cpu.16.128.240` (16 vCPU, 128 GB RAM, 240 GB disk). Plan-measured peaks: 54 GB (logn15), 114 GB (logn16). logn16 has only ~10 GB headroom on the default — override with `--flavor cpu.96.512.640` if you want margin.
- Image: `Ubuntu 22.04 (Apr 2026) [BIOS]` (auto-selected by `/vps create` for `cpu*` flavors).
- Provisioning script: `docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh` (committed; do not duplicate).
- Expected wall clock (per plan):
  - `logn15`: ~5 min bootstrap + ~11 min bench + ~30 s verify + ~30 s capture ≈ **~17 min**
  - `logn16`: ~5 min bootstrap + ~36 min bench + ~30 s verify + ~1 min capture ≈ **~42 min**

## Preflight — fail fast before spending money

1. **Config arg valid** — must be exactly `logn15` or `logn16` (matches `examples/c3ae-demo/models/params.py`).
2. **OpenStack auth healthy** — `openstack --os-cloud immers token issue`. HTTP 401 = top up balance.
3. **Local weights present** — `/home/butvinm/Dev/orion/examples/c3ae-demo/out/weights_fhe.pth` must exist. If not, tell the user to run `/train-c3ae` first.
4. **Local result collision** — refuse to start if any of these exist:
   - `/home/butvinm/Dev/orion/examples/c3ae-demo/results/<cfg>/run.jsonl`
   - `/home/butvinm/Dev/orion/examples/c3ae-demo/results/<cfg>/cleartext_vs_fhe.csv`
   - `/home/butvinm/Dev/orion/examples/c3ae-demo/out/<cfg>/compile.json`

   Ask the user to move/delete them. Do **not** auto-rename. (Plan Task 10 hit a dedup mess when `run.jsonl` got 4 rows from a racy retry — refuse + investigate is the right discipline.)

5. **Branch pushed to origin** — same check as `train-c3ae`: `setup-fhe.sh` clones from origin and `git checkout experiments`. `git status -sb` must not show "ahead". If working branch is not `experiments`, edit the script's `git checkout` line **on the VPS only**.
6. **Existing VPS check** — if `openstack server show orion-c3ae-fhe-<cfg>` succeeds, show and ask.

## Steps

### 1. Rent

```sh
CFG=<config>
FLAVOR="${FLAVOR:-cpu.16.128.240}"
openstack --os-cloud immers server create \
    --flavor "$FLAVOR" \
    --image "Ubuntu 22.04 (Apr 2026) [BIOS]" \
    --network immers --key-name butvinm --wait \
    "orion-c3ae-fhe-$CFG" -f json
```

Capture the IP. Record rent-start ISO timestamp.

### 2. Wait for SSH + clear stale host key

IPs recycle between immers.cloud rentals (plan Task 6/14 both got `195.209.214.105` reused). Always clear the host key:

```sh
IP=<from step 1>
ssh-keygen -R "$IP" 2>/dev/null
until ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=5 ubuntu@$IP true 2>/dev/null; do sleep 5; done
```

### 3. Bootstrap

```sh
scp /home/butvinm/Dev/orion/docs/plans/2026-05-09-c3ae-vps-runs/setup-fhe.sh \
    ubuntu@$IP:~/
ssh -o ServerAliveInterval=60 ubuntu@$IP 'bash setup-fhe.sh'
```

Expected ~5 min. Final line: `PROVISIONING DONE`.

**Verify:**

```sh
ssh ubuntu@$IP 'cd ~/orion && source .venv/bin/activate && \
    python -c "import torch, orion_compiler; print(\"ok\")" && \
    go version && free -h | head -2'
```

Expected: `ok`, `go1.24.0`, free shows total RAM ≈ flavor spec (e.g. ~125 GiB for `cpu.16.128.240`).

### 4. Upload weights + build bench binary

```sh
ssh ubuntu@$IP 'mkdir -p ~/orion/examples/c3ae-demo/out'
scp /home/butvinm/Dev/orion/examples/c3ae-demo/out/weights_fhe.pth \
    ubuntu@$IP:'~/orion/examples/c3ae-demo/out/'
ssh ubuntu@$IP 'cd ~/orion/examples/c3ae-demo/bench && go build && ls -la bench'
```

Expected: bench binary ~11 MB executable, weights file 136347 bytes.

### 5. Run bench (SSH-resilient nohup + 10-min polling)

The bench can take 10–40 min depending on config. Launch detached with `nohup` so SSH disconnects don't kill it, then poll from the local machine. **Do not** use `set -euxo pipefail` plus `ls | head` in any wrapper — SIGPIPE on `ls` aborts the script silently (lesson from `CLAUDE.md`).

**Launch:**

```sh
ssh ubuntu@$IP "mkdir -p ~/orion/examples/c3ae-demo/results/$CFG && \
    cd ~/orion/examples/c3ae-demo && source ../../.venv/bin/activate && \
    nohup bash scripts/run_fhe.sh $CFG \
        > results/$CFG/run_fhe.log 2>&1 < /dev/null &"
```

**Monitoring loop** — poll every 10 min, fail fast on budget overrun, emit a progress snapshot per iteration so the user sees a heartbeat:

```sh
START=$(date +%s)
case "$CFG" in
    logn15) MAX_WAIT=3600 ;;   # 1 h budget for ~11 min expected
    logn16) MAX_WAIT=7200 ;;   # 2 h budget for ~36 min expected
esac
POLL=600                       # 10 minutes
LOG=~/orion/examples/c3ae-demo/results/$CFG/run_fhe.log
JSONL=~/orion/examples/c3ae-demo/results/$CFG/run.jsonl

until ! ssh ubuntu@$IP "pgrep -f 'scripts/run_fhe.sh' >/dev/null" 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if [ $ELAPSED -gt $MAX_WAIT ]; then
        echo "TIMEOUT after ${ELAPSED}s (budget ${MAX_WAIT}s) — DO NOT teardown."
        echo "ssh ubuntu@$IP   # investigate: dmesg | tail, ps aux | grep bench, free -h"
        exit 1
    fi

    echo "=== [$(date -Is)] elapsed=${ELAPSED}s budget=${MAX_WAIT}s ==="
    ssh ubuntu@$IP "tail -3 $LOG 2>/dev/null; echo '---'; \
        echo \"jsonl_lines=\$(wc -l < $JSONL 2>/dev/null || echo 0)\"; \
        echo \"free_avail=\$(free -h | awk '/^Mem:/ {print \$7}')\""

    sleep $POLL
done
echo "=== bench completed in $(( $(date +%s) - START ))s ==="
```

**Why 10 min:** less SSH chatter, less context burned, and the bench's natural phases (compile → keygen → 3× inference) are minutes apart — 10-min polls catch each phase transition.

**Verify after exit:**

```sh
ssh ubuntu@$IP "cd ~/orion/examples/c3ae-demo && \
    wc -l results/$CFG/run.jsonl && \
    cat results/$CFG/run.jsonl && \
    cat out/$CFG/compile.json && cat out/$CFG/keys/keygen.json && \
    dmesg | tail -5"
```

Expected:

- `run.jsonl` has **exactly 3 lines** (one per boundary sample). If >3 lines, refuse to continue — investigate the cause; the plan saw 4-row races from concurrent partial runs. Do not silently `sort -u`.
- Each JSONL line has `sample_idx`, `forward_s`, `peak_rss_mb`, `result_ct_bytes`.
- `compile.json` has `compile_s`, `compile_peak_rss_mb`, `model_bytes`.
- `keygen.json` has `keygen_s`, `evk_bytes`.
- No `Out of memory` / `Killed process` in dmesg.

### 6. Correctness check

```sh
ssh ubuntu@$IP "cd ~/orion/examples/c3ae-demo && source ../../.venv/bin/activate && \
    python scripts/verify_fhe.py --config $CFG"
```

Exits 0 if all 3 samples have `abs_diff < 0.05` (default tolerance). If non-zero, **do not teardown** — preserve VPS for forensic ssh.

### 7. Capture

Pull results + tiny metric JSONs only. **Exclude** multi-GB key blobs and the model:

```sh
mkdir -p /home/butvinm/Dev/orion/examples/c3ae-demo/results/$CFG \
         /home/butvinm/Dev/orion/examples/c3ae-demo/out/$CFG/keys

rsync -av --exclude='*.bin' \
    ubuntu@$IP:"orion/examples/c3ae-demo/results/$CFG/" \
    /home/butvinm/Dev/orion/examples/c3ae-demo/results/$CFG/

rsync -av --exclude='*.bin' --exclude='model.orion' \
    ubuntu@$IP:"orion/examples/c3ae-demo/out/$CFG/" \
    /home/butvinm/Dev/orion/examples/c3ae-demo/out/$CFG/
```

**Verify local:**

```sh
cat /home/butvinm/Dev/orion/examples/c3ae-demo/results/$CFG/run.jsonl
cat /home/butvinm/Dev/orion/examples/c3ae-demo/results/$CFG/cleartext_vs_fhe.csv
cat /home/butvinm/Dev/orion/examples/c3ae-demo/out/$CFG/compile.json
cat /home/butvinm/Dev/orion/examples/c3ae-demo/out/$CFG/keys/keygen.json
```

All four present and well-formed.

### 8. Teardown (skipped if `--keep`)

```sh
openstack --os-cloud immers server delete "orion-c3ae-fhe-$CFG" --wait
openstack --os-cloud immers server list | grep "orion-c3ae-fhe-$CFG"   # must return nothing
```

Record rent-end ISO timestamp.

### 9. Report

Print to the user:

- Billed duration, flavor used.
- Bench summary: max `peak_rss_mb` (vs flavor RAM), mean and stddev `forward_s` across 3 samples, `compile_s`, `keygen_s`, `evk_bytes`.
- Correctness summary: max `abs_diff` from `cleartext_vs_fhe.csv` (saturated-sigmoid boundary samples typically read 0.000 to 6 decimals).
- OOM check: paste relevant `dmesg` tail line(s) or confirm "no OOM".
- Suggest re-running `python scripts/build_results.py` locally to regenerate `examples/c3ae-demo/results/results.md` if both `logn15` and `logn16` results are present. Do **not** auto-commit results.

## Failure handling

- **Any ssh/scp failure**: do not auto-teardown. Print `ssh ubuntu@$IP` and the failing command. The user runs `openstack server delete orion-c3ae-fhe-$CFG --wait` when done debugging.
- **Bench fails / OOM-killed**: dmesg will show `Out of memory: Killed process`. Do not teardown. If the user wants to retry on a larger flavor, the existing VPS is useless — teardown manually + invoke skill again with `--flavor cpu.96.512.640`.
- **verify_fhe.py exits nonzero**: keep VPS for inspection — comparing slot-level outputs requires the original `model.orion` and keys on disk.
- **`run.jsonl` has wrong row count**: do not auto-dedup. Surface the lines to the user.
- **Cloud-init hangs at apt lock**: SSH in, `sudo cloud-init status --wait`, re-run `bash setup-fhe.sh`.

## Cost notes

- Plan baselines: `cpu.16.128.240` rented 38 min for logn15, 56 min for logn16. Strict-minimum should be ~18 min (logn15) / ~45 min (logn16) if there's no manual intervention.
- If `cpu.16.128.240` is too small for `logn16` and 256 GB flavors are still unavailable on immers.cloud, the next step up is `cpu.96.512.640` (or `cpu.92.512.640`). Substantially more expensive per hour.
