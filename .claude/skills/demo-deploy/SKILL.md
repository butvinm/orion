# Demo Deploy (immers.cloud + thesis-vol)

Deploy and manage Orion FHE demos on immers.cloud VPS with a persistent volume.

**Arguments:** $ARGUMENTS

## Infrastructure

- Volume: `thesis-vol` (100 GB, persists across VPS rebuilds)
- Volume mount: `/data`
- Volume contents: `/data/orion` (source), `/data/venv` (Python 3.12 venv), `/data/c3ae-server` (Go binary)
- Default VPS name: `thesis-vps`
- Default flavor: `cpu.8.32.160` (8 vCPUs, 32 GB RAM — enough for C3AE FHE inference)
- SSH user: `ubuntu`

## Commands

Parse the first word of `$ARGUMENTS` as the command.

### `up [--flavor FLAVOR]`

Create VPS, attach volume, install tools, mount volume, start server.

Steps:

1. Use `/vps create --name thesis-vps --flavor FLAVOR` (default `cpu.8.32.160`)
2. Attach volume: `openstack --os-cloud immers server add volume thesis-vps thesis-vol`
3. Wait for SSH, then mount:
   ```bash
   sudo mkdir -p /data && sudo mount /dev/vdb /data && sudo chown ubuntu:ubuntu /data
   ```
4. Install Go + Python 3.12 if not present (these are on the ephemeral root disk, not volume):

   ```bash
   # Go
   wget -q https://go.dev/dl/go1.22.10.linux-amd64.tar.gz
   sudo tar -C /usr/local -xzf go1.22.10.linux-amd64.tar.gz && rm go1.22.10.linux-amd64.tar.gz
   echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

   # Python 3.12
   sudo add-apt-repository -y ppa:deadsnakes/ppa
   sudo apt-get update -qq
   sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential libgmp-dev libssl-dev pkg-config
   ```

5. Verify venv exists: `source /data/venv/bin/activate && python -c "from orion_evaluator import Evaluator"`
6. If venv broken or missing, recreate from `/data/orion`
7. Print the server IP and `ssh ubuntu@IP`

### `deploy [DEMO]`

Upload latest source and rebuild. DEMO defaults to `c3ae-demo`.

Steps:

1. Upload changed source files from the local working tree to `/data/orion/` via `scp`
   - Focus on Python packages, Go evaluator, and the specific demo directory
   - Do NOT use `git archive` (misses uncommitted changes) or full tar (overwrites volume data)
2. Rebuild Go bridges: `python tools/build_lattigo.py`
3. Reinstall Python packages: `pip install --force-reinstall --no-cache-dir --no-deps ./python/lattigo ./python/orion-compiler ./python/orion-evaluator`
4. For c3ae-demo: also rebuild the Go server and WASM client
5. Verify imports work

### `serve [DEMO]`

Start the demo server. DEMO defaults to `c3ae-demo`.

For c3ae-demo:

1. Ensure model.orion exists (compile if not: `python generate_model.py`)
2. Ensure weights.pth exists
3. Ensure dataset symlink exists
4. Build Go server if needed
5. Kill any existing server process
6. Start: `nohup /data/c3ae-server /data/orion/examples/c3ae-demo/model.orion /data/orion/examples/c3ae-demo/client :8080 > /tmp/server.log 2>&1 &`
7. Wait and verify with `curl http://localhost:8080/params`
8. Print: `http://IP:8080`

### `compile [DEMO]`

Compile the model. DEMO defaults to `c3ae-demo`.

For c3ae-demo:

1. Upload weights.pth if not on VPS (from `~/Dev/ITMO/thesis/experiments/c3ae-orion-experiment/results/c3ae_orion_s2.pth`)
2. Run: `python generate_model.py --weights weights.pth --output model.orion --stride 2`
3. Report compilation time and model size

### `bench [DEMO] [--samples N]`

Run FHE benchmark. DEMO defaults to `c3ae-demo`, N defaults to 2.

For c3ae-demo:

1. Ensure dataset exists (download with kagglehub if not)
2. Run: `python run_fhe.py --weights weights.pth --model model.orion --data-dir ./data/UTKFace --samples N`
3. Report results

### `down`

Delete VPS. Volume persists.

1. `openstack --os-cloud immers server delete thesis-vps --wait`
2. Confirm deletion
3. Remind user that `thesis-vol` is preserved

### `status`

Show current state:

1. Check if thesis-vps exists and its status
2. If running, show IP, check if server process is alive, show memory usage
3. Show volume status

### `logs`

Show server logs: `ssh ubuntu@IP "tail -50 /tmp/server.log"`

### `ssh`

Print SSH command for the running VPS.

## Important notes

- Always use `source /data/venv/bin/activate` before Python commands
- Always `export PATH=$PATH:/usr/local/go/bin` before Go commands
- Dataset needs re-downloading on each new VPS (stored in `~/.cache`, not on volume). Use kagglehub.
- Weights file may need re-uploading after volume data changes
- The Go server binary at `/data/c3ae-server` persists on the volume
- HTTP 401 from OpenStack means insufficient funds — tell user to top up
