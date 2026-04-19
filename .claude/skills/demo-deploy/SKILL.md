# Demo Deploy (immers.cloud)

Deploy and manage Orion FHE demos on immers.cloud VPS.

**Arguments:** $ARGUMENTS

## Infrastructure

- Default VPS name: `thesis-vps`
- Default flavor: `cpu.16.128.240` (16 vCPUs, 128 GB RAM — required for logN=15 FHE inference)
- SSH user: `ubuntu`
- All data lives on root disk: `~/orion` (source), `~/venv` (Python 3.12 venv), `~/c3ae-server` (Go binary)
- VPS deletion loses all data — full redeploy required on new VPS

## Commands

Parse the first word of `$ARGUMENTS` as the command.

### `up [--flavor FLAVOR]`

Create VPS, install tools, deploy source.

Steps:

1. Use `/vps create --name thesis-vps --flavor FLAVOR` (default `cpu.16.128.240`)
2. Wait for SSH
3. Install Go + Python 3.12:

   ```bash
   # Go
   wget -q https://go.dev/dl/go1.22.10.linux-amd64.tar.gz
   sudo tar -C /usr/local -xzf go1.22.10.linux-amd64.tar.gz && rm go1.22.10.linux-amd64.tar.gz
   echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

   # Python 3.12
   sudo add-apt-repository -y ppa:deadsnakes/ppa
   sudo apt-get update -qq
   sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential libgmp-dev libssl-dev pkg-config screen
   ```

4. Upload source via rsync to `~/orion/`
5. Create venv, build bridges, install packages
6. Print the server IP and `ssh ubuntu@IP`

### `deploy [DEMO]`

Upload latest source and rebuild. DEMO defaults to `c3ae-demo`.

Steps:

1. Upload changed source files from the local working tree to `~/orion/` via rsync
2. Rebuild Go bridges: `python tools/build_lattigo.py`
3. Reinstall Python packages: `pip install --force-reinstall --no-cache-dir --no-deps ./python/lattigo ./python/orion-compiler ./python/orion-evaluator`
4. For c3ae-demo: also rebuild the Go server and client
5. Verify imports work

### `serve [DEMO]`

Start the demo server. DEMO defaults to `c3ae-demo`.

For c3ae-demo:

1. Ensure model.orion exists (compile if not: `python generate_model.py`)
2. Ensure weights.pth exists
3. Build Go server if needed
4. Kill any existing server process
5. Start: `nohup ~/c3ae-server ~/orion/examples/c3ae-demo/model.orion ~/orion/examples/c3ae-demo/client :8080 > /tmp/server.log 2>&1 &`
6. Wait and verify with `curl http://localhost:8080/params`
7. Print: `http://IP:8080`

### `compile [DEMO]`

Compile the model. DEMO defaults to `c3ae-demo`.

For c3ae-demo:

1. Upload weights.pth if not on VPS (local source: `examples/c3ae-demo/weights.pth`, gitignored)
2. Run: `python generate_model.py --weights weights.pth --output model.orion --stride 2`
3. Report compilation time and model size

### `bench [DEMO] [--samples N]`

Run FHE benchmark. DEMO defaults to `c3ae-demo`, N defaults to 2.

For c3ae-demo:

1. Ensure dataset exists (download with kagglehub if not)
2. Run: `python run_fhe.py --weights weights.pth --model model.orion --data-dir ./data/UTKFace --samples N`
3. Report results

### `down`

Delete VPS.

1. `openstack --os-cloud immers server delete thesis-vps --wait`
2. Confirm deletion

### `status`

Show current state:

1. Check if thesis-vps exists and its status
2. If running, show IP, check if server process is alive, show memory usage

### `logs`

Show server logs: `ssh ubuntu@IP "tail -50 /tmp/server.log"`

### `ssh`

Print SSH command for the running VPS.

## Important notes

- Always use `source ~/venv/bin/activate` before Python commands
- Always `export PATH=$PATH:/usr/local/go/bin` before Go commands
- Dataset needs downloading on each new VPS (use kagglehub)
- Weights file needs uploading on each new VPS
- HTTP 401 from OpenStack means insufficient funds — tell user to top up
