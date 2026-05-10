#!/usr/bin/env bash
# Provisioning script for the C3AE FHE benchmark VPS (orion-c3ae-fhe-logn15 / -logn16).
# Run as ubuntu user on a fresh Ubuntu 22.04 VPS from immers.cloud (cpu.16.128.240 or larger).
set -euxo pipefail

# Wait for cloud-init to finish before touching apt
sudo cloud-init status --wait || true

# Build deps
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common ca-certificates curl gnupg
# Ubuntu 22.04 ships python3.10; we need 3.12 — pull from deadsnakes PPA
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential libgmp-dev libssl-dev pkg-config \
    python3.12 python3.12-venv python3.12-dev \
    git curl jq

# Go 1.24+ from upstream (Ubuntu 22.04 has 1.18 by default)
if ! /usr/local/go/bin/go version 2>/dev/null | grep -q 'go1.24\|go1.25\|go1.26'; then
    curl -sSLo /tmp/go.tar.gz https://go.dev/dl/go1.24.0.linux-amd64.tar.gz
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf /tmp/go.tar.gz
    rm /tmp/go.tar.gz
fi
echo 'export PATH=$PATH:/usr/local/go/bin' | sudo tee /etc/profile.d/go.sh > /dev/null
export PATH=$PATH:/usr/local/go/bin

# uv
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

# Repo
if [ ! -d ~/orion ]; then
    git clone https://github.com/butvinm/orion.git ~/orion
fi
cd ~/orion
git fetch origin
git checkout experiments
git pull --ff-only origin experiments

# Lattigo CGO shared lib (required for orion_compiler imports)
python3.12 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install --upgrade pip
python tools/build_lattigo.py

# uv sync — installs all workspace packages
uv sync

# UTKFace dataset via kagglehub (installed by `uv sync` as a workspace
# dev-dependency).  Needed by models/prepare_samples.py --boundary-band to
# dump the 3 boundary samples the FHE pipeline encrypts.
cd ~/orion/examples/c3ae-demo
python -m models.utkface

# Demo scripts (prepare_samples.py, train.py, eval.py) are run from the
# c3ae-demo dir and look for ./data/UTKFace there.
ls -la data/UTKFace/ 2>&1 | head -2

echo 'PROVISIONING DONE'
