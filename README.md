# Orion

## Installation

We tested our implementation on `Ubuntu 22.04.5 LTS`. First, install the required dependencies:

```
sudo apt update && sudo apt install -y \
    build-essential git wget curl ca-certificates \
    python3 python3-pip python3-venv \
    unzip pkg-config libgmp-dev libssl-dev
```

Install Go (for Lattigo backend):

```
cd /tmp
wget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
go version # go version go1.22.3 linux/amd64
```

### Install Orion

```bash
git clone https://github.com/baahl-nyu/orion.git
cd orion/

# Build the Go shared library
python tools/build_lattigo.py

# Install Python packages
cd python/lattigo && pip install -e .
cd python/orion-compiler && pip install -e .
cd python/orion-evaluator && pip install -e .
```

### Run the examples!

Each example is self-contained in `examples/<model>/` with `model.py`, `train.py`, `run.py`, and `README.md`.

```bash
cd examples/mlp && python run.py
cd examples/lenet && python run.py
cd examples/lola && python run.py
```

### Browser Demo

See [`examples/wasm-demo/`](examples/wasm-demo/) for a browser-based demo that performs encrypted MNIST inference. The client runs Lattigo compiled to WASM — the secret key never leaves the browser.
