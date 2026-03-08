# ResNet20 — CIFAR-10 FHE Inference Example

## Architecture

ResNet20 with BasicBlock, [3,3,3] blocks, [16,32,64] channels. Uses ReLU activation approximated via minimax sign polynomials (degrees=[15,15,27]). Residual connections via `on.Add`. Input: 3x32x32 (CIFAR-10).

Ported from `models/resnet.py` (legacy `orion.nn` API) to `orion_compiler.nn`. The 19 ReLU activations (1 stem + 18 in BasicBlocks) and residual branching DAG paths trigger ~38 bootstrap operations.

## CKKS Parameters

| Parameter | Value         |
| --------- | ------------- |
| logn      | 16            |
| logq      | [55, 40 x 10] |
| logp      | [61, 61, 61]  |
| logscale  | 40            |
| h         | 192           |
| ring_type | standard      |
| boot_logp | [61 x 8]      |
| btp_logn  | 16            |

## Memory Warning

ResNet20 at logn=16 with ~38 bootstrap operations OOMs on machines with less than 64 GB RAM. The bootstrapper generation alone (for logslots=14, 13, 12) exceeds available memory on 38 GB machines.

- Bootstrap keys: ~5.8 GB (at logn=16)
- Full FHE E2E: requires 64+ GB RAM

Use `--cleartext-only` to verify model correctness without FHE on any machine.

## Usage

```bash
cd examples/resnet

# Optional: train the model (saves weights.pt)
python train.py

# Cleartext forward pass only (verifies model correctness)
python run.py --cleartext-only

# Full FHE pipeline (requires 64+ GB RAM)
python run.py
```

## Expected Output

- Cleartext forward pass produces 10-class logits for CIFAR-10
- With trained weights, the model achieves competitive accuracy on CIFAR-10
- FHE E2E verification deferred to a machine with 64+ GB RAM
