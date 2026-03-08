# VGG16 — CIFAR-10 FHE Inference Example

## Architecture

13 Conv2d blocks with BatchNorm2d and ReLU activations (minimax sign approximation, degrees=[15,15,27]), 5 AvgPool2d downsampling layers, then a single FC layer. Input: 3x32x32 (CIFAR-10).

Ported from `models/vgg.py` (legacy `orion.nn` API) to `orion_compiler.nn`. Uses ReLU activation approximated via minimax sign polynomials — 13 ReLU activations consume many multiplicative levels, triggering many bootstrap operations.

## CKKS Parameters

| Parameter | Value         |
| --------- | ------------- |
| logn      | 16            |
| logq      | [55, 40 x 20] |
| logp      | [61, 61, 61]  |
| logscale  | 40            |
| h         | 192           |
| ring_type | standard      |
| boot_logp | [61 x 6]      |
| btp_logn  | 16            |

## Memory Warning

VGG16 compilation at logn=16 OOMs on machines with less than 64 GB RAM. Diagonal packing of 13 conv layers with up to 512 channels produces multi-GB weight matrices. Both compilation and FHE E2E require 64+ GB RAM.

- Bootstrap keys: ~5.8 GB (at logn=16)
- Full FHE E2E: requires 64+ GB RAM

Use `--cleartext-only` to verify model correctness without FHE on any machine.

## Usage

```bash
cd examples/vgg

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
