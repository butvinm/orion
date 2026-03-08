# AlexNet — CIFAR-10 FHE Inference Example

## Architecture

Conv2d(3,64,3,p=1) → BN2d → SiLU(127) → AvgPool2d(2) → Conv2d(64,192,3,p=1) → BN2d → SiLU(127) → AvgPool2d(2) → Conv2d(192,384,3,p=1) → BN2d → SiLU(127) → Conv2d(384,256,3,p=1) → BN2d → SiLU(127) → Conv2d(256,256,3,p=1) → BN2d → SiLU(127) → AdaptiveAvgPool2d(2,2) → Flatten → Linear(1024,4096) → BN1d → SiLU(127) → Linear(4096,4096) → BN1d → SiLU(127) → Linear(4096,10)

Ported from `models/alexnet.py` (legacy `orion.nn` API) to `orion_compiler.nn`. Uses SiLU activation approximated via degree-127 Chebyshev polynomials — 7 SiLU activations consume many multiplicative levels, triggering 3 bootstrap operations.

## CKKS Parameters

| Parameter | Value         |
| --------- | ------------- |
| logn      | 15            |
| logq      | [55, 40 x 20] |
| logp      | [61, 61, 61]  |
| logscale  | 40            |
| h         | 192           |
| ring_type | standard      |
| boot_logp | [61 x 6]      |
| btp_logn  | 15            |

## Compilation Results

| Metric              | Value       |
| ------------------- | ----------- |
| Input level         | 20          |
| Bootstrap count     | 3           |
| Bootstrap slots     | 4096, 16384 |
| Galois elements     | 251         |
| Compiled model size | ~4.5 GB     |

## Usage

```bash
cd examples/alexnet

# Optional: train the model (saves weights.pt)
python train.py

# Cleartext forward pass only (verifies model correctness)
python run.py --cleartext-only

# Full FHE pipeline (requires 64+ GB RAM for bootstrap keys)
python run.py
```

## Memory Requirements

AlexNet at logn=15 with 3 bootstrap operations requires significant resources:

- Bootstrap keys: ~2.6 GB (at logn=15)
- Compiled model: ~4.5 GB
- Full FHE E2E: requires 64+ GB RAM

Full FHE E2E verification is deferred to a machine with sufficient memory. Use `--cleartext-only` to verify model correctness without FHE.

## Expected Output

- Cleartext forward pass produces 10-class logits for CIFAR-10
- With trained weights, the model achieves competitive accuracy on CIFAR-10
