# LoLA — MNIST FHE Inference Example

## Architecture

Conv2d(1, 5, 2, s=2) → BN2d → Quad → Flatten → Linear(980, 100) → BN1d → Quad → Linear(100, 10)

LoLA ("Low-Latency") — a single-conv architecture from the original orion repo. Uses BatchNorm layers that fuse into the preceding linear/conv layers during compilation.

## CKKS Parameters

| Parameter | Value                                    |
| --------- | ---------------------------------------- |
| logn      | 13                                       |
| logq      | [29, 26, 26, 26, 26, 26, 26, 26, 26, 26] |
| logp      | [29, 29]                                 |
| logscale  | 26                                       |
| h         | 8192                                     |
| ring_type | conjugate_invariant                      |

Same modulus chain as MLP and LeNet (9 levels). LoLA's shallow depth (1 Conv + 2 Quad + 2 Linear) fits comfortably within this budget with no bootstrapping needed.

## Usage

```bash
cd examples/lola

# Optional: train the model (saves weights.pt)
python train.py

# Run FHE inference pipeline (works with random or trained weights)
python run.py
```

## Expected Output

- MAE < 0.1 (cleartext vs FHE output difference)
- With trained weights, the model achieves ~98% accuracy on MNIST in cleartext
