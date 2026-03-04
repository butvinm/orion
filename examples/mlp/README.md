# MLP — MNIST FHE Inference Example

## Architecture

Flatten → Linear(784, 128) → BatchNorm1d → Quad → Linear(128, 128) → BatchNorm1d → Quad → Linear(128, 10)

A 3-layer MLP with Quad (x²) activations — FHE-compatible alternative to ReLU. BatchNorm layers fuse into the preceding linear layers during compilation.

## CKKS Parameters

| Parameter | Value                                    |
| --------- | ---------------------------------------- |
| logn      | 13                                       |
| logq      | [29, 26, 26, 26, 26, 26, 26, 26, 26, 26] |
| logp      | [29, 29]                                 |
| logscale  | 26                                       |
| h         | 8192                                     |
| ring_type | conjugate_invariant                      |

The modulus chain provides 9 levels, enough for the full multiplication chain: 3 linear layers + 2 Quad activations + BatchNorm fusions.

## Usage

```bash
cd examples/mlp

# Optional: train the model (saves weights.pt)
python train.py

# Run FHE inference pipeline (works with random or trained weights)
python run.py
```

## Expected Output

- MAE < 0.1 (cleartext vs FHE output difference)
- With trained weights, the model achieves ~97% accuracy on MNIST in cleartext
