# LoLA — MNIST FHE Inference Example

## Architecture

Conv2d(1, 32, 5, s=2, p=2) → BatchNorm2d → Quad → Flatten → Linear(6272, 100) → BatchNorm1d → Quad → Linear(100, 10)

A single-conv architecture inspired by LoLA ("Low-Latency"), using Quad (x²) activations — FHE-compatible alternative to ReLU. BatchNorm layers fuse into the preceding linear/conv layers during compilation.

The original LoLA used Conv2d(1, 5, k=2, s=2, p=0) but small channel counts trigger a systematic packing error in the compiler. Adapted to 32 channels with k=5 for FHE compatibility.

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
- With trained weights, the model achieves ~96% accuracy on MNIST in cleartext
