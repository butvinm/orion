# LeNet — MNIST FHE Inference Example

## Architecture

Conv2d(1, 32, 5, s=2, p=2) → BatchNorm2d → Quad → Conv2d(32, 64, 5, s=2, p=2) → BatchNorm2d → Quad → Flatten → Linear(3136, 512) → BatchNorm1d → Quad → Linear(512, 10)

A convolutional network with 2 Conv2d layers and 2 FC layers, using Quad (x²) activations — FHE-compatible alternative to ReLU. BatchNorm layers fuse into the preceding linear/conv layers during compilation.

## CKKS Parameters

| Parameter | Value                                    |
| --------- | ---------------------------------------- |
| logn      | 13                                       |
| logq      | [29, 26, 26, 26, 26, 26, 26, 26, 26, 26] |
| logp      | [29, 29]                                 |
| logscale  | 26                                       |
| h         | 8192                                     |
| ring_type | conjugate_invariant                      |

Same modulus chain as MLP (9 levels). The compiler uses input_level=7, which fits within 9 levels with no bootstrapping needed. Despite having more layers than MLP (2 Conv2d + 3 Quad), the depth is similar because BatchNorm fusions and the packing strategy keep the multiplicative depth manageable.

## Usage

```bash
cd examples/lenet

# Optional: train the model (saves weights.pt)
python train.py

# Run FHE inference pipeline (works with random or trained weights)
python run.py
```

## Expected Output

- MAE < 0.1 (cleartext vs FHE output difference)
- With trained weights, the model achieves ~99% accuracy on MNIST in cleartext
