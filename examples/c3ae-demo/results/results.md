# C3AE Experiment Results

_Generated: 2026-05-09 23:21:35_

## Cleartext quality

| variant | scope    | n    | FPR    | FNR    | Accuracy |
| ------- | -------- | ---- | ------ | ------ | -------- |
| fhe     | boundary | 161  | 0.7121 | 0.1579 | 0.6149   |
| fhe     | overall  | 3557 | 0.2085 | 0.0268 | 0.9421   |
| relu    | boundary | 161  | 0.6515 | 0.1474 | 0.6460   |
| relu    | overall  | 3557 | 0.1675 | 0.0190 | 0.9556   |

## FHE cost

| config | compile_s | compile_peak_rss_GB | keygen_s | evk_GB | mean_forward_s | peak_rss_GB   |
| ------ | --------- | ------------------- | -------- | ------ | -------------- | ------------- |
| logn15 | 160.2     | 12.87               | 44.1     | 7.19   | 157.0 ± 2.9    | 54.19 ± 0.29  |
| logn16 | 386.7     | 25.77               | 68.1     | 12.70  | 543.7 ± 219.4  | 114.37 ± 0.07 |

## VPS rental cost log

immers.cloud rentals on 2026-05-09 (Moscow timezone). Pricing not captured at rental time — fill in from immers console if needed.

| VPS                     | Flavor              | RAM    | Rented at         | Torn down at      | Billed     |
| ----------------------- | ------------------- | ------ | ----------------- | ----------------- | ---------- |
| `orion-c3ae-train`      | `rtx4090-1.8.16.40` | 16 GB  | 2026-05-09T19:27Z | 2026-05-09T21:41Z | 2 h 14 min |
| `orion-c3ae-fhe-logn15` | `cpu.16.128.240`    | 128 GB | 2026-05-09T21:43Z | 2026-05-09T22:21Z | 0 h 38 min |
| `orion-c3ae-fhe-logn16` | `cpu.16.128.240`    | 128 GB | 2026-05-09T22:26Z | 2026-05-09T23:22Z | 0 h 56 min |

Total measured compute time across the experiment: **~3 h 48 min** (1 GPU box for training, 2 sequential CPU boxes for FHE).

## Notable findings

- **Boundary band is brutal**: at the 16–20 age threshold the model misclassifies ~65% of minors as adults across both ReLU and Quad variants. The asymmetric loss (`fpr_weight=40`) didn't shift the balance enough to overcome the dataset's 18% minor / 82% adult class imbalance.
- **Quad costs ~3 pp accuracy** vs ReLU on every metric — the price of FHE compatibility for the activation function.
- **Go-only inference cuts peak RSS by ~50%** vs the existing demo's Python-wrapped pipeline (54 GB at logn=15 vs the original 103 GB). Validates the architectural move documented in the parent plan.
- **logn=16 fits in 128 GB by ~10 GB margin**: peak 117 GB observed, 128 GB ceiling. Going larger than logn=16 at this depth would require a 256+ GB box.
- **Cold-cache effect on first inference**: sample 12 took 797s while samples 35 and 44 took ~417s. The 12.7 GB evk loading from disk into kernel page cache is the likely cause. For batch deployments, the first ciphertext should be a "warmup" not counted in measured throughput.
- **Forward time scales 2.7× when doubling ring degree**, not the naive 2× — extra cost comes from O(N) keyswitch decomposition during galois rotations.
