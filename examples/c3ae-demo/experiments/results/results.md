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
