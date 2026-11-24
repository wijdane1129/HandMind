# Benchmark BCI Competition IV 2a (test T8)

4 classes, train session T -> test session E, 22 EEG channels, 0.5-2.5 s, 8-30 Hz.

| Model | Accuracy | Kappa |
|---|---|---|
| csp_lda | 0.619 ± 0.143 | 0.492 ± 0.190 |
| fbcsp_svm | 0.615 ± 0.162 | 0.487 ± 0.216 |
| riemann | 0.674 ± 0.163 | 0.565 ± 0.217 |

Reference: competition winner (FBCSP) mean kappa ≈ 0.57.

## Kappa per subject

| Subject | csp_lda | fbcsp_svm | riemann |
|---|---|---|---|
| A01 | 0.69 | 0.76 | 0.79 |
| A02 | 0.26 | 0.25 | 0.30 |
| A03 | 0.67 | 0.81 | 0.80 |
| A04 | 0.39 | 0.43 | 0.48 |
| A05 | 0.27 | 0.20 | 0.29 |
| A06 | 0.28 | 0.33 | 0.32 |
| A07 | 0.57 | 0.42 | 0.63 |
| A08 | 0.70 | 0.59 | 0.72 |
| A09 | 0.58 | 0.59 | 0.75 |
