# E5 Full 18-Document Panel: Ceiling Sensitivity & Head Specificity (Primary Checkpoints)

Cross-dataset equal-weighted mean BPB over the full 18-document panel (8 FinePDFs, 5 PG-19, 5 Proof-Pile), matching Table 2 and E2.

| Model | Condition (Block 2 Head / Ceiling) | 2K BPB | 64K BPB | 256K BPB | 256K Recovery vs Uncapped |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Primary NoPE | `uncapped` (B2/H5, uncapped) | 1.432 | 5.886 | **8.097** | +0.000 |
| Primary NoPE | `p99_outlier` (B2/H5, H=484,606.1) | 1.432 | 5.877 | **8.126** | -0.029 |
| Primary NoPE | `p90_outlier` (B2/H5, H=70.6) | 1.470 | 1.516 | **1.591** | +6.506 |
| Primary NoPE | `hl_1024_outlier` (B2/H5, H=1024) | 1.438 | 1.833 | **2.173** | +5.923 |
| Primary NoPE | `hl_512_outlier` (B2/H5, H=512) | 1.445 | 1.531 | **1.663** | +6.434 |
| Primary NoPE | `hl_256_outlier` (B2/H5, H=256) | 1.454 | 1.506 | **1.595** | +6.501 |
| Primary NoPE | `hl_128_outlier` (B2/H5, H=128) | 1.463 | 1.509 | **1.588** | +6.509 |
| Primary NoPE | `hl_256_second_head` (B2/H7, H=256) | 1.432 | 5.921 | **8.121** | -0.024 |
| Primary NoPE | `hl_256_random_head` (B2/H4, H=256) | 1.432 | 5.886 | **8.097** | +0.000 |
| Primary Polar | `uncapped` (B2/H6, uncapped) | 1.416 | 1.602 | **1.855** | +0.000 |
| Primary Polar | `p99_outlier` (B2/H6, H=162,414.4) | 1.416 | 1.595 | **1.839** | +0.016 |
| Primary Polar | `p90_outlier` (B2/H6, H=139.2) | 1.427 | 1.411 | **1.521** | +0.334 |
| Primary Polar | `hl_1024_outlier` (B2/H6, H=1024) | 1.419 | 1.409 | **1.527** | +0.328 |
| Primary Polar | `hl_512_outlier` (B2/H6, H=512) | 1.422 | 1.408 | **1.521** | +0.334 |
| Primary Polar | `hl_256_outlier` (B2/H6, H=256) | 1.425 | 1.409 | **1.521** | +0.334 |
| Primary Polar | `hl_128_outlier` (B2/H6, H=128) | 1.427 | 1.411 | **1.521** | +0.334 |
| Primary Polar | `hl_256_second_head` (B2/H0, H=256) | 1.416 | 1.602 | **1.856** | -0.001 |
| Primary Polar | `hl_256_random_head` (B2/H5, H=256) | 1.416 | 1.602 | **1.855** | +0.000 |
