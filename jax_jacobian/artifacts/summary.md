## Jacobian relative error vs float64 exact (float32, GPU)

### torch

| lambda | score | jvp_loop | vmap_jvp | fd | diag_jvp | diag_fd |
|---|---|---|---|---|---|---|
| 0.003 | 1.0e-05 | 1.7e-06 | 1.7e-06 | 5.3e-03 | 1.7e-06 | 5.2e-03 |
| 0.03 | 1.1e-05 | 9.8e-07 | 9.9e-07 | 1.3e-03 | 9.8e-07 | 1.3e-03 |
| 0.3 | 6.9e-07 | 2.7e-07 | 3.0e-07 | 9.6e-04 | 2.7e-07 | 9.6e-04 |
| 3 | 3.3e-07 | 2.2e-07 | 2.4e-07 | 3.3e-03 | 2.2e-07 | 3.3e-03 |

### torch+compile

| lambda | score | jvp_loop | vmap_jvp | fd | diag_jvp | diag_fd |
|---|---|---|---|---|---|---|
| 0.003 | 1.0e-05 | 1.7e-06 | 1.7e-06 | 5.3e-03 | 1.7e-06 | 5.2e-03 |
| 0.03 | 1.1e-05 | 9.8e-07 | 9.9e-07 | 1.3e-03 | 9.8e-07 | 1.3e-03 |
| 0.3 | 6.9e-07 | 2.7e-07 | 3.0e-07 | 9.6e-04 | 2.7e-07 | 9.6e-04 |
| 3 | 3.3e-07 | 2.2e-07 | 2.4e-07 | 3.3e-03 | 2.2e-07 | 3.3e-03 |

### jax

| lambda | score | jacfwd_vmap | jvp_scan | diag_jvp | fd | diag_fd |
|---|---|---|---|---|---|---|
| 0.003 | 1.0e-05 | 1.4e-06 | 1.2e-06 | 1.1e-06 | 5.1e-03 | 5.1e-03 |
| 0.03 | 1.1e-05 | 1.1e-06 | 1.1e-06 | 1.0e-06 | 1.2e-03 | 1.2e-03 |
| 0.3 | 7.0e-07 | 4.0e-07 | 3.3e-07 | 3.2e-07 | 9.5e-04 | 9.5e-04 |
| 3 | 3.3e-07 | 3.5e-07 | 3.2e-07 | 3.3e-07 | 3.3e-03 | 3.3e-03 |

## Wall clock per Jacobian, median ms (in forward passes of the same run)

### torch

| rows x draws | forward | jvp_loop | vmap_jvp | fd | diag_jvp | diag_fd |
|---|---|---|---|---|---|---|
| 20 x 6 | 4.7 | 220.7 (47.1x) | 26.9 (5.7x) | 9.2 (2.0x) | 24.1 (5.2x) | 2.8 (0.6x) |
| 20 x 64 | 8.9 | 331.1 (37.4x) | 184.6 (20.8x) | 89.6 (10.1x) | 34.8 (3.9x) | 17.0 (1.9x) |
| 20 x 256 | 32.9 | 1271.5 (38.6x) | fail | 363.0 (11.0x) | 128.6 (3.9x) | 67.0 (2.0x) |

Failures: vmap_jvp@20x256: OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total capacity of 10.57 GiB of which 224.1

### torch+compile

| rows x draws | forward | jvp_loop | vmap_jvp | fd | diag_jvp | diag_fd |
|---|---|---|---|---|---|---|
| 20 x 6 | 2.7 | 214.8 (78.6x) | 28.4 (10.4x) | 9.3 (3.4x) | 25.8 (9.4x) | 2.9 (1.1x) |
| 20 x 64 | 8.9 | 334.2 (37.7x) | 184.1 (20.8x) | 90.7 (10.2x) | 34.7 (3.9x) | 17.1 (1.9x) |
| 20 x 256 | 33.3 | 1274.1 (38.3x) | fail | 383.2 (11.5x) | 139.8 (4.2x) | 73.0 (2.2x) |

Failures: vmap_jvp@20x256: OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total capacity of 10.57 GiB of which 204.1

### jax

| rows x draws | forward | jacfwd_vmap | jvp_scan | diag_jvp | fd | diag_fd |
|---|---|---|---|---|---|---|
| 20 x 6 | 1.0 | 8.3 (8.5x) | 13.3 (13.7x) | 1.7 (1.7x) | 8.0 (8.2x) | 1.4 (1.4x) |
| 20 x 64 | 7.6 | 69.6 (9.2x) | 115.0 (15.1x) | 11.8 (1.6x) | 91.6 (12.1x) | 13.1 (1.7x) |
| 20 x 256 | 28.4 | fail | 526.7 (18.5x) | 52.2 (1.8x) | fail | 65.2 (2.3x) |

Failures: jacfwd_vmap@20x256: NOT_FOUND: All configs failed during profiling or were excluded from selection.; fd@20x256: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 6.71GiB with allocator GPU_0_bfc on device 0. [tf-allocator-a

