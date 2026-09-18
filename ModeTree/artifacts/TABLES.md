### Experiment 1 - known modes, oracle clean score

| case | n | max |S| | e_MAP/sd | energy-gap err | TV (Laplace) | TV (Laplace+) | TV (Laplace+, no guard) | min ESS frac |
|---|---|---|---|---|---|---|---|---|
| symmetric | 1 | 3.3e-14 | 1.1e-11* | 9.4e-17 | 2.8e-17 | 8.3e-04 | 0.0102 | 0.963 |
| symmetric | 8 | 2.7e-15 | 7.5e-12* | 8.3e-16 | 2.2e-16 | 7.9e-05 | 7.9e-05 | 1.000 |
| symmetric | 100 | 2.5e-14 | 1.1e-11* | 6.6e-15 | 1.7e-15 | 3.9e-06 | 3.9e-06 | 1.000 |
| nearly_tied | 1 | 8.3e-17 | 1.2e-11 | 7.0e-16 | 7.8e-05 | 6.4e-04 | 0.0100 | 0.964 |
| nearly_tied | 8 | 3.3e-16 | 6.8e-12 | 6.7e-16 | 7.3e-06 | 8.2e-05 | 8.2e-05 | 1.000 |
| nearly_tied | 100 | 1.4e-14 | 4.6e-12 | 1.4e-13 | 3.2e-07 | 3.7e-06 | 3.7e-06 | 1.000 |
| separated | 1 | 0.0000 | 1.1e-13* | 6.4e-16 | 2.2e-16 | 4.4e-16 | 9.2e-16 | 1.000 |
| separated | 8 | 1.4e-15 | 1.1e-13* | 2.6e-14 | 6.5e-15 | 6.5e-15 | 1.1e-15 | 1.000 |
| separated | 100 | 7.4e-14 | 2.7e-12* | 9.8e-14 | 2.4e-14 | 2.4e-14 | 4.1e-14 | 1.000 |
| vanishing | 1 | 1.3e-13 | 6.7e-12 | 2.4e-15 | 0.0042 | 0.0016 | 0.0022 | 0.939 |
| vanishing | 8 | 3.2e-14 | 1.8e-11 | 4.0e-15 | 4.6e-04 | 4.2e-04 | 4.2e-04 | 1.000 |
| vanishing | 100 | 2.3e-14 | 4.2e-10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 |
| map_vs_mass | 1 | 4.3e-16 | 1.1e-11 | 3.4e-15 | 0.0077 | 0.0049 | 0.0049 | 0.996 |
| map_vs_mass | 8 | 5.6e-15 | 1.9e-11 | 2.7e-14 | 3.0e-04 | 8.6e-05 | 8.6e-05 | 1.000 |
| map_vs_mass | 100 | 1.1e-13 | 2.2e-11 | 7.4e-14 | 2.1e-05 | 5.3e-06 | 5.3e-06 | 1.000 |
| tilt_0.5 | 1 | 1.1e-16 | 5.7e-12 | 6.7e-16 | 0.0051 | 0.0029 | 0.0106 | 0.961 |
| tilt_0.5 | 8 | 8.9e-16 | 3.9e-12 | 1.0e-15 | 6.9e-05 | 1.0e-04 | 1.0e-04 | 1.000 |
| tilt_0.5 | 100 | 6.4e-15 | 1.6e-11 | 4.8e-14 | 2.7e-07 | 3.7e-06 | 3.7e-06 | 1.000 |
| tilt_2.0 | 1 | 2.9e-11 | 2.0e-11 | 1.8e-15 | 0.0135 | 3.6e-04 | 0.0512 | 0.872 |
| tilt_2.0 | 8 | 4.0e-16 | 3.4e-12 | 2.2e-16 | 1.2e-04 | 9.0e-05 | 9.0e-05 | 1.000 |
| tilt_2.0 | 100 | 7.0e-15 | 8.1e-12 | 2.1e-13 | 4.9e-07 | 1.8e-06 | 1.8e-06 | 1.000 |
| narrow_broad | 1 | 9.4e-13 | 3.2e-11 | 7.4e-15 | 0.0203 | 0.0057 | 0.0305 | 0.968 |
| narrow_broad | 8 | 1.1e-13 | 1.3e-10 | 1.5e-14 | 3.9e-05 | 1.2e-05 | 1.2e-05 | 1.000 |
| narrow_broad | 100 | 1.1e-12 | 7.0e-10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 |
| empty_valley | 1 | 0.0000 | 2.2e-13* | 4.0e-15 | 1.0e-15 | 1.0e-15 | 3.1e-15 | 1.000 |
| empty_valley | 8 | 4.6e-15 | 4.6e-13* | 1.7e-14 | 4.3e-15 | 4.3e-15 | 2.1e-14 | 1.000 |
| empty_valley | 100 | 1.6e-14 | 1.2e-12* | 1.7e-13 | 4.2e-14 | 4.2e-14 | 1.7e-13 | 1.000 |

### Method comparison

#### n = 8

| case | method | W1 | e_MAP/sd | weight TV | missing mass | s | field calls |
|---|---|---|---|---|---|---|---|
| empty_valley | uncorrected | 0.1053 | 1.0e-05* | 0.0065 | - | 0.78 | 200 |
| empty_valley | gauss_jacobian | 0.0646 | 2.1e-05* | 0.0055 | - | 2.04 | 200 |
| empty_valley | fnpe | 0.0960 | 1.8e-05* | 0.0080 | - | 3.91 | 1000 |
| empty_valley | modetree_known | 1.3e-08 | 0.000* | 4.0e-15 | 0.0000 | 0.03 | 4336 |
| empty_valley | modetree | 1.3e-08 | 2.1e-14* | 4.4e-15 | 0.0000 | 1.12 | 4336 |
| empty_valley | modetree_plus | 1.3e-08 | 2.1e-14* | 4.4e-15 | 0.0000 | 1.14 | 4336 |
| map_vs_mass | uncorrected | 1.3875 | 0.009 | 0.3382 | - | 0.81 | 200 |
| map_vs_mass | gauss_jacobian | 1.0790 | 4.9e-04 | 0.2692 | - | 2.12 | 200 |
| map_vs_mass | fnpe | 1.0529 | 0.001 | 0.2847 | - | 4.09 | 1000 |
| map_vs_mass | modetree_known | 0.0013 | 8.8e-11 | 3.0e-04 | 0.0000 | 0.03 | 1264 |
| map_vs_mass | modetree | 0.0013 | 8.8e-11 | 3.0e-04 | 0.0000 | 1.88 | 1136 |
| map_vs_mass | modetree_plus | 7.0e-04 | 8.8e-11 | 1.3e-04 | 0.0000 | 1.90 | 1136 |
| narrow_broad | uncorrected | 2.4753 | 7.7e-04 | 0.6258 | - | 0.78 | 200 |
| narrow_broad | gauss_jacobian | 2.1400 | 1.7e-04 | 0.5438 | - | 2.05 | 200 |
| narrow_broad | fnpe | 2.0932 | 2.0e-04 | 0.5743 | - | 3.93 | 1000 |
| narrow_broad | modetree_known | 0.0020 | 2.6e-10 | 3.9e-05 | 0.0000 | 0.02 | 944 |
| narrow_broad | modetree | 0.0020 | 2.6e-10 | 3.9e-05 | 0.0000 | 1.67 | 944 |
| narrow_broad | modetree_plus | 0.0020 | 2.6e-10 | 1.7e-05 | 0.0000 | 1.70 | 944 |
| nearly_tied | uncorrected | 0.0843 | 1.8e-04 | 0.0065 | - | 0.82 | 200 |
| nearly_tied | gauss_jacobian | 0.0225 | 1.5e-04 | 0.0015 | - | 2.17 | 200 |
| nearly_tied | fnpe | 0.1861 | 5.9e-05 | 0.0100 | - | 4.11 | 1000 |
| nearly_tied | modetree_known | 0.0027 | 2.9e-11 | 7.3e-06 | 0.0000 | 0.03 | 624 |
| nearly_tied | modetree | 0.0027 | 2.9e-11 | 7.3e-06 | 0.0000 | 2.40 | 624 |
| nearly_tied | modetree_plus | 0.0028 | 2.9e-11 | 8.2e-05 | 0.0000 | 2.43 | 624 |
| symmetric | uncorrected | 0.0731 | 6.9e-05* | 1.0e-03 | - | 0.83 | 200 |
| symmetric | gauss_jacobian | 0.0458 | 6.0e-05* | 0.0100 | - | 2.21 | 200 |
| symmetric | fnpe | 0.1793 | 2.6e-04* | 2.8e-17 | - | 3.94 | 1000 |
| symmetric | modetree_known | 0.0027 | 2.9e-11* | 1.9e-16 | 0.0000 | 0.02 | 496 |
| symmetric | modetree | 0.0027 | 2.9e-11* | 1.4e-16 | 0.0000 | 1.37 | 496 |
| symmetric | modetree_plus | 0.0028 | 2.9e-11* | 7.9e-05 | 0.0000 | 1.39 | 496 |
| vanishing | uncorrected | 0.1605 | 2.6e-04 | 0.0411 | - | 0.81 | 200 |
| vanishing | gauss_jacobian | 0.4242 | 2.0e-04 | 0.1201 | - | 2.16 | 200 |
| vanishing | fnpe | 0.5097 | 1.8e-04 | 0.1336 | - | 4.20 | 1000 |
| vanishing | modetree_known | 0.0088 | 6.7e-11 | 4.6e-04 | 0.0000 | 0.03 | 560 |
| vanishing | modetree | 0.0088 | 6.7e-11 | 4.6e-04 | 0.0000 | 2.08 | 560 |
| vanishing | modetree_plus | 0.0087 | 6.7e-11 | 4.2e-04 | 0.0000 | 2.09 | 560 |

#### n = 32

| case | method | W1 | e_MAP/sd | weight TV | missing mass | s | field calls |
|---|---|---|---|---|---|---|---|
| empty_valley | uncorrected | 0.0917 | 5.7e-06* | 0.0070 | - | 2.58 | 200 |
| empty_valley | gauss_jacobian | 0.0049 | 2.0e-05* | 0.0000 | - | 6.63 | 200 |
| empty_valley | fnpe | 0.1298 | 1.6e-05* | 0.0115 | - | 13.02 | 1000 |
| empty_valley | modetree_known | 2.5e-08 | 1.4e-13* | 3.1e-14 | 0.0000 | 0.06 | 4336 |
| empty_valley | modetree | 2.5e-08 | 1.4e-13* | 3.1e-14 | 0.0000 | 3.51 | 4336 |
| empty_valley | modetree_plus | 2.5e-08 | 1.4e-13* | 3.1e-14 | 0.0000 | 3.60 | 4336 |
| map_vs_mass | uncorrected | 1.3715 | 2.102 | 0.3390 | - | 2.69 | 200 |
| map_vs_mass | gauss_jacobian | 1.2385 | 2.8e-05 | 0.3075 | - | 6.74 | 200 |
| map_vs_mass | fnpe | 1.2938 | 2.102 | 0.3390 | - | 13.11 | 1000 |
| map_vs_mass | modetree_known | 3.1e-04 | 5.6e-11 | 6.8e-05 | 0.0000 | 0.03 | 1264 |
| map_vs_mass | modetree | 3.1e-04 | 5.6e-11 | 6.8e-05 | 0.0000 | 2.36 | 1136 |
| map_vs_mass | modetree_plus | 1.6e-04 | 5.6e-11 | 2.6e-05 | 0.0000 | 2.45 | 1136 |
| narrow_broad | uncorrected | 2.9782 | 0.004 | 0.0000 | - | 2.57 | 200 |
| narrow_broad | gauss_jacobian | 2.3506 | 8.5e-05 | 0.0000 | - | 6.68 | 200 |
| narrow_broad | fnpe | 2.2765 | 0.005 | 0.0000 | - | 13.04 | 1000 |
| narrow_broad | modetree_known | 4.1e-04 | 9.3e-10 | 0.0000 | 0.0000 | 0.00 | 0 |
| narrow_broad | modetree | 4.1e-04 | 9.3e-10 | 0.0000 | 0.0000 | 3.74 | 1072 |
| narrow_broad | modetree_plus | 4.1e-04 | 9.3e-10 | 0.0000 | 0.0000 | 3.81 | 1072 |
| nearly_tied | uncorrected | 0.0981 | 5.1e-05 | 0.0178 | - | 2.60 | 200 |
| nearly_tied | gauss_jacobian | 0.1038 | 7.4e-05 | 0.0253 | - | 6.73 | 200 |
| nearly_tied | fnpe | 0.3376 | 1.0e-04 | 0.0378 | - | 13.14 | 1000 |
| nearly_tied | modetree_known | 4.0e-04 | 9.7e-12 | 1.2e-06 | 0.0000 | 0.03 | 624 |
| nearly_tied | modetree | 4.0e-04 | 9.7e-12 | 1.2e-06 | 0.0000 | 3.68 | 624 |
| nearly_tied | modetree_plus | 4.3e-04 | 9.7e-12 | 1.3e-05 | 0.0000 | 3.74 | 624 |
| symmetric | uncorrected | 0.0469 | 2.9e-05* | 0.0020 | - | 2.59 | 200 |
| symmetric | gauss_jacobian | 0.0547 | 3.5e-05* | 0.0120 | - | 6.83 | 200 |
| symmetric | fnpe | 0.2719 | 5.9e-05* | 5.0e-04 | - | 13.20 | 1000 |
| symmetric | modetree_known | 4.0e-04 | 7.5e-12* | 5.6e-17 | 0.0000 | 0.02 | 496 |
| symmetric | modetree | 4.0e-04 | 7.5e-12* | 5.6e-17 | 0.0000 | 2.40 | 496 |
| symmetric | modetree_plus | 4.3e-04 | 7.5e-12* | 1.3e-05 | 0.0000 | 2.47 | 496 |
| vanishing | uncorrected | 0.1587 | 7.7e-05 | 0.0402 | - | 2.63 | 200 |
| vanishing | gauss_jacobian | 0.9265 | 1.6e-05 | 0.2597 | - | 6.88 | 200 |
| vanishing | fnpe | 1.0315 | 1.5e-04 | 0.2612 | - | 13.23 | 1000 |
| vanishing | modetree_known | 0.0012 | 2.6e-10 | 4.2e-06 | 0.0000 | 0.03 | 560 |
| vanishing | modetree | 0.0012 | 2.6e-10 | 4.2e-06 | 0.0000 | 3.49 | 560 |
| vanishing | modetree_plus | 0.0012 | 2.6e-10 | 3.4e-06 | 0.0000 | 3.58 | 560 |

#### n = 100

| case | method | W1 | e_MAP/sd | weight TV | missing mass | s | field calls |
|---|---|---|---|---|---|---|---|
| empty_valley | uncorrected | 15.0101 | 3.9e+16* | 0.0010 | - | 8.93 | 200 |
| empty_valley | gauss_jacobian | 0.0806 | 8.8e-07* | 0.0080 | - | 21.48 | 200 |
| empty_valley | fnpe | 0.1059 | 3.4e-06* | 0.0090 | - | 44.79 | 1000 |
| empty_valley | modetree_known | 4.4e-08 | 1.2e-12* | 4.2e-14 | 0.0000 | 0.13 | 4336 |
| empty_valley | modetree | 4.4e-08 | 1.2e-12* | 4.2e-14 | 0.0000 | 11.20 | 4336 |
| empty_valley | modetree_plus | 4.4e-08 | 1.2e-12* | 4.2e-14 | 0.0000 | 11.46 | 4336 |
| map_vs_mass | uncorrected | 10.5794 | 33.573 | 0.1577 | - | 8.98 | 200 |
| map_vs_mass | gauss_jacobian | 1.2587 | 1.8e-05 | 0.3128 | - | 21.70 | 200 |
| map_vs_mass | fnpe | 1.3062 | 2.109 | 0.3393 | - | 45.39 | 1000 |
| map_vs_mass | modetree_known | 1.0e-04 | 9.7e-11 | 2.1e-05 | 0.0000 | 0.05 | 1136 |
| map_vs_mass | modetree | 1.0e-04 | 9.7e-11 | 2.1e-05 | 0.0000 | 4.38 | 1008 |
| map_vs_mass | modetree_plus | 5.2e-05 | 9.7e-11 | 8.2e-06 | 0.0000 | 4.61 | 1008 |
| narrow_broad | uncorrected | 3.9920 | 87.123 | 0.5000 | - | 8.89 | 200 |
| narrow_broad | gauss_jacobian | 2.5457 | 0.002 | 0.0000 | - | 21.64 | 200 |
| narrow_broad | fnpe | 2.5437 | 0.003 | 0.0000 | - | 44.50 | 1000 |
| narrow_broad | modetree_known | 1.3e-04 | 4.6e-09 | 0.0000 | 0.0000 | 0.00 | 0 |
| narrow_broad | modetree | 1.3e-04 | 4.6e-09 | 0.0000 | 0.0000 | 6.58 | 1072 |
| narrow_broad | modetree_plus | 1.3e-04 | 4.6e-09 | 0.0000 | 0.0000 | 6.86 | 1072 |
| nearly_tied | uncorrected | 0.6085 | 1.6e-04 | 0.1511 | - | 8.88 | 200 |
| nearly_tied | gauss_jacobian | 0.4916 | 8.2e-06 | 0.1226 | - | 21.42 | 200 |
| nearly_tied | fnpe | 0.6815 | 7.8e-05 | 0.1256 | - | 44.07 | 1000 |
| nearly_tied | modetree_known | 1.2e-04 | 2.9e-11 | 3.2e-07 | 0.0000 | 0.03 | 624 |
| nearly_tied | modetree | 1.2e-04 | 2.9e-11 | 3.2e-07 | 0.0000 | 6.00 | 624 |
| nearly_tied | modetree_plus | 1.3e-04 | 2.9e-11 | 3.5e-06 | 0.0000 | 6.20 | 624 |
| symmetric | uncorrected | 0.0527 | 1.1e-05* | 0.0080 | - | 9.05 | 200 |
| symmetric | gauss_jacobian | 0.0075 | 1.1e-05* | 0.0000 | - | 21.98 | 200 |
| symmetric | fnpe | 0.3813 | 1.5e-04* | 0.0000 | - | 44.65 | 1000 |
| symmetric | modetree_known | 1.1e-04 | 2.9e-11* | 3.4e-15 | 0.0000 | 0.04 | 496 |
| symmetric | modetree | 1.1e-04 | 2.9e-11* | 3.4e-15 | 0.0000 | 6.25 | 496 |
| symmetric | modetree_plus | 1.3e-04 | 2.9e-11* | 3.9e-06 | 0.0000 | 6.53 | 496 |
| vanishing | uncorrected | 2.1129 | 0.004 | 0.0000 | - | 8.54 | 200 |
| vanishing | gauss_jacobian | 0.7362 | 2.9e-04 | 0.0000 | - | 21.00 | 200 |
| vanishing | fnpe | 0.7115 | 0.003 | 0.0000 | - | 42.89 | 1000 |
| vanishing | modetree_known | 3.4e-04 | 5.2e-10 | 0.0000 | 0.0000 | 0.00 | 0 |
| vanishing | modetree | 3.4e-04 | 5.2e-10 | 0.0000 | 0.0000 | 5.92 | 560 |
| vanishing | modetree_plus | 3.4e-04 | 5.2e-10 | 0.0000 | 0.0000 | 6.13 | 560 |

### Runtime vs F-NPSE

| n | gauss_jacobian (s) | F-NPSE (s) | ModeTree+ (s) | ModeTree / F-NPSE |
|---|---|---|---|---|
| 1 | 0.52 | 1.00 | 0.46 | 0.46 |
| 2 | 0.77 | 1.47 | 1.00 | 0.68 |
| 4 | 1.20 | 2.28 | 1.42 | 0.62 |
| 8 | 2.13 | 4.03 | 1.78 | 0.44 |
| 16 | 3.66 | 7.19 | 2.49 | 0.35 |
| 32 | 6.75 | 13.12 | 3.28 | 0.25 |
| 64 | 14.06 | 28.32 | 4.81 | 0.17 |
| 100 | 21.53 | 44.38 | 6.96 | 0.16 |

### Ablations

#### n = 8

| case | variant | recall | missing mass | weight TV | e_MAP/sd | B | s | JAC evals |
|---|---|---|---|---|---|---|---|---|
| map_vs_mass | full | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.39 | 1733 |
| map_vs_mass | full+laplace+ | 1.00 | 0.0000 | 1.3e-04 | 1.5e-10 | 2 | 1.39 | 1733 |
| map_vs_mass | no_sentinel | 0.50 | 0.3397 | 0.3397 | 2.070 | 1 | 0.88 | 1236 |
| map_vs_mass | no_curvature | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.36 | 1733 |
| map_vs_mass | one_path | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.35 | 1733 |
| map_vs_mass | stride2 | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.17 | 1168 |
| map_vs_mass | stride4 | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.10 | 906 |
| map_vs_mass | stride8 | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.04 | 713 |
| map_vs_mass | switch_t0.05 | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.46 | 1860 |
| map_vs_mass | switch_t0.002 | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.47 | 1863 |
| map_vs_mass | tangent | 1.00 | 0.0000 | 3.0e-04 | 1.5e-10 | 2 | 1.42 | 1817 |
| empty_valley | full | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 1.15 | 1360 |
| empty_valley | full+laplace+ | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 1.18 | 1360 |
| empty_valley | no_sentinel | 0.50 | 0.5000 | 0.5000 | 1.7e-14* | 1 | 0.56 | 777 |
| empty_valley | no_curvature | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 1.22 | 1452 |
| empty_valley | one_path | 1.00 | 0.0000 | 2.1e-14 | 1.7e-14* | 2 | 1.10 | 1360 |
| empty_valley | stride2 | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 0.98 | 931 |
| empty_valley | stride4 | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 0.91 | 716 |
| empty_valley | stride8 | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 0.83 | 571 |
| empty_valley | switch_t0.05 | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 1.22 | 1472 |
| empty_valley | switch_t0.002 | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 1.18 | 1419 |
| empty_valley | tangent | 1.00 | 0.0000 | 4.3e-15 | 1.7e-14* | 2 | 1.20 | 1466 |
| narrow_broad | full | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.68 | 2136 |
| narrow_broad | full+laplace+ | 1.00 | 0.0000 | 1.7e-05 | 1.7e-09 | 2 | 1.70 | 2136 |
| narrow_broad | no_sentinel | 0.50 | 0.9843 | 0.9843 | 7.576 | 1 | 0.91 | 1251 |
| narrow_broad | no_curvature | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.78 | 2262 |
| narrow_broad | one_path | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.67 | 2136 |
| narrow_broad | stride2 | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.45 | 1426 |
| narrow_broad | stride4 | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.34 | 1079 |
| narrow_broad | stride8 | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.08 | 722 |
| narrow_broad | switch_t0.05 | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.41 | 1796 |
| narrow_broad | switch_t0.002 | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.37 | 1769 |
| narrow_broad | tangent | 1.00 | 0.0000 | 3.9e-05 | 1.7e-09 | 2 | 1.67 | 2149 |

#### n = 32

| case | variant | recall | missing mass | weight TV | e_MAP/sd | B | s | JAC evals |
|---|---|---|---|---|---|---|---|---|
| map_vs_mass | full | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 2.46 | 2199 |
| map_vs_mass | full+laplace+ | 1.00 | 0.0000 | 2.6e-05 | 3.4e-10 | 2 | 2.55 | 2199 |
| map_vs_mass | no_sentinel | 0.50 | 0.3390 | 0.3390 | 2.102 | 1 | 1.17 | 1410 |
| map_vs_mass | no_curvature | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 2.47 | 2199 |
| map_vs_mass | one_path | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 2.47 | 2199 |
| map_vs_mass | stride2 | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 2.15 | 1495 |
| map_vs_mass | stride4 | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 2.13 | 1264 |
| map_vs_mass | stride8 | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 1.97 | 1024 |
| map_vs_mass | switch_t0.05 | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 2.53 | 2304 |
| map_vs_mass | switch_t0.002 | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 2.55 | 2325 |
| map_vs_mass | tangent | 1.00 | 0.0000 | 6.8e-05 | 3.4e-10 | 2 | 2.52 | 2309 |
| empty_valley | full | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 2.77 | 2414 |
| empty_valley | full+laplace+ | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 2.86 | 2414 |
| empty_valley | no_sentinel | 0.50 | 0.5000 | 0.5000 | 1.0e-13* | 1 | 1.04 | 1241 |
| empty_valley | no_curvature | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 2.97 | 2408 |
| empty_valley | one_path | 1.00 | 0.0000 | 8.2e-14 | 1.0e-13* | 2 | 2.74 | 2414 |
| empty_valley | stride2 | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 2.48 | 1686 |
| empty_valley | stride4 | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 2.34 | 1327 |
| empty_valley | stride8 | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 2.27 | 1160 |
| empty_valley | switch_t0.05 | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 2.71 | 2356 |
| empty_valley | switch_t0.002 | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 2.76 | 2350 |
| empty_valley | tangent | 1.00 | 0.0000 | 3.1e-14 | 1.0e-13* | 2 | 3.43 | 2997 |
| narrow_broad | full | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.91 | 2681 |
| narrow_broad | full+laplace+ | 1.00 | 0.0000 | 5.6e-17 | 1.2e-08 | 2 | 2.99 | 2681 |
| narrow_broad | no_sentinel | 0.00 | 1.0000 | 0.0000 | 49.095 | 1 | 1.37 | 1561 |
| narrow_broad | no_curvature | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.96 | 2714 |
| narrow_broad | one_path | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.92 | 2681 |
| narrow_broad | stride2 | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.60 | 1869 |
| narrow_broad | stride4 | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.56 | 1535 |
| narrow_broad | stride8 | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.47 | 1314 |
| narrow_broad | switch_t0.05 | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.83 | 2651 |
| narrow_broad | switch_t0.002 | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.91 | 2592 |
| narrow_broad | tangent | 1.00 | 0.0000 | 0.0000 | 1.2e-08 | 2 | 2.74 | 2618 |

#### n = 100

| case | variant | recall | missing mass | weight TV | e_MAP/sd | B | s | JAC evals |
|---|---|---|---|---|---|---|---|---|
| map_vs_mass | full | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 4.49 | 2578 |
| map_vs_mass | full+laplace+ | 1.00 | 0.0000 | 8.2e-06 | 1.4e-10 | 2 | 4.74 | 2578 |
| map_vs_mass | no_sentinel | 0.50 | 0.3393 | 0.3393 | 2.109 | 1 | 1.56 | 1682 |
| map_vs_mass | no_curvature | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 4.53 | 2578 |
| map_vs_mass | one_path | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 4.53 | 2578 |
| map_vs_mass | stride2 | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 4.04 | 1662 |
| map_vs_mass | stride4 | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 3.88 | 1336 |
| map_vs_mass | stride8 | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 3.83 | 1193 |
| map_vs_mass | switch_t0.05 | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 4.65 | 2641 |
| map_vs_mass | switch_t0.002 | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 4.89 | 2723 |
| map_vs_mass | tangent | 1.00 | 0.0000 | 2.1e-05 | 1.4e-10 | 2 | 4.55 | 2688 |
| empty_valley | full | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 9.37 | 6032 |
| empty_valley | full+laplace+ | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 9.63 | 6032 |
| empty_valley | no_sentinel | 0.50 | 0.5000 | 0.5000 | 5.5e-13* | 1 | 1.40 | 1336 |
| empty_valley | no_curvature | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 8.09 | 4478 |
| empty_valley | one_path | 1.00 | 0.0000 | 1.7e-13 | 5.5e-13* | 2 | 9.33 | 6032 |
| empty_valley | stride2 | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 7.93 | 3529 |
| empty_valley | stride4 | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 8.14 | 3045 |
| empty_valley | stride8 | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 7.75 | 2522 |
| empty_valley | switch_t0.05 | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 5.70 | 3252 |
| empty_valley | switch_t0.002 | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 7.58 | 4180 |
| empty_valley | tangent | 1.00 | 0.0000 | 4.2e-14 | 5.5e-13* | 2 | 11.16 | 7593 |
| narrow_broad | full | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 5.63 | 3097 |
| narrow_broad | full+laplace+ | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 5.86 | 3097 |
| narrow_broad | no_sentinel | 0.00 | 1.0000 | 0.0000 | 87.123 | 1 | 2.08 | 1823 |
| narrow_broad | no_curvature | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 5.61 | 3097 |
| narrow_broad | one_path | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 5.60 | 3097 |
| narrow_broad | stride2 | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 5.48 | 2242 |
| narrow_broad | stride4 | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 5.05 | 1770 |
| narrow_broad | stride8 | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 5.15 | 1496 |
| narrow_broad | switch_t0.05 | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 5.31 | 3326 |
| narrow_broad | switch_t0.002 | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 6.68 | 3771 |
| narrow_broad | tangent | 1.00 | 0.0000 | 0.0000 | 2.3e-08 | 2 | 6.07 | 3671 |

### Two dimensions / remote small mode

| regime | n | discovery | recall | missing mass | false | weight TV | max cycle residual | s |
|---|---|---|---|---|---|---|---|---|
| symmetric | 4 | both | 1.00 | 0.0000 | 0 | 5.3e-05 | 1.5e-15 | 2.49 |
| symmetric | 4 | curvature only | 0.00 | 1.0000 | 0 | 1.0000 | - | 0.28 |
| symmetric | 4 | sentinel only | 1.00 | 0.0000 | 0 | 5.3e-05 | 1.5e-15 | 2.49 |
| symmetric | 16 | both | 1.00 | 0.0000 | 0 | 1.6e-11 | 2.6e-15 | 4.72 |
| symmetric | 16 | curvature only | 0.00 | 1.0000 | 0 | 1.0000 | - | 1.35 |
| symmetric | 16 | sentinel only | 1.00 | 0.0000 | 0 | 1.6e-11 | 2.6e-15 | 4.36 |
| symmetric | 64 | both | 1.00 | 0.0000 | 0 | 2.6e-15 | 5.5e-15 | 16.31 |
| symmetric | 64 | curvature only | 0.00 | 1.0000 | 0 | 1.0000 | - | 7.26 |
| symmetric | 64 | sentinel only | 1.00 | 0.0000 | 0 | 2.6e-15 | 5.5e-15 | 11.43 |
| symmetric | 100 | both | 1.00 | 0.0000 | 0 | 1.1e-15 | 1.3e-14 | 25.66 |
| symmetric | 100 | curvature only | 0.00 | 1.0000 | 0 | 1.0000 | - | 10.97 |
| symmetric | 100 | sentinel only | 1.00 | 0.0000 | 0 | 1.1e-15 | 1.3e-14 | 17.88 |
| unequal_weights | 4 | both | 1.00 | 0.0000 | 0 | 0.0016 | 3.3e-16 | 2.37 |
| unequal_weights | 4 | curvature only | 0.25 | 0.4326 | 0 | 0.4326 | - | 1.05 |
| unequal_weights | 4 | sentinel only | 1.00 | 0.0000 | 0 | 0.0016 | 3.3e-16 | 2.37 |
| unequal_weights | 16 | both | 1.00 | 0.0000 | 0 | 4.8e-05 | 6.2e-15 | 5.32 |
| unequal_weights | 16 | curvature only | 0.25 | 0.7436 | 0 | 0.7436 | - | 1.76 |
| unequal_weights | 16 | sentinel only | 1.00 | 0.0000 | 0 | 4.8e-05 | 6.2e-15 | 5.73 |
| unequal_weights | 64 | both | 1.00 | 0.0000 | 0 | 2.6e-06 | 2.5e-14 | 18.96 |
| unequal_weights | 64 | curvature only | 0.50 | 0.3032 | 0 | 0.3032 | - | 4.77 |
| unequal_weights | 64 | sentinel only | 1.00 | 0.0000 | 0 | 2.6e-06 | 1.7e-14 | 16.01 |
| unequal_weights | 100 | both | 1.00 | 0.0000 | 0 | 1.0e-06 | 3.8e-14 | 30.41 |
| unequal_weights | 100 | curvature only | 0.50 | 0.3031 | 0 | 0.3031 | - | 5.99 |
| unequal_weights | 100 | sentinel only | 1.00 | 0.0000 | 0 | 1.0e-06 | 5.2e-14 | 25.10 |
| unequal_covariance | 4 | both | 1.00 | 0.0000 | 0 | 0.0030 | 1.2e-15 | 3.21 |
| unequal_covariance | 4 | curvature only | 0.25 | 0.7866 | 0 | 0.7866 | - | 1.10 |
| unequal_covariance | 4 | sentinel only | 1.00 | 0.0000 | 0 | 0.0030 | 1.2e-15 | 3.55 |
| unequal_covariance | 16 | both | 1.00 | 0.0000 | 0 | 5.2e-04 | 3.2e-15 | 6.52 |
| unequal_covariance | 16 | curvature only | 0.50 | 0.5035 | 0 | 0.5035 | - | 2.77 |
| unequal_covariance | 16 | sentinel only | 1.00 | 0.0000 | 0 | 5.2e-04 | 5.8e-15 | 5.64 |
| unequal_covariance | 64 | both | 1.00 | 0.0000 | 0 | 1.3e-04 | 1.4e-14 | 22.01 |
| unequal_covariance | 64 | curvature only | 0.50 | 0.5304 | 0 | 0.5304 | - | 6.09 |
| unequal_covariance | 64 | sentinel only | 1.00 | 0.0000 | 0 | 1.3e-04 | 1.9e-14 | 17.24 |
| unequal_covariance | 100 | both | 1.00 | 0.0000 | 0 | 8.6e-05 | 7.2e-14 | 32.13 |
| unequal_covariance | 100 | curvature only | 0.50 | 0.5497 | 0 | 0.5497 | - | 7.83 |
| unequal_covariance | 100 | sentinel only | 1.00 | 0.0000 | 0 | 8.6e-05 | 6.7e-14 | 25.60 |
| narrow_broad | 4 | both | 1.00 | 0.0000 | 0 | 0.0011 | - | 1.46 |
| narrow_broad | 4 | curvature only | 0.50 | 0.2323 | 0 | 0.2323 | - | 0.89 |
| narrow_broad | 4 | sentinel only | 1.00 | 0.0000 | 0 | 0.0011 | - | 1.46 |
| narrow_broad | 16 | both | 1.00 | 0.0000 | 0 | 2.0e-04 | - | 2.86 |
| narrow_broad | 16 | curvature only | 0.50 | 0.2434 | 0 | 0.2434 | - | 1.17 |
| narrow_broad | 16 | sentinel only | 1.00 | 0.0000 | 0 | 2.0e-04 | - | 2.87 |
| narrow_broad | 64 | both | 1.00 | 0.0000 | 0 | 5.5e-05 | - | 10.73 |
| narrow_broad | 64 | curvature only | 0.50 | 0.3069 | 0 | 0.3069 | - | 1.66 |
| narrow_broad | 64 | sentinel only | 1.00 | 0.0000 | 0 | 5.5e-05 | - | 10.61 |
| narrow_broad | 100 | both | 1.00 | 0.0000 | 0 | 3.8e-05 | - | 16.33 |
| narrow_broad | 100 | curvature only | 0.50 | 0.3607 | 0 | 0.3607 | - | 2.00 |
| narrow_broad | 100 | sentinel only | 1.00 | 0.0000 | 0 | 3.8e-05 | - | 16.39 |
| remote_small | 4 | both | 1.00 | 0.0000 | 0 | 2.0e-06 | - | 1.66 |
| remote_small | 4 | curvature only | 0.50 | 0.1013 | 0 | 0.1013 | - | 0.88 |
| remote_small | 4 | sentinel only | 1.00 | 0.0000 | 0 | 2.0e-06 | - | 1.65 |
| remote_small | 16 | both | 1.00 | 0.0000 | 0 | 1.4e-07 | - | 3.43 |
| remote_small | 16 | curvature only | 0.50 | 0.0988 | 0 | 0.0988 | - | 1.17 |
| remote_small | 16 | sentinel only | 1.00 | 0.0000 | 0 | 1.4e-07 | - | 3.39 |
| remote_small | 64 | both | 1.00 | 0.0000 | 0 | 2.7e-08 | - | 11.38 |
| remote_small | 64 | curvature only | 0.50 | 0.0982 | 0 | 0.0982 | - | 1.74 |
| remote_small | 64 | sentinel only | 1.00 | 0.0000 | 0 | 2.7e-08 | - | 11.32 |
| remote_small | 100 | both | 1.00 | 0.0000 | 0 | 1.7e-08 | - | 16.88 |
| remote_small | 100 | curvature only | 0.50 | 0.0981 | 0 | 0.0981 | - | 2.12 |
| remote_small | 100 | sentinel only | 1.00 | 0.0000 | 0 | 1.7e-08 | - | 17.00 |

### Hierarchical

| d_g x d_l | n | method | W1 (global) | recall | B / true | s |
|---|---|---|---|---|---|---|
| 1x1 | 1 | oracle_known | 0.0986 | 1.00 | 2/2 | 0.04 |
| 1x1 | 1 | oracle_tree | 0.0567 | 1.00 | 2/2 | 1.46 |
| 1x1 | 1 | gauss_jacobian_sampler | 0.1590 | - | -/- | 0.64 |
| 1x1 | 1 | fnpe_sampler | 0.1632 | - | -/- | 0.70 |
| 1x1 | 4 | oracle_known | 0.0729 | 1.00 | 2/2 | 0.03 |
| 1x1 | 4 | oracle_tree | FAILED | - | - | - |
| 1x1 | 4 | gauss_jacobian_sampler | 0.0636 | - | -/- | 1.94 |
| 1x1 | 4 | fnpe_sampler | 0.1393 | - | -/- | 1.69 |
| 1x1 | 16 | oracle_known | 0.0382 | 1.00 | 2/2 | 0.03 |
| 1x1 | 16 | oracle_tree | FAILED | - | - | - |
| 1x1 | 16 | gauss_jacobian_sampler | 0.2169 | - | -/- | 6.05 |
| 1x1 | 16 | fnpe_sampler | 0.2604 | - | -/- | 4.33 |
| 1x1 | 64 | oracle_known | 0.0081 | 1.00 | 2/2 | 0.06 |
| 1x1 | 64 | oracle_tree | FAILED | - | - | - |
| 1x1 | 64 | gauss_jacobian_sampler | 0.0706 | - | -/- | 22.13 |
| 1x1 | 64 | fnpe_sampler | 0.4793 | - | -/- | 16.82 |
| 1x1 | 100 | oracle_known | 0.0102 | 1.00 | 2/2 | 0.08 |
| 1x1 | 100 | oracle_tree | 0.0868 | 1.00 | 2/2 | 19.06 |
| 1x1 | 100 | gauss_jacobian_sampler | 0.0431 | - | -/- | 34.17 |
| 1x1 | 100 | fnpe_sampler | 0.4699 | - | -/- | 23.38 |
| 2x1 | 1 | oracle_known | - | 1.00 | 2/2 | 0.03 |
| 2x1 | 1 | oracle_tree | - | 1.00 | 2/2 | 2.04 |
| 2x1 | 1 | gauss_jacobian_sampler | - | - | -/- | 1.03 |
| 2x1 | 1 | fnpe_sampler | - | - | -/- | 0.80 |
| 2x1 | 4 | oracle_known | - | 1.00 | 2/2 | 0.03 |
| 2x1 | 4 | oracle_tree | FAILED | - | - | - |
| 2x1 | 4 | gauss_jacobian_sampler | - | - | -/- | 3.03 |
| 2x1 | 4 | fnpe_sampler | - | - | -/- | 2.08 |
| 2x1 | 16 | oracle_known | - | 1.00 | 2/2 | 0.04 |
| 2x1 | 16 | oracle_tree | - | 1.00 | 2/2 | 18.76 |
| 2x1 | 16 | gauss_jacobian_sampler | - | - | -/- | 10.04 |
| 2x1 | 16 | fnpe_sampler | - | - | -/- | 5.16 |
| 2x1 | 64 | oracle_known | - | 1.00 | 2/2 | 0.08 |
| 2x1 | 64 | oracle_tree | - | 1.00 | 2/2 | 34.47 |
| 2x1 | 64 | gauss_jacobian_sampler | - | - | -/- | 39.40 |
| 2x1 | 64 | fnpe_sampler | - | - | -/- | 18.90 |
| 2x1 | 100 | oracle_known | - | 1.00 | 2/2 | 0.10 |
| 2x1 | 100 | oracle_tree | - | 1.00 | 2/2 | 35.42 |
| 2x1 | 100 | gauss_jacobian_sampler | - | - | -/- | 61.50 |
| 2x1 | 100 | fnpe_sampler | - | - | -/- | 29.06 |
| 7x3 | 1 | oracle_known | - | 1.00 | 2/2 | 0.04 |
| 7x3 | 1 | oracle_tree | - | 1.00 | 2/2 | 2.34 |
| 7x3 | 1 | gauss_jacobian_sampler | - | - | -/- | 2.57 |
| 7x3 | 1 | fnpe_sampler | - | - | -/- | 1.26 |
| 7x3 | 4 | oracle_known | - | 1.00 | 2/2 | 0.06 |
| 7x3 | 4 | oracle_tree | - | 1.00 | 2/2 | 23.71 |
| 7x3 | 4 | gauss_jacobian_sampler | - | - | -/- | 8.94 |
| 7x3 | 4 | fnpe_sampler | - | - | -/- | 3.09 |
| 7x3 | 16 | oracle_known | - | 1.00 | 2/2 | 0.11 |
| 7x3 | 16 | oracle_tree | - | 1.00 | 2/2 | 28.60 |
| 7x3 | 16 | gauss_jacobian_sampler | - | - | -/- | 36.87 |
| 7x3 | 16 | fnpe_sampler | - | - | -/- | 10.88 |
| 7x3 | 64 | oracle_known | - | 1.00 | 2/2 | 0.34 |
| 7x3 | 64 | oracle_tree | - | 1.00 | 2/2 | 50.31 |
| 7x3 | 64 | gauss_jacobian_sampler | - | - | -/- | 169.62 |
| 7x3 | 64 | fnpe_sampler | - | - | -/- | 57.70 |
| 7x3 | 100 | oracle_known | - | 1.00 | 2/2 | 0.57 |
| 7x3 | 100 | oracle_tree | - | 1.00 | 2/2 | 65.89 |
| 7x3 | 100 | gauss_jacobian_sampler | - | - | -/- | 273.45 |
| 7x3 | 100 | fnpe_sampler | - | - | -/- | 94.37 |

### Score-error accumulation with n

| error shape | magnitude | alpha | |dS| (n=1) | |dS| (n=100) | e_MAP (n=100) | weight TV (n=100) | modes found |
|---|---|---|---|---|---|---|---|
| systematic | 0.01 | +1.000 | 0.0100 | 1.0000 | 0.0100 | 0.3665 | 2 |
| systematic | 0.05 | +1.000 | 0.0500 | 5.0000 | 0.0502 | 0.3775 | 2 |
| random | 0.01 | +0.492 | 0.0082 | 0.0743 | 2.0e-04 | 0.0186 | 2 |
| random | 0.05 | +0.492 | 0.0408 | 0.3717 | 9.9e-04 | 0.0955 | 2 |
| state | 0.01 | +1.000 | 0.0079 | 0.7854 | 0.0097 | 3.5e-06 | 2 |
| state | 0.05 | +1.000 | 0.0393 | 3.9271 | 0.0486 | 1.5e-05 | 2 |

### Calibration (188 datasets, n = 16)

| nominal | empirical |
|---|---|
| 0.5 | 0.399 |
| 0.8 | 0.862 |
| 0.9 | 0.952 |
| 0.95 | 0.973 |

(* = exactly tied mode heights; the MAP identity is ambiguous and the error is taken to the nearest tied mode.)
