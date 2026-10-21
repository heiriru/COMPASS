
### N = 16, delta = 0  (400 datasets)

| criterion | arm | rate M1 | 99% CI | predicted | agreement with exact |
|---|---|---|---|---|---|
| aic | exact | 0.160 | [0.118, 0.213] | 0.157 (yes) | 1.000 |
| aic | mle | 0.155 | [0.114, 0.207] | 0.157 (yes) | 0.995 |
| aic | map | 0.117 | [0.082, 0.165] | 0.102 (yes) | 0.958 |
| aicc | exact | 0.000 | [0.000, 0.016] | 0.002 (yes) | 1.000 |
| aicc | mle | 0.000 | [0.000, 0.016] | 0.002 (yes) | 1.000 |
| aicc | map | 0.000 | [0.000, 0.016] | 0.000 (yes) | 1.000 |
| bic | exact | 0.077 | [0.050, 0.119] | 0.063 (yes) | 1.000 |
| bic | mle | 0.077 | [0.050, 0.119] | 0.063 (yes) | 1.000 |
| bic | map | 0.033 | [0.016, 0.064] | 0.032 (yes) | 0.955 |

- LR error, MLE plug-in: mean -0.0118, sd 0.0159, max |.| 0.0817
- LR error, MAP plug-in: mean -0.2365, sd 0.3215; median LR_MAP / LR_exact (LR_exact > 1) = 0.761 (predicted 0.75)
- g MLE error: mean -1.33e-06, max |.| 2.02e-04 (0.001 posterior sd)
- M1 log-lik error at MLE vs exact (smoothed): mean +0.2619 nats/dataset; M0: +0.2673
- datasets where M1's MLE touched the box: 0; M0: 0; rows falling back to the MAP (M1): 0
- seconds per dataset (both models): 38.6

### N = 16, delta = 0.5  (200 datasets)

| criterion | arm | rate M1 | 99% CI | predicted | agreement with exact |
|---|---|---|---|---|---|
| aic | exact | 0.740 | [0.653, 0.811] | 0.721 (yes) | 1.000 |
| aic | mle | 0.735 | [0.648, 0.807] | 0.721 (yes) | 0.995 |
| aic | map | 0.670 | [0.580, 0.749] | 0.643 (yes) | 0.930 |
| aicc | exact | 0.160 | [0.104, 0.238] | 0.141 (yes) | 1.000 |
| aicc | mle | 0.155 | [0.100, 0.232] | 0.141 (yes) | 0.995 |
| aicc | map | 0.080 | [0.043, 0.144] | 0.061 (yes) | 0.920 |
| bic | exact | 0.590 | [0.499, 0.675] | 0.555 (yes) | 1.000 |
| bic | mle | 0.590 | [0.499, 0.675] | 0.555 (yes) | 1.000 |
| bic | map | 0.490 | [0.401, 0.580] | 0.441 (yes) | 0.900 |

- LR error, MLE plug-in: mean -0.0639, sd 0.0518, max |.| 0.2410
- LR error, MAP plug-in: mean -1.2602, sd 1.0285; median LR_MAP / LR_exact (LR_exact > 1) = 0.763 (predicted 0.75)
- g MLE error: mean +2.16e-06, max |.| 1.48e-04 (0.001 posterior sd)
- M1 log-lik error at MLE vs exact (smoothed): mean +0.2635 nats/dataset; M0: +0.2928
- datasets where M1's MLE touched the box: 0; M0: 0; rows falling back to the MAP (M1): 0
- seconds per dataset (both models): 39.4
