# Generalization: cross-batch temporal evaluation and SDK holdout

- Fixed protocol: primary split batch≤41 for training / >41 for testing; standardization and SDK co-occurrence weights are estimated from training events only; H contains 8 directly matched SDKs from the auxiliary list, excluding Zxing; the control draws 8 SDKs from nonlisted candidates with usage≤37, accepting the first train∈[127,139] match (seed 2024); SDK holdout is a single exploratory sensitivity analysis, and all OOD splits isolate app_group.
- Leakage controls: estimate normalization and SDK co-occurrence only from training events; attach unlabeled test events at inference with frozen co-occurrence; exclude flagged/miit features; keep application groups disjoint; purge earlier events of test applications in temporal splits; assign complete groups in SDK holdout. Remediation differences are retrospective evidence.

## Experiment 1: cross-batch temporal evaluation

### X=41 (training batch≤41, n=124 / test n=91)

| model | macro-AUC | n_classes | per-class |
|---|---|---|---|
| gnn_full | 0.606±0.005 | 6 | C1=0.58 C2=0.68 C3=0.78 C4=0.53 C5=0.60 C6=0.46 |
| lr_static | 0.623±0.000 | 6 | C1=0.53 C2=0.75 C3=0.78 C4=0.47 C5=0.61 C6=0.60 |
| lr_staticdiff | 0.613±0.000 | 6 | C1=0.54 C2=0.74 C3=0.78 C4=0.51 C5=0.55 C6=0.56 |

### X=45 (training batch≤45, n=146 / test n=69)

| model | macro-AUC | n_classes | per-class |
|---|---|---|---|
| gnn_full | 0.642±0.007 | 6 | C1=0.60 C2=0.72 C3=0.85 C4=0.48 C5=0.66 C6=0.54 |
| lr_static | 0.611±0.000 | 6 | C1=0.51 C2=0.80 C3=0.80 C4=0.33 C5=0.64 C6=0.60 |
| lr_staticdiff | 0.601±0.000 | 6 | C1=0.53 C2=0.77 C3=0.80 C4=0.33 C5=0.63 C6=0.55 |

### Label-distribution context for temporal evaluation

| period | n | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|
| early(13-40) | 125 | 0.57 | 0.43 | 0.15 | 0.02 | 0.06 | 0.10 |
| late(41-53) | 92 | 0.55 | 0.28 | 0.32 | 0.11 | 0.05 | 0.11 |

## Experiment 2: SDK holdout

| setting | train_n | test_n | gnn macro-AUC | lr macro | GNN−LR | H SDKs |
|---|---|---|---|---|---|---|
| flagged | 133 | 84 | 0.556±0.022 | 0.594 | -0.038 | MobTech(ShareSDK/MobSDK)、神策数据、七牛云 SDK、TalkingData、UBiX Merak、乐播投屏、维智地图、易观方舟分析 |
| control | 132 | 85 | 0.592±0.009 | 0.607 | -0.015 | ImmersionBar、商汤科技、vivo 广告、百度语音识别、梆梆加固、极验验证、荣耀推送、友盟 U-Verify 一键登录 |

## Interpretation and limitations

- C4/C5/C6 have <12 test positives and are interpreted descriptively; the standard deviation across 3 seeds measures seed variability only.
- All z-score statistics and inference-time SDK co-occurrence weights are estimated from training events only.
- LR is deterministic and fitted once; GNN results are mean±std across 3 seeds. Held-out SDK columns are zero throughout LR training, so their coefficients cannot learn from SDK-presence variation.
- Only one training-size-matched random control was drawn. Label composition and difficulty may differ between test sets, so the comparison does not establish dependence on risk-SDK identity.
- Generalization is limited by the absence of verified clean controls (unreported does not mean clean), static features, a single notice-source system, and retrospective differences.
