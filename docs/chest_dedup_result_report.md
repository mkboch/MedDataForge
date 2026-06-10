# Chest Deduplicated Hash-Audited Results

## Main differences

| backbone        |   c2_macro_f1 |   c2_macro_f1_sd |   c3_macro_f1 |   c3_macro_f1_sd |   c2_minus_c3_macro_f1 |
|:----------------|--------------:|-----------------:|--------------:|-----------------:|-----------------------:|
| EfficientNet-B0 |      0.989453 |      0.00150201  |      0.966904 |       0.00668236 |              0.0225496 |
| ResNet18        |      0.98729  |      0.000423809 |      0.955299 |       0.00848777 |              0.0319908 |

## Aggregate results

| backbone        | protocol   | setting                               |   n_runs |   test_rows_mean |   accuracy_mean |   accuracy_std |   balanced_accuracy_mean |   balanced_accuracy_std |   macro_f1_mean |   macro_f1_std |   weighted_f1_mean |   weighted_f1_std |
|:----------------|:-----------|:--------------------------------------|---------:|-----------------:|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|-------------------:|------------------:|
| EfficientNet-B0 | C2         | Deduplicated compiler-compatible C2   |        3 |             2603 |        0.99078  |    0.00133081  |                 0.990272 |             0.000617916 |        0.989453 |    0.00150201  |           0.990788 |       0.00132083  |
| EfficientNet-B0 | C3         | Deduplicated naive C3 on same C2 test |        3 |             2603 |        0.970675 |    0.00600506  |                 0.974093 |             0.00553948  |        0.966904 |    0.00668236  |           0.970891 |       0.00592121  |
| ResNet18        | C2         | Deduplicated compiler-compatible C2   |        3 |             2603 |        0.988859 |    0.000384172 |                 0.989485 |             0.0015536   |        0.98729  |    0.000423809 |           0.988884 |       0.000376754 |
| ResNet18        | C3         | Deduplicated naive C3 on same C2 test |        3 |             2603 |        0.960943 |    0.00691865  |                 0.95697  |             0.0157113   |        0.955299 |    0.00848777  |           0.960961 |       0.00715799  |

## Seed-level results

| backbone        | protocol   |     seed | status   | setting                               | metric_file                                                                           |   train_rows |   val_rows |   test_rows |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |
|:----------------|:-----------|---------:|:---------|:--------------------------------------|:--------------------------------------------------------------------------------------|-------------:|-----------:|------------:|-----------:|--------------------:|-----------:|--------------:|
| ResNet18        | C2         | 20260609 | ok       | Deduplicated compiler-compatible C2   | results/chest_dedup_baselines/C2_resnet18_dedup_seed20260609/test_metrics.json        |        12147 |       2603 |        2603 |     0.9889 |              0.9877 |     0.9872 |        0.9889 |
| ResNet18        | C2         | 20260610 | ok       | Deduplicated compiler-compatible C2   | results/chest_dedup_baselines/C2_resnet18_dedup_seed20260610/test_metrics.json        |        12147 |       2603 |        2603 |     0.9892 |              0.9902 |     0.9877 |        0.9893 |
| ResNet18        | C2         | 20260611 | ok       | Deduplicated compiler-compatible C2   | results/chest_dedup_baselines/C2_resnet18_dedup_seed20260611/test_metrics.json        |        12147 |       2603 |        2603 |     0.9885 |              0.9906 |     0.9869 |        0.9885 |
| ResNet18        | C3         | 20260609 | ok       | Deduplicated naive C3 on same C2 test | results/chest_dedup_baselines/C3_resnet18_dedup_seed20260609/test_metrics.json        |        20682 |       3650 |        2603 |     0.9577 |              0.9557 |     0.952  |        0.9579 |
| ResNet18        | C3         | 20260610 | ok       | Deduplicated naive C3 on same C2 test | results/chest_dedup_baselines/C3_resnet18_dedup_seed20260610/test_metrics.json        |        20682 |       3650 |        2603 |     0.9689 |              0.9733 |     0.9649 |        0.9691 |
| ResNet18        | C3         | 20260611 | ok       | Deduplicated naive C3 on same C2 test | results/chest_dedup_baselines/C3_resnet18_dedup_seed20260611/test_metrics.json        |        20682 |       3650 |        2603 |     0.9562 |              0.942  |     0.949  |        0.9558 |
| EfficientNet-B0 | C2         | 20260609 | ok       | Deduplicated compiler-compatible C2   | results/chest_dedup_baselines/C2_efficientnet_b0_dedup_seed20260609/test_metrics.json |        12147 |       2603 |        2603 |     0.9915 |              0.9906 |     0.9903 |        0.9916 |
| EfficientNet-B0 | C2         | 20260610 | ok       | Deduplicated compiler-compatible C2   | results/chest_dedup_baselines/C2_efficientnet_b0_dedup_seed20260610/test_metrics.json |        12147 |       2603 |        2603 |     0.9915 |              0.9906 |     0.9903 |        0.9916 |
| EfficientNet-B0 | C2         | 20260611 | ok       | Deduplicated compiler-compatible C2   | results/chest_dedup_baselines/C2_efficientnet_b0_dedup_seed20260611/test_metrics.json |        12147 |       2603 |        2603 |     0.9892 |              0.9896 |     0.9877 |        0.9893 |
| EfficientNet-B0 | C3         | 20260609 | ok       | Deduplicated naive C3 on same C2 test | results/chest_dedup_baselines/C3_efficientnet_b0_dedup_seed20260609/test_metrics.json |        20682 |       3650 |        2603 |     0.9762 |              0.9774 |     0.973  |        0.9763 |
| EfficientNet-B0 | C3         | 20260610 | ok       | Deduplicated naive C3 on same C2 test | results/chest_dedup_baselines/C3_efficientnet_b0_dedup_seed20260610/test_metrics.json |        20682 |       3650 |        2603 |     0.9716 |              0.9772 |     0.968  |        0.9718 |
| EfficientNet-B0 | C3         | 20260611 | ok       | Deduplicated naive C3 on same C2 test | results/chest_dedup_baselines/C3_efficientnet_b0_dedup_seed20260611/test_metrics.json |        20682 |       3650 |        2603 |     0.9643 |              0.9677 |     0.9597 |        0.9646 |


Figure: `results/figures/fig_chest_dedup_macro_f1.png`