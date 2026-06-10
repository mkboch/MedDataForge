# Final Experiment Summary with Deduplicated Chest Results

| Domain      | Backbone        | Compiler-aware protocol   | Naive protocol   |   Macro-F1 difference | Fair comparison                             |
|:------------|:----------------|:--------------------------|:-----------------|----------------------:|:--------------------------------------------|
| Skin lesion | EfficientNet-B0 | 0.7540 ± 0.0157           | 0.7440 ± 0.0147  |                0.0101 | Corrected shared seven-class test           |
| Skin lesion | ResNet18        | 0.6961 ± 0.0251           | 0.6226 ± 0.0274  |                0.0735 | Corrected shared seven-class test           |
| Chest X-ray | ResNet18        | 0.9873 ± 0.0004           | 0.9553 ± 0.0085  |                0.032  | Deduplicated same C2-compatible binary test |
| Chest X-ray | EfficientNet-B0 | 0.9895 ± 0.0015           | 0.9669 ± 0.0067  |                0.0226 | Deduplicated same C2-compatible binary test |

Chest X-ray rows use hash-level deduplicated C2/C3 splits with zero train/validation/test hash leakage.
