# MedDataForge

MedDataForge is an auditable metadata-guided experiment compiler for detecting dataset and label-space incompatibility in medical imaging experiments.

This repository contains the code needed to reproduce the paper experiments, including:

- dataset registry extraction and auditing,
- skin-lesion metadata compilation,
- chest X-ray metadata compilation,
- protocol generation,
- training and evaluation scripts,
- aggregation scripts,
- hash-level chest deduplication audit.

## What is not included

This repository does not redistribute raw medical imaging datasets, trained model checkpoints, generated paper figures, or Overleaf manuscript files. Raw datasets must be downloaded from their original sources according to their respective terms.

Large generated folders such as `data/images_raw/`, `data/images_raw_archives/`, `results/chest_baselines/`, `results/chest_dedup_baselines/`, checkpoints, logs, and paper assets are intentionally excluded.

## Environment

The experiments were run with Python 3.13.11, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.8.0, and Pillow 12.2.0 on Linux with CUDA 12.4 and NVIDIA H100 80GB GPUs.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: exact package versions are captured in `requirements.txt`. For GPU training, use a CUDA-enabled PyTorch build compatible with the local driver and hardware.

## Repository structure

```text
MedDataForge/
├── scripts/
├── docs/
├── audit_manifests/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Data requirements

Raw datasets are not included in this repository. They must be downloaded from their original sources.

The experiments used public or publicly accessible datasets including:

- HAM10000,
- ISIC 2019,
- Chest X-Ray Images (Pneumonia),
- COVID-19 Radiography Database,
- Indiana University Chest X-rays.

Some datasets may require manual download, registration, or credentialed access depending on the provider. The scripts are designed to compile manifests and protocols after the required raw files and metadata are placed in the expected local directories.

## Reproduction overview

The main scripts are in `scripts/`.

A typical reproduction sequence is shown below. Some steps depend on local dataset availability and may require adjusting paths or completing dataset-specific download steps first.

### 1. Registry extraction and candidate selection

```bash
python scripts/01_extract_registry.py
python scripts/02_select_locked_domain_candidates.py
python scripts/03_audit_candidate_links.py
python scripts/04_prepare_download_manifests.py
python scripts/05_make_registry_audit_report.py
```

### 2. Skin-lesion metadata and protocol compilation

```bash
python scripts/07_build_and_download_metadata_manifest.py
python scripts/08_redownload_and_profile_metadata.py
python scripts/09_build_label_compatibility_compiler.py
python scripts/10_compile_skin_experiment_manifests.py
python scripts/13_link_images_and_compile_protocols.py
python scripts/14_create_skin_training_splits.py
```

### 3. Skin-lesion model experiments

```bash
bash scripts/18_run_p3_p4_replicate_seeds.sh
python scripts/19_aggregate_p3_p4_seed_results.py
python scripts/21_per_class_p3_p4_analysis.py
python scripts/23_recompute_shared7_metrics_corrected.py
python scripts/25_export_predictions_for_bootstrap.py
python scripts/26_bootstrap_p3_p4_predictions.py

bash scripts/28_run_efficientnet_p3_p4_seeds.sh
python scripts/29_aggregate_efficientnet_p3_p4.py
python scripts/30_eval_efficientnet_p4_shared7.py
python scripts/31_combine_backbone_results.py
```

### 4. Chest X-ray metadata and protocol compilation

```bash
python scripts/32_prepare_chest_xray_access_plan.py
bash scripts/33_download_kaggle_chest_archives.sh
python scripts/34b_resume_extract_profile_kaggle_chest.py
python scripts/35_compile_kaggle_chest_manifests.py
python scripts/36_create_chest_binary_stress_protocols.py
```

### 5. Original chest baseline experiments

These scripts reproduce the non-deduplicated C2/C3 chest analysis used as an internal audit.

```bash
bash scripts/37_run_chest_resnet18_c2_c3_seeds.sh
python scripts/38_aggregate_chest_resnet18_c2_c3.py
python scripts/39_eval_chest_c3_on_c2_test.py

bash scripts/42_run_chest_efficientnet_b0_c2_c3_seeds.sh
python scripts/43_aggregate_chest_efficientnet_b0_c2_c3.py
python scripts/44_eval_chest_efficientnet_b0_c3_on_c2_test.py
```

### 6. Final hash-deduplicated chest analysis

The final main paper uses the hash-deduplicated chest analysis.

```bash
python scripts/46_make_chest_dedup_splits.py
bash scripts/47_run_chest_dedup_overnight.sh
python scripts/48_update_paper_assets_with_dedup_chest.py
```

## Final main results

### Skin lesion classification

Corrected shared seven-class evaluation:

- EfficientNet-B0 compiler-filtered protocol: 0.7540 ± 0.0157 macro-F1
- EfficientNet-B0 naive protocol: 0.7440 ± 0.0147 macro-F1
- Difference: +0.0101

- ResNet18 compiler-filtered protocol: 0.6961 ± 0.0251 macro-F1
- ResNet18 naive protocol: 0.6226 ± 0.0274 macro-F1
- Difference: +0.0735

### Chest X-ray classification

The final chest results use hash-deduplicated C2/C3 splits with zero train-validation-test hash overlap.

- ResNet18 C2 compiler-compatible protocol: 0.9873 ± 0.0004 macro-F1
- ResNet18 C3 naive protocol: 0.9553 ± 0.0085 macro-F1
- Difference: +0.0320

- EfficientNet-B0 C2 compiler-compatible protocol: 0.9895 ± 0.0015 macro-F1
- EfficientNet-B0 C3 naive protocol: 0.9669 ± 0.0067 macro-F1
- Difference: +0.0226

## Deduplication audit summary

The chest deduplication audit found:

- C2 input rows: 23,249
- C2 deduplicated rows: 17,353
- C2 duplicate rows removed: 5,896
- C3 input rows: 32,877
- C3 deduplicated rows: 26,935
- C3 duplicate rows removed: 5,942
- label-conflicting hashes: 0
- C2 train-validation hash overlap: 0
- C2 train-test hash overlap: 0
- C2 validation-test hash overlap: 0
- C3 train-validation hash overlap: 0
- C3 train-test hash overlap: 0
- C3 validation-test hash overlap: 0

## Notes on reproducibility

The scripts were developed for the project directory layout used during the paper experiments. If reproducing on a new machine, create the expected `data/` and `results/` folders or modify paths in the scripts as needed.

The repository intentionally excludes raw datasets and large generated outputs. This follows common paper-reproducibility practice: the repository provides the code and instructions needed to regenerate the artifacts, while raw medical datasets must be obtained from the original sources.

## Citation

Citation information will be added after publication.

## License

This code is released under the MIT License. Dataset licenses and terms remain governed by the original dataset providers.
