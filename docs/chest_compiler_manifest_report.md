# Chest Kaggle Compiler Manifest Report

## Summary

- **Chest X-Ray Images (Pneumonia)**: 11712 images
- **COVID-19 Radiography Database**: 21165 images
- **Indiana U. Chest X-rays**: 7470 images

## Compiler decisions

| dataset              |   images | task_family                                      | compiler_decision                                                         | direct_pool_with_broad_multilabel_chest   | reason                                                                                                               |
|:---------------------|---------:|:-------------------------------------------------|:--------------------------------------------------------------------------|:------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| chest_xray_pneumonia |    11712 | binary_pneumonia_classification                  | accepted only for binary pneumonia stress-test protocol                   | reject                                    | labels are normal vs pneumonia only, not a broad chest abnormality ontology                                          |
| covid19_radiography  |    21165 | covid_pneumonia_opacity_normal_multiclass        | accepted only for COVID/pneumonia/opacity multiclass stress-test protocol | reject                                    | labels are COVID, viral pneumonia, lung opacity, and normal, not equivalent to broad multilabel chest finding labels |
| indiana_chest_xrays  |     7470 | image_report_captioning_or_weak_label_extraction | reject for direct supervised classification pooling                       | reject_without_label_extraction           | dataset contains image-report pairs and projections, not directly harmonized image-level classification labels       |

## Label/task counts

| dataset              | task_family                                      | compiler_status                                | canonical_label   |   count |
|:---------------------|:-------------------------------------------------|:-----------------------------------------------|:------------------|--------:|
| chest_xray_pneumonia | binary_pneumonia_classification                  | accepted_for_binary_pneumonia_only             | pneumonia         |    8546 |
| chest_xray_pneumonia | binary_pneumonia_classification                  | accepted_for_binary_pneumonia_only             | normal            |    3166 |
| covid19_radiography  | covid_pneumonia_opacity_normal_multiclass        | accepted_for_covid_radiography_multiclass_only | normal            |   10192 |
| covid19_radiography  | covid_pneumonia_opacity_normal_multiclass        | accepted_for_covid_radiography_multiclass_only | lung_opacity      |    6012 |
| covid19_radiography  | covid_pneumonia_opacity_normal_multiclass        | accepted_for_covid_radiography_multiclass_only | covid_19          |    3616 |
| covid19_radiography  | covid_pneumonia_opacity_normal_multiclass        | accepted_for_covid_radiography_multiclass_only | viral_pneumonia   |    1345 |
| indiana_chest_xrays  | image_report_captioning_or_weak_label_extraction | rejected_for_direct_label_pooling              |                   |    7470 |

## Interpretation

These Kaggle chest datasets are all chest X-ray datasets, but the compiler does not treat them as directly equivalent. The pneumonia dataset supports a narrow binary pneumonia task, the COVID radiography dataset supports a COVID/pneumonia/opacity/normal task, and the Indiana dataset is an image-report dataset that requires label extraction before classification pooling. This provides second-domain evidence for MedDataForge's core claim: dataset compatibility must be compiled and audited before pooling.