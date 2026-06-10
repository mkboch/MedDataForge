# MedDataForge Skin-Lesion Experiment Manifest Report

## Compiler decisions

- **HAM10000 + ISIC2019**: accepted for 7-class diagnostic classification after excluding ISIC2019 extension labels SCC and UNK.
- **ISIC2020**: accepted as binary melanoma-vs-non-melanoma, not directly pooled with the 7-class task.
- **Fitzpatrick17k**: rejected for direct pooling with dermoscopy challenge datasets; retained as external stress/coarse-mapping dataset.

## Counts

- **multiclass_manifest_rows**: 35346
- **multiclass_accepted_7class_rows**: 34718
- **multiclass_flagged_extension_rows**: 628
- **isic2020_binary_rows**: 33126
- **fitzpatrick_rows**: 16577
- **fitzpatrick_mapped_overlap_rows**: 1751
- **fitzpatrick_rejected_direct_pooling_rows**: 14826

## 7-class manifest label counts

| dataset   | canonical_label                                | compiler_status         |   count |
|:----------|:-----------------------------------------------|:------------------------|--------:|
| HAM10000  | actinic_keratosis_or_intraepithelial_carcinoma | accepted_7class         |     327 |
| HAM10000  | basal_cell_carcinoma                           | accepted_7class         |     514 |
| HAM10000  | benign_keratosis_like_lesion                   | accepted_7class         |    1099 |
| HAM10000  | dermatofibroma                                 | accepted_7class         |     115 |
| HAM10000  | melanocytic_nevus                              | accepted_7class         |    6705 |
| HAM10000  | melanoma                                       | accepted_7class         |    1113 |
| HAM10000  | vascular_lesion                                | accepted_7class         |     142 |
| ISIC2019  | actinic_keratosis_or_intraepithelial_carcinoma | accepted_7class         |     867 |
| ISIC2019  | basal_cell_carcinoma                           | accepted_7class         |    3323 |
| ISIC2019  | benign_keratosis_like_lesion                   | accepted_7class         |    2624 |
| ISIC2019  | dermatofibroma                                 | accepted_7class         |     239 |
| ISIC2019  | melanocytic_nevus                              | accepted_7class         |   12875 |
| ISIC2019  | melanoma                                       | accepted_7class         |    4522 |
| ISIC2019  | vascular_lesion                                | accepted_7class         |     253 |
| ISIC2019  | squamous_cell_carcinoma                        | flagged_extension_label |     628 |


## ISIC2020 binary label counts

| dataset   | canonical_label   | compiler_status          |   count |
|:----------|:------------------|:-------------------------|--------:|
| ISIC2020  | melanoma          | accepted_binary_melanoma |     584 |
| ISIC2020  | non_melanoma      | accepted_binary_melanoma |   32542 |


## Fitzpatrick17k direct-pooling decision counts

| compiler_status           |   count |
|:--------------------------|--------:|
| reject_for_direct_pooling |   14826 |
| mapped_overlap_label      |    1751 |