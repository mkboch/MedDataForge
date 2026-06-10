# MedDataForge Registry Audit Report

## Key counts

- **total_extracted_datasets**: 659
- **datasets_with_primary_url**: 659
- **datasets_with_paper_link**: 372
- **datasets_with_leaderboard_or_challenge**: 214
- **datasets_with_license_link**: 0
- **locked_primary_candidates**: 19
- **locked_chest_xray_primary_candidates**: 9
- **locked_skin_lesion_primary_candidates**: 10
- **wave1_all_candidates**: 10
- **wave1_no_kaggle_no_physionet_scriptable**: 5
- **wave1_kaggle_candidates**: 3
- **wave2_core_gated_or_manual_candidates**: 9
- **kaggle_json_exists**: False
- **physionet_netrc_exists**: False

## Locked primary candidates by domain

| locked_domain   |   count |
|:----------------|--------:|
| skin_lesion     |      10 |
| chest_xray      |       9 |


## Download feasibility

| download_feasibility                                  |   count |
|:------------------------------------------------------|--------:|
| manual_review_needed                                  |       7 |
| likely_scriptable_or_api_download                     |       4 |
| scriptable_if_kaggle_api_configured                   |       3 |
| scriptable_if_physionet_credentials_and_dua_available |       2 |
| manual_terms_or_registration_likely                   |       1 |
| manual_review_or_registration_possible                |       1 |
| likely_scriptable_from_isic_challenge_archive         |       1 |


## Wave priority

| first_wave_priority                   |   count |
|:--------------------------------------|--------:|
| wave1_scriptable_or_easy              |      10 |
| wave2_manual_review                   |       5 |
| wave2_core_but_access_gated_or_manual |       4 |


## Wave 1 candidates

| dataset_name                   | locked_domain   | url_status   |   http_status | source_type_inferred   | download_feasibility                          | first_wave_priority      | primary_url                                                                     |
|:-------------------------------|:----------------|:-------------|--------------:|:-----------------------|:----------------------------------------------|:-------------------------|:--------------------------------------------------------------------------------|
| COVID-19 Radiography Database  | chest_xray      | alive        |           200 | kaggle                 | scriptable_if_kaggle_api_configured           | wave1_scriptable_or_easy | https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database     |
| Chest X-Ray Images (Pneumonia) | chest_xray      | alive        |           200 | kaggle                 | scriptable_if_kaggle_api_configured           | wave1_scriptable_or_easy | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia          |
| ChestX-ray14                   | chest_xray      | not_found    |           404 | www.v7labs.com         | manual_review_needed                          | wave1_scriptable_or_easy | https://www.v7labs.com/open-datasets/chestx-ray14                               |
| ChestX-ray8                    | chest_xray      | alive        |           200 | huggingface            | likely_scriptable_or_api_download             | wave1_scriptable_or_easy | https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset                |
| Indiana U. Chest X-rays        | chest_xray      | alive        |           200 | kaggle                 | scriptable_if_kaggle_api_configured           | wave1_scriptable_or_easy | https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university           |
| DDI                            | skin_lesion     | alive        |           200 | ddi-dataset.github.io  | manual_review_needed                          | wave1_scriptable_or_easy | https://ddi-dataset.github.io                                                   |
| Fitzpatrick 17k                | skin_lesion     | alive        |           200 | github                 | likely_scriptable_or_api_download             | wave1_scriptable_or_easy | https://github.com/mattgroh/fitzpatrick17k                                      |
| HAM10000                       | skin_lesion     | alive        |           202 | harvard_dataverse      | likely_scriptable_or_api_download             | wave1_scriptable_or_easy | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T |
| ISIC                           | skin_lesion     | alive        |           200 | isic                   | likely_scriptable_from_isic_challenge_archive | wave1_scriptable_or_easy | https://challenge.isic-archive.com/data                                         |
| PAD-UFES-20                    | skin_lesion     | alive        |           200 | mendeley_data          | likely_scriptable_or_api_download             | wave1_scriptable_or_easy | https://data.mendeley.com/datasets/zr7vgbcyr2/1                                 |


## Wave 2 candidates

| dataset_name                      | locked_domain   | url_status   |   http_status | source_type_inferred                     | download_feasibility                                  | first_wave_priority                   | primary_url                                                                     |
|:----------------------------------|:----------------|:-------------|--------------:|:-----------------------------------------|:------------------------------------------------------|:--------------------------------------|:--------------------------------------------------------------------------------|
| CheXpert                          | chest_xray      | alive        |           200 | stanford_aimi                            | manual_terms_or_registration_likely                   | wave2_core_but_access_gated_or_manual | https://stanfordmlgroup.github.io/competitions/chexpert/                        |
| MIMIC-CXR-JPG                     | chest_xray      | alive        |           200 | physionet                                | scriptable_if_physionet_credentials_and_dua_available | wave2_core_but_access_gated_or_manual | https://physionet.org/content/mimic-cxr-jpg                                     |
| PadChest                          | chest_xray      | alive        |           200 | bimcv                                    | manual_review_or_registration_possible                | wave2_core_but_access_gated_or_manual | https://bimcv.cipf.es/bimcv-projects/padchest                                   |
| VinDr-PCXR                        | chest_xray      | alive        |           200 | physionet                                | scriptable_if_physionet_credentials_and_dua_available | wave2_core_but_access_gated_or_manual | https://physionet.org/content/vindr-pcxr                                        |
| Dermofit Image Library            | skin_lesion     | alive        |           200 | licensing.edinburgh-innovations.ed.ac.uk | manual_review_needed                                  | wave2_manual_review                   | https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library |
| Dermoscopy and Dermatoscopy Atlas | skin_lesion     | alive        |           200 | www.dermoscopyatlas.com                  | manual_review_needed                                  | wave2_manual_review                   | https://www.dermoscopyatlas.com                                                 |
| MED-NODE                          | skin_lesion     | alive        |           200 | www.cs.rug.nl                            | manual_review_needed                                  | wave2_manual_review                   | https://www.cs.rug.nl/~imaging/databases/melanoma_naevi                         |
| Melanoma Dataset                  | skin_lesion     | alive        |           200 | www.uco.es                               | manual_review_needed                                  | wave2_manual_review                   | https://www.uco.es/grupos/ayrna/ieeetmi2015                                     |
| PH 2                              | skin_lesion     | alive        |           200 | www.fc.up.pt                             | manual_review_needed                                  | wave2_manual_review                   | https://www.fc.up.pt/addi/ph2%20database.html                                   |