# Chest Deduplicated Split Audit

## Purpose

This audit checks hash-level duplicate content and creates deduplicated C2/C3 chest X-ray splits. C3 is evaluated on the exact same deduplicated C2 test set.

## Deduplication summary

| name   |   input_rows |   valid_existing_rows |   conflict_hashes |   conflict_rows_removed |   dedup_rows |   duplicate_rows_removed |
|:-------|-------------:|----------------------:|------------------:|------------------------:|-------------:|-------------------------:|
| C2     |        23249 |                 23249 |                 0 |                       0 |        17353 |                     5896 |
| C3     |        32877 |                 32877 |                 0 |                       0 |        26935 |                     5942 |

## C2 hash leakage check

| a     | b    |   overlap_hashes |
|:------|:-----|-----------------:|
| train | val  |                0 |
| train | test |                0 |
| val   | test |                0 |

## C3 hash leakage check

| a     | b    |   overlap_hashes |
|:------|:-----|-----------------:|
| train | val  |                0 |
| train | test |                0 |
| val   | test |                0 |

## C2 split counts

| split   | dataset              | canonical_label   |   count |
|:--------|:---------------------|:------------------|--------:|
| test    | chest_xray_pneumonia | normal            |     229 |
| test    | chest_xray_pneumonia | pneumonia         |     651 |
| test    | covid19_radiography  | normal            |    1537 |
| test    | covid19_radiography  | pneumonia         |     186 |
| train   | chest_xray_pneumonia | normal            |    1133 |
| train   | chest_xray_pneumonia | pneumonia         |    2959 |
| train   | covid19_radiography  | normal            |    7105 |
| train   | covid19_radiography  | pneumonia         |     950 |
| val     | chest_xray_pneumonia | normal            |     217 |
| val     | chest_xray_pneumonia | pneumonia         |     635 |
| val     | covid19_radiography  | normal            |    1549 |
| val     | covid19_radiography  | pneumonia         |     202 |

## C3 split counts

| split   | dataset              | canonical_label   |   count |
|:--------|:---------------------|:------------------|--------:|
| test    | chest_xray_pneumonia | normal            |     229 |
| test    | chest_xray_pneumonia | pneumonia         |     651 |
| test    | covid19_radiography  | normal            |    1537 |
| test    | covid19_radiography  | pneumonia         |     186 |
| train   | chest_xray_pneumonia | normal            |    1170 |
| train   | chest_xray_pneumonia | pneumonia         |    3052 |
| train   | covid19_radiography  | normal            |    7333 |
| train   | covid19_radiography  | pneumonia         |    9127 |
| val     | chest_xray_pneumonia | normal            |     180 |
| val     | chest_xray_pneumonia | pneumonia         |     542 |
| val     | covid19_radiography  | normal            |    1321 |
| val     | covid19_radiography  | pneumonia         |    1607 |