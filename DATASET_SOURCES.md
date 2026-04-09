# Public PCOS Dataset Notes

Current structured clinical model status:
- Best local holdout accuracy is 94.50% in `results/model_metrics.csv`.
- The current production-friendly tabular model is trained on the 45-column clinical dataset in `dataset/data1.csv`.

## What can be merged directly

At the moment, I have not found a clearly additive public tabular dataset that matches the same clinical schema well enough to merge safely into `dataset/data1.csv`.

One public companion file is downloadable, but it is not a real batch of new patient rows:
- PCOS_EDA infertility subset:
  - Source: <https://github.com/powellsarahbeth/PCOS_EDA/blob/main/PCOS_infertility.csv>
  - Notes: 541 rows and only a few hormone/label fields. It overlaps heavily with the local clinical dataset, so using it as "extra data" would likely duplicate patients and inflate metrics.

## Good public datasets that are usable with a separate pipeline

### 1. Ultrasound image dataset
- Source: <https://zenodo.org/records/14592001>
- Title: `PCOSGen-train dataset`
- Format: `updated train dataset.zip`
- Size: about 72.3 MB
- License: CC BY 4.0
- Notes: 3200 labeled ultrasound images for normal/abnormal and polycystic-ovary visibility classification. This is a strong candidate for a separate image model, not for row-wise merging into the current tabular classifier.

### 2. Transcriptomics dataset
- Source: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138518>
- Title: `Transcriptome Profiling Reveals the key genes and pathways involved in polycystic ovary syndrome [RNA-seq]`
- Format: GEO series plus supplementary `GSE138518_RNA.xlsx`
- Notes: 6 granulosa-cell samples. Useful for research and biomarker exploration, but too small and too different from the app's current clinical input schema to merge directly.

### 3. Older microarray dataset
- Source: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE5850>
- Title: `Microarray analysis of NL and PCOS oocytes`
- Format: GEO series matrix and raw TAR
- Notes: 12 samples total. Useful for gene-signature exploration, not for direct integration into the current clinical model.

### 4. Small case-control biochemical dataset
- Source: <https://data.mendeley.com/datasets/ny225jgpwf/1>
- Title: `PCOS`
- License: CC BY 4.0
- Notes: 56 PCOS women and 62 controls with insulin-resistance and enzyme markers such as BuChE and PON1.
- Download check: the public archive is downloadable from Mendeley, but it only contains `Alivand-Ethics-Eng.pdf` and `Tables.docx`.
- Practical limitation: `Tables.docx` contains summary statistics tables, not row-level patient records, so it is not usable as direct training data for this repo.

## Recommendation

For this repo, the safest next step is:
1. Keep the current clinical model as the main prediction path.
2. If we want more data, add a separate ultrasound model trained on PCOSGen.
3. Keep omics datasets in a research-only pipeline unless we redesign the app inputs.

Mixing image or omics rows into the current clinical table would make the model less trustworthy, not more accurate.
