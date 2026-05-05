# Datasets

Raw datasets are not committed. Download each into the directory shown
below; the listed preprocessing script (if any) writes the processed
form consumed by the experiments.

## Adult Census

- **Source:** UCI Machine Learning Repository.
  <https://archive.ics.uci.edu/dataset/2/adult>
- **Place under:** `data/adult/`
- **Preprocessing:** None — the loader in `pcrl/data/adult.py` reads the
  UCI files directly.

## HMDA 2023 California LAR

- **Source:** FFIEC HMDA Data Publication, 2023 LAR, state of California.
  <https://ffiec.cfpb.gov/data-publication/>
- **Place under:** `data/hmda_raw/hmda_2023_ca.csv`
- **Preprocessing:** `python experiments/prepare_hmda.py` →
  writes `data/hmda_processed/`.

## Diabetes 130-US Hospitals

- **Source:** UCI Machine Learning Repository, "Diabetes 130-US hospitals
  for years 1999-2008."
  <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
- **Place under:** `data/diabetes/dataset_diabetes/` (after extracting
  the UCI archive)
- **Preprocessing:** `python experiments/preprocess_diabetes.py` →
  writes `data/diabetes_processed/`.

## CelebA

- **Source:** MMLAB CelebA project page (CUHK).
  <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
- **Place under:** `data/celeba/` containing `img_align_celeba/` plus
  `list_attr_celeba.csv` and `list_eval_partition.csv`.
- **Preprocessing:** None — `pcrl/vision/dataset.py` reads the official
  layout directly.

## BIOS (Bias in Bios, De-Arteaga et al. 2019)

- **Source:** Reconstruction code from the original authors at
  <https://github.com/microsoft/biosbias>. Run that scraper to produce
  the BIOS pickle file from Common Crawl.
- **Place under:** `data/bios/` (top-10 occupations subset is built
  by the loader at runtime).
- **Preprocessing:** None — the loader in `pcrl/language/bios.py` reads
  the reconstructed pickle.
