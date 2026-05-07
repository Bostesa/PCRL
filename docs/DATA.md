# Datasets

Raw datasets are not committed. Download each into the directory shown
below; the listed preprocessing script (if any) writes the processed form
consumed by the experiments. The repository is set up so that no command
phones home — every dataset must be acquired directly from its primary
source.

## Adult Census

- **Source:** UCI Machine Learning Repository.
  <https://archive.ics.uci.edu/dataset/2/adult>
- **License:** Donor-provided, free for research use (UCI ML Repository terms).
- **Place under:** `data/adult/`
- **Preprocessing:** None — `pcrl/data/adult.py` reads the UCI files directly.
- **Approx. size on disk:** 5 MB.

## HMDA 2023 California LAR

- **Source:** FFIEC HMDA Data Publication, 2023 LAR, state of California.
  <https://ffiec.cfpb.gov/data-publication/>
- **License:** US public-domain (FFIEC public release).
- **Place under:** `data/hmda_raw/hmda_2023_ca.csv`
- **Preprocessing:**
  ```
  python experiments/prepare_hmda.py
  ```
  writes `data/hmda_processed/`. Wall-clock ≈ 2 min.
- **Approx. size on disk:** 250 MB raw, 80 MB processed.

## Diabetes 130-US Hospitals

- **Source:** UCI Machine Learning Repository, "Diabetes 130-US hospitals
  for years 1999-2008."
  <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
- **License:** Donor-provided, free for research use (UCI ML Repository terms).
- **Place under:** `data/diabetes/dataset_diabetes/` (after extracting the
  UCI archive).
- **Preprocessing:**
  ```
  python experiments/preprocess_diabetes.py
  ```
  writes `data/diabetes_processed/`. Wall-clock ≈ 30 s.
- **Approx. size on disk:** 25 MB.

## CelebA

- **Source:** MMLAB CelebA project page (CUHK).
  <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
- **License:** Per the official release: research, non-commercial use only.
  Read the project page for details before redistribution.
- **Place under:** `data/celeba/` containing `img_align_celeba/` plus
  `list_attr_celeba.csv` and `list_eval_partition.csv`.
- **Preprocessing:** None — `pcrl/vision/dataset.py` reads the official
  layout directly.
- **Approx. size on disk:** 1.4 GB.

## BIOS (Bias in Bios, De-Arteaga et al. 2019)

- **Source:** Reconstruction code from the original authors at
  <https://github.com/microsoft/biosbias>. Run that scraper to produce
  the BIOS pickle file from Common Crawl.
- **License:** Per the upstream repository.
- **Place under:** `data/bios/` (top-10 occupations subset is built by
  the loader at runtime).
- **Preprocessing:** None — `pcrl/language/bios.py` reads the
  reconstructed pickle.
- **Approx. size on disk:** 1.2 GB pickle (depends on Common Crawl scrape).

## Folktables (American Community Survey, ACS PUMS)

- **Source:** Folktables benchmark suite, <https://github.com/socialfoundations/folktables>.
  Pulls ACS PUMS tabular data via the official `folktables` Python package.
- **License:** US Census Bureau public-domain ACS PUMS release.
- **Place under:** `data/folktables/<year>/<period>/` matching the
  Folktables convention.
- **Preprocessing:** None — `pcrl/data/folktables.py` reads the layout
  produced by the `folktables` package.
- **Approx. size on disk:** 800 MB per state-year (CA 2018 1-Year).

## UCI HAR (Human Activity Recognition)

- **Source:** UCI Machine Learning Repository, "Human Activity Recognition
  Using Smartphones."
  <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
- **License:** Donor-provided, free for research use.
- **Place under:** `data/UCI HAR Dataset/` (extract the archive directly).
- **Preprocessing:** None — `pcrl/data/har.py` reads the raw text files.
- **Approx. size on disk:** 60 MB.
