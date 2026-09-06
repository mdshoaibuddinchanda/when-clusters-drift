# Dataset Acquisition, Provenance, and Validation Architecture

This directory contains the dataset acquisition, storage, unpacking, and validation infrastructure for the benchmark study:
**When Clusters Drift: Label-Free Failure Prediction and Risk-Guided Adaptation for Soft Clustering**.

---

## 1. Quick Summary of Datasets

When everything is fully set up, the repository manages **47 total datasets and benchmark families**:

| Category | Registered | Currently Canonicalized | Status | Notes |
|---|---|---|---|---|
| **Controlled Real Datasets** | **30** | **30 / 30** | ✅ 100% Complete | Standard tabular benchmarks (sklearn + OpenML + UCI) |
| **Natural-Shift Real Datasets** | **9** | **4 / 9** (16 state/domain partitions) | 🟡 In Progress / Raw Placed | 4 canonicalized + 4 raw placed (HELOC, ASSISTments, Scorecard, Accidents) + 1 pending (Taxi train) |
| **Synthetic Benchmark Families** | **8** | **8 / 8** | ✅ 100% Complete | Gaussian mixtures & Student-t with exact soft posterior truth |
| **Total Benchmark Universe** | **47** | **42 active units / partitions** | — | Over **1,685,789 observations** in canonical format |

---

## 2. Directory Structure

```text
data/
├── raw/                      # Untransformed raw downloads cached from upstream providers (GITIGNORED)
│   ├── controlled/           # Raw OpenML / UCI downloads for controlled datasets
│   └── natural/              # Raw WhyShift and TableShift partitions
│       ├── whyshift/
│       │   ├── income/       # Raw Census PUMS state CSVs (CA, FL, NY, PA, TX)
│       │   ├── mobility/     # Raw Census PUMS state CSVs (CA, FL, NY, PA, TX)
│       │   ├── pubcov/       # Raw Census PUMS state CSVs (CA, FL, NY, PA, TX)
│       │   ├── whyshift_us_accidents/ # US_Accidents_March23.csv (~2.9 GB)
│       │   └── whyshift_taxi/         # NYC Taxi Trip Duration (test.csv placed, train.csv pending)
│       └── tableshift/
│           ├── tableshift_hospital_readmission/ # diabetes_130_us_hospitals.zip
│           ├── tableshift_college_scorecard/    # Most-Recent-Cohorts & historical files
│           ├── tableshift_heloc/                # heloc_dataset_v1.csv
│           └── tableshift_assistments/          # 2012-2013-data-with-predictions-4-final.csv (~3.0 GB)
│
├── canonical/                # Standardized, label-isolated Parquet partitions (GITIGNORED)
│   ├── controlled/           # 30 controlled real datasets (features.parquet, labels.parquet, metadata.json, optional groups.parquet)
│   └── natural/              # Natural-shift datasets separated by domain/state (optional domains.parquet)
│       ├── whyshift/         # 15 state domain partitions across income, mobility, pubcov
│       └── tableshift/       # tableshift_hospital_readmission (17 admission source domains)
│
├── synthetic/                # 8 controlled synthetic benchmark families (GITIGNORED)
│   ├── s01_balanced_gmm/     # features.parquet, hard_labels.parquet, soft_memberships.npy, parameters.json
│   ├── s02_overlap_gmm/
│   ├── s03_imbalanced_gmm/
│   ├── s04_heteroscedastic_gmm/
│   ├── s05_anisotropic_gmm/
│   ├── s06_high_dimensional_gmm/
│   ├── s07_irrelevant_features_gmm/
│   └── s08_student_t_mixture/
│
├── processed/                # Reserved strictly for experiment-time scalers/folds (Phase 2+)
├── splits/                   # Reserved strictly for source/target rotation indices (Phase 2+)
│
└── manifests/                # Cryptographic manifests, provenance records, and reports (TRACKED BY GIT)
    ├── datasets.json         # Manifest entries for all real datasets with SHA256 bundle hashes
    ├── synthetic_manifest.json # Manifest entries for synthetic generators with bundle hashes
    ├── download_report.json  # Timestamped execution report from 01_download_real_datasets.py
    ├── unavailable_datasets.json # Instructions and access endpoints for credential-gated datasets
    └── dataset_summary.csv   # Comprehensive tabular audit of row/feature counts, missingness, and hashes
```

---

## 3. How to Run, Check, and Unpack Data

### A. How to Run Data Acquisition

To acquire and canonicalize all available automated datasets:

```bash
# Acquire all registered real datasets
python scripts/01_download_real_datasets.py --all

# Or acquire a specific group:
python scripts/01_download_real_datasets.py --group controlled
python scripts/01_download_real_datasets.py --group natural

# Or acquire an individual dataset:
python scripts/01_download_real_datasets.py --dataset wine
```

To generate the 8 synthetic benchmark families:

```bash
python scripts/02_generate_synthetic_datasets.py --all
```

---

### B. How to Check and Validate Data Integrity

To audit all canonical datasets, verify label and domain isolation, and generate cryptographic manifests:

```bash
# Run comprehensive dataset validator and update manifests
python scripts/03_validate_datasets.py

# Run unit tests verifying checksums and schema integrity
pytest tests/test_dataset_checksums.py -v
pytest tests/test_dataset_integrity.py -v
```

Validation asserts that:
1. Every dataset has isolated `features.parquet`, `labels.parquet`, and `metadata.json`.
2. Group-structured datasets (`human_activity_recognition`, `mice_protein_expression`) isolate `groups.parquet` (`subject_id` and `mouse_subject_id`).
3. Domain-shift datasets (`tableshift_hospital_readmission`) isolate `domains.parquet` (`admission_source_id`).
4. Constituent files match Schema v2 cryptographic bundle hashes.

---

### C. How to Unpack and Place Credentialed / Manual Datasets

For datasets requiring manual download or platform credentials, place raw files into the appropriate folder under `data/raw/natural/`:

| Dataset | Expected Directory | File(s) Needed | Source / Instructions |
|---|---|---|---|
| **FICO HELOC** (`tableshift_heloc`) | `data/raw/natural/tableshift/tableshift_heloc/` | `heloc_dataset_v1.csv` | Accept terms at [FICO Community](https://community.fico.com/s/explainable-machine-learning-challenge) or download from [GitHub mirror](https://raw.githubusercontent.com/patrickmthisi/FICO-Homeloan-credit-classification/main/heloc_dataset_v1.csv). |
| **ASSISTments** (`tableshift_assistments`) | `data/raw/natural/tableshift/tableshift_assistments/` | `2012-2013-data-with-predictions-4-final.csv` | Download from [Kaggle ASSISTments](https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010), extract CSV. |
| **US Accidents** (`whyshift_us_accidents`) | `data/raw/natural/whyshift/whyshift_us_accidents/` | `US_Accidents_March23.csv` | Download `archive.zip` from [Kaggle US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), extract CSV. |
| **College Scorecard** (`tableshift_college_scorecard`) | `data/raw/natural/tableshift/tableshift_college_scorecard/` | `Most-Recent-Cohorts-Institution.csv` & cohorts | Download `College_Scorecard_Raw_Data_*.zip` from [College Scorecard](https://collegescorecard.ed.gov/data/), extract all files. |
| **NYC Taxi** (`whyshift_taxi`) | `data/raw/natural/whyshift/whyshift_taxi/` | `train.csv` | Download `train.zip` from [Kaggle NYC Taxi](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data), extract `train.csv`. |

#### Safe Unpacking Pattern
Always extract downloaded archives directly into their destination folder and remove the `.zip` file to save disk space and keep the repository root clean:
```powershell
# Example: Extract and clean up College Scorecard
Expand-Archive -Path "College_Scorecard_Raw_Data_06102026.zip" -DestinationPath "data/raw/natural/tableshift/tableshift_college_scorecard/"
Remove-Item "College_Scorecard_Raw_Data_06102026.zip"
```

---

## 4. Upstream Repositories and Direct Source Links

### A. Controlled Real Datasets (30 Datasets)

| Slug | Provider | Source ID | Upstream Link |
|---|---|---|---|
| `iris` | scikit-learn | `load_iris` | [scikit-learn Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) |
| `wine` | scikit-learn | `load_wine` | [scikit-learn Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) |
| `breast_cancer_wisconsin_diagnostic` | scikit-learn | `load_breast_cancer` | [scikit-learn Breast Cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |
| `seeds` | OpenML | 1499 | [OpenML 1499](https://www.openml.org/d/1499) |
| `glass` | OpenML | 41 | [OpenML 41](https://www.openml.org/d/41) |
| `ecoli` | OpenML | 39 | [OpenML 39](https://www.openml.org/d/39) |
| `yeast` | OpenML | 181 | [OpenML 181](https://www.openml.org/d/181) |
| `vehicle_silhouettes` | OpenML | 54 | [OpenML 54](https://www.openml.org/d/54) |
| `image_segmentation` | OpenML | 40984 | [OpenML 40984](https://www.openml.org/d/40984) |
| `satimage` | OpenML | 182 | [OpenML 182](https://www.openml.org/d/182) |
| `pendigits` | OpenML | 32 | [OpenML 32](https://www.openml.org/d/32) |
| `optdigits` | OpenML | 28 | [OpenML 28](https://www.openml.org/d/28) |
| `letter_recognition` | OpenML | 6 | [OpenML 6](https://www.openml.org/d/6) |
| `banknote_authentication` | OpenML | 1462 | [OpenML 1462](https://www.openml.org/d/1462) |
| `ionosphere` | OpenML | 59 | [OpenML 59](https://www.openml.org/d/59) |
| `sonar` | OpenML | 40 | [OpenML 40](https://www.openml.org/d/40) |
| `pima_diabetes` | OpenML | 37 | [OpenML 37](https://www.openml.org/d/37) |
| `heart_disease` | OpenML | 1565 | [OpenML 1565](https://www.openml.org/d/1565) |
| `haberman_survival` | OpenML | 43 | [OpenML 43](https://www.openml.org/d/43) |
| `dermatology` | OpenML | 35 | [OpenML 35](https://www.openml.org/d/35) |
| `balance_scale` | OpenML | 11 | [OpenML 11](https://www.openml.org/d/11) |
| `waveform` | OpenML | 60 | [OpenML 60](https://www.openml.org/d/60) |
| `spambase` | OpenML | 44 | [OpenML 44](https://www.openml.org/d/44) |
| `isolet` | OpenML | 300 | [OpenML 300](https://www.openml.org/d/300) |
| `madelon` | OpenML | 1485 | [OpenML 1485](https://www.openml.org/d/1485) |
| `electricity` | OpenML | 151 | [OpenML 151](https://www.openml.org/d/151) |
| `bank_marketing` | OpenML | 1461 | [OpenML 1461](https://www.openml.org/d/1461) |
| `aps_failure` | OpenML | 41138 | [OpenML 41138](https://www.openml.org/d/41138) |
| `human_activity_recognition` | UCI | 240 | [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| `mice_protein_expression` | UCI | 342 | [UCI Mice Protein Dataset](https://archive.ics.uci.edu/dataset/342/mice+protein+expression) |

---

### B. Natural-Shift Real Datasets (9 Datasets)

| Slug | Benchmark Provider | Upstream Source Link |
|---|---|---|
| `whyshift_acs_income` | WhyShift / Census PUMS | [namkoong-lab/whyshift](https://github.com/namkoong-lab/whyshift) |
| `whyshift_acs_pubcov` | WhyShift / Census PUMS | [namkoong-lab/whyshift](https://github.com/namkoong-lab/whyshift) |
| `whyshift_acs_mobility` | WhyShift / Census PUMS | [namkoong-lab/whyshift](https://github.com/namkoong-lab/whyshift) |
| `whyshift_us_accidents` | WhyShift / Kaggle | [Kaggle US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) |
| `whyshift_taxi` | WhyShift / Kaggle | [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data) |
| `tableshift_hospital_readmission` | TableShift / UCI 296 | [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| `tableshift_heloc` | TableShift / FICO | [FICO Challenge](https://community.fico.com/s/explainable-machine-learning-challenge) / [Mirror](https://raw.githubusercontent.com/patrickmthisi/FICO-Homeloan-credit-classification/main/heloc_dataset_v1.csv) |
| `tableshift_college_scorecard` | TableShift / Dept. of Ed. | [College Scorecard Data](https://collegescorecard.ed.gov/data/) |
| `tableshift_assistments` | TableShift / Kaggle | [Kaggle ASSISTments 2012–2013](https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010) |

---

## 5. Why Each Dataset is Included

### 1. 30 Controlled Real Datasets
* **Purpose**: Establish baseline clustering performance and evaluate how standard algorithms (k-means, FCM, GMM) behave under controlled artificial drift (feature dropout, covariance scaling, mean shift).
* **Composition**: Classic benchmarks spanning diverse domains (biology, medicine, computer vision, physical sensors, text processing):
  - *Small/Classical*: `iris`, `wine`, `seeds`, `glass`, `ecoli`, `yeast`, `balance_scale`, `haberman_survival`
  - *Sensor/Signal*: `sonar`, `ionosphere`, `waveform`
  - *Vision/Digits*: `vehicle_silhouettes`, `image_segmentation`, `satimage`, `pendigits`, `optdigits`, `letter_recognition`
  - *Medical/Clinical*: `breast_cancer_wisconsin_diagnostic`, `pima_diabetes`, `heart_disease`, `dermatology`
  - *Financial/Industrial*: `banknote_authentication`, `spambase`, `electricity`, `bank_marketing`, `aps_failure`, `madelon`, `isolet`
  - *Group-Structured*:
    - `human_activity_recognition` (UCI 240): $10,299 \times 561$, 6 activity classes, grouped across 30 distinct human subjects (`subject_id`).
    - `mice_protein_expression` (UCI 342): $1,080 \times 77$, 8 genotype/treatment classes, grouped across 72 biological mice (`mouse_subject_id`).

### 2. 9 Natural-Shift Real Datasets
* **Purpose**: Evaluate label-free failure prediction under genuine, naturally occurring distribution shifts across geographic states, hospital systems, educational cohorts, and socio-economic tiers.
* **Composition**:
  - *WhyShift Spatial Partitions*:
    - `whyshift_acs_income` (5 state domains: CA, TX, NY, FL, PA; N=601,843)
    - `whyshift_acs_pubcov` (5 state domains: CA, TX, NY, FL, PA; N=420,411)
    - `whyshift_acs_mobility` (5 state domains: CA, TX, NY, FL, PA; N=232,512)
    - `whyshift_us_accidents` (Traffic accident severity across states)
    - `whyshift_taxi` (Ride duration shifted across metropolitan areas)
  - *TableShift Domain Partitions*:
    - `tableshift_hospital_readmission` (Diabetic inpatient encounters, 46 predictors, shifted across 17 clinical admission source IDs)
    - `tableshift_heloc` (FICO credit risk performance shifted across risk tiers)
    - `tableshift_college_scorecard` (Graduation outcomes shifted between public and private universities)
    - `tableshift_assistments` (Online tutoring accuracy shifted across different school cohorts)

### 3. 8 Synthetic Benchmark Families
* **Purpose**: Provide mathematical ground truth for soft clustering memberships ($\sum_k \tau_{ik} = 1.0$), enabling exact evaluation of failure prediction metrics without confounding estimation errors.
* **Composition**:
  - `s01_balanced_gmm`: Balanced, well-separated spherical Gaussians ($K=3, d=10, n=10000$).
  - `s02_overlap_gmm`: Reduced separation to stress boundary ambiguity ($K=3, d=10, n=10000$).
  - `s03_imbalanced_gmm`: Severe cluster prior imbalance ($\pi=[0.70, 0.20, 0.10]$).
  - `s04_heteroscedastic_gmm`: Unequal cluster covariance variances ($\sigma^2 \in \{0.5, 1.5, 3.0\}$).
  - `s05_anisotropic_gmm`: Rotated non-spherical clusters violating Euclidean distance assumptions.
  - `s06_high_dimensional_gmm`: High-dimensional mixture testing curse of dimensionality ($d=100, K=5$).
  - `s07_irrelevant_features_gmm`: 20 informative dimensions embedded in 80 pure noise dimensions.
  - `s08_student_t_mixture`: Heavy-tailed Student-t clusters ($\nu=3$) modeling natural outliers.

---

## 6. Physical Artifact Isolation Rules

Every canonicalized dataset strictly isolates its variables into separate files:

- `features.parquet`: Pure input features $X$. Target, domain, and group columns are strictly excluded.
- `labels.parquet`: Ground-truth evaluation labels $y$. Completely separated from clustering inputs.
- `domains.parquet` *(optional)*: Preserved domain indicator for natural distribution shifts.
- `groups.parquet` *(optional)*: Preserved repeated-measurement group identifiers for GroupKFold validation.
- `metadata.json`: Feature roles (`numeric`, `categorical`, `ordinal`, `boolean`), split policy, and provenance.
