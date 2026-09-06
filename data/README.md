# Dataset Acquisition, Provenance, and Validation Architecture

This directory contains the dataset acquisition, storage, and validation infrastructure for the benchmark study:
**When Clusters Drift: Label-Free Failure Prediction and Risk-Guided Adaptation for Soft Clustering**.

---

## 1. Directory Structure

```text
data/
├── raw/                      # Untransformed raw downloads cached from upstream providers
│   ├── controlled/           # Raw OpenML / UCI downloads for controlled datasets
│   └── natural/              # Raw WhyShift and TableShift partitions
│       ├── whyshift/
│       └── tableshift/
│
├── canonical/                # Standardized, label-isolated Parquet partitions
│   ├── controlled/           # 30 controlled real datasets (features.parquet, labels.parquet, metadata.json, optional groups.parquet)
│   └── natural/              # Natural-shift datasets separated by domain/state (optional domains.parquet)
│       ├── whyshift/
│       └── tableshift/
│
├── synthetic/                # 8 controlled synthetic benchmark families
│   ├── s01_balanced_gmm/
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
└── manifests/                # Cryptographic manifests, provenance records, and reports
    ├── datasets.json         # Manifest entries for all real datasets with SHA256 hashes
    ├── synthetic_manifest.json # Manifest entries for synthetic generators and parameters
    ├── download_report.json  # Timestamped execution report from 01_download_real_datasets.py
    ├── unavailable_datasets.json # Instructions and access endpoints for credential-gated datasets
    └── dataset_summary.csv   # Comprehensive tabular audit of row/feature counts, missingness, and hashes
```

> **Important**: `data/processed/` is strictly prohibited from holding downloaded or canonical data. It is reserved for later experiment-specific preprocessing (e.g. source-fitted standard scalers and imputers).

---

## 2. One-Command Reproduction Workflows

### A. Download Real Datasets

To acquire and canonicalize all available real-world datasets:

```bash
python scripts/01_download_real_datasets.py --all
```

Options:

- `--group controlled`: Download all 30 controlled real datasets.
- `--group natural`: Download natural-shift datasets.
- `--dataset <slug>`: Download a specific dataset (e.g., `--dataset wine` or `--dataset seeds`).
- `--force`: Overwrite existing canonical files.
- `--dry-run`: Preview planned downloads without network execution.

### B. Generate Deterministic Synthetic Datasets

To generate all eight synthetic benchmark families with exact soft posterior ground truth:

```bash
python scripts/02_generate_synthetic_datasets.py --all
```

Options:

- `--dataset <slug>`: Generate a specific family (e.g., `s01_balanced_gmm`).
- `--force`: Overwrite existing synthetic artifacts.
- `--dry-run`: Preview generator specifications.

### C. Validate Integrity and Generate Manifests

To audit structural schemas, verify label isolation, compute hashes, and generate `dataset_summary.csv`:

```bash
python scripts/03_validate_datasets.py
```

---

## 3. Dataset Taxonomy

### 30 Controlled Real Datasets (`controlled_real`)

All 30 datasets are acquired from scikit-learn built-ins or official OpenML data IDs pinned in `configs/datasets.yaml`:

1. `iris` (sklearn)
2. `wine` (sklearn)
3. `seeds` (OpenML 1499)
4. `glass` (OpenML 41)
5. `ecoli` (OpenML 39)
6. `yeast` (OpenML 181)
7. `vehicle_silhouettes` (OpenML 54)
8. `image_segmentation` (OpenML 40984)
9. `satimage` (OpenML 182)
10. `pendigits` (OpenML 32)
11. `optdigits` (OpenML 28)
12. `letter_recognition` (OpenML 6)
13. `banknote_authentication` (OpenML 1462)
14. `ionosphere` (OpenML 59)
15. `sonar` (OpenML 40)
16. `breast_cancer_wisconsin_diagnostic` (sklearn)
17. `pima_diabetes` (OpenML 37)
18. `heart_disease` (OpenML 1565)
19. `haberman_survival` (OpenML 43)
20. `dermatology` (OpenML 35)
21. `balance_scale` (OpenML 11)
22. `waveform` (OpenML 60, 40 predictors incl. 19 noise attributes)
23. `spambase` (OpenML 44)
24. `mice_protein_expression` (UCI 342, Mice Protein Expression, 1,080 x 77, 8 classes, 72 biological mice, path: `data/canonical/controlled/mice_protein_expression/`)
25. `human_activity_recognition` (UCI 240, HAR Using Smartphones, 10,299 x 561, 6 classes, 30 subjects, path: `data/canonical/controlled/human_activity_recognition/`)
26. `isolet` (OpenML 300)
27. `madelon` (OpenML 1485)
28. `electricity` (OpenML 151)
29. `bank_marketing` (OpenML 1461)
30. `aps_failure` (OpenML 41138)

### 10 Natural-Shift Real Datasets (`natural_shift`)

Preserve genuine spatial, institutional, and clinical domain partitions:

- **WhyShift Spatial Partitions**:
  - `whyshift_acs_income` (5 state domains: CA, TX, NY, FL, PA; N=601,843)
  - `whyshift_acs_pubcov` (5 state domains: CA, TX, NY, FL, PA; N=420,411)
  - `whyshift_acs_mobility` (5 state domains: CA, TX, NY, FL, PA; N=232,512)
  - `whyshift_taxi` (Cities: nyc, bog, mex, uio; credential-gated source)
  - `whyshift_us_accidents` (States: CA, TX, FL, NY; credential-gated Kaggle source)
- **TableShift Domain Partitions**:
  - `tableshift_hospital_readmission` (UCI 296 / TableShift diabetes readmission, N=99,493, 46 predictors: 8 numeric, 3 ordinal, 35 categorical, 17 admission source domains)
  - `tableshift_college_scorecard` (Higher education; public vs. private domain shift; auth required)
  - `tableshift_childhood_lead` (CDC blood lead; county poverty level shift; auth required)
  - `tableshift_heloc` (FICO credit risk; community license acceptance required)
  - `tableshift_assistments` (Online education; school cohort shift; auth required)

### 8 Synthetic Benchmark Families (`synthetic`)

Each family is generated with deterministic seeds (1001-1008), varying cluster counts ($K \in \{3, 4, 5\}$), and exact soft posterior memberships ($\sum_k \tau_{ik} = 1.0$):

- `s01_balanced_gmm` (seed 1001, $K=3, d=10, n=10000$, balanced spherical Gaussian)
- `s02_overlap_gmm` (seed 1002, $K=3, d=10, n=10000$, high overlap / boundary ambiguity)
- `s03_imbalanced_gmm` (seed 1003, $K=3, d=10, n=10000, \pi=[0.70, 0.20, 0.10]$)
- `s04_heteroscedastic_gmm` (seed 1004, $K=3, d=10, n=10000$, unequal cluster variances)
- `s05_anisotropic_gmm` (seed 1005, $K=3, d=10, n=10000$, rotated elliptical covariance)
- `s06_high_dimensional_gmm` (seed 1006, $K=5, d=100, n=10000$, high-dimensional structure)
- `s07_irrelevant_features_gmm` (seed 1007, $K=4, d=100, n=10000$, 20 informative + 80 noise dimensions)
- `s08_student_t_mixture` (seed 1008, $K=3, d=10, n=10000, \nu=3$, heavy-tailed natural outliers)

---

## 4. Credential-Gated & Manual Access Handling

To maintain research integrity and prevent legal/license violations, the downloader never scrapes or circumvents authentication:

- **`AUTO`**: Downloaded directly via official APIs (scikit-learn, OpenML, WhyShift).
- **`AUTH_REQUIRED`**: Datasets requiring Kaggle API tokens or data portal registration (e.g. `whyshift_taxi`, `whyshift_us_accidents`). The downloader records exact instructions and continues without crashing.
- **`MANUAL_LICENSE_ACCEPTANCE`**: Datasets requiring explicit user agreement (e.g. `tableshift_heloc` under FICO Community License).

Place any manually acquired archives into their respective directory under `data/raw/` to have them automatically processed on the next run.

---

## 5. Physical Label, Domain, and Group Separation

Every canonicalized dataset is split into separate files:

- `features.parquet`: Pure input features $\tilde{X}$. Target, domain, and group columns are strictly omitted (`assert target_column not in X.columns`, `assert domain_column not in X.columns`, `assert group_column not in X.columns`).
- `labels.parquet`: Evaluation ground truth $y$. Kept strictly isolated from clustering predictors.
- `domains.parquet` *(optional)*: Preserved natural-shift partition variable (e.g. `admission_source_id` for TableShift).
- `groups.parquet` *(optional)*: Preserved experimental unit repeated-measurement grouping (e.g. `subject_id` for HAR, `MouseID` for Mice Protein Expression).
- `metadata.json`: Feature names, exhaustive feature roles (`numeric`, `categorical`, `ordinal`), row counts, split strategies, domain indicators, and license provenance.
