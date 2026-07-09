# Phase 1 Build Instructions: Temporal Split → Feature Engineering → Baseline Models

Read `CLAUDE.md` before starting. All hard rules from Phase 0 still apply.
Phase 0 must be fully complete (43/43 tests green, AACT mirror populated) before starting here.

## Pre-flight checks — run these before touching a file

```bash
# AACT mirror must be populated — these files must exist and be non-empty
python -c "
import pandas as pd, sys
df = pd.read_parquet('data/raw/aact_studies_snapshot.parquet')
if len(df) < 10000:
    print('ERROR: AACT mirror looks empty. Run dvc repro ingest with real credentials first.')
    sys.exit(1)
print(f'AACT mirror OK: {len(df)} studies')
"

# Gold set leakage review — run this and confirm BEFORE any feature engineering
python -c "
import pandas as pd
from cto.features.leakage import LEAKAGE_BLOCKLIST
df = pd.read_parquet('data/raw/cto_gold.parquet')
# Load the full gold CSV columns (all 73) to check for leakage
# overall_status and last_update_posted_date must be in blocklist
required_in_blocklist = ['overall_status', 'last_update_posted_date']
missing = [c for c in required_in_blocklist if c not in LEAKAGE_BLOCKLIST]
if missing:
    print(f'ACTION REQUIRED: Add these to config/leakage_blocklist.yaml: {missing}')
else:
    print('Gold set leakage check: OK')
"
```

If either check fails, fix it before proceeding.

---

## Exit criteria — Phase 1 is done when ALL of these are true

- [ ] `uv run pytest tests/ -v` exits 0 (all existing + new tests)
- [ ] `dvc repro featurize` completes, producing 6 parquet files:
      `data/processed/features_phase{1,2,3}_{train,val,test}.parquet`
- [ ] No test split contains any trial whose `completion_date` is earlier than
      the latest `completion_date` in the train split (temporal integrity check)
- [ ] Baseline XGBoost per-phase runs and logs to MLflow:
      Phase III PR-AUC ≥ 0.60 (anything below suggests a data/leakage problem)
- [ ] Calibration curves saved to `reports/figures/calibration_phase{1,2,3}.png`
- [ ] SHAP summary plots saved to `reports/figures/shap_phase{1,2,3}.png`
- [ ] Top SHAP feature for any phase is NOT a feature with suspiciously high importance
      (>0.4 mean |SHAP| for a single feature → treat as suspected leakage, audit it)
- [ ] MLflow experiment `cto_baseline` has three logged runs (phase1, phase2, phase3)
      each with `feature_schema_version`, `phase`, `prauc`, `auroc`, `brier`, `ece` logged

---

## Step 1 — Add to config/leakage_blocklist.yaml

Before any feature work, extend the blocklist with columns confirmed post-hoc
from the gold set audit:

```yaml
# Add these to the aact_post_hoc_columns section:
aact_post_hoc_columns:
  # ... existing entries ...
  - overall_status          # encodes COMPLETED/TERMINATED — only known post-completion
  - last_update_posted_date # definitionally post-registration
  - last_update_submitted_date
  - last_update_posted_date_type
  - disposition_first_posted_date
  - disposition_first_submitted_date
```

After editing, run `uv run pytest tests/test_leakage.py` — must stay green.
Add a new parametrize entry for `overall_status` in `test_known_offenders_in_blocklist`.

---

## Step 2 — Temporal split

### 2.1 src/cto/features/split.py

Requirements:
- `make_temporal_splits(df: pd.DataFrame, date_col: str = "completion_date") -> dict`
  - Sorts df by `date_col` ascending.
  - Train: trials completed before TRAIN_CUTOFF
  - Val: trials completed between TRAIN_CUTOFF and VAL_CUTOFF
  - Test: trials completed after VAL_CUTOFF
  - Cutoffs come from `params.yaml` under `split:` (see 2.2 below).
  - Returns `{"train": df_train, "val": df_val, "test": df_test}`.
  - Raises `ValueError` if any split is empty.
  - Logs split sizes and date ranges to stdout.

- `assert_temporal_integrity(train: pd.DataFrame, val: pd.DataFrame,
                              test: pd.DataFrame, date_col: str) -> None`
  - Asserts `train[date_col].max() <= val[date_col].min()`.
  - Asserts `val[date_col].max() <= test[date_col].min()`.
  - Raises `ValueError` with specific dates if violated.
  - This is the temporal analog of `assert_no_leakage()` — call it everywhere.

### 2.2 Add to params.yaml under split:
```yaml
split:
  train_cutoff: "2021-12-31"   # trials completed on/before this → train
  val_cutoff: "2022-12-31"     # trials completed 2022 → val
  # test: trials completed 2023+ (plus the CTO gold set 2020-2024 for final eval)
  date_col: "completion_date"
  min_split_size: 100          # raise if any split has fewer rows
```

Rationale: CTO's paper trains on pre-2022 trials and tests on 2022–2024.
The gold set (2020–2024) is reserved for final evaluation only — never used
in any training or hyperparameter tuning decision.

### 2.3 tests/test_split.py
```python
import pytest
import pandas as pd
import numpy as np
from cto.features.split import make_temporal_splits, assert_temporal_integrity

def make_fake_df(n=300):
    dates = pd.date_range("2015-01-01", "2024-12-31", periods=n)
    return pd.DataFrame({
        "nct_id": [f"NCT{i:07d}" for i in range(n)],
        "completion_date": dates,
        "y": np.random.randint(0, 2, n),
    })

def test_splits_are_nonempty():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    for name, split in splits.items():
        assert len(split) > 0, f"{name} split is empty"

def test_temporal_integrity_holds():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    assert_temporal_integrity(splits["train"], splits["val"], splits["test"],
                               date_col="completion_date")  # must not raise

def test_no_row_appears_in_two_splits():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    all_ids = [set(s["nct_id"]) for s in splits.values()]
    for i, s1 in enumerate(all_ids):
        for j, s2 in enumerate(all_ids):
            if i != j:
                overlap = s1 & s2
                assert not overlap, f"Splits {i} and {j} share nct_ids: {overlap}"

def test_train_predates_val():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    assert splits["train"]["completion_date"].max() <= \
           splits["val"]["completion_date"].min()

def test_val_predates_test():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    assert splits["val"]["completion_date"].max() <= \
           splits["test"]["completion_date"].min()

def test_temporal_integrity_raises_on_violation():
    train = pd.DataFrame({"nct_id": ["A"], "completion_date":
                          pd.to_datetime(["2023-01-01"])})
    val = pd.DataFrame({"nct_id": ["B"], "completion_date":
                        pd.to_datetime(["2020-01-01"])})  # before train!
    test = pd.DataFrame({"nct_id": ["C"], "completion_date":
                         pd.to_datetime(["2024-01-01"])})
    with pytest.raises(ValueError, match="temporal"):
        assert_temporal_integrity(train, val, test, "completion_date")

def test_empty_split_raises():
    # Create df where all dates are in train range
    df = pd.DataFrame({
        "nct_id": ["A", "B"],
        "completion_date": pd.to_datetime(["2015-01-01", "2016-01-01"]),
        "y": [0, 1],
    })
    with pytest.raises(ValueError):
        make_temporal_splits(df)  # val and test will be empty
```

---

## Step 3 — Full feature engineering

### 3.1 src/cto/features/text.py

TF-IDF features from eligibility criteria text.

Requirements:
- `fit_tfidf(texts: pd.Series, max_features: int, ngram_range: tuple) -> TfidfVectorizer`
  - Fits on training texts only. Returns the fitted vectorizer.
  - Persists the fitted vectorizer to `models/tfidf_vectorizer.joblib` using `joblib.dump`.
- `transform_tfidf(texts: pd.Series, vectorizer: TfidfVectorizer) -> pd.DataFrame`
  - Transforms texts using the already-fitted vectorizer.
  - Returns a DataFrame with columns named `tfidf_{n}` for n in range(max_features).
  - Replaces NaN criteria with empty string before transforming.
- `load_tfidf() -> TfidfVectorizer`
  - Loads from `models/tfidf_vectorizer.joblib`. Raises FileNotFoundError with a clear
    message if not found (means fit_tfidf was not run yet).

### 3.2 src/cto/features/build.py — full implementation

Implement `build_features(phase: int, split: str) -> pd.DataFrame` that:

1. Loads the raw joined DataFrame for this phase (from Phase 0's `build_raw_joined`).

2. Joins to AACT mirror tables:
   - `aact_designs_snapshot.parquet` → allocation, intervention_model, masking, primary_purpose
   - `aact_eligibilities_snapshot.parquet` → gender, minimum_age, maximum_age,
     healthy_volunteers, criteria (text)
   - `aact_sponsors_snapshot.parquet` → agency_class (lead sponsor only:
     `WHERE lead_or_collaborator = 'lead'`)
   - `aact_browse_conditions_snapshot.parquet` → aggregate MeSH terms per nct_id
   - `aact_interventions_snapshot.parquet` → intervention_type, count per nct_id
   - `aact_calculated_values_snapshot.parquet` → number_of_facilities ONLY
     (DO NOT pull actual_duration or were_results_reported)

3. Engineer each feature from `config/features.yaml`:

   **Numeric (apply log1p to enrollment):**
   - `enrollment_log`: `log1p(enrollment)` where `enrollment_type = 'ANTICIPATED'`;
     set to NaN if `enrollment_type = 'ACTUAL'` (actual is post-registration)
   - `number_of_arms`: fill NaN with 2 (most common)
   - `number_of_groups`: fill NaN with 1
   - `number_of_facilities`: fill NaN with median
   - `criteria_length`: `len(criteria)` if not null else 0
   - `num_inclusion_criteria`: count lines starting with inclusion indicators
   - `num_exclusion_criteria`: count lines starting with exclusion indicators
   - `min_age_years`: parse "18 Years" → 18.0; "N/A" → NaN
   - `max_age_years`: same parsing
   - `num_countries`: from facilities join (count distinct countries)
   - `num_primary_outcomes`: count rows in design_outcomes where outcome_type='primary'
   - `registration_year`: `study_first_posted_date.dt.year`

   **Categorical (ordinal or one-hot — see below):**
   - `sponsor_class`: map to int: INDUSTRY=0, NIH=1, FED=2, OTHER_GOV=3,
     NETWORK=4, OTHER=5, UNKNOWN=6; fill NaN with 6
   - `allocation`: RANDOMIZED=1, NON_RANDOMIZED=0, null=0
   - `intervention_model`: PARALLEL=0, CROSSOVER=1, SINGLE_GROUP=2,
     FACTORIAL=3, SEQUENTIAL=4, OTHER=5; fill NaN with 5
   - `masking_ordinal`: NONE=0, SINGLE=1, DOUBLE=2, TRIPLE=3, QUADRUPLE=4;
     fill NaN with 0
   - `primary_purpose`: TREATMENT=0, PREVENTION=1, BASIC_SCIENCE=2,
     DIAGNOSTIC=3, SUPPORTIVE_CARE=4, SCREENING=5, OTHER=6; fill NaN with 6
   - `intervention_type_primary`: DRUG=0, BIOLOGICAL=1, DEVICE=2,
     BEHAVIORAL=3, PROCEDURE=4, RADIATION=5, DIETARY_SUPPLEMENT=6, OTHER=7;
     determined by most common intervention_type per nct_id; fill NaN with 7
   - `gender`: ALL=0, FEMALE=1, MALE=2; fill NaN with 0

   **Flags (binary int, 0/1):**
   - `has_industry_lead`: `(sponsor_class == 0).astype(int)`
   - `has_nih_lead`: `(sponsor_class == 1).astype(int)`
   - `has_industry_collaborator`: any sponsor with agency_class=INDUSTRY
     and lead_or_collaborator='collaborator'
   - `has_nih_collaborator`: same for NIH
   - `is_randomized`: `(allocation == 'RANDOMIZED').astype(int)`
   - `is_blinded`: `(masking != 'NONE').astype(int)`
   - `is_multinational`: `(num_countries > 1).astype(int)`
   - `accepts_healthy_volunteers`: `(healthy_volunteers == 'Yes').astype(int)`
   - `has_drug_intervention`: any intervention with type=DRUG
   - `has_biological_intervention`: any intervention with type=BIOLOGICAL
   - `has_combination_therapy`: `(count_interventions > 1).astype(int)`
   - `has_survival_endpoint`: primary outcome text contains any of
     ["overall survival", " os ", "progression-free", " pfs ", "event-free",
      "disease-free", "time to", "hazard"]

   **Text features:**
   - If split == "train": call `fit_tfidf(criteria_series, max_features, ngram_range)`
     — this fits AND transforms. Save the vectorizer.
   - If split in {"val", "test"}: call `transform_tfidf` using the loaded vectorizer.
   - Concatenate TF-IDF columns to the feature DataFrame.

4. Drop `nct_id`, `completion_date`, and any remaining date columns before returning.
   Keep `y` as a separate Series, not in the feature DataFrame.

5. Call `assert_no_leakage(X, context=f"build_features(phase={phase}, split={split})")`.

6. Return `X` (features only, no `y`, no `nct_id`). Caller accesses `y` separately.

### 3.3 src/cto/pipelines/featurize.py

Orchestrates the full featurize stage called by `dvc repro featurize`.

Requirements:
- For each phase in [1, 2, 3]:
  - Call `build_raw_joined(phase)` to get the full phase DataFrame.
  - Call `make_temporal_splits(df)` to split by completion_date.
  - Call `assert_temporal_integrity(train, val, test)`.
  - For each split in ["train", "val", "test"]:
    - Call `build_features(phase, split)` to get X.
    - Extract y from the split DataFrame (do NOT include y in the features file).
    - Save X to `data/processed/features_phase{phase}_{split}.parquet`.
    - Save y to `data/processed/labels_phase{phase}_{split}.parquet`
      (single column `y`, indexed by row number).
  - Log: phase, split sizes, feature count, date ranges.
- Save the fitted TF-IDF vectorizer inside the train call for phase 1
  (phases 2 and 3 reuse the same vectorizer — call `load_tfidf()` for their val/test).

### 3.4 Update dvc.yaml featurize stage outputs
Add label files to the `outs` section:
```yaml
  - data/processed/labels_phase1_train.parquet
  - data/processed/labels_phase1_val.parquet
  - data/processed/labels_phase1_test.parquet
  # ... same for phase2, phase3
  - models/tfidf_vectorizer.joblib
```

---

## Step 4 — Baseline model training

### 4.1 src/cto/models/train.py

Train one XGBoost model per phase. MLflow-tracked. No Optuna yet — that's Phase 2.
Phase 1 goal: reproduce the published baselines and confirm no leakage.

Requirements:
- `train_phase(phase: int, experiment_name: str = "cto_baseline") -> mlflow.ActiveRun`
  - Loads `features_phase{phase}_train.parquet` and `labels_phase{phase}_train.parquet`.
  - Loads `features_phase{phase}_val.parquet` and `labels_phase{phase}_val.parquet`.
  - Computes `scale_pos_weight = n_neg / n_pos` from training labels.
  - Fits XGBoost `XGBClassifier` with:
    ```python
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "tree_method": "hist",
    }
    ```
  - Evaluates on val set: PR-AUC, AUROC, F1 at threshold=0.5, Brier score, ECE.
  - Logs to MLflow: all params, all metrics, `phase`, `feature_schema_version`,
    `n_train`, `n_val`, `n_features`, `scale_pos_weight`.
  - Saves model artifact to MLflow run (not to `models/` — MLflow manages it).
  - Prints a clear summary: phase, PR-AUC val, AUROC val.
  - Returns the MLflow run.

- `compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold: float = 0.5) -> dict`
  - Returns: `prauc`, `auroc`, `f1`, `brier`, `ece`.
  - ECE = Expected Calibration Error: bin predictions into 10 bins,
    compute |avg_predicted_prob - avg_actual| per bin, weighted average.
  - Use `sklearn.metrics` for prauc, auroc, f1, brier.

### 4.2 src/cto/models/calibrate.py

Calibrate the raw XGBoost probabilities using isotonic regression.

Requirements:
- `calibrate_model(model, X_val, y_val) -> CalibratedClassifierCV`
  - Wraps the fitted XGBoost model with `CalibratedClassifierCV(cv="prefit", method="isotonic")`.
  - Fits on the validation set (NOT training — training data was used to fit the base model).
  - Returns the calibrated wrapper.
- `plot_calibration(y_true, y_prob_raw, y_prob_cal, phase: int, output_dir: Path) -> None`
  - Plots raw vs calibrated reliability curve (10 bins).
  - Plots histogram of predicted probabilities (raw vs calibrated).
  - Saves to `output_dir/calibration_phase{phase}.png`.
  - The calibrated curve should be closer to the diagonal than the raw curve.
  - If it's NOT closer, log a warning — isotonic may be overfitting the validation set.

### 4.3 src/cto/models/explain.py

SHAP-based feature importance for the trained models.

Requirements:
- `compute_shap(model, X: pd.DataFrame) -> np.ndarray`
  - Uses `shap.TreeExplainer(model)`.
  - Returns SHAP values array (shape: n_samples × n_features).
  - Uses a sample of max 2,000 rows from X to keep it fast (random sample, seed=42).
- `plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame,
                     phase: int, output_dir: Path) -> None`
  - `shap.summary_plot` (beeswarm), saved to `output_dir/shap_phase{phase}.png`.
  - Also saves a bar plot (mean |SHAP|) to `output_dir/shap_bar_phase{phase}.png`.
- `get_top_features(shap_values: np.ndarray, feature_names: list[str],
                    n: int = 10) -> list[dict]`
  - Returns top N features by mean |SHAP| as `[{"feature": name, "mean_abs_shap": val}]`.
  - Used by the serving layer.

### 4.4 src/cto/pipelines/train_baseline.py

Orchestrates the full train baseline stage.

Requirements:
- For each phase in [1, 2, 3]:
  - Call `train_phase(phase)` — returns MLflow run.
  - Load the trained model from the MLflow run.
  - Calibrate using val set.
  - Evaluate the calibrated model on the TEST set (not val): log `test_prauc`,
    `test_auroc`, `test_brier`, `test_ece` as MLflow metrics.
  - Evaluate the calibrated model on the GOLD set
    (`features` built from gold nct_ids, labels from `data/raw/cto_gold.parquet`):
    log `gold_prauc`, `gold_auroc` as MLflow metrics.
  - Compute SHAP values and save plots.
  - Plot and save calibration curves.
  - Print summary table:

    ```
    Phase I:   Val PR-AUC=0.xxx  Test PR-AUC=0.xxx  Gold PR-AUC=0.xxx
    Phase II:  Val PR-AUC=0.xxx  Test PR-AUC=0.xxx  Gold PR-AUC=0.xxx
    Phase III: Val PR-AUC=0.xxx  Test PR-AUC=0.xxx  Gold PR-AUC=0.xxx

    Baselines to beat (PyTrial XGBoost):
    Phase I:   0.513  Phase II: 0.586  Phase III: 0.697
    ```

- Register each model in MLflow with `cto_phase{n}` name and alias `@challenger`
  (not `@champion` yet — that requires the promotion gate in Phase 2).

### 4.5 Add train_baseline stage to dvc.yaml
```yaml
  train_baseline:
    cmd: python -m cto.pipelines.train_baseline
    deps:
      - src/cto/pipelines/train_baseline.py
      - src/cto/models/train.py
      - src/cto/models/calibrate.py
      - src/cto/models/explain.py
      - data/processed/
      - models/tfidf_vectorizer.joblib
    outs:
      - reports/figures/calibration_phase1.png
      - reports/figures/calibration_phase2.png
      - reports/figures/calibration_phase3.png
      - reports/figures/shap_phase1.png
      - reports/figures/shap_phase2.png
      - reports/figures/shap_phase3.png
      - reports/figures/shap_bar_phase1.png
      - reports/figures/shap_bar_phase2.png
      - reports/figures/shap_bar_phase3.png
    params:
      - params.yaml:
        - model
```

---

## Step 5 — Leakage check on SHAP output

This step has no code — it's a required human review gate.

After `dvc repro train_baseline` completes, open `reports/figures/shap_bar_phase3.png`
(Phase III is most interpretable) and apply this check:

**Expected top features (plausible, registation-time, domain-sensible):**
- `is_randomized` / `allocation` — Phase III trials almost always randomized
- `has_industry_lead` / `sponsor_class` — industry trials complete at higher rates
- `enrollment_log` — larger trials more often complete
- `masking_ordinal` — blinding correlates with rigor
- `has_drug_intervention` — drug vs device/behavioral have different success rates
- `registration_year` — era effects (CTO's documented distribution shift)
- Any `tfidf_*` features — text signal from eligibility criteria

**Red flag — suspected leakage if ANY of these appear in top 10:**
- Any date column (means the temporal split is wrong)
- `number_of_facilities` with implausibly high SHAP (facilities count can be updated post-registration)
- Any column whose name you don't recognise as registration-time
- `mean |SHAP| > 0.4` for a single feature (too dominant — check it)

If you see a red flag, add the column to `config/leakage_blocklist.yaml`,
re-run `dvc repro featurize train_baseline`, and repeat until the top features
are all domain-sensible.

---

## Step 6 — Sanity checks and tests

### 6.1 tests/test_features.py
```python
import pytest
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")

@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_feature_files_exist():
    for phase in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            assert (PROCESSED / f"features_phase{phase}_{split}.parquet").exists()
            assert (PROCESSED / f"labels_phase{phase}_{split}.parquet").exists()

@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_no_leakage_in_processed_features():
    from cto.features.leakage import LEAKAGE_BLOCKLIST
    for phase in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            df = pd.read_parquet(PROCESSED / f"features_phase{phase}_{split}.parquet")
            leaked = set(df.columns) & LEAKAGE_BLOCKLIST
            assert not leaked, (
                f"Phase {phase} {split}: leaked columns found: {leaked}"
            )

@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_y_not_in_feature_files():
    for phase in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            df = pd.read_parquet(PROCESSED / f"features_phase{phase}_{split}.parquet")
            assert "y" not in df.columns, \
                f"Phase {phase} {split}: 'y' label found in feature file — must be separate"

@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_temporal_integrity_of_splits():
    from cto.features.split import assert_temporal_integrity
    # Check using label files (they retain completion_date)
    # This test requires completion_date to be saved in the label files
    pass  # Implement if label parquets include completion_date

@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_train_larger_than_val_larger_than_test():
    for phase in [1, 2, 3]:
        n_train = len(pd.read_parquet(PROCESSED / f"features_phase{phase}_train.parquet"))
        n_val   = len(pd.read_parquet(PROCESSED / f"features_phase{phase}_val.parquet"))
        n_test  = len(pd.read_parquet(PROCESSED / f"features_phase{phase}_test.parquet"))
        assert n_train > n_val, f"Phase {phase}: train ({n_train}) not larger than val ({n_val})"
        assert n_val > 0, f"Phase {phase}: val split is empty"
        assert n_test > 0, f"Phase {phase}: test split is empty"

@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_feature_schema_consistent_across_splits():
    for phase in [1, 2, 3]:
        cols_train = set(pd.read_parquet(PROCESSED / f"features_phase{phase}_train.parquet").columns)
        cols_val   = set(pd.read_parquet(PROCESSED / f"features_phase{phase}_val.parquet").columns)
        cols_test  = set(pd.read_parquet(PROCESSED / f"features_phase{phase}_test.parquet").columns)
        assert cols_train == cols_val == cols_test, \
            f"Phase {phase}: feature schemas differ across splits"
```

### 6.2 tests/test_models.py
```python
import pytest
import numpy as np
from cto.models.train import compute_metrics

def test_compute_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    m = compute_metrics(y, p)
    assert m["prauc"] == pytest.approx(1.0)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(0.0)

def test_compute_metrics_random():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 100)
    p = rng.uniform(0, 1, 100)
    m = compute_metrics(y, p)
    # Random should be near 0.5 for AUROC
    assert 0.3 < m["auroc"] < 0.7
    assert 0.0 <= m["ece"] <= 1.0

def test_compute_metrics_returns_all_keys():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    m = compute_metrics(y, p)
    for key in ["prauc", "auroc", "f1", "brier", "ece"]:
        assert key in m, f"Missing key: {key}"
```

---

## Full Phase 1 verification checklist

```bash
# 1. All tests
uv run pytest tests/ -v --tb=short

# 2. Lint
uv run ruff check src/ tests/

# 3. Full pipeline
dvc repro

# 4. Confirm feature files
python -c "
import pandas as pd
from pathlib import Path
for phase in [1, 2, 3]:
    for split in ['train', 'val', 'test']:
        X = pd.read_parquet(f'data/processed/features_phase{phase}_{split}.parquet')
        y = pd.read_parquet(f'data/processed/labels_phase{phase}_{split}.parquet')
        print(f'Phase {phase} {split}: X={X.shape}, y={len(y)}, pos_rate={y[\"y\"].mean():.2f}')
"

# 5. Check MLflow runs
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
runs = mlflow.search_runs(experiment_names=['cto_baseline'])
print(runs[['tags.mlflow.runName','metrics.prauc','metrics.test_prauc','metrics.gold_prauc']].to_string())
"

# 6. Phase III PR-AUC gate (must be >= 0.60)
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
runs = mlflow.search_runs(experiment_names=['cto_baseline'])
phase3 = runs[runs['params.phase'] == '3']
prauc = float(phase3['metrics.test_prauc'].iloc[0])
print(f'Phase III test PR-AUC: {prauc:.3f}')
if prauc < 0.60:
    print('WARNING: Below baseline threshold. Investigate feature engineering or possible leakage.')
else:
    print('PASS: Above minimum threshold for Phase 1.')
"

# 7. SHAP leakage review (human review — no script)
open reports/figures/shap_bar_phase3.png
# Review manually against the red-flag list in Step 5
```

---

## Notes for Claude Code

- Write `tests/test_split.py` (Step 2.3) before implementing `split.py` (Step 2.1) — TDD.
- Write `tests/test_models.py` (Step 6.2) before `train.py` (Step 4.1) — TDD.
- The TF-IDF vectorizer is fit on Phase 1 train text only, then reused for Phases 2 and 3.
  This is intentional — one shared vocabulary. If features.yaml `tfidf_max_features` changes,
  delete `models/tfidf_vectorizer.joblib` and re-run featurize.
- `enrollment_type = 'ACTUAL'` must be set to NaN for `enrollment_log`. This is not optional —
  actual enrollment is only known after the trial completes.
- The gold set is NEVER used for training, validation, or hyperparameter decisions.
  It is evaluation-only. Do not pass it to `make_temporal_splits`.
- If Phase III test PR-AUC is below 0.55, stop and audit. Check:
  (1) Is `enrollment_type` filtered correctly? (2) Are any date columns surviving into features?
  (3) Is the val set being used to fit the TF-IDF?
- `ultrathink` before implementing `build_features` — it has many moving parts and
  is where most bugs will be introduced.
- After `dvc repro` completes, run the full verification checklist before declaring Phase 1 done.
