# Phase 0 Build Instructions: Scaffold → Leakage Gate → Data Load/EDA

Read `CLAUDE.md` first. Every hard rule there applies throughout this phase.

## Exit criteria — Phase 0 is done when ALL of these are true
- [ ] `uv run pytest tests/ -v` exits 0, including `test_leakage.py`
- [ ] `uv run ruff check src/ tests/` exits 0
- [ ] `dvc repro` runs the `ingest` stage end-to-end without error
- [ ] CTO labels load: `data/raw/cto_phase1.parquet`, `cto_phase2.parquet`,
      `cto_phase3.parquet`, `cto_gold.parquet` all exist with expected row counts
- [ ] AACT connection test passes: `python -m cto.data.aact_client --test` prints
      the count of studies updated in the last 7 days
- [ ] `notebooks/eda.py` runs to completion and writes all figures to `reports/figures/`
- [ ] No column from `LEAKAGE_BLOCKLIST` appears in any of the three feature DataFrames
      produced by `src/cto/features/build.py`

---

## Step 1 — Repository scaffold

### 1.1 Init project with uv
```bash
mkdir cto-predict && cd cto-predict
uv init --python 3.11
uv add xgboost lightgbm catboost scikit-learn optuna shap \
        pandas pyarrow psycopg2-binary sqlalchemy \
        datasets huggingface_hub mlflow "dvc[s3]" \
        fastapi uvicorn "pydantic>=2.0" pydantic-settings \
        evidently streamlit
uv add --dev ruff pytest pytest-cov pre-commit
```

### 1.2 Create full directory structure
Create every directory and a `.gitkeep` in empty ones:
```
cto-predict/
├── config/
├── data/raw/
├── data/processed/
├── data/interim/
├── models/
├── reports/figures/
├── notebooks/
├── docs/
├── src/cto/data/
├── src/cto/features/
├── src/cto/models/
├── src/cto/pipelines/
├── src/cto/serving/
├── src/cto/common/
├── tests/
├── dashboard/
└── .github/workflows/
```

Create `src/cto/__init__.py` and an `__init__.py` in every `src/cto/` subdirectory.

### 1.3 pyproject.toml — add tool configs
Append to `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

### 1.4 DVC init
```bash
git init
dvc init
# Configure remote later when cloud storage is set up.
# For now use a local remote so dvc repro works:
dvc remote add -d local_remote /tmp/dvc-remote-cto
```

Create `dvc.yaml` with this skeleton — stages will be filled in as modules are built:
```yaml
stages:
  ingest:
    cmd: python -m cto.pipelines.ingest
    deps:
      - src/cto/pipelines/ingest.py
      - src/cto/data/cto_labels.py
      - src/cto/data/aact_client.py
      - config/aact.yaml
    outs:
      - data/raw/cto_phase1.parquet
      - data/raw/cto_phase2.parquet
      - data/raw/cto_phase3.parquet
      - data/raw/cto_gold.parquet
      - data/raw/aact_studies_snapshot.parquet
    params:
      - params.yaml:
        - ingest

  featurize:
    cmd: python -m cto.pipelines.featurize
    deps:
      - src/cto/pipelines/featurize.py
      - src/cto/features/build.py
      - src/cto/features/leakage.py
      - src/cto/features/text.py
      - data/raw/cto_phase1.parquet
      - data/raw/cto_phase2.parquet
      - data/raw/cto_phase3.parquet
      - data/raw/aact_studies_snapshot.parquet
      - config/leakage_blocklist.yaml
      - config/features.yaml
    outs:
      - data/processed/features_phase1.parquet
      - data/processed/features_phase2.parquet
      - data/processed/features_phase3.parquet
    params:
      - params.yaml:
        - features
```

### 1.5 params.yaml
```yaml
ingest:
  cto_hf_repo: "chufangao/CTO"
  cto_phase_configs:
    - name: phase1
      hf_config: phase1_CTO_preds
    - name: phase2
      hf_config: phase2_CTO_preds
    - name: phase3
      hf_config: phase3_CTO_preds
    - name: gold
      hf_config: human_labels
  aact:
    lookback_days: 7          # for incremental sync
    full_pull: false           # set true for initial historical pull

features:
  schema_version: "1.0.0"
  tfidf_max_features: 400
  tfidf_ngram_range: [1, 2]
  min_enrollment: 1
  phases: [1, 2, 3]

model:
  phases: [1, 2, 3]
  n_optuna_trials: 100
  early_stopping_rounds: 50
  n_estimators: 2000
  random_state: 42

monitoring:
  psi_amber: 0.10
  psi_red: 0.20
  prauc_drop_amber: 0.03
  prauc_drop_red: 0.07
  min_new_outcomes_for_retrain: 250
```

### 1.6 .env.example
```
AACT_USER=your_aact_username
AACT_PASSWORD=your_aact_password
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
HF_TOKEN=                          # optional, CTO dataset is public
```

Add `.env` to `.gitignore`.

### 1.7 pre-commit config
Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
```
Run `uv run pre-commit install`.

### 1.8 GitHub Actions — CI stub
Create `.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.11"
      - run: uv sync --frozen
      - run: uv run ruff check src/ tests/
      - run: uv run pytest tests/ -v --tb=short
```

---

## Step 2 — Config files

### 2.1 config/leakage_blocklist.yaml
This is the canonical list. Any column in this list that appears in a feature
DataFrame causes `assert_no_leakage()` to raise immediately.

```yaml
# CTO labeling-function columns — used to GENERATE labels, never features
cto_lf_columns:
  - hint_train
  - hint_train2
  - hint_train3
  - status
  - status2
  - gpt
  - gpt2
  - linkage
  - linkage2
  - stock_price
  - results_reported
  - new_headlines
  - pvalues
  - update_more_recent
  - sites
  - serious_ae
  - patient_drop
  - num_patients
  - death_ae
  - amendments
  - all_ae
  - pred_proba

# AACT post-completion columns — only exist after trial ends
aact_post_hoc_columns:
  - why_stopped
  - results_first_posted_date
  - results_first_submitted_date
  - results_first_posted_date_type
  - last_known_status
  - fdaaa801_violation
  - limitations_and_caveats
  - actual_duration             # from calculated_values
  - were_results_reported       # from calculated_values

# AACT result tables — never query these for features
aact_result_tables:
  - outcomes
  - outcome_analyses
  - outcome_measurements
  - outcome_counts
  - result_groups
  - reported_events
  - reported_event_totals
  - baseline_measurements
  - baseline_counts
  - participant_flows
  - milestones
```

### 2.2 config/aact.yaml
```yaml
host: aact-db.ctti-clinicaltrials.org
port: 5432
dbname: aact
schema: ctgov
sslmode: require

# Registration-time tables (safe to query for features)
feature_tables:
  - studies
  - designs
  - eligibilities
  - sponsors
  - conditions
  - interventions
  - browse_conditions
  - browse_interventions
  - calculated_values
  - facilities
  - countries
  - design_outcomes
  - design_groups
```

### 2.3 config/features.yaml
```yaml
schema_version: "1.0.0"

# Registration-time feature definitions
numeric_features:
  - enrollment_log              # log1p(studies.enrollment) where enrollment_type=ANTICIPATED
  - number_of_arms
  - number_of_groups
  - number_of_facilities
  - criteria_length             # char count of eligibility criteria text
  - num_inclusion_criteria      # count of inclusion bullet points
  - num_exclusion_criteria      # count of exclusion bullet points
  - min_age_years               # parsed from eligibilities.minimum_age
  - max_age_years               # parsed from eligibilities.maximum_age
  - num_countries
  - num_primary_outcomes
  - registration_year           # year of study_first_posted_date

categorical_features:
  - phase_clean                 # 1/2/3 (routing key, one-hot for combined phases)
  - sponsor_class               # INDUSTRY/NIH/FED/OTHER_GOV/NETWORK/OTHER/UNKNOWN
  - allocation                  # RANDOMIZED/NON_RANDOMIZED/N_A
  - intervention_model          # PARALLEL/CROSSOVER/SINGLE_GROUP/FACTORIAL/SEQUENTIAL
  - masking_ordinal             # 0=OPEN, 1=SINGLE, 2=DOUBLE, 3=TRIPLE, 4=QUADRUPLE
  - primary_purpose             # TREATMENT/PREVENTION/BASIC_SCIENCE/DIAGNOSTIC/etc.
  - intervention_type_primary   # DRUG/BIOLOGICAL/DEVICE/BEHAVIORAL/OTHER
  - gender                      # ALL/FEMALE/MALE
  - healthy_volunteers          # Yes/No/Unknown

flag_features:
  - has_industry_lead           # sponsor_class == INDUSTRY
  - has_nih_lead                # sponsor_class == NIH
  - has_industry_collaborator
  - has_nih_collaborator
  - is_randomized
  - is_blinded                  # masking != NONE
  - is_multinational            # num_countries > 1
  - accepts_healthy_volunteers
  - has_drug_intervention
  - has_biological_intervention
  - has_combination_therapy     # num_interventions > 1
  - has_survival_endpoint       # primary outcome contains OS/PFS/survival keywords

text_features:
  - eligibility_tfidf           # TF-IDF on criteria field, 400 dims, 1-2 grams
```

---

## Step 3 — Leakage gate

### 3.1 src/cto/features/leakage.py
Implement the following (complete, production-ready code):

```python
"""
Leakage gate — the single most important module in this project.
Every feature-building function MUST call assert_no_leakage() before returning.
"""
from __future__ import annotations
import yaml
from pathlib import Path
import pandas as pd

# Load blocklist at import time so it's always current
_BLOCKLIST_PATH = Path(__file__).parents[4] / "config" / "leakage_blocklist.yaml"


def _load_blocklist() -> set[str]:
    with open(_BLOCKLIST_PATH) as f:
        raw = yaml.safe_load(f)
    columns: set[str] = set()
    for section_cols in raw.values():
        if isinstance(section_cols, list):
            columns.update(section_cols)
    return columns


LEAKAGE_BLOCKLIST: set[str] = _load_blocklist()


def assert_no_leakage(df: pd.DataFrame, context: str = "") -> None:
    """
    Raise ValueError if any column in df is in LEAKAGE_BLOCKLIST.
    Call this at the end of every feature-building function.

    Args:
        df: The feature DataFrame to check.
        context: Optional string describing where this check is called from.

    Raises:
        ValueError: If any post-hoc column survives into the feature matrix.
    """
    leaked = set(df.columns) & LEAKAGE_BLOCKLIST
    if leaked:
        raise ValueError(
            f"LEAKAGE DETECTED{' in ' + context if context else ''}. "
            f"The following post-hoc columns must never appear in features: {sorted(leaked)}\n"
            f"Check config/leakage_blocklist.yaml and remove these columns "
            f"from the feature-building pipeline."
        )


def drop_leakage_columns(df: pd.DataFrame, warn: bool = True) -> pd.DataFrame:
    """
    Drop all blocklisted columns from df and return the cleaned DataFrame.
    Prefer assert_no_leakage() (fail-fast); use this only for raw data cleaning
    before feature engineering.
    """
    to_drop = [c for c in df.columns if c in LEAKAGE_BLOCKLIST]
    if to_drop and warn:
        import warnings
        warnings.warn(
            f"Dropping {len(to_drop)} leakage column(s): {to_drop}. "
            f"These should never enter the feature pipeline.",
            stacklevel=2,
        )
    return df.drop(columns=to_drop)
```

### 3.2 tests/test_leakage.py
Implement every assertion below — all must pass for Phase 0 to be complete:

```python
"""
Leakage gate tests — must be green before any model work begins.
These are the most important tests in the project.
"""
import pytest
import pandas as pd
from cto.features.leakage import (
    LEAKAGE_BLOCKLIST,
    assert_no_leakage,
    drop_leakage_columns,
)

# ── Blocklist sanity ──────────────────────────────────────────────────────────

def test_blocklist_is_nonempty():
    assert len(LEAKAGE_BLOCKLIST) >= 20, "Blocklist suspiciously small"


@pytest.mark.parametrize("col", [
    # CTO LF columns
    "why_stopped", "pvalues", "serious_ae", "death_ae", "all_ae",
    "results_reported", "results_first_posted_date", "patient_drop",
    "stock_price", "new_headlines", "num_patients", "amendments",
    "pred_proba", "hint_train", "hint_train2", "hint_train3",
    "status", "status2", "gpt", "gpt2", "linkage", "linkage2",
    "update_more_recent", "sites",
    # AACT post-hoc
    "actual_duration", "were_results_reported", "last_known_status",
    "fdaaa801_violation", "limitations_and_caveats",
])
def test_known_offenders_in_blocklist(col):
    assert col in LEAKAGE_BLOCKLIST, f"'{col}' must be in LEAKAGE_BLOCKLIST"


# ── assert_no_leakage ─────────────────────────────────────────────────────────

def test_clean_df_passes():
    df = pd.DataFrame({"phase": [1, 2], "enrollment_log": [3.5, 4.1]})
    assert_no_leakage(df)  # must not raise


def test_single_leaked_column_raises():
    df = pd.DataFrame({"phase": [1], "pvalues": [0.03]})
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        assert_no_leakage(df)


def test_multiple_leaked_columns_all_reported():
    df = pd.DataFrame({"phase": [1], "pvalues": [0.03], "serious_ae": [2]})
    with pytest.raises(ValueError) as exc_info:
        assert_no_leakage(df)
    msg = str(exc_info.value)
    assert "pvalues" in msg
    assert "serious_ae" in msg


def test_context_string_appears_in_error():
    df = pd.DataFrame({"why_stopped": ["business"]})
    with pytest.raises(ValueError, match="build_phase1_features"):
        assert_no_leakage(df, context="build_phase1_features")


def test_nct_id_is_allowed():
    """nct_id is an identifier, not a feature — but it must not be blocked."""
    df = pd.DataFrame({"nct_id": ["NCT0001"], "phase": [1]})
    assert_no_leakage(df)  # must not raise


# ── drop_leakage_columns ──────────────────────────────────────────────────────

def test_drop_removes_blocked_columns():
    df = pd.DataFrame({"phase": [1], "pvalues": [0.03], "enrollment_log": [3.5]})
    cleaned = drop_leakage_columns(df, warn=False)
    assert "pvalues" not in cleaned.columns
    assert "phase" in cleaned.columns
    assert "enrollment_log" in cleaned.columns


def test_drop_on_clean_df_is_noop():
    df = pd.DataFrame({"phase": [1], "enrollment_log": [3.5]})
    cleaned = drop_leakage_columns(df, warn=False)
    assert list(cleaned.columns) == list(df.columns)


# ── CTO label-side isolation ──────────────────────────────────────────────────

CTO_LF_COLUMNS = [
    "hint_train", "hint_train2", "hint_train3",
    "status", "status2", "gpt", "gpt2",
    "linkage", "linkage2", "stock_price",
    "results_reported", "new_headlines", "pvalues",
    "update_more_recent", "sites", "serious_ae",
    "patient_drop", "num_patients", "death_ae",
    "amendments", "all_ae", "pred_proba",
]

def test_all_cto_lf_columns_in_blocklist():
    """Every CTO labeling-function column must be blocked from feature use."""
    missing = [c for c in CTO_LF_COLUMNS if c not in LEAKAGE_BLOCKLIST]
    assert not missing, (
        f"CTO LF columns not in blocklist: {missing}\n"
        f"These columns were used to *generate* the weak labels and "
        f"must never be used as features."
    )


def test_pred_proba_is_blocked():
    """pred_proba is CTO's output label proxy — must be blocked."""
    assert "pred_proba" in LEAKAGE_BLOCKLIST


def test_nct_id_not_blocked():
    """nct_id is the join key — must not be accidentally blocked."""
    assert "nct_id" not in LEAKAGE_BLOCKLIST
```

---

## Step 4 — CTO data loader

### 4.1 src/cto/data/cto_labels.py
Load CTO from Hugging Face and produce clean label DataFrames joined on `nct_id`.

Requirements:
- Load each phase CSV using `datasets.load_dataset("chufangao/CTO", config_name, split="test")`
  and convert to pandas. The available `config_name` values are in `params.yaml`:
  `phase1_CTO_preds`, `phase2_CTO_preds`, `phase3_CTO_preds`, `human_labels`.
- Keep ONLY `["nct_id", "pred_proba", "labels"]` (where they exist — `pred_proba` is in
  phase files; `labels` is in the gold `human_labels` file). Drop everything else.
  The remaining CTO columns are LF votes used to produce pred_proba; they are blocklisted.
- For phase files: derive a binary label `y` = `(pred_proba >= 0.5).astype(int)`.
  For the gold file: use `labels` directly as `y`.
- De-duplicate on `nct_id` — keep the last row per nct_id (some nct_ids appear in
  multiple phase files; this is expected — route by phase at join time).
- Save to `data/raw/cto_phase{n}.parquet` and `data/raw/cto_gold.parquet`.
- Return a dict `{"phase1": df1, "phase2": df2, "phase3": df3, "gold": df_gold}`.
- Log row counts per phase file after dedup.
- Expose a `if __name__ == "__main__"` entrypoint that loads and saves all four files.

### 4.2 Entrypoint test
Add to `tests/test_data.py`:
```python
def test_cto_labels_schema():
    """Minimal schema check — does not require network access (uses fixture or mock)."""
    import pandas as pd
    from cto.data.cto_labels import derive_binary_label

    df = pd.DataFrame({"nct_id": ["NCT001", "NCT002"], "pred_proba": [0.3, 0.8]})
    out = derive_binary_label(df)
    assert "y" in out.columns
    assert out["y"].isin([0, 1]).all()
    assert "pred_proba" not in out.columns  # should be dropped after deriving y
```

---

## Step 5 — AACT client

### 5.1 src/cto/data/aact_client.py
PostgreSQL client for AACT with incremental-pull support.

Requirements:
- Read host/port/dbname/schema/sslmode from `config/aact.yaml`.
- Read `AACT_USER` / `AACT_PASSWORD` from env (via `pydantic-settings`).
- `get_engine()` → SQLAlchemy engine with `sslmode=require`, `pool_pre_ping=True`.
- `get_studies(since: datetime | None = None) → pd.DataFrame`
  - If `since` is None: pull ALL studies (initial historical pull, ~400k rows — warn user this is slow).
  - If `since` is a datetime: pull only rows where `last_update_posted_date > since`.
  - Always select ONLY registration-time columns from `studies`:
    `nct_id, phase, study_type, overall_status, enrollment, enrollment_type,
     number_of_arms, number_of_groups, source_class,
     study_first_posted_date, primary_completion_date, completion_date,
     last_update_posted_date`
  - NEVER select post-hoc columns. If a post-hoc column is accidentally included,
    `assert_no_leakage()` downstream will catch it.
- `get_designs(nct_ids: list[str]) → pd.DataFrame`
  - Pull `nct_id, allocation, intervention_model, masking, masking_description,
     primary_purpose` from `ctgov.designs`.
- `get_eligibilities(nct_ids: list[str]) → pd.DataFrame`
  - Pull `nct_id, gender, minimum_age, maximum_age, healthy_volunteers, criteria`
    from `ctgov.eligibilities`.
- `get_sponsors(nct_ids: list[str]) → pd.DataFrame`
  - Pull `nct_id, agency_class, lead_or_collaborator` from `ctgov.sponsors`.
- `get_conditions(nct_ids: list[str]) → pd.DataFrame`
  - Pull `nct_id, name, downcase_mesh_term` from `ctgov.browse_conditions`.
- `get_interventions(nct_ids: list[str]) → pd.DataFrame`
  - Pull `nct_id, intervention_type, name` from `ctgov.interventions`.
- `get_calculated_values(nct_ids: list[str]) → pd.DataFrame`
  - Pull ONLY: `nct_id, number_of_facilities`.
  - DO NOT pull `actual_duration` or `were_results_reported` — those are blocklisted.
- `test_connection() → dict` — returns row count of recent studies (last 7 days),
  confirms SSL, prints confirmation. Expose as `python -m cto.data.aact_client --test`.
- Use chunked queries (chunk size 5,000 nct_ids) for the `get_*` helpers to avoid
  PostgreSQL query size limits.
- Handle `psycopg2.OperationalError` with a clear message pointing the user to
  the AACT signup page if credentials fail.

### 5.2 src/cto/data/mirror.py
Local Parquet snapshot of AACT data for fast offline feature building.

Requirements:
- `build_mirror(since: datetime | None = None) → None`
  - Calls all `get_*` helpers from `aact_client.py`.
  - Saves to `data/raw/aact_{table}_snapshot.parquet` for each table.
  - Saves `data/raw/sync_state.json` with `{"last_sync": "<ISO datetime>"}`.
- `load_mirror(table: str) → pd.DataFrame`
  - Reads `data/raw/aact_{table}_snapshot.parquet`.
- `get_last_sync() → datetime | None`
  - Reads `sync_state.json`; returns None if not found.

---

## Step 6 — Feature builder stub (enough for EDA)

### 6.1 src/cto/features/build.py
Build registration-time feature matrices from the AACT mirror + CTO labels.

For Phase 0, implement only enough to produce a joinable DataFrame for EDA.
Full feature engineering happens in Phase 1.

Requirements:
- `build_raw_joined(phase: int) → pd.DataFrame`
  - Loads `cto_phase{phase}.parquet` (labels) and joins to `aact_studies_snapshot.parquet`
    on `nct_id` (inner join — only keep trials present in both).
  - Filters to the correct phase: `phase_clean in {str(phase), f"PHASE{phase}",
    f"Phase {phase}"}` (AACT phase strings vary; normalise them).
  - Calls `drop_leakage_columns()` on the raw CTO side before joining.
  - Calls `assert_no_leakage(df, context=f"build_raw_joined(phase={phase})")` before returning.
  - Returns a DataFrame with columns: `nct_id, y, overall_status, enrollment,
    enrollment_type, number_of_arms, source_class, study_first_posted_date,
    primary_completion_date, completion_date, last_update_posted_date`.
  - Logs join stats: rows before/after join, null rate per column.

---

## Step 7 — Ingest pipeline

### 7.1 src/cto/pipelines/ingest.py
Orchestrates the full data ingestion step called by `dvc repro ingest`.

Requirements:
- Load params from `params.yaml` (using `OmegaConf` or plain `yaml.safe_load`).
- Call `cto_labels.load_all()` → saves all four CTO parquet files.
- Call `mirror.build_mirror(since=mirror.get_last_sync())` — incremental on re-runs,
  full pull only on first run.
- Log total row counts for each saved file.
- Exit with code 1 and a clear error message if AACT credentials are missing.
- Expose `if __name__ == "__main__"` so DVC can call it as `python -m cto.pipelines.ingest`.

---

## Step 8 — EDA

### 8.1 notebooks/eda.py
Write as a plain Python script (not a notebook) using `matplotlib`/`seaborn` so it
runs headlessly in CI and saves figures. Use `# %%` cell markers for IDE support.

Implement the following sections — each saves a figure to `reports/figures/`:

**Section 1 — Label distribution per phase**
- Bar chart: success rate (%) per phase (I, II, III) from CTO phase files.
- Bar chart: label counts (success vs failure) per phase.
- Expected output: Phase III should be ~68% success, Phase II ~50%, Phase I ~56%.
- Save as `reports/figures/label_distribution.png`.

**Section 2 — CTO label quality check**
- For each phase file, print: total rows, unique nct_ids, null rate in `pred_proba`.
- Histogram of `pred_proba` per phase (distribution of the weak-supervision probability).
- Save as `reports/figures/pred_proba_distribution.png`.

**Section 3 — AACT join rate**
- For each phase, show: CTO rows, AACT rows, join rate (inner join %).
- A low join rate (<50%) indicates a problem — flag it visually.
- Save as `reports/figures/join_rate.png`.

**Section 4 — Registration-time feature distributions**
- `enrollment` histogram (log scale) — flag trials with `enrollment_type=ACTUAL`
  (these should be excluded from features; `ANTICIPATED` is registration-time).
- `study_first_posted_date` histogram — shows temporal distribution of the dataset
  (important for understanding train/test split dynamics).
- `source_class` bar chart (INDUSTRY/NIH/etc.) per phase.
- `overall_status` bar chart — confirms the class-label distribution makes sense.
- Save as `reports/figures/feature_distributions.png`.

**Section 5 — Temporal coverage**
- Timeline plot: number of trials registered per year per phase (2000–2024).
- Highlight the proposed temporal split cutoff (e.g. 2022 line).
- This directly informs the train/val/test split strategy.
- Save as `reports/figures/temporal_coverage.png`.

**Section 6 — Leakage surface audit**
- Load all four CTO raw files (before any processing).
- Print all column names per file.
- Highlight in red any column that is in `LEAKAGE_BLOCKLIST`.
- Print the count of blocklisted columns found.
- This is a human-readable audit trail confirming the leakage gate is necessary.
- Save output as a text file `reports/figures/leakage_audit.txt`.

**Section 7 — Missing-value heatmap**
- For the joined Phase III DataFrame (most predictable phase), show a heatmap of
  null rates per registration-time column.
- Columns with >50% null rate should be flagged — they may need imputation strategy.
- Save as `reports/figures/missing_values_phase3.png`.

---

## Step 9 — Common utilities

### 9.1 src/cto/common/settings.py
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    aact_user: str
    aact_password: str
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    hf_token: str = ""

settings = Settings()
```

### 9.2 src/cto/common/db.py
SQLAlchemy Postgres client for the predictions/metrics database (Render Postgres in prod,
SQLite locally). Leave as a stub with `get_engine()` returning a SQLite engine for now.

---

## Verification checklist (run in order)

```bash
# 1. All tests pass
uv run pytest tests/ -v --tb=short

# 2. Linting clean
uv run ruff check src/ tests/

# 3. Ingest pipeline runs
dvc repro ingest

# 4. Confirm output files exist and have expected shapes
python -c "
import pandas as pd
for phase in [1, 2, 3]:
    df = pd.read_parquet(f'data/raw/cto_phase{phase}.parquet')
    print(f'Phase {phase}: {len(df)} rows, cols={list(df.columns)}')
df_gold = pd.read_parquet('data/raw/cto_gold.parquet')
print(f'Gold: {len(df_gold)} rows')
"

# 5. AACT connection test
python -m cto.data.aact_client --test

# 6. EDA runs headlessly
python notebooks/eda.py
ls reports/figures/
```

All six commands must complete without error before Phase 0 is closed.

---

## Notes for Claude Code

- Start with Step 1 (scaffold) before any source code.
- Write `test_leakage.py` (Step 3.2) before `leakage.py` (Step 3.1) — TDD the safety rule.
- After writing `leakage.py`, immediately run `uv run pytest tests/test_leakage.py` and
  iterate until green. Do not proceed to Step 4 until leakage tests pass.
- The EDA (Step 8) should be run last — it requires Steps 4–7 to produce data first.
- If the AACT connection fails (wrong credentials / network), implement `mirror.py` to
  load from a cached local snapshot and mock the AACT calls in tests with `pytest-mock`.
- Use `ultrathink` for any step where the correct approach is ambiguous.
- After completing each step, run the verification checklist for that step before moving on.
- Do not add libraries not in the tech stack in `CLAUDE.md` without asking first.
