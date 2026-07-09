# Phase 2 Build Instructions: Definition Audit → Sponsor Features → Gold Retraining → Ensemble

Read `CLAUDE.md` before starting. All hard rules apply throughout.
Phase 1 must be complete (60+ tests green, gold PR-AUC logged in MLflow) before starting.

---

## Exit criteria — Phase 2 is done when ALL of these are true

- [ ] `uv run pytest tests/ -v` exits 0 (all existing + new tests)
- [ ] `reports/definition_audit.md` exists and documents the success-definition finding
- [ ] `dvc repro featurize_gold` completes with sponsor + design features included
- [ ] `data/processed/features_gold_phase{1,2,3}_{train,val,test}.parquet` all exist
- [ ] Feature count increased from 427 (Phase 1) — sponsor/design features added
- [ ] Gold models trained with Optuna logged in MLflow experiment `cto_gold`
- [ ] Phase III gold PR-AUC > 0.700 (must beat Phase 1 baseline of 0.700)
- [ ] Ensemble model (XGBoost + LightGBM + CatBoost stack) beats best single model per phase
- [ ] AutoGluon benchmark run logged to MLflow experiment `cto_autogluon`
- [ ] TabPFN run logged for Phase I and II (not III — n is large enough for GBDTs)
- [ ] Calibration switched to CalibratedClassifierCV(cv=5) for all phases
- [ ] MLflow `@champion` alias updated for each phase (only if challenger beats incumbent)
- [ ] `reports/phase2_results.md` written with side-by-side Phase 1 vs Phase 2 comparison

---

## AMENDMENTS (override this document where they conflict) — applied 2026-07-04

Adopted after the pre-Phase-2 audit (`reports/pre_phase2_audit.md`). These take precedence
over the step text below.

1. **Class weights (Step 5.2): DONE.** `params.yaml model.class_weight_by_phase` is now the
   gold-derived 4.02 / 2.23 / 0.89 (was 0.78 / 1.00 / 0.47). Use `scale_pos_weight` from params.

2. **Promotion gate (supersedes Step 9's bare `margin=0.01`).** Use `src/cto/models/gate.py`.
   PR-AUC **and** AUROC are co-primary — challenger must beat champion on both, never on a
   bare point difference. Phase I: walk-forward temporal CV (4 folds, test 2021-2024) + paired
   Nadeau-Bengio corrected t-test (mean ΔPR-AUC>0 and p<0.05). Phase II/III: single temporal
   split + Boyd et al. (2013) logit CI on PR-AUC (challenger CI lower bound > champion point,
   or non-overlapping CIs). Bootstrap/CV CIs are invalid for AUPRC — do not use them.

3. **Sponsor-history temporal check is a HARD GATE (amends Step 2).** `build_sponsor_history.py`
   must, after computing, sample ≥500 TEST-set trials and RAISE if any has
   `sponsor_prior_trial_count` including a trial with `study_first_posted_date >= its own`.
   Not a deferred manual check.

4. **Headline reporting split = gold cutoffs 2022/2023 for ALL phases** (test = 2024+). Keep a
   single shared cutoff config — do NOT use per-phase cutoffs (breaks the combo-trial
   same-split guarantee). Phase I headline test PR-AUC (n=20 positives) is still reported but
   ALWAYS with the caveat "n=20 positives, wide CI, not used for promotion — see walk-forward
   gate."

5. **Feature additions (Step 2/3/4 build.py):** `num_countries`, `is_multinational`,
   `num_primary_outcomes`, `has_survival_endpoint` require AACT `facilities` and
   `design_outcomes` tables that are NOT in the current 7-table mirror. `healthy_volunteers`
   IS available (`eligibilities`). Resolve the mirror-extension decision before implementing
   these (see audit). Target feature count 432 assumes all five land.

6. **TRAINING STRATEGY RESOLVED — single track, gold-only. Track B DROPPED.** During the
   pre-training audits a two-track plan was considered (Track A = gold-only; Track B =
   weak+gold with covariate correction). Track B is ruled out by two independent analyses:
   `pre_training_audit.md` Item 1 (importance weighting collapses effective sample size to 5%,
   domain AUC 0.90) and `failure_class_shift.md` (failure-class domain AUC 0.940, ABOVE the
   whole-population 0.912, uniform across phases; only 9.1% of weak failures are gold-like).
   The weak failures are MORE shifted than the population — the gold set was expert-enriched for
   hard failures; weak failures are routine terminations. **Train GOLD-ONLY (Step 5).** Weak
   data's value is realized as FEATURES (sponsor/indication historical rates over all ~117k weak
   trials), NOT as training rows. Do NOT build any weak-augmented training path or the
   contamination-guarded weak-row pipeline. `contamination_guard.py` stays as tested, unused
   infrastructure only. This supersedes any Track-A-vs-Track-B comparison, and Step 10's Table 1
   "Ensemble" comparison remains within the gold-only track (XGBoost vs LGBM vs CatBoost stack).

---

## Step 0 — CLAUDE.md update (do this first, before any code)

Update CLAUDE.md's "Active phase" line:
```
Phase 2: definition audit → sponsor features → gold-label retraining → ensemble
```

Then install new deps:
```bash
uv add betacal nannyml
uv add --optional bench autogluon.tabular tabpfn
# tabpfn and autogluon are heavy — add as optional so they don't bloat prod image
```

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
bench = ["autogluon.tabular", "tabpfn>=2.0"]
```

---

## Step 1 — Definition audit (mandatory before any retraining)

This is not optional and must complete before Step 2.
The research finding: CTO weak labels may encode "phase-progression" (~60–66% success)
while gold labels may encode "approval/efficacy" (~10–20% success).
If the definitions differ, the supervision leakage finding reframes entirely.

### 1.1 Create `reports/definition_audit.md` programmatically

Write `scripts/definition_audit.py` that produces this report. Requirements:

**Section A — Gold label distribution analysis**
```python
gold = pd.read_parquet("data/raw/cto_gold.parquet")

# 1. Print all columns in the gold file
# 2. Per phase: n, pos_rate, overall_status value_counts,
#    study_type value_counts
# 3. For label=1 trials: what is the distribution of overall_status?
#    If label=1 mostly means "COMPLETED" → suggests completion-based definition
#    If label=1 has a mix of COMPLETED + TERMINATED → suggests efficacy-based
# 4. For label=0 trials: are they mostly TERMINATED or also COMPLETED?
#    Many COMPLETED+label=0 → efficacy definition (completed but failed primary endpoint)
#    Mostly TERMINATED+label=0 → termination definition
# 5. Print: among COMPLETED trials, what % are labeled 1 vs 0?
# 6. Print: among TERMINATED trials, what % are labeled 1 vs 0?
```

**Section B — CTO weak label analysis**
```python
for phase_name, phase_file in [
    ("phase1", "data/raw/cto_phase1.parquet"),
    ...
]:
    df = pd.read_parquet(phase_file)
    # 1. pos_rate (pred_proba >= 0.5)
    # 2. Distribution of pred_proba (histogram bins)
    # 3. If overall_status is available: cross-tab y vs overall_status
```

**Section C — Write conclusion**
Based on the output, determine and write:
- If gold label=1 requires COMPLETED + results showing efficacy → "approval/efficacy definition"
- If gold label=1 = any COMPLETED trial that advanced → "phase-progression definition"
- Record the actual base rates and conclude whether the 83% vs 20% gap is
  (a) definitional mismatch, (b) systematic optimism in CTO labeling functions, or (c) both
- This conclusion goes into `reports/definition_audit.md` and is referenced in the README

Run `python scripts/definition_audit.py > reports/definition_audit.md` and
review the output manually before proceeding to Step 2.

**Interpretation guide:**
- If COMPLETED trials split ~50/50 on label=1 → efficacy definition (hard, correct)
- If COMPLETED trials are ~80%+ label=1 → phase-progression or completion definition
- If TERMINATED trials are mostly label=0 → consistent with any definition
- The 20% gold positive rate for Phase I is consistent with the FDA's reported
  ~14% Phase I→approval success rate — supporting an efficacy/approval definition

---

## Step 2 — Sponsor track record features (highest ROI feature addition)

### 2.1 Why this is high ROI
Research by Lo, Siah & Wong (*Biostatistics* 2019) and Aliper et al. (2023)
both identify sponsor type and track record as among the top predictors.
Your AACT mirror has 591k trials — enough to compute meaningful historical rates.

### 2.2 src/cto/features/sponsor_history.py

**TEMPORAL LEAKAGE RULE (hardcoded check):**
All lookups must filter to: `prior_trial.study_first_posted_date < current_trial.study_first_posted_date`
This is the sponsor analog of the completion_date split. Violating this leaks future sponsor success.

Implement:
- `compute_sponsor_history(studies_df: pd.DataFrame) -> pd.DataFrame`
  - Input: full AACT studies snapshot with columns:
    `nct_id, study_first_posted_date, source_class, overall_status, phase`
  - For each trial, compute from PRIOR trials by the same sponsor (source):
    - `sponsor_prior_trial_count`: total prior trials (all phases)
    - `sponsor_prior_phase_count`: prior trials of the same phase
    - `sponsor_prior_completion_rate`: fraction of prior trials with overall_status=COMPLETED
    - `sponsor_prior_same_phase_completion_rate`: completion rate for same phase only
    - `sponsor_is_established`: int(sponsor_prior_trial_count >= 5)
    - `sponsor_is_large`: int(sponsor_prior_trial_count >= 20)
  - This requires a self-join / grouped time-windowed aggregation.
    For 591k trials, do this efficiently with a sort + cumulative groupby, not nested loops.
    Recommended approach: sort by study_first_posted_date, then use
    `df.groupby('source').apply(lambda g: g.expanding().agg(...))` or a
    vectorized cumsum approach per sponsor group.
  - For sponsors with no prior history: fill all rates with the phase-level median
    and `sponsor_prior_trial_count=0`, `sponsor_is_established=0`.
  - **ASSERT**: for every row, verify that prior trials counted have
    `study_first_posted_date < current trial's study_first_posted_date`.
    Raise if violation found.
  - Return DataFrame with columns:
    `nct_id, sponsor_prior_trial_count, sponsor_prior_phase_count,
     sponsor_prior_completion_rate, sponsor_prior_same_phase_completion_rate,
     sponsor_is_established, sponsor_is_large`
  - Save to `data/interim/sponsor_history.parquet` (DVC-track this).

- `load_sponsor_history() -> pd.DataFrame`
  - Reads `data/interim/sponsor_history.parquet`.

### 2.3 tests/test_sponsor_history.py (write this FIRST — TDD)

```python
import pytest
import pandas as pd
import numpy as np
from cto.features.sponsor_history import compute_sponsor_history

def make_sponsor_df():
    """Three sponsors: big_pharma (many trials), startup (few), unknown (1)."""
    return pd.DataFrame({
        "nct_id": [f"NCT{i:07d}" for i in range(8)],
        "source": [
            "BIG_PHARMA", "BIG_PHARMA", "BIG_PHARMA", "BIG_PHARMA",  # 4 trials
            "STARTUP", "STARTUP",                                        # 2 trials
            "UNKNOWN_SPONSOR",                                           # 1 trial
            "BIG_PHARMA",                                                # 5th trial
        ],
        "study_first_posted_date": pd.to_datetime([
            "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01",
            "2015-06-01", "2017-06-01",
            "2016-03-01",
            "2019-01-01",
        ]),
        "overall_status": [
            "COMPLETED", "COMPLETED", "TERMINATED", "COMPLETED",
            "COMPLETED", "TERMINATED",
            "COMPLETED",
            "COMPLETED",
        ],
        "phase": ["PHASE1"] * 8,
    })

def test_no_future_leakage():
    """Prior trial counts must only include trials registered before the current one."""
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    # First BIG_PHARMA trial has no prior history
    first_row = result[result["nct_id"] == "NCT0000000"].iloc[0]
    assert first_row["sponsor_prior_trial_count"] == 0

def test_prior_counts_increase_over_time():
    """Each successive BIG_PHARMA trial should see more prior trials."""
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    bp_rows = result[result["nct_id"].isin(
        ["NCT0000000", "NCT0000001", "NCT0000002", "NCT0000003", "NCT0000007"]
    )].sort_values("nct_id")
    counts = bp_rows["sponsor_prior_trial_count"].values
    assert counts[0] == 0     # first trial: no prior
    assert counts[1] == 1     # second: 1 prior
    assert counts[4] == 4     # fifth: 4 prior

def test_unknown_sponsor_gets_zero_history():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    unk = result[result["nct_id"] == "NCT0000006"].iloc[0]
    assert unk["sponsor_prior_trial_count"] == 0
    assert unk["sponsor_is_established"] == 0

def test_completion_rate_computed_correctly():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    # For NCT0000003 (4th BIG_PHARMA trial, posted 2018):
    # Prior: NCT0 (COMPLETED), NCT1 (COMPLETED), NCT2 (TERMINATED) → rate = 2/3
    row = result[result["nct_id"] == "NCT0000003"].iloc[0]
    assert abs(row["sponsor_prior_completion_rate"] - (2/3)) < 0.01

def test_is_established_threshold():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    # NCT0000007 is the 5th BIG_PHARMA trial — prior count = 4, not yet established
    row_4_prior = result[result["nct_id"] == "NCT0000007"].iloc[0]
    assert row_4_prior["sponsor_prior_trial_count"] == 4
    assert row_4_prior["sponsor_is_established"] == 0  # threshold is 5

def test_output_columns_present():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    expected = {
        "nct_id", "sponsor_prior_trial_count", "sponsor_prior_phase_count",
        "sponsor_prior_completion_rate", "sponsor_prior_same_phase_completion_rate",
        "sponsor_is_established", "sponsor_is_large",
    }
    assert expected.issubset(set(result.columns))

def test_no_nct_id_in_leakage_blocklist():
    """sponsor_history columns must not conflict with blocklist."""
    from cto.features.leakage import LEAKAGE_BLOCKLIST
    sponsor_cols = {
        "sponsor_prior_trial_count", "sponsor_prior_phase_count",
        "sponsor_prior_completion_rate", "sponsor_prior_same_phase_completion_rate",
        "sponsor_is_established", "sponsor_is_large",
    }
    assert not sponsor_cols.intersection(LEAKAGE_BLOCKLIST)
```

### 2.4 Add sponsor history columns to config/features.yaml

Add under `numeric_features`:
```yaml
  - sponsor_prior_trial_count
  - sponsor_prior_phase_count
  - sponsor_prior_completion_rate
  - sponsor_prior_same_phase_completion_rate
```

Add under `flag_features`:
```yaml
  - sponsor_is_established
  - sponsor_is_large
```

### 2.5 Add to dvc.yaml — new interim stage

```yaml
  sponsor_history:
    cmd: python -m cto.pipelines.build_sponsor_history
    deps:
      - src/cto/features/sponsor_history.py
      - data/raw/aact_studies_snapshot.parquet
    outs:
      - data/interim/sponsor_history.parquet
```

Create `src/cto/pipelines/build_sponsor_history.py` — one-liner that calls
`compute_sponsor_history()` and saves to `data/interim/sponsor_history.parquet`.

---

## Step 3 — Therapeutic area success rate features

### 3.1 src/cto/features/indication_history.py

Same temporal leakage rule as sponsor history — only prior trials count.

Implement:
- `compute_indication_history(studies_df: pd.DataFrame,
                              conditions_df: pd.DataFrame) -> pd.DataFrame`
  - Maps MeSH terms to therapeutic areas (buckets: ONCOLOGY, CNS, CV, INFECTIOUS,
    METABOLIC, RARE, IMMUNOLOGY, OTHER) using a keyword mapping dict.
  - For each trial, computes from PRIOR trials (registered before current) in same TA:
    - `ta_prior_trial_count`
    - `ta_prior_completion_rate`
    - `ta_prior_same_phase_completion_rate`
    - `ta_bucket` (int: 0–7, from TA mapping)
  - Fill unknown/no-history with phase-level medians.
  - Return DataFrame keyed by `nct_id`.

Therapeutic area mapping (seed — Claude Code may expand):
```python
TA_KEYWORDS = {
    "ONCOLOGY":    ["cancer", "tumor", "carcinoma", "leukemia", "lymphoma",
                    "melanoma", "sarcoma", "neoplasm", "oncology", "glioma"],
    "CNS":         ["alzheimer", "parkinson", "dementia", "schizophrenia",
                    "depression", "epilepsy", "migraine", "neurology", "brain"],
    "CV":          ["cardiac", "heart", "hypertension", "coronary", "atrial",
                    "stroke", "vascular", "cardiovascular"],
    "INFECTIOUS":  ["hiv", "hepatitis", "tuberculosis", "covid", "infection",
                    "bacterial", "viral", "fungal", "antibiotic"],
    "METABOLIC":   ["diabetes", "obesity", "insulin", "lipid", "metabolic",
                    "thyroid", "adrenal"],
    "RARE":        ["orphan", "rare disease", "genetic disorder", "lysosomal"],
    "IMMUNOLOGY":  ["rheumatoid", "lupus", "autoimmune", "crohn", "psoriasis",
                    "colitis", "immunology"],
}
```

Add to `config/features.yaml` numeric: `ta_prior_trial_count`,
`ta_prior_completion_rate`, `ta_prior_same_phase_completion_rate`;
categorical: `ta_bucket`.

Add to `dvc.yaml` as `indication_history` stage (same pattern as sponsor_history).

---

## Step 4 — Gold-label featurize pipeline

### 4.1 src/cto/pipelines/featurize_gold.py

This is a NEW pipeline stage separate from the Phase 1 featurize stage.
Key differences from Phase 1 featurize:
- Source labels: `data/raw/cto_gold.parquet`, column `labels` (not `pred_proba`)
- Temporal split cutoffs: train_cutoff=2022-12-31, val_cutoff=2023-12-31 (within gold 2020–2024)
- Include sponsor history and indication history features (join by nct_id)
- TF-IDF: DO NOT refit — load the existing `models/tfidf_vectorizer.joblib`
  (vocabulary must stay identical for feature schema consistency)
- `feature_schema_version`: bump to "2.0.0" in `config/features.yaml`
  (new features added: sponsor history + indication history)

Requirements:
- Load `cto_gold.parquet`, extract per-phase subsets using `phase_clean`
- For each phase, call `make_temporal_splits()` on gold data
- Call `assert_temporal_integrity()`
- Call `build_features(phase, split)` — this must now join sponsor_history and
  indication_history parquets in addition to AACT mirror tables
- Assert new feature count > 427 (sanity check that new features were added)
- Assert `models/tfidf_vectorizer.joblib` exists before running (fail with clear error if not)
- Save to `data/processed/features_gold_phase{1,2,3}_{train,val,test}.parquet`
- Save labels to `data/processed/labels_gold_phase{1,2,3}_{train,val,test}.parquet`
- Log: phase, split sizes, feature count, positive rate per split

### 4.2 Update build.py to join sponsor + indication features

In `build_features()`, after the existing AACT table joins, add:
```python
# Join sponsor history (pre-computed, no leakage — computation enforces temporal ordering)
sponsor_hist = load_sponsor_history()
df = df.merge(sponsor_hist, on="nct_id", how="left")

# Join indication history
indication_hist = load_indication_history()
df = df.merge(indication_hist, on="nct_id", how="left")
```

### 4.3 Update dvc.yaml — featurize_gold stage

```yaml
  featurize_gold:
    cmd: python -m cto.pipelines.featurize_gold
    deps:
      - src/cto/pipelines/featurize_gold.py
      - src/cto/features/build.py
      - src/cto/features/leakage.py
      - data/raw/cto_gold.parquet
      - data/raw/aact_studies_snapshot.parquet
      - data/interim/sponsor_history.parquet
      - data/interim/indication_history.parquet
      - models/tfidf_vectorizer.joblib
      - config/leakage_blocklist.yaml
      - config/features.yaml
    outs:
      - data/processed/features_gold_phase1_train.parquet
      - data/processed/features_gold_phase1_val.parquet
      - data/processed/features_gold_phase1_test.parquet
      - data/processed/features_gold_phase2_train.parquet
      - data/processed/features_gold_phase2_val.parquet
      - data/processed/features_gold_phase2_test.parquet
      - data/processed/features_gold_phase3_train.parquet
      - data/processed/features_gold_phase3_val.parquet
      - data/processed/features_gold_phase3_test.parquet
      - data/processed/labels_gold_phase1_train.parquet
      - data/processed/labels_gold_phase1_val.parquet
      - data/processed/labels_gold_phase1_test.parquet
      - data/processed/labels_gold_phase2_train.parquet
      - data/processed/labels_gold_phase2_val.parquet
      - data/processed/labels_gold_phase2_test.parquet
      - data/processed/labels_gold_phase3_train.parquet
      - data/processed/labels_gold_phase3_val.parquet
      - data/processed/labels_gold_phase3_test.parquet
    params:
      - params.yaml:
        - split
        - features
```

---

## Step 5 — Gold-label model training with Optuna

### 5.1 src/cto/models/train_gold.py

Train per-phase XGBoost models on gold labels with Optuna HPO.

**Critical differences from Phase 1 train.py:**
- Source: `features_gold_phase{n}_{split}.parquet`
- `scale_pos_weight` from `params.yaml: model.class_weight_by_phase` (gold base rates)
  NOT computed from training data
- Optuna budget: 50 trials for Phase I/II, 100 for Phase III
- Search space: shallow and regularization-focused (prevents overfitting at n~800–900)
- Calibration: `CalibratedClassifierCV(cv=5, method="sigmoid")` for Phase I/II;
  `CalibratedClassifierCV(cv=5, method="isotonic")` for Phase III
- Evaluation objective for Optuna: PR-AUC on val set using calibrated probabilities
- MLflow experiment: `cto_gold`

**Optuna search space (hardcode this — do not let Claude Code expand it):**
```python
params = {
    "max_depth":        trial.suggest_int("max_depth", 2, 5),
    "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 15.0, log=True),
    "subsample":        trial.suggest_float("subsample", 0.6, 0.9),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
    "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
    "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
    "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
    "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
    # Fixed — do NOT add to search space:
    "n_estimators":     2000,    # early stopping controls actual count
    "tree_method":      "exact", # recommended for small n
    "eval_metric":      "aucpr",
    "early_stopping_rounds": 50,
    "scale_pos_weight": scale_pos_weight,  # from params.yaml gold rates
    "random_state":     42,
}
```

**Repeated CV for Phase I (n_train ≈ 800):**
For Phase I specifically, use `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` within
the training period instead of a single val split for Optuna's objective function.
This reduces variance on the 800-row training set. The held-out temporal val/test splits
are still used for final honest evaluation only.

**Function signature:**
```python
def train_gold_phase(phase: int,
                     use_repeated_cv: bool = None,  # auto: True if n_train < 1000
                     experiment_name: str = "cto_gold") -> dict:
    # Returns: {"run_id": str, "val_prauc": float, "test_prauc": float}
```

Log to MLflow per run:
- All Optuna best params
- `n_train`, `n_val`, `n_test`, `n_features`
- `scale_pos_weight` (from params, not computed)
- `feature_schema_version`
- `training_labels`: "gold"
- `calibration_method`: "sigmoid_cv5" or "isotonic_cv5"
- Metrics: `val_prauc`, `val_auroc`, `val_brier`, `val_ece`
- Metrics: `test_prauc`, `test_auroc`, `test_brier`, `test_ece`
- Metrics: `gold_prauc`, `gold_auroc` (gold = the full gold set, held-out)

### 5.2 Update params.yaml

```yaml
model:
  phases: [1, 2, 3]
  class_weight_by_phase:
    # From gold label base rates — DO NOT use weak-label rates
    # Phase I: pos_rate=0.199 → (1-0.199)/0.199 = 4.02
    # Phase II: pos_rate=0.309 → (1-0.309)/0.309 = 2.23
    # Phase III: pos_rate=0.529 → (1-0.529)/0.529 = 0.89
    1: 4.02
    2: 2.23
    3: 0.89
  n_optuna_trials_small: 50    # for Phase I/II (n < 2000)
  n_optuna_trials_large: 100   # for Phase III
  early_stopping_rounds: 50
  n_estimators: 2000
  random_state: 42
  calibration_method_small_n: "sigmoid"   # Phase I/II
  calibration_method_large_n: "isotonic"  # Phase III
  calibration_cv: 5
```

---

## Step 6 — Ensemble: XGBoost + LightGBM + CatBoost

### 6.1 src/cto/models/ensemble.py

**What to build:** a 3-model stack with a logistic meta-learner trained on
out-of-fold (OOF) predicted probabilities.

**Why OOF (not val predictions):**
Training the meta-learner on val predictions exposes it to the same temporal split,
inflating meta-learner performance. OOF predictions from the training set give an
honest estimate of each base model's predictions while containing no future information.

Requirements:
- `train_ensemble(phase: int, experiment_name: str = "cto_gold") -> dict`
  - Loads gold train features/labels for this phase
  - Trains XGBoost, LightGBM, and CatBoost using best Optuna params from `cto_gold` run
    (load from MLflow by tag: `phase={phase}, model_type={name}`)
  - Generates OOF predictions using `StratifiedKFold(n_splits=5)` for each base model
  - Stacks OOF predictions (3 columns) + trains a `LogisticRegression(C=0.1)` meta-learner
  - Evaluates ensemble on val and test (same temporal held-outs as base models)
  - Calibrates ensemble probabilities using `CalibratedClassifierCV(cv=5)`
  - Logs to MLflow experiment `cto_gold` with tag `model_type="ensemble"`
  - Returns test_prauc for comparison

**LightGBM training (use same search space bounds as XGBoost, adapted):**
```python
lgb_params = {
    "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
    "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
    "subsample":         same as XGBoost,
    "colsample_bytree":  same,
    "reg_lambda":        same,
    "reg_alpha":         same,
    "learning_rate":     same,
    "is_unbalance":      True,  # instead of scale_pos_weight for LightGBM
    "n_estimators":      2000,
    "early_stopping_rounds": 50,
}
```

**CatBoost training:**
```python
cat_params = {
    "depth":           trial.suggest_int("depth", 2, 6),
    "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    "learning_rate":   same bounds,
    "subsample":       same,
    "auto_class_weights": "Balanced",  # CatBoost handles imbalance natively
    "iterations":      2000,
    "early_stopping_rounds": 50,
    "eval_metric":     "PRAUC",
    "verbose":         False,
}
```

### 6.2 Promotion gate check

After ensemble trains, check:
```python
if ensemble_test_prauc > best_single_model_prauc + 0.005:
    print(f"Phase {phase}: Ensemble PASSES promotion gate")
    # Update MLflow alias: @best_model_{phase} → ensemble run
else:
    print(f"Phase {phase}: Ensemble does NOT beat single model — keep best single model")
    # Do not promote ensemble
```

---

## Step 7 — AutoGluon benchmark (optional dep, benchmarking only)

### 7.1 scripts/run_autogluon.py

**Run this separately from the main pipeline — do not add to dvc.yaml.**
AutoGluon is 2GB+ and should not be in the prod image.

```python
"""
AutoGluon benchmark — run once, compare to ensemble, do not add to dvc.yaml.
Install: uv pip install --optional bench
"""
from autogluon.tabular import TabularPredictor
import pandas as pd, mlflow

for phase in [1, 2, 3]:
    X_train = pd.read_parquet(f"data/processed/features_gold_phase{phase}_train.parquet")
    y_train = pd.read_parquet(f"data/processed/labels_gold_phase{phase}_train.parquet")["y"]
    X_test  = pd.read_parquet(f"data/processed/features_gold_phase{phase}_test.parquet")
    y_test  = pd.read_parquet(f"data/processed/labels_gold_phase{phase}_test.parquet")["y"]

    train_df = X_train.copy()
    train_df["label"] = y_train

    predictor = TabularPredictor(
        label="label",
        eval_metric="average_precision",  # PR-AUC
        path=f"models/autogluon_phase{phase}",
    ).fit(
        train_df,
        presets="best_quality",
        time_limit=4 * 3600,  # 4 hours per phase
        excluded_model_types=["KNN"],  # KNN is slow and poor at this scale
    )

    probs = predictor.predict_proba(X_test)[1]
    from sklearn.metrics import average_precision_score
    test_prauc = average_precision_score(y_test, probs)

    # Log to MLflow
    with mlflow.start_run(experiment_id=..., tags={"model_type": "autogluon", "phase": str(phase)}):
        mlflow.log_metric("test_prauc", test_prauc)
        mlflow.log_param("preset", "best_quality")
        print(f"Phase {phase} AutoGluon test PR-AUC: {test_prauc:.3f}")
```

**Decision rule after AutoGluon:**
- If AutoGluon beats ensemble by > 0.02 on test PR-AUC: adopt as production model
- If within 0.02: keep ensemble (interpretable, smaller, no 2GB dep)
- Never adopt AutoGluon unless it clearly wins — deployment complexity cost is real

---

## Step 8 — TabPFN for Phase I and II (small-n)

### 8.1 scripts/run_tabpfn.py

**Phase I and II only.** Phase III n_train is large enough for GBDTs.
**Feature limit:** TabPFN v2 supports max ~500 features and ~10k samples.
Your gold Phase I train is ~800 rows with ~430+ features — fine on both counts,
but drop or PCA the TF-IDF block to reduce noise.

```python
"""
TabPFN v2 benchmark for Phase I and II.
Requires: uv pip install --optional bench (includes tabpfn>=2.0)
GPU recommended but not required (5× slower on CPU).
"""
from tabpfn import TabPFNClassifier
from sklearn.decomposition import PCA
import pandas as pd, numpy as np, mlflow
from sklearn.metrics import average_precision_score

for phase in [1, 2]:  # NOT phase 3
    X_train = pd.read_parquet(f"data/processed/features_gold_phase{phase}_train.parquet")
    y_train = pd.read_parquet(f"data/processed/labels_gold_phase{phase}_train.parquet")["y"]
    X_test  = pd.read_parquet(f"data/processed/features_gold_phase{phase}_test.parquet")
    y_test  = pd.read_parquet(f"data/processed/labels_gold_phase{phase}_test.parquet")["y"]

    # Separate TF-IDF columns (high-dim, noisy) from structured features
    tfidf_cols = [c for c in X_train.columns if c.startswith("tfidf_")]
    struct_cols = [c for c in X_train.columns if not c.startswith("tfidf_")]

    # PCA on TF-IDF block (400 dims → 30 components)
    pca = PCA(n_components=30, random_state=42)
    tfidf_train_pca = pca.fit_transform(X_train[tfidf_cols].fillna(0))
    tfidf_test_pca  = pca.transform(X_test[tfidf_cols].fillna(0))

    X_train_pca = np.hstack([X_train[struct_cols].values, tfidf_train_pca])
    X_test_pca  = np.hstack([X_test[struct_cols].values, tfidf_test_pca])

    clf = TabPFNClassifier(n_estimators=32)  # default ensemble size
    clf.fit(X_train_pca, y_train.values)
    probs = clf.predict_proba(X_test_pca)[:, 1]
    test_prauc = average_precision_score(y_test, probs)

    with mlflow.start_run(tags={"model_type": "tabpfn", "phase": str(phase)}):
        mlflow.log_metric("test_prauc", test_prauc)
        mlflow.log_param("pca_components", 30)
        print(f"Phase {phase} TabPFN test PR-AUC: {test_prauc:.3f}")

    # Decision: adopt TabPFN if it beats ensemble by > 0.01 on Phase I test PR-AUC
```

---

## Step 9 — MLflow champion promotion gate

> **SUPERSEDED by AMENDMENT 2.** The bare `margin=0.01` rule below is NOT the gate. Use the
> interval-based gate in `src/cto/models/gate.py` (walk-forward + Nadeau-Bengio for Phase I;
> logit-CI single split for Phase II/III; PR-AUC and AUROC co-primary). `registry.py` should
> call `gate.promotion_decision_*` to get the promote/retain decision, then only flip the
> MLflow `@champion` alias when the gate returns `promote=True`. The snippet below is kept for
> the alias-flipping mechanics only — replace its `challenger_prauc >= champion_prauc + margin`
> check with the gate decision.

### 9.1 src/cto/models/registry.py

```python
def promote_if_better(phase: int, challenger_prauc: float,
                       challenger_run_id: str,
                       margin: float = 0.01) -> bool:
    """
    Promote challenger to @champion only if it beats current champion
    on the SAME gold test set by at least `margin` PR-AUC.

    Returns True if promoted, False if champion retained.
    """
    client = mlflow.tracking.MlflowClient()
    model_name = f"cto_phase{phase}"

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        # Load champion's test_prauc from its run
        champion_run = client.get_run(champion.run_id)
        champion_prauc = champion_run.data.metrics.get("test_prauc", 0.0)
    except mlflow.exceptions.MlflowException:
        champion_prauc = 0.0  # no champion yet

    if challenger_prauc >= champion_prauc + margin:
        # Promote: register challenger, assign @champion alias
        model_uri = f"runs:/{challenger_run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        client.set_registered_model_alias(model_name, "champion", mv.version)
        client.set_registered_model_alias(model_name, "challenger", mv.version)
        print(f"Phase {phase}: PROMOTED — challenger ({challenger_prauc:.4f}) "
              f"beats champion ({champion_prauc:.4f}) by {challenger_prauc-champion_prauc:.4f}")
        return True
    else:
        print(f"Phase {phase}: RETAINED champion ({champion_prauc:.4f}) — "
              f"challenger ({challenger_prauc:.4f}) did not beat margin={margin}")
        return False
```

---

## Step 10 — Results report

### 10.1 scripts/generate_phase2_report.py

Write a script that queries MLflow and generates `reports/phase2_results.md`.

The report must include:

**Table 1: Phase 1 vs Phase 2 comparison**
```
| Phase | Training labels | PR-AUC (test) | PR-AUC (gold) | Model type  |
|-------|-----------------|---------------|---------------|-------------|
| I     | CTO weak        | 0.908         | 0.358         | XGBoost     |
| I     | Gold            | ?             | ?             | XGBoost+Opt |
| I     | Gold            | ?             | ?             | Ensemble    |
| II    | CTO weak        | 0.762         | 0.514         | XGBoost     |
| ...   | ...             | ...           | ...           | ...         |
| III   | CTO weak        | 0.808         | 0.700         | XGBoost     |
| III   | Gold            | ?             | ?             | XGBoost+Opt |
```

**Table 2: Definition audit summary**
- Finding from Step 1 in 3–5 bullet points
- Conclusion on whether divergence is definitional mismatch vs systematic optimism

**Table 3: Feature importance changes**
- Top 10 SHAP features for Phase III model (Phase 1 vs Phase 2)
- Did sponsor features appear in top 10?

**Table 4: Calibration improvement**
- Brier score and ECE per phase before/after calibration fix

---

## Step 11 — Tests for Phase 2

### 11.1 tests/test_gold_pipeline.py

```python
import pytest
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
INTERIM = Path("data/interim")

@pytest.mark.skipif(not (INTERIM / "sponsor_history.parquet").exists(),
                    reason="Run dvc repro sponsor_history first")
def test_sponsor_history_no_future_leakage():
    """Spot-check: no trial should have a prior_count > its rank in date-sorted order."""
    from cto.features.sponsor_history import compute_sponsor_history
    studies = pd.read_parquet("data/raw/aact_studies_snapshot.parquet",
                              columns=["nct_id", "source", "study_first_posted_date",
                                       "overall_status", "phase"])
    result = pd.read_parquet(INTERIM / "sponsor_history.parquet")
    # Sample 1000 random trials and verify prior count <= number of earlier trials
    # by same sponsor in the full studies df
    sample = result.sample(min(1000, len(result)), random_state=42)
    merged = sample.merge(studies[["nct_id", "source", "study_first_posted_date"]],
                          on="nct_id")
    for _, row in merged.head(50).iterrows():
        actual_prior = len(studies[
            (studies["source"] == row["source"]) &
            (studies["study_first_posted_date"] < row["study_first_posted_date"])
        ])
        assert row["sponsor_prior_trial_count"] <= actual_prior, (
            f"Temporal leakage in sponsor history for {row['nct_id']}: "
            f"computed={row['sponsor_prior_trial_count']}, actual_prior={actual_prior}"
        )

@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize_gold first")
def test_gold_feature_count_increased():
    """Gold features must include sponsor/indication features — count should exceed Phase 1."""
    phase1_features = len(pd.read_parquet(
        PROCESSED / "features_phase1_train.parquet").columns)
    gold_features = len(pd.read_parquet(
        PROCESSED / "features_gold_phase1_train.parquet").columns)
    assert gold_features > phase1_features, (
        f"Gold feature count ({gold_features}) not greater than Phase 1 ({phase1_features}). "
        f"Sponsor/indication features may not have been added."
    )

@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize_gold first")
def test_gold_positive_rates_match_expected():
    """Gold label positive rates must match the confirmed gold base rates."""
    expected = {1: (0.15, 0.25), 2: (0.25, 0.38), 3: (0.45, 0.62)}
    for phase, (lo, hi) in expected.items():
        for split in ["train", "val", "test"]:
            y = pd.read_parquet(PROCESSED / f"labels_gold_phase{phase}_{split}.parquet")["y"]
            rate = y.mean()
            assert lo <= rate <= hi, (
                f"Phase {phase} {split}: positive rate {rate:.3f} outside "
                f"expected range [{lo}, {hi}]. Check label column and phase filter."
            )
```

---

## Full Phase 2 verification checklist

```bash
# 1. Tests
uv run pytest tests/ -v --tb=short

# 2. Lint
uv run ruff check src/ tests/

# 3. Definition audit (review manually)
python scripts/definition_audit.py > reports/definition_audit.md
cat reports/definition_audit.md  # read and understand before proceeding

# 4. Full pipeline
dvc repro

# 5. Check gold feature files and positive rates
python -c "
import pandas as pd
for phase in [1, 2, 3]:
    for split in ['train', 'val', 'test']:
        X = pd.read_parquet(f'data/processed/features_gold_phase{phase}_{split}.parquet')
        y = pd.read_parquet(f'data/processed/labels_gold_phase{phase}_{split}.parquet')
        print(f'Phase {phase} {split}: X={X.shape}, pos_rate={y[\"y\"].mean():.2f}')
"

# 6. Check MLflow results
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
runs = mlflow.search_runs(experiment_names=['cto_gold'])
print(runs[['tags.phase','tags.model_type',
            'metrics.test_prauc','metrics.gold_prauc']].sort_values(
    ['tags.phase','metrics.test_prauc']).to_string())
"

# 7. Phase III gold PR-AUC gate (must beat 0.700)
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
runs = mlflow.search_runs(experiment_names=['cto_gold'])
phase3 = runs[runs['tags.phase'] == '3'].sort_values('metrics.gold_prauc', ascending=False)
best = phase3.iloc[0]
prauc = best['metrics.gold_prauc']
model = best.get('tags.model_type', 'unknown')
print(f'Best Phase III gold PR-AUC: {prauc:.4f} ({model})')
assert prauc > 0.700, f'FAIL: {prauc:.4f} does not beat Phase 1 baseline of 0.700'
print('PASS: Phase 2 beats Phase 1 baseline')
"

# 8. Sponsor feature leakage spot-check
python -c "
import pandas as pd
from cto.features.leakage import LEAKAGE_BLOCKLIST
X = pd.read_parquet('data/processed/features_gold_phase3_train.parquet')
sponsor_cols = [c for c in X.columns if 'sponsor' in c or 'ta_' in c]
print(f'Sponsor/TA features present: {sponsor_cols}')
leaked = set(X.columns) & LEAKAGE_BLOCKLIST
assert not leaked, f'Leakage found: {leaked}'
print('No leakage detected')
"

# 9. Generate final report
python scripts/generate_phase2_report.py
cat reports/phase2_results.md
```

---

## Notes for Claude Code

- **Step 1 (definition audit) is mandatory.** Do not skip or defer it.
  Read and interpret the output before writing any model code.
- Write `tests/test_sponsor_history.py` before `sponsor_history.py` — TDD.
- Write `tests/test_gold_pipeline.py` before `featurize_gold.py` — TDD.
- The TF-IDF vectorizer in `models/tfidf_vectorizer.joblib` must NOT be refit.
  If it is missing, raise `FileNotFoundError` with: "Run Phase 1 featurize first."
- `ultrathink` before implementing `compute_sponsor_history` — the temporal
  self-join is the most complex data operation in the project and the most
  likely to introduce silent leakage. Verify with the test before trusting it.
- LightGBM and CatBoost should reuse the same Optuna search space bounds as
  XGBoost where possible — don't invent new ranges without justification.
- AutoGluon and TabPFN are benchmarking tools, not production models.
  Do not add them to dvc.yaml or the FastAPI serving layer.
- If Phase III gold PR-AUC after ensemble is still ≤ 0.700, do NOT tune further
  or add complexity — document the result honestly. The sponsor features not helping
  is itself a finding worth reporting.
- After dvc repro completes, run the full verification checklist before
  declaring Phase 2 done.
