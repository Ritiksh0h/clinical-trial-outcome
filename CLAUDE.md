# cto-predict — Clinical Trial Outcome Prediction System

## Project in one sentence
Binary classification (success/failure) of drug/biologics clinical trials per phase
(I, II, III) using only registration-time metadata — no post-hoc data, ever.

## Active phase
Phase 2: definition audit → sponsor features → gold-label retraining → ensemble
Full build plan: @docs/PHASE2.md
Previous phases: @docs/PHASE0.md @docs/PHASE1.md

## Confirmed findings (do not re-investigate)
- Phase 0/1 complete: 43→60→N tests green, AACT mirror 591k studies
- The weak-vs-gold gap is POPULATION-driven (covariate shift), NOT label-driven.
  On overlap trials, weak and gold labels agree 96-98% (label effect ~3pp;
  population effect ~45pp; population is 94% of the gap). The Phase 1 model did
  not learn labeling artifacts — it trained on ~83%-completed trials and was
  evaluated on ~20%-completed trials. Weak LF is conservative: weak=0/gold=1
  cell is exactly 0 in all phases (never calls a real success a failure).
  See reports/label_vs_population.md and reports/weak_failure_pool.md.
- Gold set distribution (use THESE for scale_pos_weight, not weak-label rates):
    Phase I:   n=1,933  pos_rate=0.199  scale_pos_weight≈4.02
    Phase II:  n=3,646  pos_rate=0.309  scale_pos_weight≈2.23
    Phase III: n=2,418  pos_rate=0.529  scale_pos_weight≈0.89
- Phase 1 baseline (gold eval): Phase I=0.358, Phase II=0.514, Phase III=0.700 PR-AUC
- Phase III beats published XGBoost baseline (0.697) — this is the one honest result
- Features: gold build (schema 2.1.0) produces 438 (27 structured + 400 TF-IDF + 11 history).
  TF-IDF vectorizer fitted on Phase 1 train — DO NOT refit.
- number_of_facilities, num_countries, is_multinational DROPPED as soft leakage — all derive
  from AACT tables (facilities, countries) that ACCRUE during trial conduct, not
  registration-time. WITHDRAWN trials show truncated counts (median ~1) because they never
  opened sites — the count is a consequence of the outcome. Confirmed by impact test:
  dropping facilities cost only 0.04 Phase III PR-AUC (0.819→0.779), so not load-bearing;
  dropping all three left Phase III ~0.83 (unchanged). This corrects the Phase 1 baseline,
  which was mildly inflated by these features. Blocklisted (aact_conduct_accrued_soft_leak).
  See reports/facilities_leak_check.md.
- THIRD silent data bug found and fixed (dead-feature scan, scripts/dead_feature_scan.py):
  enrollment_log used the wrong magic string ('ANTICIPATED' vs AACT's 'ESTIMATED') — dead
  (all-NaN) since Phase 1. Fixed to ESTIMATED (registration-time, leakage-safe). ACTUAL
  enrollment is soft leakage (completion-time) and excluded via the ESTIMATED gate
  (see enrollment_conditional_note in leakage_blocklist.yaml). enrollment_log stays
  mostly-NaN by design (98.6% of gold is ACTUAL/overwritten) but carries real signal for
  ESTIMATED rows; XGBoost handles NaN natively. Two features (healthy_volunteers,
  enrollment_log) were dead in Phase 1 — the honest live structured count was 25, not 27.
  Full-population trial size would need an AACT enrollment-history pull (deferred).
- Phase I gold test (2024, single split) has only 20 positives — too few for a reliable
  gate (±0.22 CI). Phase I promotion uses walk-forward temporal CV across 2021-2024 to pool
  positives while keeping train-past/test-future validity. PR-AUC CIs use Boyd et al. logit
  method; bootstrap/CV CIs are invalid for AUPRC. See src/cto/models/gate.py.
- Track B (weak-augmented training) DROPPED. Two independent analyses agree the weak failures
  cannot help: (a) pre_training_audit Item 1 — importance weighting collapses effective sample
  size to 5% (domain AUC 0.90); (b) failure_class_shift diagnostic — failure-class domain AUC
  0.940 (ABOVE the whole-population 0.912), uniform across phases (I 0.925 / II 0.938 /
  III 0.948); only 9.1% of weak failures reach gold-like territory. Weak failures are MORE
  shifted than the population, not less — the gold set was expert-enriched for hard/informative
  failures, weak failures are routine terminations. Ship gold-only (Track A). Weak data's value
  is realized as FEATURES (sponsor/indication historical rates from all 117k weak trials), NOT
  as training rows. See reports/failure_class_shift.md and reports/pre_training_audit.md.

- Gold phase-routing = CTO MEMBERSHIP (trial in phase N iff its nct_id is in
  cto_phaseN.parquet; combo trials appear in BOTH their phases). Chosen over _PHASE_MAP:
  consistent with all prior analysis (which used membership — switching invalidates it);
  _PHASE_MAP drops ~40% of Phase I gold (the scarcest phase); combo trials in two phases are
  harmless — the temporal split is a pure function of completion_date with shared cutoffs, so
  a combo trial lands in the SAME split in both phases (pre_phase2_audit Item 1), no
  cross-phase leakage. `featurize_gold` is the SOLE gold-split computer: one split object per
  phase builds the test matrix AND freezes the test nct_ids to
  `data/processed/gold_test_nct_ids.json` (AUTHORITATIVE — phase1=410, phase2=555, phase3=275,
  all=1073; gold cutoffs train≤2022 / val 2023 / test 2024+; schema 2.1.0, 438 features).
  NEVER recompute the gold split elsewhere — read the frozen file
  (contamination_guard.load_gold_test_nct_ids).
- Model bake-off (XGBoost/LightGBM/CatBoost/CatBoost-native) on clean 438-feature set:
  all statistically indistinguishable. No challenger passes the promotion gate on any
  phase (Phase III a literal tie at 0.828; LightGBM tied but CI lower bound 0.739 <
  champion point → gate correctly refused). XGBoost retained. CatBoost-native (categoricals
  + has_time) did NOT beat engineered features — the leakage-safe sponsor/TA rate features
  already capture the categorical signal. Final model-agnostic headline: Phase III PR-AUC
  ~0.83, Phase II ~0.26, Phase I ~0.31 walk-forward — leak-free, four-algorithm-robust.

## Hard rules — never violate these

### Data integrity
- NEVER use any column from `LEAKAGE_BLOCKLIST` in `config/leakage_blocklist.yaml` as a feature
- NEVER random-split data — always split by `completion_date` ascending
- NEVER fit any transformer (TF-IDF, scaler, encoder) on validation or test data
- NEVER overwrite `data/raw/` with fewer rows than it already has (mirror guard)
- NEVER delete or modify `data/processed/` without bumping `feature_schema_version`
- CTO CSV columns (except `nct_id`) are label-side only — NOT features
- NEVER compute scale_pos_weight from CTO weak labels — use gold base rates above
- NEVER overwrite a non-empty parquet snapshot with an empty DataFrame:
  mirror.py must assert len(df) > 0 before any snapshot write

### Sponsor/history features (TWO temporal windows — registration vs outcome-known)
- COUNT features (sponsor_prior_trial_count, sponsor_prior_phase_count, is_established,
  is_large) use REGISTRATION order: a prior counts iff
  other_trial.study_first_posted_date < this_trial.study_first_posted_date. Valid for counts.
- RATE features (sponsor_prior_completion_rate, sponsor_prior_same_phase_completion_rate)
  use OUTCOME-KNOWN order: a prior counts toward the rate ONLY if
  other_trial.completion_date < this_trial.study_first_posted_date. "Registered before" is
  necessary but NOT sufficient — a prior still running at the current registration has no
  known outcome, so using its eventual status leaks the future (audit Item 3: was ~37.6% of
  counted priors; fixed). Null completion or completion on/after registration → EXCLUDED.
- Same TWO-WINDOW rule applies to indication_history (ta_prior_completion_rate) — build it
  outcome-known from the start; do NOT repeat the Item 3 leak.
- HARD GATE in the build (not manual): build_sponsor_history.py samples ≥500 TEST-set trials
  and runs BOTH gates — assert_no_future_leakage (count: no future-registered prior) AND
  assert_rate_outcome_known (rate: stored rate == honest outcome-known rate). RAISE on either.

### Model integrity
- Promotion gate (implemented in `src/cto/models/gate.py`) — NEVER promote on a bare
  point-estimate difference. PR-AUC AND AUROC are co-primary; the challenger must beat the
  champion on BOTH. Phase I: walk-forward temporal CV (4 folds, test 2021/2022/2023/2024) +
  paired Nadeau-Bengio corrected-resampled t-test (promote only if mean ΔPR-AUC>0 and p<0.05).
  Phase II/III: single temporal split (2022/2023 cutoffs) + Boyd et al. logit CI on PR-AUC
  (promote only if challenger CI lower bound > champion point estimate, or non-overlapping CIs).
- NEVER retrain without running `tests/test_leakage.py` first (must be green)
- NEVER trust PR-AUC > 0.90 without a leakage audit (Phase I val=0.931 was suspicious)
- Gold set is EVALUATION ONLY — never used for training or HPO decisions
- Optuna: max 50 trials for Phase I/II (n<2k). 100 trials only for Phase III.
- Do NOT tune n_estimators — use early_stopping_rounds=50 instead

### Calibration rule (small-n)
- Phase I and II: use Platt (sigmoid) or beta calibration via CalibratedClassifierCV(cv=5)
  Isotonic regression requires ≥1,000 calibration points to avoid overfitting
- Phase III: isotonic is acceptable (val n ≈ 500+), but cross-validated is still better
- NEVER use cv="prefit" — sklearn ≥ 1.9 removed it. Use cv=5 or a manual wrapper.

### Code integrity
- ALL feature-building code calls assert_no_leakage() before returning
- ALL pipeline stages must be reproducible via dvc repro
- ALL secrets go in .env — never hardcoded
- Always use sqlalchemy.engine.URL.create() for DB connections — f-strings break
  on passwords containing @, #, or other special characters

## AACT schema gotchas (verified against live schema)
- browse_conditions: column is `mesh_term` NOT `name`
- conditions: column is `name` (free-text condition name)
- number_of_groups: 100% null for interventional trials — removed from features.yaml
- calculated_values / countries: `number_of_facilities`, `num_countries`, `is_multinational`
  are now BLOCKLISTED as conduct-accrued soft leakage (see Confirmed findings). `actual_duration`
  and `were_results_reported` remain hard post-hoc blocks.

## Tech stack
- Python 3.11, uv; xgboost>=2.0, lightgbm>=4.0, catboost>=1.2
- scikit-learn>=1.9, optuna>=3.6, shap>=0.45, betacal (beta calibration)
- tabpfn>=2.0 (Phase I/II only — needs GPU or tolerant of 5× slower CPU)
- autogluon.tabular (optional dep, heavy — install separately for benchmarking only)
- pandas>=2.0, pyarrow>=14.0, psycopg2-binary, sqlalchemy>=2.0
- datasets, huggingface_hub; mlflow>=2.12 (aliases not stages)
- dvc>=3.0, fastapi, uvicorn, pydantic>=2.0, pydantic-settings
- evidently>=0.4, nannyml (monitoring — NannyML estimates perf without ground truth)
- streamlit, ruff, pytest>=8.0, pre-commit

## Key commands
```bash
uv run pytest tests/ -v --tb=short
uv run ruff check src/ tests/
dvc repro
dvc repro -s featurize_gold
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Published baselines (on TOP benchmark — different split/definition, not directly comparable)
- XGBoost Phase III PR-AUC: 0.697  ← we matched this (0.700 on gold)
- HINT (graph model): 0.811
- SPOT: 0.856  |  AnyPredict: 0.881
- These use a 2014 split and different success definitions — cite honestly

## What NOT to do
- Do not add molecular/SMILES features to tree models — they help graph/deep models, not GBDTs
- Do not use SMOTE — use class weights; SMOTE breaks calibrated probabilities
- Do not use MLflow model stages (deprecated 2.9+) — use aliases
- Do not put AutoGluon in the production image — it is 2GB+, benchmarking only
- Do not run TabPFN on all 427 features — drop or PCA the TF-IDF block first (500 feature limit)
- Do not use weak-label pos_rate to set scale_pos_weight — use gold rates (listed above)
- Do not put secrets in CLAUDE.md
