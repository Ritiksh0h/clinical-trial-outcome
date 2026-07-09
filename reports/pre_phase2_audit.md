# Pre-Phase-2 Audit

Generated 2026-07-04. **DIAGNOSTIC ONLY — nothing was fixed.** Evidence below is from
read-only checks against the live data snapshots and current code.

## Severity summary

| # | Item | Verdict | Severity |
|---|------|---------|----------|
| — | `params.yaml` class_weight_by_phase is stale (0.78/1.00/0.47, not gold 4.02/2.23/0.89) | **FAIL** | **BLOCKER** (before training) |
| 5 | Gold split positive counts — Phase I test = 20 positives under gold cutoffs | **FAIL** | **BLOCKER** (before promotion gate) |
| 3 | Sponsor/indication history temporal check on real data | **N/A** | **WARN** (not built yet) |
| — | `config/features.yaml` lists 5 features not built by `build.py` | **WARN** | WARN |
| 1 | Combo-trial cross-phase leak | **PASS** | — |
| 2 | TF-IDF temporal provenance | **PASS** | — |
| 4 | Snapshot integrity | **PASS** | — |
| 6 | Stale feature-file reads in Phase 2 code | **PASS** | — |
| 7 | Leakage blocklist completeness | **PASS** | — |

---

## Item 1 — Combo-trial cross-phase leak → **PASS**

Combo trials (in two CTO phase files): **7,706** in phase1∩phase2, **3,360** in phase2∩phase3,
**0** in phase1∩phase3.

The temporal split (`src/cto/features/split.py`) assigns a split purely from
`completion_date` against a single shared cutoff config in `params.yaml`. A trial has exactly
one `completion_date`, so it lands in the **same** split for every phase it belongs to.
Cross-phase "train in phase A / test in phase B" is therefore structurally impossible.
Verified empirically under both cutoff schemes:

| cutoffs | combo p1&p2 split dist | combo p2&p3 split dist |
|---------|------------------------|------------------------|
| weak 2021/2022 | train 5916 / val 542 / test 998 / dropped(NaT) 250 | train 2709 / val 225 / test 328 / dropped 98 |
| gold 2022/2023 | train 6458 / val 574 / test 424 / dropped 250 | train 2934 / val 203 / test 125 / dropped 98 |

Cross-phase train/test pairs for the same trial: **0**.

Caveats (not leaks today, but preconditions of the guarantee): (a) it holds only while all
phases share one cutoff config — do not introduce per-phase cutoffs; (b) the one genuinely
shared fitted artifact across phases is the TF-IDF vectorizer — see Item 2.

---

## Item 2 — TF-IDF temporal provenance → **PASS**

`models/tfidf_vectorizer.joblib`: vocab=400, max_features=400, ngram=(1,2), mtime 2026-07-01.

Fit logic: `build_features()` fits only when `split=="train" and not _TFIDF_PATH.exists()`,
and `featurize.py` iterates phase 1 first — so the vectorizer was fit **once, on the Phase 1
train split only**.

- Fit population: Phase 1 weak trials with `completion_date <= 2021-12-31`, **n=24,756**.
- Date range of fit population: **1987-05-31 → 2021-12-31**.
- Phase 1 train rows with `completion_date > 2022-01-01`: **0**.

No val/test-period (2022+) eligibility text entered the fit. The Phase 2 gold pipeline will
reuse this vectorizer (transform-only); since gold val/test are `completion_date >= 2023`,
their text was never in the fit set. **No temporal leakage.**

---

## Item 3 — Sponsor/indication history temporal check → **N/A (WARN)**

Cannot run: the artifacts do not exist yet (Phase 2 Step 2 not started).

MISSING: `src/cto/features/sponsor_history.py`, `data/interim/sponsor_history.parquet`,
`src/cto/features/indication_history.py`, `data/interim/indication_history.parquet`,
`tests/test_sponsor_history.py`.

**Action:** the toy-fixture test passing is not sufficient assurance. Once Step 2 builds
`sponsor_history.parquet`, the temporal-leakage assertion MUST be run against a real TEST-set
sample (verify no test trial's `sponsor_prior_trial_count` includes trials registered on/after
its own `study_first_posted_date`). This check is deferred, not passed.

---

## Item 4 — Snapshot integrity → **PASS**

| snapshot | rows | unique nct_id | coverage vs studies |
|----------|------|---------------|---------------------|
| studies | 591,828 | 591,828 | — |
| designs | 587,051 | 587,051 | 99.2% |
| eligibilities | 590,853 | 590,853 | 99.8% |
| sponsors | 944,561 | 591,828 | 100.0% (multi-row/trial) |
| conditions | 4,297,630 | 467,301 | 79.0% |
| interventions | 1,000,801 | 531,957 | 89.9% |
| calculated_values | 591,828 | 591,828 | 100.0% |

**Eligibility criteria coverage: 99.8%** of studies have a matching eligibilities row with
non-null criteria (590,761 / 591,828). No truncation. The lower `conditions` (79.0%) and
`interventions` (89.9%) coverage is expected — not all trials carry MeSH conditions or
interventions — and both feed only optional features. No snapshot appears partial.

---

## Item 5 — Gold split positive counts → **FAIL (BLOCKER for promotion gate)**

Membership-routed gold, temporal split. Counts under **PHASE2.md gold cutoffs (2022/2023)**:

| phase | split | pos (y=1) | neg (y=0) | total | flag |
|-------|-------|-----------|-----------|-------|------|
| I | train | 479 | 1607 | 2086 | |
| I | val | 119 | 624 | 743 | |
| I | **test** | **20** | 390 | 410 | **pos < 30** |
| II | train | 1120 | 2370 | 3490 | |
| II | val | 250 | 765 | 1015 | |
| II | test | 65 | 490 | 555 | |
| III | train | 1062 | 932 | 1994 | |
| III | val | 243 | 311 | 554 | |
| III | test | 96 | 179 | 275 | |

(0 gold trials dropped for NaT completion_date in any phase.)

**Phase I gold test has only 20 positives.** A ±0.01 PR-AUC promotion decision on 20 positives
is inside the noise floor — the Phase I gate would be unreliable.

Critically, this is **cutoff-dependent**. The current `params.yaml` still has the weak cutoffs
(2021/2022); PHASE2.md Step 4.1 prescribes moving gold to 2022/2023. Phase I test positives:

| cutoffs | P1 test | P2 test | P3 test |
|---------|---------|---------|---------|
| current params.yaml (2021/2022) | pos=139, neg=1014 | pos=315, neg=1255 | pos=339, neg=490 |
| PHASE2.md gold (2022/2023) | **pos=20**, neg=390 | pos=65, neg=490 | pos=96, neg=179 |

The 2022/2023 cutoffs buy a cleaner "recent-years" test at the cost of Phase I test power.
**Resolve before relying on the Phase I gold gate** — options include keeping 2021/2022 for
Phase I, using repeated CV for the Phase I gate instead of the single temporal test split
(PHASE2.md Step 5.1 already prescribes RepeatedStratifiedKFold for Phase I training), or
reporting Phase I with an explicit small-n confidence caveat.

---

## Item 6 — Stale feature-file reads in Phase 2 code → **PASS**

No Phase 2 code exists yet (`featurize_gold.py`, `train_gold.py`, `ensemble.py` all MISSING),
so nothing reads the schema-1.0.0 files incorrectly. All `features_phase{N}_{split}.parquet`
references are in legitimate Phase 1 paths: `featurize.py`, `train.py`, `train_baseline.py`,
`tests/test_features.py`, `dvc.yaml`.

**Forward-looking note:** when `featurize_gold.py`/`train_gold.py` are written they must read
`features_gold_phase{N}_{split}.parquet` (not `features_phase{N}`), and `feature_schema_version`
should bump to 2.0.0 (currently 1.0.0 in both `params.yaml`-adjacent `config/features.yaml`) —
correct for now since sponsor/indication features aren't added yet.

---

## Item 7 — Leakage blocklist completeness → **PASS**

`uv run pytest tests/test_leakage.py` → **42 passed**. Blocklist size 48. All required columns
blocked:

`overall_status` ✓, `last_update_posted_date` ✓, `last_update_submitted_date` ✓,
`last_update_posted_date_type` ✓, `disposition_first_posted_date` ✓,
`disposition_first_submitted_date` ✓.

`build_raw_joined` also emits a runtime warning confirming it drops `overall_status` and
`last_update_posted_date` before features are built.

---

## Additional findings (outside the 7 items, surfaced by the audit)

### A1 — `params.yaml` class weights are stale → **BLOCKER (before Phase 2 training)**

`model.class_weight_by_phase` is currently **1: 0.78, 2: 1.00, 3: 0.47**, derived from the
published ~56%/50%/68% success rates (comment in file). CLAUDE.md and PHASE2.md Step 5.2
mandate the **gold** base rates: **1: 4.02, 2: 2.23, 3: 0.89**. Training Phase 2 gold models
as-is would under-weight the positive (success) class ~5× for Phase I. The corrected block is
already written out in PHASE2.md Step 5.2 — it just hasn't been applied. Must fix before Step 5.

### A2 — `config/features.yaml` lists 5 features `build.py` never produces → **WARN**

Built matrix = 427 (27 structured + 400 TF-IDF). Listed in `features.yaml` but not built:
`has_survival_endpoint`, `healthy_volunteers`, `is_multinational`, `num_countries`,
`num_primary_outcomes`. `build.py` is the source of truth and has been consistent through
Phase 1, so this is documentation drift, not a runtime bug — but `featurize_gold` reuses
`build.py`, so the drift will persist into Phase 2. Reconcile the config or implement the
features when sponsor/indication work touches `build.py`.

### A3 — NaT completion dates silently dropped from splits → **WARN (low impact)**

`make_temporal_splits` drops trials with null `completion_date` from all splits (NaT fails both
`<= cutoff` and `> cutoff`). Impact on the gold pipeline is nil (0 gold trials have NaT
completion), but 250 phase1∩phase2 and 98 phase2∩phase3 weak combo trials are dropped. Not a
leak; noting for completeness since it silently reduces the weak training pool.
