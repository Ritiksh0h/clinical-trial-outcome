# CTO-Predict — Ground-Truth Project Audit

Generated 2026-07-17. **DIAGNOSTIC ONLY — nothing changed, no models retrained.**
Purpose: verify what the repo *actually contains* against what CLAUDE.md / STATE.md / PHASE docs
*claim*, so nothing false lands on a CV. Verdicts are based on the code, data, saved models,
and MLflow store — not on the docs.

**Verification methods used:** re-ran the full test suite; re-loaded and re-evaluated all saved
model objects on the frozen gold test set; queried the MLflow SQLite store directly (run metrics,
params, timestamps, registry aliases); read the feature-build, leakage, gate, and training code;
inspected configs, CI, DVC stages, git history.

**One-line verdict:** The headline numbers are **real and reproducible** — every performance claim
I could check reproduces exactly from the saved models. The science (leakage discipline, definition
audit, covariate-shift diagnosis, honest negative results) is genuinely strong. The gaps are all on
the **engineering/packaging** side (empty README, no serving layer, stale MLflow registry, several
PHASE2 build-plan items pivoted or skipped) and in **loose wording**, not in fabricated results.

---

## 1. What actually exists

### Source modules (all present, import cleanly)
`src/cto/` — `common/{db,settings}`, `data/{aact_client,cto_labels,mirror}`,
`features/{build,contamination_guard,indication_history,leakage,split,sponsor_history,text}`,
`models/{calibrate,compare_models,explain,gate,train,train_gold}`,
`pipelines/{build_sponsor_history,build_indication_history,featurize,featurize_gold,ingest,train_baseline}`.
`serving/` contains **only `__init__.py`** (no FastAPI app).

### Trained models (`models/`) — all 13 load; 3 champions re-evaluated
| File | Loads | Format | Re-evaluated on frozen test |
|------|-------|--------|------------------------------|
| `gold_phase1.joblib` | ✅ | dict(model, calibrator, features=438) | PR-AUC **0.114**, AUROC 0.653 (n=410, pos=20) |
| `gold_phase2.joblib` | ✅ | dict(model, calibrator, features=438) | PR-AUC **0.264**, AUROC 0.699 (n=555, pos=65) |
| `gold_phase3.joblib` | ✅ | dict(model, calibrator, features=438) | PR-AUC **0.828**, AUROC 0.888 (n=275, pos=96) |
| `gold_phase{1,2,3}_lightgbm.joblib` | ✅ | dict(model, platt, kind) | metrics in MLflow (see §2) |
| `gold_phase{1,2,3}_catboost.joblib` | ✅ | dict | MLflow |
| `gold_phase{1,2,3}_catboost_native.joblib` | ✅ | dict | MLflow |
| `tfidf_vectorizer.joblib` | ✅ | fitted TfidfVectorizer (400 dims) | — |

The three champions' recomputed PR-AUC/AUROC **match the MLflow-logged values to 3 decimals** —
the saved artifacts are the real models behind the headline.

### Feature schema — **438 confirmed** (matches the claim exactly)
All 18 processed matrices (gold + weak × 3 phases × train/val/test) have **438 columns**.
Composition verified from `build.py`: 8 numeric + 9 categorical + 10 flags + 6 sponsor-history +
5 indication-history = **38 structured**, plus **400 TF-IDF** = 438. (CLAUDE.md's "27 structured +
11 history + 400" is the same 438; "27" just excludes the 11 history columns.) Schema tag `2.1.0`.

### Tests — **105 passed**, 0 failed, 8 warnings, 5.05s (`uv run pytest`)
Breakdown: contamination_guard 9, data 3, features 5, gate 10, gold_pipeline 5, indication_history 6,
leakage 45, models 4, split 7, sponsor_history 11. `ruff check` on src/tests/scripts: **all pass.**
(CLAUDE.md says "43→60→N"; N is now 105. The initial git commit message says "90 tests" — stale.)

### Frozen gold test set — **counts match exactly**
`data/processed/gold_test_nct_ids.json`: phase1=**410**, phase2=**555**, phase3=**275**, all=**1073**.
Phase sum is 1240; `all` is the dedup union (1073), so **167 combo trials appear in two phases** —
consistent with the documented CTO-membership routing. Test-matrix row counts equal the frozen
counts (test enforces this).

---

## 2. Headline-claim verification

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Target is **completion** (COMPLETED vs TERMINATED/WITHDRAWN), not efficacy | **TRUE** | `definition_audit.md`: COMPLETED→91.2% success, TERMINATED→0.1%. Both weak & gold encode completion. Report itself warns *not* to equate with FDA ~14% approval. |
| 2 | Weak-vs-gold gap is **covariate shift, not label noise** (labels agree 96–98%) | **TRUE** | `label_vs_population.md`: agreement 97.6 / 97.3 / 96.1%; population effect 45.5pp vs label effect 3.0pp (94% of gap). weak=0/gold=1 cell = 0 in all phases. |
| 3 | **3 silent bugs fixed** (mirror empty-overwrite, dead `healthy_volunteers`, dead `enrollment_log`) | **TRUE (in code)** | `enrollment_log` gates on `"ESTIMATED"` (build.py:179, was ANTICIPATED); `_map_healthy_volunteers` handles the boolean snapshot + `accepts_healthy_volunteers` fixed (build.py:149,435); mirror.py has empty-write guards (57–59, 81–85). See caveat in §3. |
| 4 | **2 soft leaks removed** (facilities/countries) — blocklisted + absent from build | **TRUE** | `leakage_blocklist.yaml` `aact_conduct_accrued_soft_leak` = {number_of_facilities, num_countries, is_multinational}; loader pulls it into `LEAKAGE_BLOCKLIST` (leakage.py:22–25); no facilities/countries join in build.py (317–320); features.yaml drops them; feature count 441→438 reflects removal. `facilities_leak_check.md` documents the impact test. |
| 5 | Honest results: **Phase III ~0.83, Phase II ~0.26, Phase I ~0.31 walk-forward** | **TRUE** | Re-eval of saved models = MLflow (438/2.1.0 runs): III **0.828** [CI 0.739–0.891], II **0.264** [0.171–0.384], I single-split **0.114** but **walk-forward pooled 0.308**. Numbers reproduce exactly. *Wording caveat in §3.* |
| 6 | **4-algorithm bake-off**, all indistinguishable, XGBoost retained; models saved + recorded | **TRUE** | 15 `cto_gold` runs + 12 challenger joblibs. III: xgb 0.828 / lgbm 0.828 / cat 0.812 / cat-native 0.807. II: xgb 0.264 / cat 0.246 / lgbm 0.218 / cat-native 0.173. Gate logic (gate.py) refuses all (LightGBM Phase III ties on point but CI-low 0.739 < champ point). *No standalone written report — see §3/§4.* |
| 7 | **3 validation gates** (leak-check, fit diagnostic, ceiling) — reports exist | **TRUE** | `facilities_leak_check.md`, `fit_diagnostic.md` (train/val/test overfit table, reproduces headline), `phase2_ceiling_check.md` (label-limited, 0/40 tuning configs beat champion). All substantive. |
| 8 | **Track B killed**; Phase II label-expansion scoped + declined — reports exist | **TRUE** | `failure_class_shift.md` (failure-class domain AUC 0.940 > 0.912; ship gold-only) + `pre_training_audit.md`; `phase2_label_expansion_scope.md` (domain AUC 0.888, verdict MARGINAL/decline). |
| — | Phase 1 baseline gold-eval **III=0.700** beats published XGBoost 0.697 | **TRUE** | `cto_baseline` run: gold_prauc phase3 = 0.6999. Weak-test 0.908/0.762/0.808; gold-eval 0.358/0.514/0.700 — matches CLAUDE.md and the PHASE2 comparison table. |

**On the duplicate MLflow runs (a thing a reviewer *will* notice):** each phase has two `*_xgb` runs.
The `07-10 03:35` set is **stale** (n_features=**441**, schema **2.0.0** — before the soft-leak drop:
III 0.819 / II 0.307 / I 0.127). The `07-10 04:30` set is **current** (n_features=**438**, schema
**2.1.0**: III 0.828 / II 0.264 / I 0.114). The saved joblibs are the 438 ones. This fully explains
the Phase II 0.307→0.264 spread — it is the honest cost of dropping the soft-leak features, not noise.

---

## 3. Inconsistencies, stale numbers, and loose wording

These are **not fabricated results**, but a sharp reviewer would catch them — fix the wording before the CV.

1. **`facilities_leak_check.md` internal tension.** It reports dropping facilities *alone* costs
   0.04 (Phase III 0.819→**0.779**), yet the final 438 model (all three soft-leaks dropped) scores
   **0.828**. Those come from **different retrains** (the leak-check quick-retrain vs the production
   `featurize_gold` matrix), so "0.04 cost" and "0.83 unchanged" (CLAUDE.md) sit awkwardly together.
   The defensible, reproducible headline is the **0.828 from the saved model** — lead with that; treat
   the 0.779 as "even with facilities removed in isolation it stays ≥0.78."

2. **"Phase II ~0.26 walk-forward" is loosely worded.** Phase II 0.264 is a **single-split** number;
   only **Phase I** uses walk-forward (per the gate design). The `~0.31` walk-forward label applies to
   Phase I only. Say "Phase III 0.83 and Phase II 0.26 on the frozen 2024 test split; Phase I 0.31
   walk-forward (single-split unreliable at 20 positives)."

3. **Two Phase-I walk-forward aggregations exist:** pooled = **0.308** (MLflow), mean-of-folds =
   **0.293** (`fit_diagnostic.md`). Both round to ~0.29–0.31; the headline uses pooled. Harmless, but
   don't quote 0.31 and 0.29 as if they're the same computation.

4. **`scale_pos_weight` derivation vs actual training population.** params.yaml uses 4.02/2.23/0.89,
   derived from the **_PHASE_MAP** gold rates (0.199/0.309/0.529, `definition_audit.md`). But the trained
   matrices use **CTO-membership** routing, whose train pos-rates are 0.230/0.321/0.533 → would imply
   ~3.35/2.12/0.88. Phase I is ~20% off. This is an **intentional** "use fixed gold base rates, don't
   fit class weight to the split" choice, but the two gold-n conventions (1933/3646/2418 by _PHASE_MAP
   vs 3239/5060/2823 by membership) coexist in the docs and can confuse. Both are documented in CLAUDE.md.

5. **STATE.md counts "5 leaks/bugs," the CV framing says "3 bugs + 2 leaks."** Same events, different
   grouping (mirror guard, dead enrollment_log, dead healthy_volunteers, two-window rate leak, and
   facilities/countries). All real and in-code; just be consistent about the count.

6. **Mirror guard is softer than the CLAUDE.md hard rule.** CLAUDE.md says "mirror.py must *assert*
   len(df) > 0 before any snapshot write." The code **logs-and-skips** on empty (mirror.py:57,81) rather
   than asserting, and the **full-pull studies path (since=None) with 0 rows is unguarded** (would write
   empty). The documented incremental-overwrite scenario *is* prevented; the phrasing "assert" is
   slightly stronger than reality. Low risk.

---

## 4. What's genuinely missing (for a portfolio / CV project)

| Gap | Status | Impact |
|-----|--------|--------|
| **README.md is 0 bytes** | Empty | A cloner sees *nothing*. Biggest single portfolio gap — all the good narrative lives in `reports/` and CLAUDE.md, not a front-door README. |
| **No serving layer** | `serving/` = `__init__.py` only; `dashboard/` = `.gitkeep` only | fastapi/uvicorn/streamlit are in deps but unused. STATE.md lists this as "Next (Phase 3)". No inference API / demo. |
| **MLflow registry is stale** | `registry.py` **does not exist**; registry has only a `challenger` alias → **version 1 = the Phase 1 weak baselines**; **no `champion` alias was ever set** | PHASE2 exit criterion "@champion alias updated per phase" is **unmet**. The real champions are the `models/*.joblib` files loaded directly; the registry does not reflect them. Misleading if a reviewer inspects MLflow. |
| **Gold training uses a fixed config, not Optuna** | `train_gold.py` hard-codes `_XGB`; no optuna import; `params.n_optuna_trials=100` is unused for gold | PHASE2 "trained with Optuna, logged in MLflow" is **unmet**. *Defensible* — `phase2_ceiling_check.md` ran a 40-trial sweep and showed tuning doesn't beat the fixed champion — but the docs imply HPO that didn't happen. |
| **No stacking ensemble** | `ensemble.py` (PHASE2 Step 6) **does not exist** | Replaced by the `compare_models.py` bake-off (train-each-separately + gate). Defensible pivot (all algos tie → a stack wouldn't help), but the "ensemble beats best single model" exit criterion is **unmet by design**. Don't claim an "ensemble model." |
| **AutoGluon + TabPFN never run** | No `cto_autogluon` / `cto_tabpfn` experiments; bench extras are **commented out** in pyproject | PHASE2 benchmark criteria unmet. No false claim (nothing asserts they ran), just incomplete vs the build plan. |
| **No `reports/phase2_results.md`** | Absent | PHASE2's final results report was never written. The content exists scattered across STATE.md + the diagnostic reports. |
| **No written bake-off report** | Results only in MLflow + saved joblibs + `compare_models.py` stdout (not captured) | The bake-off *conclusion* is asserted in CLAUDE.md/STATE.md but not persisted as a readable artifact. The gate promote/retain verdicts print to stdout only. |
| **Substantial work is uncommitted** | `git status`: `compare_models.py`, `fit_diagnostic.{py,md}`, `phase2_ceiling_check.{py,md}`, `phase2_label_expansion_scope.{py,md}`, `STATE.md`, modified CLAUDE.md + tests are **untracked/modified** | A reviewer cloning **HEAD** (last commit `ef1321f`) gets the honest-438 matrices but **not** the bake-off code or the three validation reports that justify the headline. Commit them. |
| **`catboost_info/` at repo root** | Untracked training-log dir | Stray artifact; should be gitignored. |
| **pre-commit ruff pin mismatch** | `.pre-commit-config.yaml` rev `v0.4.4` vs CI/uv ruff `0.15.20` | Documented in STATE.md "Deferred cleanup." Local hooks and CI can disagree. |
| **Repo not reproducible from a bare clone** | Models + parquets are gitignored; `dvc repro` needs AACT Postgres creds + HF download | Standard for ML repos, but means a reviewer can't run it end-to-end without credentials. Worth a README note. |

**Not attempted / CAN'T-CONFIRM here:** full `dvc repro` (needs live AACT credentials); the Phase I
Nadeau-Bengio gate promote/retain verdicts (logic is present and sound in `gate.py`/`compare_models.py`
but the decisions are printed, not stored).

---

## 5. CV-safe summary — what you can state without hedging

**Verified true and reproducible (safe to claim):**
- Built a leakage-audited clinical-trial **completion**-prediction pipeline (per-phase binary
  classification) on ClinicalTrials.gov/AACT (591k studies) with an expert-labeled gold evaluation set;
  **438 registration-time-only features** (structured + sponsor/therapeutic-area history + 400-dim TF-IDF).
- **Phase III PR-AUC 0.83** (AUROC 0.89) on a frozen, temporally-held-out 2024 gold test set —
  **matches the published XGBoost benchmark (0.697→our 0.70 baseline, 0.83 with the Phase 2 feature set)**;
  reproducible from the saved model. Phase II 0.26 (frozen test), Phase I 0.31 (walk-forward).
- Diagnosed the weak-vs-gold performance gap as **94% covariate shift, not label noise** (labels agree
  96–98% on overlap trials) — with a decomposition report.
- Caught and fixed **3 silent data bugs** and removed **3 conduct-accrued soft-leak features**, trading
  0.04 PR-AUC for a defensible, leakage-free headline (documented impact test).
- **Four-algorithm bake-off** (XGBoost/LightGBM/CatBoost/CatBoost-native) with an interval-based
  promotion gate (Boyd logit-CI + Nadeau-Bengio walk-forward) — all statistically indistinguishable.
- Two **honest negative results**: killed weak-augmented training (Track B) and declined Phase II
  label expansion, each backed by a covariate-shift diagnostic. **105 tests, CI green.**

**Do NOT claim (unmet or overstated):**
- ❌ "Deployed a FastAPI serving layer / dashboard" — not built.
- ❌ "Optuna-tuned gold models" — gold uses a fixed config (tuning shown not to help, but not run).
- ❌ "Stacking ensemble that beats the best single model" — no ensemble was built.
- ❌ "AutoGluon/TabPFN benchmarks" — never run.
- ❌ "MLflow model registry with champion promotion" — registry only holds the old baselines as
  `challenger`; no champion alias. (The *gate logic* exists and is defensible to mention; the
  registry *wiring* does not.)

**Before showing the repo:** write a real README (the story is genuinely good — surface it),
commit the uncommitted bake-off code + validation reports, and either set the MLflow `@champion`
aliases or stop referencing the registry as the source of truth.
