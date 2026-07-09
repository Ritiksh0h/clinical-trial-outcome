# Failure-Class Covariate-Shift Diagnostic

Generated 2026-07-09. **DIAGNOSTIC ONLY — no training path, no pipeline changes.**
Decides whether the extra weak-only failures are usable as targeted minority-class
augmentation, or whether we ship gold-only and confirm a negative result.

## Method

Domain classifiers (weak-only = 1 vs gold = 0), XGBoost, stratified 5-fold OOF, AUC with a
300-sample bootstrap 95% CI. Class membership is by `overall_status` (consistent both sides):
**failure = TERMINATED or WITHDRAWN**, **success = COMPLETED**. "Weak-only" = trials in the
CTO phase files but NOT in the gold set (disjoint populations).

**Feature set (stated explicitly, per the rigor requirement):** only raw registration-time
features that mean the same thing in both populations —
`phase_clean, sponsor_class, allocation, intervention_model, masking_ordinal, primary_purpose,
intervention_type_primary, gender, num_countries, is_multinational, number_of_arms,
registration_year` (12 structured) **+ 400 TF-IDF eligibility-text dims**.
**Excluded** (would let the classifier detect construction artifacts, not genuine shift): all
engineered history aggregates (`sponsor_prior_completion_rate`, indication rates, …) and
`enrollment_log`. Structured features built via `build_features` (source of truth), subset taken.

## Experiment 1 — per-class domain separability

| Domain classifier | AUC | 95% CI | n_weak | n_gold |
|-------------------|-----|--------|--------|--------|
| Whole-population (same feature set) | 0.912 | [0.909, 0.914] | 37,272 | 9,710 |
| **FAILURE class** (weak-fail vs gold-fail) | **0.940** | [0.937, 0.943] | 13,768 | 6,119 |
| SUCCESS class (weak-succ vs gold-succ) | 0.904 | [0.900, 0.909] | 23,504 | 3,436 |

(Whole-population AUC 0.912 here is the recomputed baseline with this feature set; consistent
with the earlier phase-1/6-feature figure of 0.904.)

**The failure class is MORE separable than the whole population, not less** (0.940 > 0.912).
Weak-only failures and gold failures are highly distinguishable on registration-time features —
the opposite of the hoped-for "weak failures are gold-like."

## Experiment 2 — how many weak failures are usable via instance selection

Unique weak-only failures across phases: **13,768** (the audit's 15,599 was the per-phase
*sum*, which double-counts combo trials that appear in two phase files; 13,768 is the honest
unique count).

Gold-like probability P(gold | x) assigned by the failure-class classifier to weak-only failures:

| statistic | value |
|-----------|-------|
| p10 / median / p90 | 0.00 / **0.02** / 0.46 |
| P(gold-like) ≥ 0.3 | 1,988 (14.4%) |
| P(gold-like) ≥ 0.4 | 1,575 (11.4%) |
| P(gold-like) ≥ 0.5 | 1,256 (9.1%) |

The median weak failure gets P(gold-like) = 0.02 — the classifier is highly confident it is
NOT gold-like. Only ~9–14% (1,256–1,988) fall in anything resembling gold support, and at an
AUC of 0.94 even those assignments are unreliable. For context, gold already contains **6,119**
failures, so the instance-selectable ~1,300 add little and carry real selection risk.

## Experiment 3 — per-phase failure-class separability

| Phase | Failure-class AUC | 95% CI | weak_fail | gold_fail |
|-------|-------------------|--------|-----------|-----------|
| I   | 0.925 | [0.918, 0.931] | 4,553 | 2,478 |
| II  | 0.938 | [0.933, 0.943] | 7,795 | 3,397 |
| III | 0.948 | [0.940, 0.956] | 3,251 | 1,272 |

All three phases are ≥ 0.925 — the shift is uniform across phases; none is gold-like.

## Decision (per phase)

Decision rule: AUC ≤ 0.75 → augment; AUC ≥ 0.85 → ship gold-only; 0.75–0.85 → instance-select.

| Phase | Failure-class AUC | Branch | Recommendation |
|-------|-------------------|--------|----------------|
| I   | 0.925 | ≥ 0.85 | **Ship gold-only** |
| II  | 0.938 | ≥ 0.85 | **Ship gold-only** |
| III | 0.948 | ≥ 0.85 | **Ship gold-only** |

**All phases land in the ship-gold-only branch. Confirmed negative result: the weak-only
failures are as shifted as the whole population (more so), so adding them — even the
minority-class failures — would reintroduce the covariate shift the gold-only design avoids.**

This converges with the earlier `pre_training_audit.md` Item 1 finding (importance weighting
toward gold collapsed effective sample size to 5%). Two independent analyses now agree: **do not
build the weak-augmented Track B. Train gold-only (Track A).** The 13,768 weak failures are not
usable augmentation; ship gold-only and report this as an honest negative result.
