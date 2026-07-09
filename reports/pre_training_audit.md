# Pre-Training Methodological Audit

Generated 2026-07-08. **DIAGNOSTIC ONLY — nothing changed.** Not a bug hunt (that was
`pre_phase2_audit.md`); this reviews methodological soundness — structural flaws that would
produce plausible-but-WRONG results. Every verdict below is backed by evidence run against
the real data, not reasoning from assumptions.

## Severity summary

| Item | Topic | Verdict |
|------|-------|---------|
| **3** | Sponsor completion-rate uses **future outcomes** (outcome-leak) | **BLOCKER** |
| **5** | Track B train/test contamination (91% of gold-test in weak files) | **BLOCKER** (Track B) |
| **1** | Track B importance-weighting degenerates (ESS collapses to 5%) | **SERIOUS** (Track B) |
| **6a** | `gold is EVALUATION ONLY` rule contradicts Phase 2 gold training | **MINOR** (governance) |
| **6b** | `indication_history` will inherit Item 3's leak when built | **SERIOUS** (preemptive) |
| **2** | Gold vs weak label definition consistency | **MINOR** (96.8% consistent) |
| **4** | Walk-forward gate on real Phase I data | **SOUND** (minor caveat) |

**Verdict: fix these before training —**
1. **Item 3** before `featurize_gold` (it contaminates the shared feature matrix → inflates *both* tracks).
2. **Items 5 + 1** before building Track B (Track A/gold-only can proceed once Item 3 is fixed).
3. Item 6b when `indication_history` is written; Items 2, 4, 6a are notes, not blockers.

---

## ITEM 3 — Sponsor completion-rate outcome-leak → **BLOCKER** (highest priority)

**Code fact:** `sponsor_history.py` sets `_completed = (overall_status == "COMPLETED")` from the
**snapshot-time (2026) status** and orders priors only by `study_first_posted_date`. It enforces
*registration-before* but not *outcome-known-before*.

**Why that leaks:** a prior trial registered before the current trial may not have *completed*
until years later. Using its eventual outcome in the current trial's sponsor rate uses
information that did not exist at the current trial's registration.

**Evidence (387 gold trials with ≥1 prior):**
- **37.6%** of counted priors (mean; median 33.3%) had an outcome **not knowable** at the current
  trial's registration (`completion_date ≥ current.study_first_posted_date` or never-terminal).
- **19.4%** of trials have >50% of their priors leaked.
- The leaked `sponsor_prior_completion_rate` differs from the honest (outcome-known-before)
  value by **~8pp per trial** on average (leaked mean 0.730 vs honest 0.799).

**Scope:** affects the **two rate features** (`sponsor_prior_completion_rate`,
`sponsor_prior_same_phase_completion_rate`). The **count/flag features**
(`sponsor_prior_trial_count`, `sponsor_prior_phase_count`, `sponsor_is_established`,
`sponsor_is_large`) are **SOUND** — registration-order is valid for counting, no outcome
involved. So the "dense, 71% have 20+ priors" result stands; only the rates are wrong.

**Fix (for later):** in the rate aggregation, count a prior only if its
`completion_date < current.study_first_posted_date` (this single condition also implies
registration-before). `completion_date` is already in the studies snapshot, so no new pull is
needed. Expect more NaN rates (trials with no terminal-before priors → phase-median fill), and
the Amendment-3 gate should be extended to check the completion-date condition, not just
registration order. The exact PR-AUC inflation from this leak is **UNVERIFIABLE-YET** (needs
training), but the leak is definite and must be fixed regardless.

---

## ITEM 5 — Two-track evaluation contamination → **BLOCKER** (Track B)

Both tracks must evaluate on the same untouched gold **test** split (completion 2024+). Track A
(gold-only) trains on the gold train split (≤2022) and is clean by the temporal split.

**Track B is not clean as specified.** Track B trains on weak+gold. Of the **1,175** gold
test-set trials, **1,073 (91.3%)** also appear in the weak phase files. If Track B trains on all
weak trials, 91% of the evaluation set is in its training set (as weak-labeled rows) — direct
train/test contamination that would inflate Track B's apparent gold-test performance.

**Fix (for later):** Track B's weak training set must exclude every nct_id in the gold
val+test splits before training. This is a data-assembly requirement for the (not-yet-built)
Track B pipeline.

---

## ITEM 1 — Track B covariate-correction coherence → **SERIOUS** (Track B)

**Is importance weighting the right tool?** In principle yes — the confirmed finding is covariate
shift (P(x) differs, P(y|x) ~same), and IW by w(x)=P_gold(x)/P_weak(x) is the textbook
correction for covariate shift. But the empirics show it **degenerates** here.

**Evidence — gold-vs-weak domain classifier (Phase I population, covariate features):**
- Domain AUC = **0.904** (0.5 = identical, 1.0 = perfectly separable). Gold and weak are highly
  separable on covariates (largely era: gold is 2020–2024 completions).
- Resulting IW weights: **median 0.01**, p99 4.09, max 31.6.
- **Effective sample size after weighting = 5.0% of n.** The ~41k weak trials collapse to an
  effective ~2,000, dominated by a handful of high-weight trials.

**So both failure modes the plan worried about are real:** (a) a handful of gold-like weak trials
dominate, and (b) the effective signal collapses toward near-gold-only — making Track B nearly
pointless while adding variance. Separately, **IW and `scale_pos_weight` do interact**: because
the population shift is label-correlated (gold is failure-enriched), reweighting toward gold also
shifts the effective class balance toward failures, and then pinning `scale_pos_weight` to gold
rates applies a *second* balance correction on top — a double-touch that is not specified.

**Recommendation before building Track B:** don't ship naive full IW. Options: (i) clip/trim
weights (e.g. cap at p95) and report ESS; (ii) drop IW and instead just augment gold with the
~15.6k trustworthy weak **failures** (per `weak_failure_pool.md`) at moderate weight; (iii) if IW
is kept, fix the class prior via sample weights that already incorporate the label so the two
mechanisms don't stack. Specify and unit-test the weighting before building.

---

## ITEM 2 — Label definition consistency → **MINOR** (coherent enough)

**Evidence:**
- **Gold**: `y == (overall_status==COMPLETED)` agreement **96.8%**. COMPLETED→y=1 is 91.2%
  (342 COMPLETED-but-y=0 — the efficacy downgrade), TERMINATED→y=1 is 0.1% (only 7).
- **Weak (Phase 1)**: agreement **99.8%**. COMPLETED→y=1 = 99.9%, TERMINATED→y=1 = 1.4%.

Both encode completion, consistent with the confirmed finding. The 342 COMPLETED-but-0 gold rows
are **not silent noise** — they are the intentional ~9% efficacy adjustment (completed but failed
endpoint). The subtle mismatch for a combined track: **gold applies this efficacy downgrade, weak
does not**, so a COMPLETED trial carries y=1 under its weak label but may be y=0 under gold. This
is the known ~3pp label effect, localized to COMPLETED trials. It is small and does not break a
combined-training track, but Track B mixes two slightly different label definitions — report it
honestly rather than claiming a single clean target.

---

## ITEM 4 — Walk-forward gate on real Phase I data → **SOUND** (minor caveat)

**Per-fold positive counts (Phase I, membership-routed, by completion year):**

| test fold | n | positives | negatives |
|-----------|---|-----------|-----------|
| 2021 | 664 | **168** | 496 |
| 2022 | 778 | **155** | 623 |
| 2023 | 743 | **119** | 624 |
| 2024 | 388 | **19** | 369 |

Three of four folds are robust (119–168 positives); the walk-forward achieves its purpose —
pooling **461 positives** across folds vs the **19** in the single-2024 split. No fold is <10, so
the "computed on noise" trigger does not fire. The 2024 fold (19 pos) is thin, but a noisy fold
**widens** the Nadeau-Bengio paired-test variance → the gate becomes *more* conservative (harder
to promote), never falsely confident. `walk_forward_folds` raises on a 0-positive fold. **Minor
caveat:** it does not warn on small-but-nonzero folds — adding a per-fold `n_pos` floor + warning
(e.g. <30) would make the thinness explicit in logs.

---

## ITEM 6 — Other methodological risks

**6a (MINOR, governance):** CLAUDE.md hard rule says *"Gold set is EVALUATION ONLY — never used
for training or HPO decisions,"* but Phase 2 (both tracks, `featurize_gold`, `train_gold`) trains
on the gold **train** split. The mechanism is sound (temporal split holds the gold test set out),
but the rule wording is stale from Phase 1 and now contradicts the plan. Reconcile it to *"the
gold **test** split is evaluation-only; gold train/val may be used for Phase 2 training"* before
training, so nobody trips over the contradiction or accidentally trains on the test split.

**6b (SERIOUS, preemptive):** `indication_history` (Step 3, not yet built) computes
`ta_prior_completion_rate` — the **exact same outcome-leak as Item 3**. Build it with the
completion-date condition from the start, and give it the same Amendment-3-style gate. Flagging
now so it is not re-introduced.

Nothing else surfaced beyond these. Enrollment sparsity, Phase I small-n, and calibration choices
are already documented and handled.

---

## Final verdict: **fix these before training**

- **Before `featurize_gold` (blocks both tracks):** fix **Item 3** (sponsor rate outcome-leak).
- **Before building Track B:** fix **Item 5** (exclude gold val/test nct_ids from weak training)
  and **Item 1** (don't ship naive IW — clip/trim or switch to weak-failure augmentation).
- **Notes:** clarify the gold-eval-only rule (6a); build `indication_history` leak-safe (6b);
  report the mixed-label caveat (2); optional per-fold warning in the gate (4).

Track A (gold-only) is clear to train **once Item 3 is fixed**. Track B needs Items 3 + 5 + 1.
