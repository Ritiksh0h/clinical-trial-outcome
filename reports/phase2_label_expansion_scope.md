# Phase II label-expansion scope (Stage 1)

Generated 2026-07-10. DIAGNOSTIC ONLY — no training, no matrix changed, no labels added.

## 0. Label-definition linchpin (is the target mechanical?)

- Gold Phase II (membership `cto_gold ∩ cto_phase2`): **n=5060**, gold pos_rate **0.284**.
- Agreement `gold_y == (overall_status=='COMPLETED')`: **0.9704**.
- → Gold is ~mechanical completion; the ~3% gap is COMPLETED-but-labeled-failure (faint efficacy signal). Mechanical `overall_status` labeling is label-clean to ~97% — same order as the known weak-vs-gold agreement. **Label axis is safe; the only risk is population shift.**

## 1. Available expansion (counts)

- Routing set (matches CTO Phase II membership): `['PHASE1/PHASE2', 'PHASE2', 'PHASE2/PHASE3']`.
- Interventional Phase II-routed trials (any status): **89,072**.
- ... with terminal status (['COMPLETED', 'TERMINATED', 'WITHDRAWN']): **58,371**.
- ... NOT already in gold Phase II → **NEW labelable: 53,411**.
- Completion base rate — new pool **0.811** vs gold **0.284**  (**Δ=+0.528** — the Track-B population-prior signature).

## 2. Covariate shift (the critical risk check)

- Registration-year median: gold **2019** vs new pool **2012** (see `phase2_expansion_year.png`).
- **Domain classifier AUC (gold vs new, 438 features, 5-fold OOF): 0.888**.
  - Comparable to the Track-B kill (whole-pop 0.912 / failure-class 0.940). >0.75 → strong shift, easily separable → risky bulk add.

## 3. Overlap region size

- New-pool trials the domain classifier confuses with gold (P(new)<0.5): **19.8%** of sample → **~10,595** of 53,411 (extrapolated).
- Permissive upper estimate (within gold's central support, P(new)<0.60 = gold 90th pct): **24.2%** → **~12,912**.
- **Completion rate WITHIN the overlap: 0.802** (vs gold 0.284, vs full new pool 0.811). Still prior-shifted even within the covariate overlap — IW on P(x) alone won't fix it.
  (Sample-based extrapolation; see `phase2_expansion_domain.png`.)

## 4. Temporal / leakage hygiene

- New-pool trials missing `study_first_posted_date` (no registration-time features): **0**.
- Excluded as right-censored / non-terminal (RECRUITING/ACTIVE/UNKNOWN/etc.): **30,701** — never counted as completions. Breakdown: {'UNKNOWN': 11589, 'RECRUITING': 10236, 'ACTIVE_NOT_RECRUITING': 4463, 'NOT_YET_RECRUITING': 3741, 'SUSPENDED': 363, 'ENROLLING_BY_INVITATION': 309}.
- Features would be built by the audited 438-feature pipeline (registration-time only); `overall_status` is the LABEL, never a feature. No new leakage surface.

## Verdict

**MARGINAL — large overlap by covariates (~10,595) BUT it stays prior-shifted (overlap completion 0.80 ≫ gold 0.28). Importance-weighting corrects P(x), not this P(y) gap, so the addable-and-safe slice is smaller than the covariate overlap suggests. Low expected upside; pursue only as a scoped experiment, else document and go to writeup.**