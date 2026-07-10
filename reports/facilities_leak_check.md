# `number_of_facilities` Leak Check

Generated 2026-07-09. **DIAGNOSTIC ONLY — nothing changed, no models retrained/saved.**
Question: is `number_of_facilities` the registration-time PLANNED site count, or the
actual/accrued count that reflects conduct (and thus partially encodes the outcome)?
It is the #1 SHAP feature for Phase II (0.36) and Phase III (0.51), tripping the
mean|SHAP| > 0.4 leakage red-flag — so before trusting Phase III 0.819 we check it.

## 1. What it is / when AACT sets it → conduct-time, not registration

`calculated_values.number_of_facilities` is an **exact count of the `facilities` table**
(verified: sampled trials show number_of_facilities == COUNT(facilities), e.g. 1==1, 24==24).
The `facilities` table is the list of **actual participating sites** and carries a per-site
`status` column (Recruiting / Completed / Withdrawn / Terminated / …). AACT regenerates
`calculated_values` on every refresh from the current record, so the count reflects the
sites that have actually been opened/recorded — it **accrues and changes as the trial runs**.
There is no registration-time "planned sites" field and no per-facility date in the standard
schema, so a registration-time count cannot be reconstructed from the mirror. This is the
same class of field as ACTUAL enrollment (already treated as soft leakage).

## 2. Outcome correlation (gold TRAIN trials, per phase)

| Phase | group | n | median | mean | p25 | p75 |
|-------|-------|---|--------|------|-----|-----|
| I | COMPLETED | 535 | 1.0 | 7.0 | 1 | 6 |
| I | TERM/WD | 1531 | 1.0 | 4.1 | 1 | 4 |
| II | COMPLETED | 1220 | **7.0** | 25.0 | 1 | 30 |
| II | TERM/WD | 2239 | **1.0** | 9.3 | 1 | 7 |
| III | COMPLETED | 1139 | **43.0** | 77.9 | 9 | 99 |
| III | TERM/WD | 832 | **2.0** | 32.6 | 1 | 26 |

Huge separation, growing with phase (Phase III: 43 vs 2 median). Decisive tell:
**WITHDRAWN trials have median 1 facility in every phase** — a withdrawn trial never opened
its planned sites, so the count is truncated *by* the withdrawal. The feature value is a
consequence of the outcome, not a registration-time input.

## 3. Temporal knowability → NOT registration-knowable (soft leak)

The count is the number of sites that actually appeared in the record over the trial's life
(per-site status confirms conduct tracking). For a terminated/withdrawn trial it reflects
"sites opened before it stopped," which is only known at/after termination. There is no
registration-time snapshot in the mirror. So the feature is **conduct-time contaminated**:
it partially encodes completion vs termination. Not a hard label-derived field, but soft
leakage by the same mechanism as ACTUAL enrollment.

## 4. Impact test — retrain without the feature (gold test)

| Phase | PR-AUC with | PR-AUC without | Δ | AUROC with | AUROC without |
|-------|-------------|----------------|-----|-----------|---------------|
| I | 0.127 | 0.094 | −0.033 | 0.660 | 0.651 |
| II | 0.307 | 0.283 | −0.024 | 0.734 | 0.718 |
| **III** | **0.819** | **0.779** | **−0.040** | 0.891 | 0.865 |

Despite being #1 by SHAP, the feature is **not load-bearing** — Phase III loses only 0.04
PR-AUC and stays at **0.779** (still above the 0.700 prior and far above the 0.35 no-skill).
The model's other features (sponsor history, TA history, text, design) carry most of the
signal; facilities was largely redundant.

## Verdict: **SOFT-LEAK** (conduct-contaminated, not registration-knowable)

All three source/behavior checks agree: `number_of_facilities` is the accrued count of
actual participating sites, truncated by early termination/withdrawal, with no
registration-time version available. It partially encodes the outcome.

## Recommendation: **DROP it** (and re-examine its siblings)

- **Drop `number_of_facilities`** from the feature set and blocklist it as soft leakage —
  the same treatment already applied to ACTUAL enrollment. The cost is small and known:
  Phase III 0.819 → **0.779** (honest), II 0.307 → 0.283, I 0.127 → 0.094. We trade 0.04 of
  partly-leaked PR-AUC for a defensible headline.
- **Related concern — `num_countries` / `is_multinational`:** these are derived from the
  `countries` table, which accrues the same way (countries appear as sites open). Same
  soft-leak mechanism. SHAP impact is minor (num_countries was #8 in Phase III, ~0.055), but
  for consistency they should be re-checked and most likely dropped/blocklisted too.
- This revises the CLAUDE.md note that permitted `number_of_facilities` from
  calculated_values — it should move to the soft-leakage list alongside actual_duration /
  were_results_reported. (Not changed here — diagnostic only.)

Net: the Phase III result survives the removal (0.779, still the honest headline), so
dropping the leaky feature strengthens the project's credibility at minimal cost.
