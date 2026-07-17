# Project State

_Last updated: 2026-07-10_

## Status: Modeling COMPLETE

XGBoost champion, honest leak-free baseline, bake-off confirms.

- **Track A (gold-only) is the model.** Track B (weak-augmented) dropped — two independent
  analyses agreed weak failures can't help (see `reports/failure_class_shift.md`,
  `reports/pre_training_audit.md`).
- **Champion: XGBoost, per phase**, on the clean 438-feature set (schema 2.1.0). Trained on
  gold, calibrated (Platt on val), evaluated on the frozen gold test set
  (`data/processed/gold_test_nct_ids.json` — authoritative, never recompute the split).
- **Bake-off (XGBoost / LightGBM / CatBoost / CatBoost-native)**: all statistically
  indistinguishable; no challenger passes the promotion gate on any phase. XGBoost retained
  for simplicity/interpretability. CatBoost-native categoricals gave no lift over the
  engineered sponsor/TA features.
- **Headline (model-agnostic, leak-free):** Phase III PR-AUC ~0.83, Phase II ~0.26,
  Phase I ~0.31 (walk-forward). Phase I single-split (n=20 positives) is unreliable — use
  the walk-forward number.
- **Leaks found & removed** during hardening: mirror empty-overwrite guard, dead
  `enrollment_log` (magic-string), dead `accepts_healthy_volunteers` (boolean-vs-string),
  sponsor/indication rate outcome-leak (two-window fix), and conduct-accrued
  `number_of_facilities` / `num_countries` / `is_multinational` (blocklisted).

## Next

- Phase 2 writeup (results report: `reports/phase2_results.md`), and/or
- Phase 3 (serving: FastAPI + the saved `models/gold_phase{N}.joblib`).

## Deferred cleanup

- **pre-commit ruff version mismatch with CI ruff** — `.pre-commit-config.yaml` pins
  `astral-sh/ruff` rev `v0.4.4`, but the uv/CI ruff is `0.15.20`. Update the pre-commit `rev`
  to match `uv run ruff` so local hooks and CI agree. Later — not blocking.
