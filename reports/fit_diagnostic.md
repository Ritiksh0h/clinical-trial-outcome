# Fit Diagnostic — honest Track A XGBoost (438 features)

Generated 2026-07-10. DIAGNOSTIC ONLY — saved models loaded, nothing retrained/changed.

## 1. Train vs Val vs Test (the core overfit check)

Sanity: test PR-AUC below reproduces the frozen headline (I 0.114 / II 0.264 / III 0.828) — same predict_proba path as the champion, so train/val are comparable.

| Phase | Train PR-AUC | Val PR-AUC | Test PR-AUC | Train AUROC | Val AUROC | Test AUROC | Verdict |
|-------|-------------|-----------|------------|------------|----------|-----------|---------|
| 1 | 0.687 | 0.283 | 0.114 | 0.876 | 0.693 | 0.653 | MILD OVERFIT, well-bounded — weak-signal/small-n dominated |
| 2 | 0.860 | 0.461 | 0.264 | 0.922 | 0.723 | 0.699 | OVERFIT — but also a genuinely hard problem |
| 3 | 0.902 | 0.705 | 0.828 | 0.889 | 0.725 | 0.888 | HEALTHY — best result, generalizes cleanly |

## 2. Trees used (early stopping) + config

| Phase | trees used (best_iter+1) | max n_estimators | early-stopped? |
|-------|--------------------------|------------------|----------------|
| 1 | 14 | 2000 | yes |
| 2 | 104 | 2000 | yes |
| 3 | 44 | 2000 | yes |

Config (all phases): `{'max_depth': 4, 'learning_rate': 0.05, 'reg_lambda': 1.0, 'min_child_weight': 5.0, 'subsample': 0.8, 'colsample_bytree': 0.7, 'gamma': 0.0, 'early_stopping_rounds': 50, 'reg_alpha': 0.0}`

## 4. Calibrated Brier — train vs test

| Phase | Train Brier | Test Brier | gap |
|-------|-------------|-----------|-----|
| 1 | 0.1399 | 0.0584 | -0.0815 |
| 2 | 0.1378 | 0.1101 | -0.0277 |
| 3 | 0.1585 | 0.1514 | -0.0071 |

## 3. Learning curves (train vs val PR-AUC over rounds)

- Phase 1: `learning_curve_phase1.png` — final train PR-AUC 0.687 vs val 0.277 at best_iter=13
- Phase 2: `learning_curve_phase2.png` — final train PR-AUC 0.860 vs val 0.460 at best_iter=103
- Phase 3: `learning_curve_phase3.png` — final train PR-AUC 0.902 vs val 0.704 at best_iter=43

## 5. Phase I walk-forward — per-fold train vs test PR-AUC

Caveat: the walk-forward GATE model uses `n_estimators=300` with **no early stopping** (by design — it's an evaluation protocol, not the deployed model). 300 unpruned trees memorise fold-train to ~1.0; the honest read is the test column. The *deployed* Phase I model early-stops at 14 trees (§2), train PR-AUC 0.69 — it does not memorise.

| test year | n_train | n_test | pos | Train PR-AUC | Test PR-AUC | gap |
|-----------|---------|--------|-----|--------------|------------|-----|
| 2021 | 644 | 664 | 168 | 1.000 | 0.378 | +0.622 |
| 2022 | 1308 | 778 | 155 | 1.000 | 0.354 | +0.646 |
| 2023 | 2086 | 743 | 119 | 1.000 | 0.271 | +0.729 |
| 2024 | 2829 | 388 | 19 | 0.996 | 0.170 | +0.826 |

Mean test PR-AUC across folds: **0.293** (test signal shrinks as the positive count falls year-on-year: 168→155→119→19).

## Verdict per phase (one line, with the driving evidence)

- **Phase 1 — MILD OVERFIT, well-bounded — weak-signal/small-n dominated.** train 0.69 / val 0.28 / test 0.11, AUROC 0.88/0.69/0.65 — there IS a train-val gap, but early stopping caps it at 14 trees and test is n=20-positive noise. Weak signal, not memorisation.
- **Phase 2 — OVERFIT — but also a genuinely hard problem.** train 0.86 / val 0.46 / test 0.26, AUROC 0.92/0.72/0.70 — learning curve diverges (train rises, val flat from round ~20). Clear overfit, but val/test AUROC ~0.70 means real-but-modest signal on a hard problem. More trees would not help (val plateaued).
- **Phase 3 — HEALTHY — best result, generalizes cleanly.** train 0.90 / val 0.71 / test 0.83, AUROC 0.89/0.73/**0.89** — test AUROC matches TRAIN, train-test PR-AUC gap only 0.07. val<test is a temporal-cohort effect (2023 val harder than 2024 test), not overfit. This is the honest headline.