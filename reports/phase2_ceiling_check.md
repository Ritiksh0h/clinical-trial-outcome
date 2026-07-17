# Phase II Stage-0 ceiling check

Generated 2026-07-10. DIAGNOSTIC ONLY — champion `models/gold_phase2.joblib` untouched.

Val set: n=1015, positives=250. Train max size=3490. spw=2.23.

## 1. Coarse hyperparameter sweep (40 Optuna/TPE trials)

- Champion val PR-AUC: **0.4611**  (Boyd 95% CI [0.4002, 0.5231])
- Best sweep val PR-AUC: **0.4713**  (config: `{'learning_rate': 0.0764136186923332, 'max_depth': 8, 'min_child_weight': 1.2707942999213684, 'reg_lambda': 0.28246357083904894, 'reg_alpha': 0.45227288910538066, 'gamma': 1.6266516538163218}`)
- Sweep val distribution: min 0.3931 / median 0.4442 / max 0.4713
- Configs beating the champion Boyd upper bound (0.5231): **0 / 40**
- Paired val-bootstrap of (best − champion): mean +0.0108, 95% CI [-0.0242, +0.0459]

PRIMARY (best > Boyd upper 0.5231): **FAIL** (0.4713 ≤ 0.5231)
SECONDARY (bootstrap Δ CI excludes 0): **FAIL** (lower bound -0.0242)
→ Sweep lifts plateau meaningfully: **NO**

## 2. Learning curve over training-set size

`phase2_size_curve.png` — val PR-AUC (mean±std over seeds 42/7/123), champion Boyd band shaded.

| train size | val PR-AUC (mean) | std |
|-----------|-------------------|-----|
| 500 | 0.3645 | 0.0095 |
| 1000 | 0.3937 | 0.0078 |
| 1500 | 0.4020 | 0.0160 |
| 2000 | 0.4255 | 0.0060 |
| 2500 | 0.4368 | 0.0108 |
| 3000 | 0.4430 | 0.0056 |
| 3490 | 0.4611 | 0.0000 |

- Tail slope (size 2500→3490): +0.0242  vs seed-noise threshold 2×0.0087=0.0173 (increments live on the fixed val set)
- Curve still rising at max size: **YES**

## Verdict

**LABEL-LIMITED, NOT A SIGNAL CEILING. Tuning is exhausted (0/40 configs clear the champion's noise band; best is +0.010, bootstrap CI includes 0) — so the fit_diagnostic 'more trees/depth won't help' call stands. BUT the size curve is still rising monotonically at the full 3490 rows (+0.018 over the last 490, above seed noise), so this is a DATA ceiling, not a feature/signal ceiling. Test PR-AUC 0.264 is the honest ceiling AT 3490 gold labels — more gold Phase II labels are the lever, not tuning and not (yet) new features.**