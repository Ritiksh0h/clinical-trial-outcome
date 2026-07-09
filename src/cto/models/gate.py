"""
Promotion gate: walk-forward temporal CV (Phase I) + interval-based promotion (all phases).

Why this exists: the single 2024 gold test split has only 20 Phase I positives — a bare
0.01 PR-AUC threshold on that is inside the noise floor (±0.22 CI). This module:

  - Phase I: walk-forward temporal CV across 2021-2024 (expanding train window) to pool
    positives while keeping train-past / test-future validity, judged with a paired
    Nadeau-Bengio corrected-resampled t-test.
  - Phase II/III: single temporal test split (they have enough positives) judged with
    logit confidence intervals for AUPRC.
  - All phases: PR-AUC AND AUROC are co-primary; never promote on a bare point difference.

PR-AUC confidence intervals use the Boyd, Eng & Page (2013) logit interval with a binomial
standard error on the positive count. Bootstrap and CV-based intervals are statistically
invalid for AUPRC (Boyd et al. 2013) — do not substitute them.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def evaluate(y_true, y_prob) -> dict[str, float]:
    """PR-AUC (average precision) and AUROC for one split."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "prauc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
    }


def walk_forward_folds(
    dates: pd.Series,
    y,
    test_years: tuple[int, ...] = (2021, 2022, 2023, 2024),
) -> list[dict]:
    """
    Expanding-window walk-forward folds by completion year.

    For each year Y in `test_years`: train = completion_date <= Dec 31 (Y-1),
    test = completion_date within calendar year Y. Enforces a strict temporal gap
    (train max date < test min date) and requires >0 positives in every test fold.

    Returns a list of {test_year, train_idx, test_idx, n_pos_test}.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    y = np.asarray(y)
    folds = []
    for yr in test_years:
        train_end = pd.Timestamp(f"{yr - 1}-12-31")
        test_start = pd.Timestamp(f"{yr}-01-01")
        test_end = pd.Timestamp(f"{yr}-12-31")

        train_idx = np.where(dates <= train_end)[0]
        test_idx = np.where((dates >= test_start) & (dates <= test_end))[0]

        if len(train_idx) == 0:
            raise ValueError(f"walk-forward fold {yr}: empty train window (<= {train_end.date()})")
        if len(test_idx) == 0:
            raise ValueError(f"walk-forward fold {yr}: empty test window ({yr})")

        # strict temporal gap: no train trial on/after the test window start
        train_max = dates.iloc[train_idx].max()
        test_min = dates.iloc[test_idx].min()
        if not (train_max < test_min):
            raise ValueError(
                f"walk-forward fold {yr}: temporal gap violated — "
                f"train max {train_max.date()} not < test min {test_min.date()}"
            )

        n_pos = int(y[test_idx].sum())
        if n_pos == 0:
            raise ValueError(
                f"walk-forward fold {yr}: 0 positive examples in test window — "
                f"cannot evaluate a gate fold with no positives"
            )

        folds.append(
            {
                "test_year": yr,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "n_pos_test": n_pos,
            }
        )
    return folds


def auprc_logit_ci(auprc: float, n_pos: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Boyd, Eng & Page (2013) logit confidence interval for AUPRC.

    Logit-transforms the point estimate (keeping the interval inside [0, 1]) with a
    binomial standard error on the positive count: SE(logit) = 1 / sqrt(n_pos * θ(1-θ)).
    n_pos is the effective sample size — this is why 20 positives gives a very wide CI.
    """
    from scipy.stats import norm

    if n_pos <= 0:
        return (0.0, 1.0)
    theta = min(max(float(auprc), 1e-6), 1.0 - 1e-6)
    z = norm.ppf(1.0 - alpha / 2.0)
    eta = math.log(theta / (1.0 - theta))
    se_eta = 1.0 / math.sqrt(n_pos * theta * (1.0 - theta))
    lo = 1.0 / (1.0 + math.exp(-(eta - z * se_eta)))
    hi = 1.0 / (1.0 + math.exp(-(eta + z * se_eta)))
    return (lo, hi)


def nadeau_bengio_ttest(diffs, n_train, n_test) -> tuple[float, float]:
    """
    Paired corrected-resampled t-test (Nadeau & Bengio 2003).

    Corrects the naive paired-t variance for the dependence between overlapping
    train sets: variance is scaled by (1/K + n_test/n_train) instead of 1/K.
    `diffs` are per-fold (challenger - champion) metric values. n_train / n_test may
    be scalars or per-fold lists (their means are used for the correction ratio).
    Returns (t_stat, two_sided_p).
    """
    from scipy.stats import t as t_dist

    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    if k < 2:
        raise ValueError("Nadeau-Bengio test needs at least 2 folds")
    mean_d = float(diffs.mean())
    var_d = float(diffs.var(ddof=1))
    ratio = float(np.mean(np.asarray(n_test, dtype=float))) / float(
        np.mean(np.asarray(n_train, dtype=float))
    )
    corrected_var = var_d * (1.0 / k + ratio)
    if corrected_var <= 0:
        # zero variance: significant only if the mean differs from zero
        return (
            math.inf if mean_d > 0 else (-math.inf if mean_d < 0 else 0.0),
            0.0 if mean_d != 0 else 1.0,
        )
    t_stat = mean_d / math.sqrt(corrected_var)
    p = 2.0 * t_dist.sf(abs(t_stat), df=k - 1)
    return (t_stat, float(p))


def promotion_decision_single(y_true, champ_prob, chall_prob, alpha: float = 0.05) -> dict:
    """
    Phase II/III gate on a single temporal test split.

    Promote only if the challenger beats the champion on BOTH PR-AUC and AUROC AND
    the PR-AUC improvement is interval-supported: challenger's logit-CI lower bound
    exceeds the champion's PR-AUC point estimate (or the two CIs do not overlap).
    """
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    champ = evaluate(y_true, champ_prob)
    chall = evaluate(y_true, chall_prob)
    champ_ci = auprc_logit_ci(champ["prauc"], n_pos, alpha)
    chall_ci = auprc_logit_ci(chall["prauc"], n_pos, alpha)

    beats_both = (chall["prauc"] > champ["prauc"]) and (chall["auroc"] > champ["auroc"])
    ci_supported = (chall_ci[0] > champ_ci[1]) or (chall_ci[0] > champ["prauc"])
    promote = bool(beats_both and ci_supported)

    return {
        "promote": promote,
        "method": "single_split_logit_ci",
        "champion": champ,
        "challenger": chall,
        "champion_prauc_ci": champ_ci,
        "challenger_prauc_ci": chall_ci,
        "n_pos": n_pos,
        "reason": (
            f"beats_both={beats_both}, ci_supported={ci_supported} "
            f"(chall PR-AUC CI {chall_ci[0]:.3f}-{chall_ci[1]:.3f} vs "
            f"champ point {champ['prauc']:.3f})"
        ),
    }


def promotion_decision_walkforward(
    y_by_fold, champ_by_fold, chall_by_fold, n_train, n_test, alpha: float = 0.05
) -> dict:
    """
    Phase I gate across walk-forward folds.

    Promote only if the challenger beats the champion on BOTH mean PR-AUC and mean
    AUROC across folds AND the per-fold PR-AUC improvement is significant under the
    paired Nadeau-Bengio corrected-resampled t-test (p < alpha, improvement positive).
    """
    champ_prauc, chall_prauc, champ_auroc, chall_auroc, prauc_diffs = [], [], [], [], []
    for yt, cp, xp in zip(y_by_fold, champ_by_fold, chall_by_fold, strict=True):
        c, x = evaluate(yt, cp), evaluate(yt, xp)
        champ_prauc.append(c["prauc"])
        chall_prauc.append(x["prauc"])
        champ_auroc.append(c["auroc"])
        chall_auroc.append(x["auroc"])
        prauc_diffs.append(x["prauc"] - c["prauc"])

    mean_champ_prauc = float(np.mean(champ_prauc))
    mean_chall_prauc = float(np.mean(chall_prauc))
    mean_champ_auroc = float(np.mean(champ_auroc))
    mean_chall_auroc = float(np.mean(chall_auroc))

    t_stat, p_value = nadeau_bengio_ttest(prauc_diffs, n_train, n_test)
    beats_both = (mean_chall_prauc > mean_champ_prauc) and (mean_chall_auroc > mean_champ_auroc)
    significant = (float(np.mean(prauc_diffs)) > 0) and (p_value < alpha)
    promote = bool(beats_both and significant)

    return {
        "promote": promote,
        "method": "walk_forward_nadeau_bengio",
        "mean_champion_prauc": mean_champ_prauc,
        "mean_challenger_prauc": mean_chall_prauc,
        "mean_champion_auroc": mean_champ_auroc,
        "mean_challenger_auroc": mean_chall_auroc,
        "prauc_diffs": [float(d) for d in prauc_diffs],
        "nb_t_stat": float(t_stat),
        "nb_p_value": float(p_value),
        "reason": (
            f"beats_both={beats_both}, significant={significant} "
            f"(mean ΔPR-AUC={np.mean(prauc_diffs):+.3f}, NB p={p_value:.3f})"
        ),
    }
