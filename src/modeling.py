"""Ordinal credit rating models, nonlinear benchmarks, IG-Cliff classifiers, and Dual CV."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.data import (
    BROAD_TIERS,
    COL_COMPANY,
    COL_MACRO,
    COL_RATING,
    COL_SECTOR,
    IG_MAX_NOTCH,
    NOTCH_TO_RATING,
    NOTCH_TO_TIER,
    RATING_LADDER,
    TIER_TO_NOTCHES,
    notch_to_rating,
    notch_to_tier,
)
from src.features import (
    FEATURE_LABELS,
    FEATURE_MONOTONIC_SIGN,
    FEATURE_NAMES,
    PILLAR_FEATURES,
    CreditFeatureTransformer,
)

N_NOTCHES = len(RATING_LADDER)  # 18
NOTCH_MONOTONIC_SIGNS = np.array(
    [FEATURE_MONOTONIC_SIGN[f] for f in FEATURE_NAMES], dtype=int
)
IG_MONOTONIC_SIGNS = -NOTCH_MONOTONIC_SIGNS  # Higher feature = stronger credit => higher P(IG)


class MonotonicLogisticClassifier(BaseEstimator, ClassifierMixin):
    """L2-regularized binary logistic regression with credit-direction sign constraints.

    Solves the exact convex L2-penalized log-loss via L-BFGS-B with optional sign bounds
    (`sign = -1` => beta <= 0, `sign = +1` => beta >= 0) so collinear financial ratios can
    never invert economic intuition in scorecards or stress tests.
    """

    def __init__(
        self,
        C: float = 0.5,
        monotonic_signs: np.ndarray | list[int] | None = None,
        max_iter: int = 500,
    ) -> None:
        self.C = C
        self.monotonic_signs = monotonic_signs
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MonotonicLogisticClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape
        self.classes_ = np.array([0, 1], dtype=int)

        pos_rate = float(np.clip(np.mean(y_arr), 1e-6, 1.0 - 1e-6))
        if np.all(y_arr == 0) or np.all(y_arr == 1):
            self.intercept_ = np.array([np.log(pos_rate / (1.0 - pos_rate))], dtype=float)
            self.coef_ = np.zeros((1, n_features), dtype=float)
            return self

        bounds: list[tuple[float | None, float | None]] = [(None, None)]
        if self.monotonic_signs is None:
            bounds.extend([(None, None)] * n_features)
        else:
            signs = np.asarray(self.monotonic_signs, dtype=int)
            for s in signs:
                if s < 0:
                    bounds.append((None, 0.0))
                elif s > 0:
                    bounds.append((0.0, None))
                else:
                    bounds.append((None, None))

        reg_strength = 1.0 / max(float(self.C), 1e-6)
        init_params = np.zeros(n_features + 1, dtype=float)
        init_params[0] = np.log(pos_rate / (1.0 - pos_rate))

        def _loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            b0 = params[0]
            w = params[1:]
            z = b0 + X_arr @ w
            # Numerically stable binary cross-entropy: log(1 + exp(z)) - y * z
            loss = float(np.sum(np.logaddexp(0.0, z) - y_arr * z)) + 0.5 * reg_strength * float(
                np.dot(w, w)
            )
            p = expit(z)
            diff = p - y_arr
            grad = np.empty_like(params)
            grad[0] = float(np.sum(diff))
            grad[1:] = X_arr.T @ diff + reg_strength * w
            return loss, grad

        res = minimize(
            _loss_and_grad,
            init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )
        self.intercept_ = np.array([float(res.x[0])], dtype=float)
        self.coef_ = res.x[1:].reshape(1, -1).astype(float)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return float(self.intercept_[0]) + X_arr @ self.coef_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class OrdinalLogistic(BaseEstimator, RegressorMixin):
    """Cumulative ordinal logistic scorecard over 18 ordered credit notches (AAA=1 .. D=18).

    Fits 17 L2-regularized cumulative thresholds P(Y >= k) for k in 2..18, enforces
    monotonic tail probabilities via cumulative-min projection, and computes exact
    additive linear pillar attributions via path integration from the universe mean.
    """

    def __init__(
        self,
        C: float = 0.25,
        n_notches: int = N_NOTCHES,
        monotonic_signs: np.ndarray | list[int] | None = None,
    ) -> None:
        self.C = C
        self.n_notches = n_notches
        self.monotonic_signs = monotonic_signs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalLogistic":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        signs = (
            NOTCH_MONOTONIC_SIGNS
            if self.monotonic_signs is None and X_arr.shape[1] == len(NOTCH_MONOTONIC_SIGNS)
            else self.monotonic_signs
        )

        self.thresholds_: list[int] = list(range(2, self.n_notches + 1))
        self.models_: list[MonotonicLogisticClassifier] = []
        coef_rows = []
        intercept_vals = []

        for k in self.thresholds_:
            y_bin = (y_arr >= k).astype(int)
            clf = MonotonicLogisticClassifier(C=self.C, monotonic_signs=signs)
            clf.fit(X_arr, y_bin)
            self.models_.append(clf)
            coef_rows.append(clf.coef_[0])
            intercept_vals.append(float(clf.intercept_[0]))

        self.coef_matrix_ = np.vstack(coef_rows)  # shape: (17, n_features)
        self.intercepts_ = np.array(intercept_vals, dtype=float)  # shape: (17,)
        self.mean_weights_ = np.mean(self.coef_matrix_, axis=0)
        return self

    def predict_cumulative_proba(self, X: np.ndarray) -> np.ndarray:
        """Return monotonic cumulative probabilities P(Y >= k) for k = 2..18, shape (n, 17)."""
        X_arr = np.asarray(X, dtype=float)
        logits = X_arr @ self.coef_matrix_.T + self.intercepts_[None, :]
        raw_q = np.clip(expit(logits), 0.0, 1.0)
        # Monotonic projection: P(Y >= 2) >= P(Y >= 3) >= ... >= P(Y >= 18) >= 0
        return np.minimum.accumulate(raw_q, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return valid 18-notch probability distribution per row summing identically to 1."""
        q = self.predict_cumulative_proba(X)
        n = q.shape[0]
        q_ext = np.hstack([np.ones((n, 1), dtype=float), q, np.zeros((n, 1), dtype=float)])
        probs = np.clip(q_ext[:, :-1] - q_ext[:, 1:], 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0.0, probs / row_sums, 1.0 / self.n_notches)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Expected continuous notch score E[Y|x] = 1 + sum_{k=2..18} P(Y >= k) in [1, 18]."""
        q = self.predict_cumulative_proba(X)
        return np.clip(1.0 + np.sum(q, axis=1), 1.0, float(self.n_notches))

    def predict_tier_proba(self, X: np.ndarray) -> np.ndarray:
        """Return 7-broad-tier probabilities aggregated from the 18-notch distribution."""
        p18 = self.predict_proba(X)
        tier_cols = [
            np.sum(p18[:, [n - 1 for n in TIER_TO_NOTCHES[tier]]], axis=1)
            for tier in BROAD_TIERS
        ]
        return np.column_stack(tier_cols)

    def predict_tier(self, X: np.ndarray) -> np.ndarray:
        """Return predicted broad tier based on nearest expected notch."""
        notches = self.predict(X)
        return np.array([notch_to_tier(v) for v in notches], dtype=object)

    def predict_ig_proba(self, X: np.ndarray) -> np.ndarray:
        """Implied P(Investment Grade) = P(Y <= 10) = 1 - P(Y >= 11)."""
        q = self.predict_cumulative_proba(X)
        # Index 9 corresponds to threshold k = 11 (since thresholds start at k = 2)
        return 1.0 - q[:, IG_MAX_NOTCH - 1]

    def explain_notch_attributions(
        self, X_scaled: np.ndarray, n_steps: int = 30
    ) -> dict[str, Any]:
        """Decompose expected notch E[Y|z] - E[Y|0] into exact additive feature & pillar shifts.

        Uses midpoint path integration over alpha in [0, 1] along the standardized vector z.
        When monotonic clamping is active at any threshold, distributes the tiny residual
        proportionally so `baseline_notch + sum(pillar_contribs) == expected_notch` holds exactly.
        """
        z = np.asarray(X_scaled, dtype=float)
        if z.ndim == 2:
            z = z[0]
        n_features = z.shape[0]

        zero_vec = np.zeros((1, n_features), dtype=float)
        baseline_notch = float(self.predict(zero_vec)[0])
        target_notch = float(self.predict(z.reshape(1, -1))[0])
        exact_delta = target_notch - baseline_notch

        alphas = (np.arange(n_steps, dtype=float) + 0.5) / n_steps
        z_path = np.outer(alphas, z)  # (n_steps, n_features)
        logits_path = z_path @ self.coef_matrix_.T + self.intercepts_[None, :]
        q_raw = expit(logits_path)
        dq_deta = q_raw * (1.0 - q_raw)  # (n_steps, 17)
        # Average gradient wrt z across path: mean_alpha [ sum_t q_t(1-q_t) * beta_{t,j} ]
        avg_grad = np.mean(dq_deta @ self.coef_matrix_, axis=0)
        raw_contribs = z * avg_grad

        raw_sum = float(np.sum(raw_contribs))
        if abs(raw_sum) > 1e-9:
            # Absorb any minor monotonic-projection discrepancy strictly proportionally
            scale_adj = exact_delta / raw_sum
            # Guard against extreme scaling if exact_delta and raw_sum nearly cancel
            if 0.5 <= scale_adj <= 1.5:
                feature_contribs = raw_contribs * scale_adj
            else:
                feature_contribs = raw_contribs + (exact_delta - raw_sum) / n_features
        else:
            feature_contribs = np.zeros(n_features, dtype=float)

        feat_map = {
            FEATURE_NAMES[i]: float(feature_contribs[i]) for i in range(n_features)
        }
        pillar_map = {
            pillar: float(sum(feat_map[f] for f in feats))
            for pillar, feats in PILLAR_FEATURES.items()
        }
        return {
            "baseline_notch": baseline_notch,
            "expected_notch": target_notch,
            "feature_attributions": feat_map,
            "pillar_attributions": pillar_map,
        }


def compute_ordinal_metrics(
    y_true_notch: np.ndarray,
    y_pred_notch: np.ndarray,
    y_pred_tier: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute core ordinal credit rating evaluation metrics."""
    y_true = np.asarray(y_true_notch, dtype=float)
    y_pred = np.clip(np.asarray(y_pred_notch, dtype=float), 1.0, float(N_NOTCHES))
    y_pred_round = np.rint(y_pred).astype(int)
    y_true_int = y_true.astype(int)

    abs_err = np.abs(y_true - y_pred)
    true_tiers = np.array([NOTCH_TO_TIER[n] for n in y_true_int])
    if y_pred_tier is None:
        pred_tiers = np.array([NOTCH_TO_TIER[n] for n in y_pred_round])
    else:
        pred_tiers = np.asarray(y_pred_tier)

    tier_to_idx = {t: i for i, t in enumerate(BROAD_TIERS)}
    true_tier_idx = np.array([tier_to_idx[t] for t in true_tiers])
    pred_tier_idx = np.array([tier_to_idx[t] for t in pred_tiers])

    return {
        "notch_mae": float(mean_absolute_error(y_true, y_pred)),
        "within_1_notch_pct": float(np.mean(abs_err <= 1.0) * 100.0),
        "within_2_notch_pct": float(np.mean(abs_err <= 2.0) * 100.0),
        "broad_tier_acc_pct": float(accuracy_score(true_tiers, pred_tiers) * 100.0),
        "qwk_18": float(cohen_kappa_score(y_true_int, y_pred_round, weights="quadratic")),
        "qwk_7": float(cohen_kappa_score(true_tier_idx, pred_tier_idx, weights="quadratic")),
    }


def compute_ig_metrics(y_true_ig: np.ndarray, p_ig: np.ndarray) -> dict[str, float]:
    """Compute IG-vs-HY discrimination and probability calibration metrics."""
    y_ig = np.asarray(y_true_ig, dtype=int)
    p = np.clip(np.asarray(p_ig, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "ig_roc_auc": float(roc_auc_score(y_ig, p)),
        "ig_pr_auc": float(average_precision_score(y_ig, p)),
        "hy_pr_auc": float(average_precision_score(1 - y_ig, 1.0 - p)),
        "ig_brier": float(brier_score_loss(y_ig, p)),
    }


def _make_hgb_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def _make_hgb_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def _make_calibrated_ig_logistic() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=MonotonicLogisticClassifier(C=0.3, monotonic_signs=IG_MONOTONIC_SIGNS),
        method="sigmoid",
        cv=5,
    )


def _make_calibrated_ig_hgb() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=_make_hgb_classifier(),
        method="sigmoid",
        cv=5,
    )


def _evaluate_cv_scheme(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Run strictly leakage-free OOF evaluation across all models for a given split list."""
    n = len(df)
    y_notch = df["notch_num"].to_numpy(dtype=int)
    y_tier = df["broad_tier"].to_numpy(dtype=object)
    y_ig = df["is_ig"].to_numpy(dtype=int)

    oof_ord_notch = np.zeros(n, dtype=float)
    oof_ord_p_ig = np.zeros(n, dtype=float)
    oof_ord_probs = np.zeros((n, N_NOTCHES), dtype=float)
    oof_hgb_notch = np.zeros(n, dtype=float)
    oof_hgb_tier = np.empty(n, dtype=object)
    oof_ig_log_p = np.zeros(n, dtype=float)
    oof_ig_hgb_p = np.zeros(n, dtype=float)

    for train_idx, val_idx in splits:
        df_tr = df.iloc[train_idx]
        df_va = df.iloc[val_idx]

        # Fit transformers strictly on training fold — zero CV leakage
        tf_scaled = CreditFeatureTransformer(scale=True)
        X_tr_s = tf_scaled.fit_transform(df_tr)
        X_va_s = tf_scaled.transform(df_va)

        tf_raw = CreditFeatureTransformer(scale=False)
        X_tr_r = tf_raw.fit_transform(df_tr)
        X_va_r = tf_raw.transform(df_va)

        # 1. Primary OrdinalLogistic
        ord_model = OrdinalLogistic(C=0.25)
        ord_model.fit(X_tr_s, y_notch[train_idx])
        oof_ord_notch[val_idx] = ord_model.predict(X_va_s)
        oof_ord_p_ig[val_idx] = ord_model.predict_ig_proba(X_va_s)
        oof_ord_probs[val_idx] = ord_model.predict_proba(X_va_s)

        # 2. Nonlinear Benchmarks (HistGradientBoosting)
        hgb_reg = _make_hgb_regressor()
        hgb_reg.fit(X_tr_r, y_notch[train_idx])
        oof_hgb_notch[val_idx] = np.clip(hgb_reg.predict(X_va_r), 1.0, float(N_NOTCHES))

        hgb_clf = _make_hgb_classifier()
        hgb_clf.fit(X_tr_r, y_tier[train_idx])
        oof_hgb_tier[val_idx] = hgb_clf.predict(X_va_r)

        # 3. IG-Cliff Models (Calibrated Logistic & Calibrated HGB)
        ig_log = _make_calibrated_ig_logistic()
        ig_log.fit(X_tr_s, y_ig[train_idx])
        oof_ig_log_p[val_idx] = ig_log.predict_proba(X_va_s)[:, 1]

        ig_hgb = _make_calibrated_ig_hgb()
        ig_hgb.fit(X_tr_r, y_ig[train_idx])
        oof_ig_hgb_p[val_idx] = ig_hgb.predict_proba(X_va_r)[:, 1]

    ord_metrics = {
        **compute_ordinal_metrics(y_notch, oof_ord_notch),
        **compute_ig_metrics(y_ig, oof_ord_p_ig),
    }
    hgb_reg_metrics = compute_ordinal_metrics(y_notch, oof_hgb_notch)
    hgb_tier_metrics = compute_ordinal_metrics(y_notch, oof_hgb_notch, y_pred_tier=oof_hgb_tier)
    hgb_combined = {
        **hgb_reg_metrics,
        "hgb_clf_broad_tier_acc_pct": hgb_tier_metrics["broad_tier_acc_pct"],
        "hgb_clf_qwk_7": hgb_tier_metrics["qwk_7"],
        **compute_ig_metrics(y_ig, oof_ig_hgb_p),
    }
    ig_log_metrics = compute_ig_metrics(y_ig, oof_ig_log_p)
    ig_hgb_metrics = compute_ig_metrics(y_ig, oof_ig_hgb_p)

    # 7-tier confusion matrix for primary OrdinalLogistic OOF
    pred_tiers_ord = np.array([notch_to_tier(v) for v in oof_ord_notch])
    cm_7 = confusion_matrix(y_tier, pred_tiers_ord, labels=list(BROAD_TIERS))

    # Reliability / calibration curves for IG-Cliff models
    prob_true_log, prob_pred_log = calibration_curve(y_ig, oof_ig_log_p, n_bins=10, strategy="quantile")
    prob_true_hgb, prob_pred_hgb = calibration_curve(y_ig, oof_ig_hgb_p, n_bins=10, strategy="quantile")
    prob_true_ord, prob_pred_ord = calibration_curve(y_ig, oof_ord_p_ig, n_bins=10, strategy="quantile")

    return {
        "metrics": {
            "ordinal_logistic": ord_metrics,
            "hist_gradient_boosting": hgb_combined,
            "ig_cliff_logistic": ig_log_metrics,
            "ig_cliff_hgb": ig_hgb_metrics,
        },
        "oof": {
            "shadow_notch": oof_ord_notch,
            "shadow_p_ig_ordinal": oof_ord_p_ig,
            "shadow_probs_18": oof_ord_probs,
            "hgb_shadow_notch": oof_hgb_notch,
            "hgb_shadow_tier": oof_hgb_tier,
            "p_ig_logistic": oof_ig_log_p,
            "p_ig_hgb": oof_ig_hgb_p,
        },
        "confusion_matrix_7tier": cm_7.tolist(),
        "calibration": {
            "ig_logistic": {
                "prob_true": prob_true_log.tolist(),
                "prob_pred": prob_pred_log.tolist(),
            },
            "ig_hgb": {
                "prob_true": prob_true_hgb.tolist(),
                "prob_pred": prob_pred_hgb.tolist(),
            },
            "ordinal_implied": {
                "prob_true": prob_true_ord.tolist(),
                "prob_pred": prob_pred_ord.tolist(),
            },
        },
    }


def run_dual_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute 5-Fold Stratified CV and 5-Fold Out-of-Sector GroupKFold by NSE Macro Industry."""
    y_notch = df["notch_num"].to_numpy(dtype=int)
    groups = df[COL_MACRO].astype(str).to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat_splits = list(skf.split(df, y_notch))

    gkf = GroupKFold(n_splits=5)
    group_splits = list(gkf.split(df, y_notch, groups))

    strat_res = _evaluate_cv_scheme(df, strat_splits)
    group_res = _evaluate_cv_scheme(df, group_splits)

    # Assemble strict Stratified OOF company-level shadow ratings & Rating-Gap flags
    oof = strat_res["oof"]
    agency_notch = df["notch_num"].to_numpy(dtype=float)
    shadow_notch = oof["shadow_notch"]
    raw_notch_diff = shadow_notch - agency_notch
    # Positive rating_gap_notches => shadow is stronger than agency => Rising Star (Upgrade Watch)
    rating_gap_notches = agency_notch - shadow_notch

    watch_status = np.where(
        rating_gap_notches >= 2.0,
        "Rising Star (Upgrade Watch)",
        np.where(
            rating_gap_notches <= -2.0,
            "Fallen Angel (Downgrade Watch)",
            "In-Line",
        ),
    )

    oof_df = pd.DataFrame(
        {
            COL_COMPANY: df[COL_COMPANY].to_numpy(),
            COL_MACRO: df[COL_MACRO].to_numpy(),
            COL_SECTOR: df[COL_SECTOR].to_numpy(),
            "peer_group": df["peer_group"].to_numpy(),
            "agency_rating": df[COL_RATING].to_numpy(),
            "agency_notch": agency_notch.astype(int),
            "agency_tier": df["broad_tier"].to_numpy(),
            "is_ig": df["is_ig"].to_numpy(),
            "shadow_notch": np.round(shadow_notch, 3),
            "shadow_rating": [notch_to_rating(v) for v in shadow_notch],
            "shadow_tier": [notch_to_tier(v) for v in shadow_notch],
            "raw_notch_diff": np.round(raw_notch_diff, 3),
            "rating_gap_notches": np.round(rating_gap_notches, 3),
            "watch_status": watch_status,
            "p_ig_logistic": np.round(oof["p_ig_logistic"], 4),
            "p_ig_ordinal": np.round(oof["shadow_p_ig_ordinal"], 4),
            "p_ig_hgb": np.round(oof["p_ig_hgb"], 4),
            "hgb_shadow_notch": np.round(oof["hgb_shadow_notch"], 3),
            "hgb_shadow_tier": oof["hgb_shadow_tier"],
        }
    )

    validation_report: dict[str, Any] = {
        "n_valid_corporates": int(len(df)),
        "stratified_5fold": strat_res["metrics"],
        "group_kfold_macro": group_res["metrics"],
        "confusion_matrix_7tier_labels": list(BROAD_TIERS),
        "confusion_matrix_7tier_stratified": strat_res["confusion_matrix_7tier"],
        "confusion_matrix_7tier_group": group_res["confusion_matrix_7tier"],
        "calibration_stratified": strat_res["calibration"],
        "calibration_group": group_res["calibration"],
    }
    return oof_df, validation_report


def fit_production_bundle(df: pd.DataFrame) -> dict[str, Any]:
    """Fit full-universe models and transformers for live underwriting & stress-testing."""
    y_notch = df["notch_num"].to_numpy(dtype=int)
    y_tier = df["broad_tier"].to_numpy(dtype=object)
    y_ig = df["is_ig"].to_numpy(dtype=int)

    tf_scaled = CreditFeatureTransformer(scale=True)
    X_s = tf_scaled.fit_transform(df)

    tf_raw = CreditFeatureTransformer(scale=False)
    X_r = tf_raw.fit_transform(df)

    ord_model = OrdinalLogistic(C=0.25)
    ord_model.fit(X_s, y_notch)

    hgb_reg = _make_hgb_regressor().fit(X_r, y_notch)
    hgb_clf = _make_hgb_classifier().fit(X_r, y_tier)
    ig_log = _make_calibrated_ig_logistic().fit(X_s, y_ig)
    ig_hgb = _make_calibrated_ig_hgb().fit(X_r, y_ig)

    # Scorecard feature weights table for Tab 4 audit
    scorecard_weights = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "label": [FEATURE_LABELS[f] for f in FEATURE_NAMES],
            "pillar": [
                next(p for p, fs in PILLAR_FEATURES.items() if f in fs)
                for f in FEATURE_NAMES
            ],
            "mean_log_odds_weight": ord_model.mean_weights_,
            "abs_weight": np.abs(ord_model.mean_weights_),
        }
    ).sort_values("abs_weight", ascending=False).reset_index(drop=True)

    # Universe median implied cost of debt for debt-free stress-test fallback
    valid_cod = df["implied_cost_of_debt"].dropna()
    median_cod = float(valid_cod.median()) if len(valid_cod) else 0.1056

    return {
        "transformer_scaled": tf_scaled,
        "transformer_raw": tf_raw,
        "ordinal_model": ord_model,
        "hgb_regressor": hgb_reg,
        "hgb_tier_classifier": hgb_clf,
        "ig_cliff_logistic": ig_log,
        "ig_cliff_hgb": ig_hgb,
        "scorecard_weights": scorecard_weights,
        "median_cost_of_debt": median_cod,
        "sector_to_macro": (
            df[[COL_SECTOR, COL_MACRO]]
            .drop_duplicates()
            .set_index(COL_SECTOR)[COL_MACRO]
            .to_dict()
        ),
    }


def score_company_record(
    record_df: pd.DataFrame, bundle: dict[str, Any]
) -> dict[str, Any]:
    """Score a single-row corporate DataFrame using the production model bundle."""
    tf_s: CreditFeatureTransformer = bundle["transformer_scaled"]
    tf_r: CreditFeatureTransformer = bundle["transformer_raw"]
    ord_model: OrdinalLogistic = bundle["ordinal_model"]
    ig_log: CalibratedClassifierCV = bundle["ig_cliff_logistic"]
    hgb_reg: HistGradientBoostingRegressor = bundle["hgb_regressor"]

    X_s = tf_s.transform(record_df)
    X_r = tf_r.transform(record_df)

    expected_notch = float(ord_model.predict(X_s)[0])
    probs_18 = ord_model.predict_proba(X_s)[0]
    tier_probs = ord_model.predict_tier_proba(X_s)[0]
    p_ig_cal = float(ig_log.predict_proba(X_s)[0, 1])
    p_ig_ord = float(ord_model.predict_ig_proba(X_s)[0])
    hgb_notch = float(np.clip(hgb_reg.predict(X_r)[0], 1.0, float(N_NOTCHES)))
    attributions = ord_model.explain_notch_attributions(X_s[0])

    return {
        "expected_notch": expected_notch,
        "shadow_rating": notch_to_rating(expected_notch),
        "shadow_tier": notch_to_tier(expected_notch),
        "probs_18": probs_18,
        "tier_probs": dict(zip(BROAD_TIERS, tier_probs.tolist())),
        "p_ig": p_ig_cal,
        "p_ig_ordinal": p_ig_ord,
        "hgb_notch": hgb_notch,
        "baseline_notch": attributions["baseline_notch"],
        "pillar_attributions": attributions["pillar_attributions"],
        "feature_attributions": attributions["feature_attributions"],
    }
