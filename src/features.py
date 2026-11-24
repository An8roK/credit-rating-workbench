"""Leakage-free credit feature engineering and financial stress-shock propagation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.data import (
    COL_CURR_RATIO,
    COL_GCA,
    COL_INC_GROWTH,
    COL_INT_COV,
    COL_INTEREST,
    COL_MACRO,
    COL_NETWORTH,
    COL_OP_CYCLE,
    COL_PAT,
    COL_PAT_GROWTH,
    COL_PAT_MARGIN,
    COL_PBILDT,
    COL_PBILDT_MARGIN,
    COL_QUICK_RATIO,
    COL_ROCE,
    COL_SECTOR,
    COL_TCE,
    COL_TOI,
    MIN_PEER_GROUP_SIZE,
)

# 7 credit pillars -> 22 engineered features (exact docs/ARCHITECTURE.md schema)
PILLAR_FEATURES: dict[str, list[str]] = {
    "Scale": [
        "log_toi",
        "signed_log_networth",
        "log_tce",
    ],
    "Profitability": [
        "pbildt_margin_clip",
        "pat_margin_clip",
        "roce_clip",
    ],
    "Solvency & Coverage": [
        "gca_to_debt",
        "pbildt_to_debt",
        "networth_to_tce",
        "asinh_interest_coverage",
        "short_term_debt_share",
    ],
    "Distress Flags": [
        "is_negative_networth",
        "is_negative_gca",
        "is_negative_pbildt",
    ],
    "Liquidity & Working Capital": [
        "current_ratio_clip",
        "quick_ratio_clip",
        "op_cycle_clip",
    ],
    "Growth": [
        "income_growth_clip",
        "pat_growth_clip",
    ],
    "Sector-Relative": [
        "rel_pbildt_margin_z",
        "rel_roce_z",
        "rel_leverage_z",
    ],
}

FEATURE_NAMES: list[str] = [
    feat for feats in PILLAR_FEATURES.values() for feat in feats
]
FEATURE_TO_PILLAR: dict[str, str] = {
    feat: pillar for pillar, feats in PILLAR_FEATURES.items() for feat in feats
}

# Credit direction for notch_num (1=AAA .. 18=D):
# -1 => higher feature value improves credit (lowers notch_num / P(Y >= k))
# +1 => higher feature value weakens credit (raises notch_num / P(Y >= k))
FEATURE_MONOTONIC_SIGN: dict[str, int] = {
    "log_toi": -1,
    "signed_log_networth": -1,
    "log_tce": -1,
    "pbildt_margin_clip": -1,
    "pat_margin_clip": -1,
    "roce_clip": -1,
    "gca_to_debt": -1,
    "pbildt_to_debt": -1,
    "networth_to_tce": -1,
    "asinh_interest_coverage": -1,
    "short_term_debt_share": 1,
    "is_negative_networth": 1,
    "is_negative_gca": 1,
    "is_negative_pbildt": 1,
    "current_ratio_clip": -1,
    "quick_ratio_clip": -1,
    "op_cycle_clip": 1,
    "income_growth_clip": -1,
    "pat_growth_clip": -1,
    "rel_pbildt_margin_z": -1,
    "rel_roce_z": -1,
    "rel_leverage_z": -1,
}

FEATURE_LABELS: dict[str, str] = {
    "log_toi": "Log Total Operating Income",
    "signed_log_networth": "Signed Log Networth",
    "log_tce": "Log Total Capital Employed",
    "pbildt_margin_clip": "PBILDT Margin (%)",
    "pat_margin_clip": "PAT Margin (%)",
    "roce_clip": "ROCE (%)",
    "gca_to_debt": "GCA / Implied Total Debt",
    "pbildt_to_debt": "PBILDT / Implied Total Debt",
    "networth_to_tce": "Networth / TCE Cushion",
    "asinh_interest_coverage": "Interest Coverage (asinh)",
    "short_term_debt_share": "Short-Term Debt Share",
    "is_negative_networth": "Negative Networth Flag",
    "is_negative_gca": "Negative GCA Flag",
    "is_negative_pbildt": "Negative PBILDT Flag",
    "current_ratio_clip": "Current Ratio (x)",
    "quick_ratio_clip": "Quick Ratio (x)",
    "op_cycle_clip": "Operating Cycle (days)",
    "income_growth_clip": "Total Income Growth (%)",
    "pat_growth_clip": "PAT Growth (%)",
    "rel_pbildt_margin_z": "Sector-Rel PBILDT Margin (z)",
    "rel_roce_z": "Sector-Rel ROCE (z)",
    "rel_leverage_z": "Sector-Rel PBILDT/Debt (z)",
}

# Mapping from relative z-score feature -> underlying base feature
RELATIVE_SOURCE_MAP: dict[str, str] = {
    "rel_pbildt_margin_z": "pbildt_margin_clip",
    "rel_roce_z": "roce_clip",
    "rel_leverage_z": "pbildt_to_debt",
}


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Order-preserving signed log transform for variables that can cross zero (e.g. Networth)."""
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 19 deterministic (non-peer-relative) credit features."""
    toi = pd.to_numeric(df[COL_TOI], errors="coerce").to_numpy(dtype=float)
    nw = pd.to_numeric(df[COL_NETWORTH], errors="coerce").to_numpy(dtype=float)
    tce = pd.to_numeric(df[COL_TCE], errors="coerce").to_numpy(dtype=float)
    # Under stress testing, optional _scale_tce holds pre-shock TCE so debt bloat never boosts Scale
    if "_scale_tce" in df.columns:
        scale_tce = pd.to_numeric(df["_scale_tce"], errors="coerce").to_numpy(dtype=float)
    else:
        scale_tce = tce

    pbildt = pd.to_numeric(df[COL_PBILDT], errors="coerce").to_numpy(dtype=float)
    gca = pd.to_numeric(df[COL_GCA], errors="coerce").to_numpy(dtype=float)
    pbildt_margin = pd.to_numeric(df[COL_PBILDT_MARGIN], errors="coerce").to_numpy(dtype=float)
    pat_margin = pd.to_numeric(df[COL_PAT_MARGIN], errors="coerce").to_numpy(dtype=float)
    roce = pd.to_numeric(df[COL_ROCE], errors="coerce").to_numpy(dtype=float)
    int_cov = pd.to_numeric(df[COL_INT_COV], errors="coerce").to_numpy(dtype=float)
    curr_ratio = pd.to_numeric(df[COL_CURR_RATIO], errors="coerce").to_numpy(dtype=float)
    quick_ratio = pd.to_numeric(df[COL_QUICK_RATIO], errors="coerce").to_numpy(dtype=float)
    op_cycle = pd.to_numeric(df[COL_OP_CYCLE], errors="coerce").to_numpy(dtype=float)
    inc_growth = pd.to_numeric(df[COL_INC_GROWTH], errors="coerce").to_numpy(dtype=float)
    pat_growth = pd.to_numeric(df[COL_PAT_GROWTH], errors="coerce").to_numpy(dtype=float)

    imp_td = pd.to_numeric(df["implied_total_debt"], errors="coerce").to_numpy(dtype=float)
    imp_term = pd.to_numeric(df["implied_term_debt"], errors="coerce").to_numpy(dtype=float)

    # Negative-denominator-proof solvency & coverage ratios
    debt_denom = np.maximum(imp_td, 1.0)
    gca_to_debt = np.clip(gca / debt_denom, -2.0, 5.0)
    pbildt_to_debt = np.clip(pbildt / debt_denom, -2.0, 5.0)
    if "_max_gca_to_debt" in df.columns:
        gca_to_debt = np.minimum(
            gca_to_debt, pd.to_numeric(df["_max_gca_to_debt"], errors="coerce").to_numpy(dtype=float)
        )
    if "_max_pbildt_to_debt" in df.columns:
        pbildt_to_debt = np.minimum(
            pbildt_to_debt, pd.to_numeric(df["_max_pbildt_to_debt"], errors="coerce").to_numpy(dtype=float)
        )

    # Equity cushion: strictly negative when NW < 0, never flips sign when TCE <= 0
    raw_nw_tce = np.where(
        tce > 1.0,
        np.clip(nw / np.maximum(tce, 1.0), -1.0, 1.2),
        np.where(nw < 0.0, -1.0, 0.0),
    )
    if "_max_nw_to_tce" in df.columns:
        cap = pd.to_numeric(df["_max_nw_to_tce"], errors="coerce").to_numpy(dtype=float)
        raw_nw_tce = np.minimum(raw_nw_tce, cap)

    short_term_share = np.where(
        imp_td > 0.0,
        np.clip((imp_td - imp_term) / np.maximum(imp_td, 1e-9), 0.0, 1.0),
        0.0,
    )

    out = pd.DataFrame(
        {
            "log_toi": np.log1p(np.maximum(toi, 0.0)),
            "signed_log_networth": signed_log1p(nw),
            "log_tce": np.log1p(np.maximum(scale_tce, 0.0)),
            "pbildt_margin_clip": np.clip(pbildt_margin, -50.0, 75.0),
            "pat_margin_clip": np.clip(pat_margin, -50.0, 50.0),
            "roce_clip": np.clip(roce, -40.0, 80.0),
            "gca_to_debt": gca_to_debt,
            "pbildt_to_debt": pbildt_to_debt,
            "networth_to_tce": raw_nw_tce,
            "asinh_interest_coverage": np.arcsinh(np.clip(int_cov, -50.0, 500.0)),
            "short_term_debt_share": short_term_share,
            "is_negative_networth": (nw < 0.0).astype(float),
            "is_negative_gca": (gca < 0.0).astype(float),
            "is_negative_pbildt": (pbildt < 0.0).astype(float),
            "current_ratio_clip": np.clip(curr_ratio, 0.0, 10.0),
            "quick_ratio_clip": np.clip(quick_ratio, 0.0, 8.0),
            "op_cycle_clip": np.clip(op_cycle, -180.0, 730.0),
            "income_growth_clip": np.clip(inc_growth, -100.0, 200.0),
            "pat_growth_clip": np.clip(pat_growth, -100.0, 200.0),
        },
        index=df.index,
    )
    return out.fillna(0.0)


def _robust_center_scale(values: np.ndarray, floor: float = 1e-2) -> tuple[float, float]:
    """Compute (median, robust_scale) using MAD / IQR with a safe floor."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_scale = float(q75 - q25) / 1.349
    scale = max(mad, iqr_scale, floor)
    return med, scale


class CreditFeatureTransformer(BaseEstimator, TransformerMixin):
    """Leakage-free sklearn transformer for corporate credit features.

    Fits sector/macro/universe robust medians & scales and StandardScaler strictly
    on the training fold passed to `fit()`, guaranteeing zero CV leakage.
    """

    def __init__(
        self,
        scale: bool = True,
        min_peer_group_size: int = MIN_PEER_GROUP_SIZE,
        z_clip: float = 4.0,
    ) -> None:
        self.scale = scale
        self.min_peer_group_size = min_peer_group_size
        self.z_clip = z_clip

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CreditFeatureTransformer":
        base = compute_base_features(X)
        sectors = X[COL_SECTOR].astype(str).to_numpy()
        macros = X[COL_MACRO].astype(str).to_numpy()

        self.universe_stats_: dict[str, tuple[float, float]] = {}
        self.macro_stats_: dict[str, dict[str, tuple[float, float]]] = {}
        self.sector_stats_: dict[str, dict[str, tuple[float, float]]] = {}

        # Fit hierarchical peer statistics strictly on X_train
        for rel_feat, src_feat in RELATIVE_SOURCE_MAP.items():
            vals = base[src_feat].to_numpy(dtype=float)
            self.universe_stats_[rel_feat] = _robust_center_scale(vals)

            macro_map: dict[str, tuple[float, float]] = {}
            for macro_name in np.unique(macros):
                mask = macros == macro_name
                if int(mask.sum()) >= self.min_peer_group_size:
                    macro_map[str(macro_name)] = _robust_center_scale(vals[mask])
            self.macro_stats_[rel_feat] = macro_map

            sector_map: dict[str, tuple[float, float]] = {}
            for sec_name in np.unique(sectors):
                mask = sectors == sec_name
                if int(mask.sum()) >= self.min_peer_group_size:
                    sector_map[str(sec_name)] = _robust_center_scale(vals[mask])
            self.sector_stats_[rel_feat] = sector_map

        full_df = self._assemble_unscaled(X, base)
        self.feature_names_out_: list[str] = list(FEATURE_NAMES)
        self.scaler_: StandardScaler | None = None
        if self.scale:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(full_df.to_numpy(dtype=float))
        return self

    def _assemble_unscaled(
        self, X: pd.DataFrame, base: pd.DataFrame
    ) -> pd.DataFrame:
        out = base.copy()
        sectors = X[COL_SECTOR].astype(str).to_numpy()
        macros = X[COL_MACRO].astype(str).to_numpy()

        for rel_feat, src_feat in RELATIVE_SOURCE_MAP.items():
            vals = base[src_feat].to_numpy(dtype=float)
            sec_map = self.sector_stats_[rel_feat]
            mac_map = self.macro_stats_[rel_feat]
            uni_med, uni_scale = self.universe_stats_[rel_feat]

            z_vals = np.empty(len(vals), dtype=float)
            for i in range(len(vals)):
                sec = sectors[i]
                mac = macros[i]
                if sec in sec_map:
                    med, sc = sec_map[sec]
                elif mac in mac_map:
                    med, sc = mac_map[mac]
                else:
                    med, sc = uni_med, uni_scale
                z_vals[i] = (vals[i] - med) / sc
            out[rel_feat] = np.clip(z_vals, -self.z_clip, self.z_clip)

        return out[FEATURE_NAMES]

    def transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame into a 22-column engineered feature DataFrame."""
        base = compute_base_features(X)
        unscaled = self._assemble_unscaled(X, base)
        if self.scale and self.scaler_ is not None:
            scaled_arr = self.scaler_.transform(unscaled.to_numpy(dtype=float))
            return pd.DataFrame(scaled_arr, columns=FEATURE_NAMES, index=X.index)
        return unscaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-compatible transform returning float64 ndarray of shape (n, 22)."""
        return self.transform_df(X).to_numpy(dtype=float)


def apply_stress_shocks(
    record: pd.Series | dict[str, Any],
    rate_shock_bps: float = 0.0,
    pbildt_shock_pct: float = 0.0,
    debt_shock_pct: float = 0.0,
    wc_days_shock: float = 0.0,
    fallback_cost_of_debt: float = 0.1056,
) -> pd.DataFrame:
    """Propagate live underwriting stress shocks cleanly through a company's financials.

    Returns a single-row DataFrame ready for a frozen `CreditFeatureTransformer`.
    All shocks are financially self-consistent and directionally monotonic under stress:
    - Rate shock (+bps) applies to post-shock implied_total_debt.
    - PBILDT shock (%) flows into PBILDT, PBILDT Margin, PAT, PAT Margin, GCA, ROCE, and ICR.
    - Debt shock (%) raises implied debt, incremental interest, and leverage without inflating
      structural franchise Scale (`_scale_tce` remains anchored at pre-shock TCE).
    - Working-capital days shock shifts Operating Cycle directly.
    """
    row = dict(record)

    base_toi = float(row[COL_TOI])
    base_pbildt = float(row[COL_PBILDT])
    base_interest = float(row[COL_INTEREST])
    base_pat = float(row[COL_PAT])
    base_gca = float(row[COL_GCA])
    base_nw = float(row[COL_NETWORTH])
    base_tce = float(row[COL_TCE])
    base_pbildt_margin = float(row[COL_PBILDT_MARGIN])
    base_pat_margin = float(row[COL_PAT_MARGIN])
    base_roce = float(row[COL_ROCE])
    base_ic = float(row[COL_INT_COV])
    base_op_cycle = float(row[COL_OP_CYCLE])

    base_debt = float(row["implied_total_debt"])
    base_term_debt = float(row["implied_term_debt"])
    raw_cod = row.get("implied_cost_of_debt", np.nan)
    eff_cod = (
        float(np.clip(float(raw_cod), 0.06, 0.18))
        if raw_cod is not None and np.isfinite(float(raw_cod)) and float(raw_cod) > 0
        else fallback_cost_of_debt
    )

    # 1. Debt & Interest Shock
    debt_new = max(0.0, base_debt * (1.0 + debt_shock_pct / 100.0))
    delta_debt = debt_new - base_debt
    if base_debt > 0.0:
        term_debt_new = float(np.clip(base_term_debt * (debt_new / base_debt), 0.0, debt_new))
    else:
        term_debt_new = 0.0

    rate_interest_add = debt_new * (rate_shock_bps / 10000.0)
    debt_interest_add = max(0.0, delta_debt) * eff_cod
    interest_new = max(0.0, base_interest + rate_interest_add + debt_interest_add)
    delta_interest = interest_new - base_interest

    # 2. Operating Earnings (PBILDT) Shock
    # Using |base_pbildt| ensures already-loss-making firms worsen rather than improve on negative shocks
    delta_pbildt = abs(base_pbildt) * (pbildt_shock_pct / 100.0)
    pbildt_new = base_pbildt + delta_pbildt
    toi_denom = max(base_toi, 1.0)
    pbildt_margin_new = base_pbildt_margin + (delta_pbildt / toi_denom) * 100.0

    # 3. Cashflow & Profitability Pass-Through (PAT, GCA, ROCE, Interest Coverage)
    delta_cashflow = delta_pbildt - delta_interest
    gca_new = base_gca + delta_cashflow
    pat_new = base_pat + delta_cashflow
    pat_margin_new = base_pat_margin + (delta_cashflow / toi_denom) * 100.0

    tce_new = base_tce + max(0.0, delta_debt)
    if base_tce > 1.0:
        est_ebit_base = (base_roce / 100.0) * base_tce
        roce_new = ((est_ebit_base + delta_pbildt) / max(tce_new, 1.0)) * 100.0
    else:
        roce_new = base_roce + (delta_pbildt / max(abs(base_tce), 10.0)) * 100.0

    adverse_earnings_or_debt = (
        pbildt_shock_pct <= 0.0 and rate_shock_bps >= 0.0 and debt_shock_pct >= 0.0
    )
    if adverse_earnings_or_debt:
        # Prevent negative-numerator / larger-denominator dilution from artificially lifting ROCE
        roce_new = min(base_roce, roce_new)

    ic_new = pbildt_new / max(interest_new, 0.1)
    if adverse_earnings_or_debt:
        ic_new = min(base_ic, ic_new)

    # 4. Working Capital Cycle Shock
    op_cycle_new = base_op_cycle + wc_days_shock

    # Base ratio caps prevent negative-numerator dilution when debt/TCE rises under stress
    base_nw_to_tce = (
        float(np.clip(base_nw / max(base_tce, 1.0), -1.0, 1.2))
        if base_tce > 1.0
        else (-1.0 if base_nw < 0.0 else 0.0)
    )
    base_debt_denom = max(base_debt, 1.0)
    base_gca_to_debt = float(np.clip(base_gca / base_debt_denom, -2.0, 5.0))
    base_pbildt_to_debt = float(np.clip(base_pbildt / base_debt_denom, -2.0, 5.0))

    row[COL_PBILDT] = pbildt_new
    row[COL_INTEREST] = interest_new
    row[COL_PAT] = pat_new
    row[COL_GCA] = gca_new
    row[COL_TCE] = tce_new
    row[COL_PBILDT_MARGIN] = pbildt_margin_new
    row[COL_PAT_MARGIN] = pat_margin_new
    row[COL_ROCE] = roce_new
    row[COL_INT_COV] = ic_new
    row[COL_OP_CYCLE] = op_cycle_new
    row["implied_total_debt"] = debt_new
    row["implied_term_debt"] = term_debt_new
    row["implied_cost_of_debt"] = interest_new / debt_new if debt_new > 0.0 else np.nan
    # Hold structural Scale TCE at pre-shock level under debt increases
    row["_scale_tce"] = min(base_tce, tce_new) if debt_shock_pct >= 0.0 else tce_new
    if adverse_earnings_or_debt:
        row["_max_nw_to_tce"] = base_nw_to_tce
        row["_max_gca_to_debt"] = base_gca_to_debt
        row["_max_pbildt_to_debt"] = base_pbildt_to_debt

    return pd.DataFrame([row])
