"""Data ingestion, cleaning, quarantine, implied debt reconstruction, and target mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/Dataoffinancialsindustryrating.xlsx")
SHEET_NAME = "OOS Listed 3"
MIN_PEER_GROUP_SIZE = 5

# Canonical 18-notch Indian corporate credit rating scale (1 = strongest, 18 = default)
RATING_LADDER: tuple[str, ...] = (
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "C",
    "D",
)
RATING_TO_NOTCH: dict[str, int] = {r: idx + 1 for idx, r in enumerate(RATING_LADDER)}
NOTCH_TO_RATING: dict[int, str] = {idx + 1: r for idx, r in enumerate(RATING_LADDER)}

BROAD_TIERS: tuple[str, ...] = ("AAA", "AA", "A", "BBB", "BB", "B", "C/D")
RATING_TO_TIER: dict[str, str] = {
    "AAA": "AAA",
    "AA+": "AA",
    "AA": "AA",
    "AA-": "AA",
    "A+": "A",
    "A": "A",
    "A-": "A",
    "BBB+": "BBB",
    "BBB": "BBB",
    "BBB-": "BBB",
    "BB+": "BB",
    "BB": "BB",
    "BB-": "BB",
    "B+": "B",
    "B": "B",
    "B-": "B",
    "C": "C/D",
    "D": "C/D",
}
NOTCH_TO_TIER: dict[int, str] = {
    notch: RATING_TO_TIER[rating] for notch, rating in NOTCH_TO_RATING.items()
}
TIER_TO_NOTCHES: dict[str, list[int]] = {
    tier: [n for n, t in NOTCH_TO_TIER.items() if t == tier] for tier in BROAD_TIERS
}
IG_MAX_NOTCH = 10  # BBB- and above

# Canonical column names after stripping header whitespace
COL_COMPANY = "Company"
COL_MACRO = "NSE Macro Industry"
COL_SECTOR = "NSE Sector"
COL_INDUSTRY = "NSE Industry"
COL_BASIC_IND = "NSE Basic Industry"
COL_STATE = "State"
COL_DATABASE = "Database"
COL_RESULT_TYPE = "Result Type"
COL_PERIOD = "Period"
COL_MONTHS = "Months"
COL_NET_SALES = "Net Sales (Rs. crore)"
COL_TOI = "Total Operating Income (Rs. crore)"
COL_PBILDT = "PBILDT (Rs. crore)"
COL_DEPR = "Depreciation (Rs. crore)"
COL_INTEREST = "Interest (Rs. crore)"
COL_PBT = "PBT (Rs. crore)"
COL_PAT = "PAT (after deferred tax) (Rs. crore)"
COL_GCA = "Gross Cash Accruals (GCA) (Rs. crore)"
COL_EQUITY_CAP = "Equity Share Capital (Rs. crore)"
COL_NETWORTH = "Networth (Rs. crore)"
COL_TCE = "Total Capital Employed (Rs. crore)"
COL_INC_GROWTH = "Growth in Total Income (%)"
COL_PAT_GROWTH = "Growth in PAT (after deferred tax) (%)"
COL_PBILDT_MARGIN = "PBILDT Margin (%)"
COL_PAT_MARGIN = "PAT Margin (%)"
COL_ROCE = "ROCE (%)"
COL_LT_DE = "Long Term Debt Equity Ratio (times)"
COL_GEARING = "Overall gearing ratio (times)"
COL_INT_COV = "Interest coverage (times)"
COL_TERM_GCA = "Term debt/GCA (times)"
COL_TD_GCA = "Total debt/GCA (times)"
COL_CURR_RATIO = "Current ratio (times)"
COL_QUICK_RATIO = "Quick ratio (times)"
COL_COLLECTION_DAYS = "Average collection period (days)"
COL_INVENTORY_DAYS = "Average inventory (days)"
COL_CREDITORS_DAYS = "Average Creditors (days)"
COL_OP_CYCLE = "Operating Cycle (days)"
COL_RATING = "Rating"

NUMERIC_COLUMNS: tuple[str, ...] = (
    COL_MONTHS,
    COL_NET_SALES,
    COL_TOI,
    COL_PBILDT,
    COL_DEPR,
    COL_INTEREST,
    COL_PBT,
    COL_PAT,
    COL_GCA,
    COL_EQUITY_CAP,
    COL_NETWORTH,
    COL_TCE,
    COL_INC_GROWTH,
    COL_PAT_GROWTH,
    COL_PBILDT_MARGIN,
    COL_PAT_MARGIN,
    COL_ROCE,
    COL_LT_DE,
    COL_GEARING,
    COL_INT_COV,
    COL_TERM_GCA,
    COL_TD_GCA,
    COL_CURR_RATIO,
    COL_QUICK_RATIO,
    COL_COLLECTION_DAYS,
    COL_INVENTORY_DAYS,
    COL_CREDITORS_DAYS,
    COL_OP_CYCLE,
)


def notch_to_rating(notch: float) -> str:
    """Convert a continuous or integer notch score into the nearest rating label."""
    idx = int(np.clip(np.rint(notch), 1, len(RATING_LADDER)))
    return NOTCH_TO_RATING[idx]


def notch_to_tier(notch: float) -> str:
    """Convert a continuous or integer notch score into its 7-tier broad band."""
    idx = int(np.clip(np.rint(notch), 1, len(RATING_LADDER)))
    return NOTCH_TO_TIER[idx]


def load_raw_workbook(
    path: Path | str = RAW_DATA_PATH, sheet_name: str = SHEET_NAME
) -> pd.DataFrame:
    """Load the raw Excel sheet via openpyxl without mutating headers yet."""
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def strip_schema_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from headers (e.g. 'Rating ') and all string fields."""
    out = df.copy()
    out.columns = [c.strip() if isinstance(c, str) else c for c in out.columns]
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    return out


def reconstruct_implied_debt(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct implied total debt, term debt, and implied borrowing cost.

    Primary formula `Overall gearing ratio * Networth` recovers positive debt even for
    negative-Networth corporates because `(-gearing) * (-Networth) > 0`. Falls back to
    `Total debt/GCA * GCA` when `|Networth| < 1` or when gearing rounds to 0 while
    `Total debt/GCA > 0`.
    """
    out = df.copy()

    nw = pd.to_numeric(out[COL_NETWORTH], errors="coerce").to_numpy(dtype=float)
    gearing = pd.to_numeric(out[COL_GEARING], errors="coerce").to_numpy(dtype=float)
    gca = pd.to_numeric(out[COL_GCA], errors="coerce").to_numpy(dtype=float)
    td_gca = pd.to_numeric(out[COL_TD_GCA], errors="coerce").to_numpy(dtype=float)
    lt_de = pd.to_numeric(out[COL_LT_DE], errors="coerce").to_numpy(dtype=float)
    term_gca = pd.to_numeric(out[COL_TERM_GCA], errors="coerce").to_numpy(dtype=float)
    interest = pd.to_numeric(out[COL_INTEREST], errors="coerce").to_numpy(dtype=float)

    use_td_fallback = (np.abs(nw) < 1.0) | ((gearing == 0.0) & (td_gca > 0.0))
    raw_total_debt = np.where(use_td_fallback, td_gca * gca, gearing * nw)
    implied_total_debt = np.clip(np.nan_to_num(raw_total_debt, nan=0.0), 0.0, None)

    use_term_fallback = (np.abs(nw) < 1.0) | ((lt_de == 0.0) & (term_gca > 0.0))
    raw_term_debt = np.where(use_term_fallback, term_gca * gca, lt_de * nw)
    # Clip term debt <= total debt to absorb minor 2-decimal rounding discrepancies
    implied_term_debt = np.clip(
        np.nan_to_num(raw_term_debt, nan=0.0), 0.0, implied_total_debt
    )

    implied_cost_of_debt = np.divide(
        interest,
        implied_total_debt,
        out=np.full_like(interest, np.nan, dtype=float),
        where=implied_total_debt > 0.0,
    )

    out["implied_total_debt"] = implied_total_debt
    out["implied_term_debt"] = implied_term_debt
    out["implied_cost_of_debt"] = implied_cost_of_debt
    out["debt_reconstruction_fallback"] = use_td_fallback
    return out


def resolve_peer_groups(
    df: pd.DataFrame, min_group_size: int = MIN_PEER_GROUP_SIZE
) -> pd.Series:
    """Assign each firm its reporting peer group using Sector -> Macro -> Universe pooling."""
    sector_counts = df[COL_SECTOR].value_counts()
    macro_counts = df[COL_MACRO].value_counts()

    def _pick(row: pd.Series) -> str:
        sec = row[COL_SECTOR]
        mac = row[COL_MACRO]
        if sector_counts.get(sec, 0) >= min_group_size:
            return str(sec)
        if macro_counts.get(mac, 0) >= min_group_size:
            return f"{mac} (Macro Pool)"
        return "Universe (Fallback)"

    return df.apply(_pick, axis=1)


def clean_and_quarantine(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw workbook rows, quarantine Months != 12, reconstruct debt, and map targets."""
    df = strip_schema_whitespace(df_raw)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_mask = df[COL_MONTHS] == 12
    quarantined = df.loc[~valid_mask].copy().reset_index(drop=True)
    quarantined["quarantine_reason"] = "Non-12-month reporting period (Months != 12)"

    clean = df.loc[valid_mask].copy().reset_index(drop=True)
    clean = reconstruct_implied_debt(clean)

    # Validate all rating strings map cleanly to the 18-notch ladder
    unknown_ratings = set(clean[COL_RATING].dropna().unique()) - set(RATING_TO_NOTCH)
    if unknown_ratings:
        raise ValueError(f"Unmapped rating labels encountered: {sorted(unknown_ratings)}")

    clean["notch_num"] = clean[COL_RATING].map(RATING_TO_NOTCH).astype(int)
    clean["broad_tier"] = clean[COL_RATING].map(RATING_TO_TIER)
    clean["is_ig"] = (clean["notch_num"] <= IG_MAX_NOTCH).astype(int)
    clean["peer_group"] = resolve_peer_groups(clean, MIN_PEER_GROUP_SIZE)

    return clean, quarantined


def load_cleaned_dataset(
    path: Path | str = RAW_DATA_PATH, sheet_name: str = SHEET_NAME
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: read raw Excel workbook and return (clean_df, quarantined_df)."""
    raw = load_raw_workbook(path=path, sheet_name=sheet_name)
    return clean_and_quarantine(raw)
