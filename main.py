"""Project 3: Corporate Credit Shadow-Rating & Underwriting Workbench (Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.charts import (
    plot_calibration_curve,
    plot_confusion_matrix_7tier,
    plot_credit_ladder_medians,
    plot_ig_cliff_comparison,
    plot_peer_percentiles,
    plot_rating_distribution,
    plot_rating_gap_scatter,
    plot_scorecard_waterfall,
    plot_scorecard_weights,
    plot_sector_breakdown,
)
from src.data import (
    BROAD_TIERS,
    COL_COMPANY,
    COL_CURR_RATIO,
    COL_GCA,
    COL_GEARING,
    COL_INC_GROWTH,
    COL_INT_COV,
    COL_INTEREST,
    COL_LT_DE,
    COL_MACRO,
    COL_NETWORTH,
    COL_OP_CYCLE,
    COL_PAT,
    COL_PAT_GROWTH,
    COL_PAT_MARGIN,
    COL_PBILDT,
    COL_PBILDT_MARGIN,
    COL_QUICK_RATIO,
    COL_RATING,
    COL_ROCE,
    COL_SECTOR,
    COL_TCE,
    COL_TD_GCA,
    COL_TERM_GCA,
    COL_TOI,
    MIN_PEER_GROUP_SIZE,
    notch_to_rating,
)
from src.features import apply_stress_shocks
from src.modeling import score_company_record

ARTIFACTS_DIR = Path("artifacts")


@st.cache_resource
def load_model_bundle(artifacts_dir: str = str(ARTIFACTS_DIR)) -> dict[str, Any]:
    """Load pre-fitted production model bundle once per container lifecycle."""
    return joblib.load(Path(artifacts_dir) / "model_bundle.joblib")


@st.cache_data
def load_data_artifacts(
    artifacts_dir: str = str(ARTIFACTS_DIR),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load precomputed CSV/JSON artifacts in < 1s without touching the raw workbook."""
    base = Path(artifacts_dir)
    df_clean = pd.read_csv(base / "cleaned_universe.csv")
    df_quarantined = pd.read_csv(base / "quarantined_rows.csv")
    df_oof = pd.read_csv(base / "oof_predictions.csv")
    with open(base / "validation_metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    return df_clean, df_quarantined, df_oof, metrics


def build_credit_ladder_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute median financial profile stepping from AAA down to C/D."""
    rows = []
    for tier in BROAD_TIERS:
        sub = df_clean[df_clean["broad_tier"] == tier]
        if sub.empty:
            continue
        gca_to_debt = np.clip(
            sub[COL_GCA] / np.maximum(sub["implied_total_debt"], 1.0), -2.0, 5.0
        )
        rows.append(
            {
                "Broad Tier": tier,
                "Count": int(len(sub)),
                "Share (%)": round(len(sub) / len(df_clean) * 100.0, 1),
                "Median TOI (Rs. Cr)": round(float(sub[COL_TOI].median()), 1),
                "Median Networth (Rs. Cr)": round(float(sub[COL_NETWORTH].median()), 1),
                "Median Implied Debt (Rs. Cr)": round(
                    float(sub["implied_total_debt"].median()), 1
                ),
                "Median PBILDT Margin (%)": round(
                    float(sub[COL_PBILDT_MARGIN].median()), 2
                ),
                "Median ROCE (%)": round(float(sub[COL_ROCE].median()), 2),
                "Median GCA / Debt (x)": round(float(np.median(gca_to_debt)), 2),
                "Median Interest Cov. (x)": round(float(sub[COL_INT_COV].median()), 2),
                "Median Op. Cycle (days)": round(float(sub[COL_OP_CYCLE].median()), 0),
            }
        )
    return pd.DataFrame(rows)


def compute_peer_percentiles(
    record: pd.Series | dict[str, Any], df_clean: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Compute credit-oriented peer percentiles using Sector -> Macro -> Universe pooling."""
    sec = str(record[COL_SECTOR])
    mac = str(record[COL_MACRO])
    sec_peers = df_clean[df_clean[COL_SECTOR] == sec]
    if len(sec_peers) >= MIN_PEER_GROUP_SIZE:
        peers = sec_peers
        label = f"{sec} (n={len(peers)})"
    else:
        mac_peers = df_clean[df_clean[COL_MACRO] == mac]
        if len(mac_peers) >= MIN_PEER_GROUP_SIZE:
            peers = mac_peers
            label = f"{mac} Macro Pool (n={len(peers)})"
        else:
            peers = df_clean
            label = f"Universe Fallback (n={len(peers)})"

    comp_gca_debt = float(
        np.clip(float(record[COL_GCA]) / max(float(record["implied_total_debt"]), 1.0), -2.0, 5.0)
    )
    peer_gca_debt = np.clip(
        peers[COL_GCA].to_numpy(dtype=float)
        / np.maximum(peers["implied_total_debt"].to_numpy(dtype=float), 1.0),
        -2.0,
        5.0,
    )

    specs = [
        ("Total Operating Income", float(record[COL_TOI]), peers[COL_TOI].to_numpy(dtype=float), False, "{:,.0f} Cr"),
        ("PBILDT Margin", float(record[COL_PBILDT_MARGIN]), peers[COL_PBILDT_MARGIN].to_numpy(dtype=float), False, "{:.1f}%"),
        ("ROCE", float(record[COL_ROCE]), peers[COL_ROCE].to_numpy(dtype=float), False, "{:.1f}%"),
        ("GCA / Implied Debt", comp_gca_debt, peer_gca_debt, False, "{:.2f}x"),
        ("Interest Coverage", float(record[COL_INT_COV]), peers[COL_INT_COV].to_numpy(dtype=float), False, "{:.2f}x"),
        ("Current Ratio", float(record[COL_CURR_RATIO]), peers[COL_CURR_RATIO].to_numpy(dtype=float), False, "{:.2f}x"),
        ("Operating Cycle (Lower=Better)", float(record[COL_OP_CYCLE]), peers[COL_OP_CYCLE].to_numpy(dtype=float), True, "{:.0f} d"),
    ]
    rows = []
    for name, val, arr, invert, fmt in specs:
        pct = float(np.mean(arr >= val) * 100.0) if invert else float(np.mean(arr <= val) * 100.0)
        rows.append(
            {
                "Metric": name,
                "Company Value": fmt.format(val),
                "Percentile": round(pct, 1),
            }
        )
    return pd.DataFrame(rows), label


def build_custom_company_record(
    company_name: str,
    sector: str,
    macro: str,
    toi: float,
    pbildt_margin: float,
    pat_margin: float,
    roce: float,
    interest: float,
    gca: float,
    networth: float,
    total_debt: float,
    term_debt: float,
    current_ratio: float,
    quick_ratio: float,
    op_cycle: float,
    inc_growth: float,
    pat_growth: float,
) -> dict[str, Any]:
    """Build a schema-compatible financial record from custom private-company inputs."""
    pbildt = toi * (pbildt_margin / 100.0)
    pat = toi * (pat_margin / 100.0)
    tce = networth + total_debt
    int_cov = pbildt / max(interest, 0.1)
    safe_nw = networth if abs(networth) >= 1.0 else (1.0 if networth >= 0 else -1.0)
    safe_gca = gca if abs(gca) >= 1.0 else (1.0 if gca >= 0 else -1.0)
    term_debt_clipped = min(max(term_debt, 0.0), max(total_debt, 0.0))

    return {
        COL_COMPANY: company_name.strip() or "Custom Private Corporate",
        COL_MACRO: macro,
        COL_SECTOR: sector,
        COL_TOI: float(toi),
        COL_PBILDT: float(pbildt),
        COL_INTEREST: float(interest),
        COL_PAT: float(pat),
        COL_GCA: float(gca),
        COL_NETWORTH: float(networth),
        COL_TCE: float(tce),
        COL_INC_GROWTH: float(inc_growth),
        COL_PAT_GROWTH: float(pat_growth),
        COL_PBILDT_MARGIN: float(pbildt_margin),
        COL_PAT_MARGIN: float(pat_margin),
        COL_ROCE: float(roce),
        COL_LT_DE: float(term_debt_clipped / safe_nw),
        COL_GEARING: float(total_debt / safe_nw),
        COL_INT_COV: float(int_cov),
        COL_TERM_GCA: float(term_debt_clipped / safe_gca),
        COL_TD_GCA: float(total_debt / safe_gca),
        COL_CURR_RATIO: float(current_ratio),
        COL_QUICK_RATIO: float(quick_ratio),
        COL_OP_CYCLE: float(op_cycle),
        "implied_total_debt": float(max(total_debt, 0.0)),
        "implied_term_debt": float(term_debt_clipped),
        "implied_cost_of_debt": (
            float(interest / total_debt) if total_debt > 0.0 else np.nan
        ),
    }


def _render_tab_universe(
    df_clean: pd.DataFrame, df_quarantined: pd.DataFrame
) -> None:
    st.subheader("Universe & Credit Ladder")
    ig_pct = df_clean["is_ig"].mean() * 100.0
    neg_nw = int((df_clean[COL_NETWORTH] < 0).sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Valid Corporates", f"{len(df_clean):,}", f"{len(df_quarantined)} quarantined")
    c2.metric("Investment Grade Share", f"{ig_pct:.1f}%", f"{int(df_clean['is_ig'].sum())} IG / {int((1 - df_clean['is_ig']).sum())} HY")
    c3.metric("Median TOI", f"Rs. {df_clean[COL_TOI].median():,.0f} Cr")
    c4.metric("Median Implied Debt", f"Rs. {df_clean['implied_total_debt'].median():,.0f} Cr")
    c5.metric("Negative Networth Firms", f"{neg_nw}", "Debt recovered cleanly")

    st.plotly_chart(plot_rating_distribution(df_clean), width="stretch")

    col_left, col_right = st.columns([1.15, 1.0])
    with col_left:
        group_choice = st.radio(
            "Group Breakdown By",
            [COL_MACRO, COL_SECTOR],
            horizontal=True,
            key="tab1_group_choice",
        )
        st.plotly_chart(
            plot_sector_breakdown(df_clean, group_col=group_choice),
            width="stretch",
        )
    with col_right:
        ladder_df = build_credit_ladder_table(df_clean)
        st.plotly_chart(plot_credit_ladder_medians(ladder_df), width="stretch")

    st.markdown("#### Median Financial Ratios Across the Credit Ladder (AAA to C/D)")
    st.dataframe(ladder_df, width="stretch", hide_index=True)

    if not df_quarantined.empty:
        with st.expander(f"Quarantined Records ({len(df_quarantined)} row with Months != 12)"):
            st.dataframe(
                df_quarantined[[COL_COMPANY, COL_SECTOR, "Period", "Months", COL_RATING, "quarantine_reason"]],
                width="stretch",
                hide_index=True,
            )


def _render_tab_screener(df_clean: pd.DataFrame, df_oof: pd.DataFrame) -> None:
    st.subheader("Strict Out-of-Fold Rating-Gap Screener")
    st.caption(
        "All shadow ratings and gap flags come strictly from 5-Fold Stratified Out-of-Fold predictions. "
        "Positive Gap (Agency Notch - Shadow Notch >= +2) signals Rising Star upgrade potential; "
        "Negative Gap (<= -2) signals Fallen Angel downgrade risk."
    )

    merged = df_oof.merge(
        df_clean[
            [
                COL_COMPANY,
                COL_TOI,
                COL_PBILDT_MARGIN,
                COL_ROCE,
                COL_INT_COV,
                "implied_total_debt",
            ]
        ],
        on=COL_COMPANY,
        how="left",
    )

    f1, f2, f3, f4 = st.columns(4)
    macros = ["All"] + sorted(merged[COL_MACRO].unique().tolist())
    sel_macro = f1.selectbox("NSE Macro Industry", macros, key="scr_macro")

    sec_pool = (
        merged[merged[COL_MACRO] == sel_macro][COL_SECTOR].unique().tolist()
        if sel_macro != "All"
        else merged[COL_SECTOR].unique().tolist()
    )
    sel_sector = f2.selectbox("NSE Sector", ["All"] + sorted(sec_pool), key="scr_sector")
    sel_tiers = f3.multiselect(
        "Agency Broad Tier",
        list(BROAD_TIERS),
        default=list(BROAD_TIERS),
        key="scr_tiers",
    )
    ig_filter = f4.selectbox(
        "IG-Cliff Filter",
        ["All Corporates", "Agency IG Only", "Agency HY Only", "IG-Cliff Watch (BBB- / BB+)"],
        key="scr_ig",
    )

    filt = merged.copy()
    if sel_macro != "All":
        filt = filt[filt[COL_MACRO] == sel_macro]
    if sel_sector != "All":
        filt = filt[filt[COL_SECTOR] == sel_sector]
    if sel_tiers:
        filt = filt[filt["agency_tier"].isin(sel_tiers)]
    if ig_filter == "Agency IG Only":
        filt = filt[filt["is_ig"] == 1]
    elif ig_filter == "Agency HY Only":
        filt = filt[filt["is_ig"] == 0]
    elif ig_filter == "IG-Cliff Watch (BBB- / BB+)":
        filt = filt[filt["agency_rating"].isin(["BBB-", "BB+"])]

    rising = filt[filt["rating_gap_notches"] >= 2.0].sort_values(
        "rating_gap_notches", ascending=False
    )
    fallen = filt[filt["rating_gap_notches"] <= -2.0].sort_values(
        "rating_gap_notches", ascending=True
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filtered Corporates", f"{len(filt):,}")
    k2.metric("Rising Stars (>= +2 Notches)", f"{len(rising):,}")
    k3.metric("Fallen Angels (<= -2 Notches)", f"{len(fallen):,}")
    k4.metric("In-Line (< 2 Notches)", f"{len(filt) - len(rising) - len(fallen):,}")

    st.plotly_chart(plot_rating_gap_scatter(filt), width="stretch")

    display_cols = [
        COL_COMPANY,
        COL_SECTOR,
        "agency_rating",
        "shadow_rating",
        "shadow_notch",
        "rating_gap_notches",
        "p_ig_logistic",
        COL_TOI,
        COL_PBILDT_MARGIN,
        "implied_total_debt",
    ]
    rename_map = {
        "agency_rating": "Agency",
        "shadow_rating": "OOF Shadow",
        "shadow_notch": "Shadow Notch",
        "rating_gap_notches": "Notch Gap",
        "p_ig_logistic": "P(IG)",
        COL_TOI: "TOI (Cr)",
        COL_PBILDT_MARGIN: "PBILDT Margin (%)",
        "implied_total_debt": "Implied Debt (Cr)",
    }

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### Rising Stars — Upgrade Watch (Gap >= +2 Notches)")
        st.dataframe(
            rising[display_cols].rename(columns=rename_map),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Export Rising Stars CSV",
            data=rising[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="rising_stars_oof.csv",
            mime="text/csv",
            width="stretch",
        )
    with t_col2:
        st.markdown("#### Fallen Angels — Downgrade Watch (Gap <= -2 Notches)")
        st.dataframe(
            fallen[display_cols].rename(columns=rename_map),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Export Fallen Angels CSV",
            data=fallen[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="fallen_angels_oof.csv",
            mime="text/csv",
            width="stretch",
        )


def _render_tab_underwriting(
    df_clean: pd.DataFrame, df_oof: pd.DataFrame, bundle: dict[str, Any]
) -> None:
    st.subheader("Underwriting & Live Stress-Test Simulator")
    mode = st.radio(
        "Entity Input Mode",
        ["Select Listed Corporate (1,224 Universe)", "Custom Private Company Financials"],
        horizontal=True,
        key="uw_mode",
    )

    agency_label: str | None = None
    oof_shadow_note: str | None = None

    if mode.startswith("Select"):
        companies = sorted(df_clean[COL_COMPANY].tolist())
        sel_company = st.selectbox("Select Corporate", companies, index=0, key="uw_company")
        base_row = df_clean[df_clean[COL_COMPANY] == sel_company].iloc[0].to_dict()
        oof_row = df_oof[df_oof[COL_COMPANY] == sel_company].iloc[0]
        agency_label = f"{base_row[COL_RATING]} (Notch {int(base_row['notch_num'])})"
        oof_shadow_note = f"{oof_row['shadow_rating']} ({oof_row['shadow_notch']:.2f})"
    else:
        st.markdown("##### Enter Private Company Financials (Rs. Crore / Ratios)")
        sector_to_macro: dict[str, str] = bundle["sector_to_macro"]
        sectors = sorted(sector_to_macro.keys())
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        cust_name = r1c1.text_input("Company Name", "Acme Industrial Pvt Ltd")
        cust_sector = r1c2.selectbox("NSE Sector", sectors, index=0)
        cust_toi = r1c3.number_input("Total Operating Income (Cr)", min_value=1.0, value=1500.0, step=50.0)
        cust_nw = r1c4.number_input("Networth (Cr)", value=650.0, step=25.0)

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        cust_debt = r2c1.number_input("Total Debt (Cr)", min_value=0.0, value=450.0, step=25.0)
        cust_term = r2c2.number_input("Term Debt (Cr)", min_value=0.0, value=250.0, step=25.0)
        cust_int = r2c3.number_input("Interest Expense (Cr)", min_value=0.0, value=45.0, step=5.0)
        cust_gca = r2c4.number_input("Gross Cash Accruals (Cr)", value=140.0, step=10.0)

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        cust_pm = r3c1.number_input("PBILDT Margin (%)", value=14.0, step=0.5)
        cust_patm = r3c2.number_input("PAT Margin (%)", value=6.5, step=0.5)
        cust_roce = r3c3.number_input("ROCE (%)", value=15.0, step=0.5)
        cust_oc = r3c4.number_input("Operating Cycle (days)", value=75.0, step=5.0)

        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        cust_cr = r4c1.number_input("Current Ratio (x)", min_value=0.0, value=1.6, step=0.1)
        cust_qr = r4c2.number_input("Quick Ratio (x)", min_value=0.0, value=1.1, step=0.1)
        cust_ig = r4c3.number_input("Total Income Growth (%)", value=11.0, step=1.0)
        cust_pg = r4c4.number_input("PAT Growth (%)", value=9.0, step=1.0)

        base_row = build_custom_company_record(
            company_name=cust_name,
            sector=cust_sector,
            macro=sector_to_macro[cust_sector],
            toi=cust_toi,
            pbildt_margin=cust_pm,
            pat_margin=cust_patm,
            roce=cust_roce,
            interest=cust_int,
            gca=cust_gca,
            networth=cust_nw,
            total_debt=cust_debt,
            term_debt=cust_term,
            current_ratio=cust_cr,
            quick_ratio=cust_qr,
            op_cycle=cust_oc,
            inc_growth=cust_ig,
            pat_growth=cust_pg,
        )

    # Live Stress-Test Sliders
    st.markdown("#### Live Underwriting Stress Shocks")
    s1, s2, s3, s4 = st.columns(4)
    rate_bps = s1.slider("Interest Rate Shock (+bps)", 0, 500, 0, step=25)
    pbildt_pct = s2.slider("PBILDT Shock (%)", -60, 0, 0, step=5)
    debt_pct = s3.slider("Debt Shock (%)", 0, 100, 0, step=5)
    wc_days = s4.slider("Working Capital Days (+days)", 0, 120, 0, step=10)

    base_df = pd.DataFrame([base_row])
    stressed_df = apply_stress_shocks(
        base_row,
        rate_shock_bps=rate_bps,
        pbildt_shock_pct=pbildt_pct,
        debt_shock_pct=debt_pct,
        wc_days_shock=wc_days,
        fallback_cost_of_debt=bundle["median_cost_of_debt"],
    )

    base_score = score_company_record(base_df, bundle)
    stressed_score = score_company_record(stressed_df, bundle)
    notch_migration = stressed_score["expected_notch"] - base_score["expected_notch"]
    ig_delta_pp = (stressed_score["p_ig"] - base_score["p_ig"]) * 100.0

    # Headline KPI Summary
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Agency Rating" if agency_label else "Entity Mode",
        agency_label if agency_label else "Private Underwriting",
        f"OOF Shadow: {oof_shadow_note}" if oof_shadow_note else base_row[COL_SECTOR],
    )
    m2.metric(
        "Base Shadow Rating",
        f"{base_score['shadow_rating']} ({base_score['expected_notch']:.2f})",
        f"Tier: {base_score['shadow_tier']}",
    )
    m3.metric(
        "Stressed Shadow Rating",
        f"{stressed_score['shadow_rating']} ({stressed_score['expected_notch']:.2f})",
        f"{notch_migration:+.2f} notches",
        delta_color="inverse",
    )
    m4.metric(
        "Base -> Stressed P(IG)",
        f"{stressed_score['p_ig'] * 100:.1f}%",
        f"{ig_delta_pp:+.1f} pp (from {base_score['p_ig'] * 100:.1f}%)",
    )
    m5.metric(
        "Implied Debt (Base -> Stress)",
        f"Rs. {stressed_df['implied_total_debt'].iloc[0]:,.0f} Cr",
        f"Int: Rs. {base_row[COL_INTEREST]:,.1f} -> {stressed_df[COL_INTEREST].iloc[0]:,.1f} Cr",
    )

    # Row 1: Peer Percentiles & IG-Cliff Gauge
    p_col, g_col = st.columns([1.4, 1.0])
    with p_col:
        pct_df, peer_label = compute_peer_percentiles(base_row, df_clean)
        st.plotly_chart(plot_peer_percentiles(pct_df, peer_label), width="stretch")
    with g_col:
        st.plotly_chart(
            plot_ig_cliff_comparison(base_score["p_ig"], stressed_score["p_ig"]),
            width="stretch",
        )

    # Row 2: Linear Scorecard Waterfall (Base vs Stressed)
    any_shock = any([rate_bps != 0, pbildt_pct != 0, debt_pct != 0, wc_days != 0])
    if any_shock:
        w1, w2 = st.columns(2)
        with w1:
            st.plotly_chart(
                plot_scorecard_waterfall(
                    base_score["baseline_notch"],
                    base_score["pillar_attributions"],
                    base_score["expected_notch"],
                    title=f"Base Case Scorecard Waterfall ({base_score['shadow_rating']})",
                ),
                width="stretch",
            )
        with w2:
            st.plotly_chart(
                plot_scorecard_waterfall(
                    stressed_score["baseline_notch"],
                    stressed_score["pillar_attributions"],
                    stressed_score["expected_notch"],
                    title=f"Stressed Case Scorecard Waterfall ({stressed_score['shadow_rating']})",
                ),
                width="stretch",
            )
    else:
        st.plotly_chart(
            plot_scorecard_waterfall(
                base_score["baseline_notch"],
                base_score["pillar_attributions"],
                base_score["expected_notch"],
                title=f"Linear Scorecard Notch Driver Waterfall — {base_row[COL_COMPANY]}",
            ),
            width="stretch",
        )

    # Financial Snapshot Comparison (Base vs Stressed)
    with st.expander("Base vs. Stressed Financial Metrics Audit"):
        comp_table = pd.DataFrame(
            [
                ("TOI (Rs. Cr)", base_row[COL_TOI], stressed_df[COL_TOI].iloc[0]),
                ("PBILDT (Rs. Cr)", base_row[COL_PBILDT], stressed_df[COL_PBILDT].iloc[0]),
                ("Interest (Rs. Cr)", base_row[COL_INTEREST], stressed_df[COL_INTEREST].iloc[0]),
                ("GCA (Rs. Cr)", base_row[COL_GCA], stressed_df[COL_GCA].iloc[0]),
                ("Implied Total Debt (Rs. Cr)", base_row["implied_total_debt"], stressed_df["implied_total_debt"].iloc[0]),
                ("PBILDT Margin (%)", base_row[COL_PBILDT_MARGIN], stressed_df[COL_PBILDT_MARGIN].iloc[0]),
                ("ROCE (%)", base_row[COL_ROCE], stressed_df[COL_ROCE].iloc[0]),
                ("Interest Coverage (x)", base_row[COL_INT_COV], stressed_df[COL_INT_COV].iloc[0]),
                ("Operating Cycle (days)", base_row[COL_OP_CYCLE], stressed_df[COL_OP_CYCLE].iloc[0]),
            ],
            columns=["Metric", "Base Case", "Stressed Case"],
        )
        comp_table["Delta"] = comp_table["Stressed Case"] - comp_table["Base Case"]
        st.dataframe(comp_table, width="stretch", hide_index=True)


def _render_tab_validation(
    metrics: dict[str, Any], bundle: dict[str, Any]
) -> None:
    st.subheader("Model Validation & Scorecard Audit")
    st.caption(
        "Dual Cross-Validation: 5-Fold Stratified CV (in-universe shadow accuracy) vs. "
        "5-Fold GroupKFold by NSE Macro Industry (out-of-sector generalization with zero leakage)."
    )

    # Dual CV Summary Table
    strat = metrics["stratified_5fold"]
    grp = metrics["group_kfold_macro"]
    cv_rows = []
    specs = [
        ("OrdinalLogistic (Primary)", "Stratified 5-Fold", strat["ordinal_logistic"]),
        ("OrdinalLogistic (Primary)", "GroupKFold (Macro)", grp["ordinal_logistic"]),
        ("HistGradientBoosting (Benchmark)", "Stratified 5-Fold", strat["hist_gradient_boosting"]),
        ("HistGradientBoosting (Benchmark)", "GroupKFold (Macro)", grp["hist_gradient_boosting"]),
        ("IG-Cliff Calibrated Logistic", "Stratified 5-Fold", strat["ig_cliff_logistic"]),
        ("IG-Cliff Calibrated Logistic", "GroupKFold (Macro)", grp["ig_cliff_logistic"]),
        ("IG-Cliff Calibrated HGB", "Stratified 5-Fold", strat["ig_cliff_hgb"]),
        ("IG-Cliff Calibrated HGB", "GroupKFold (Macro)", grp["ig_cliff_hgb"]),
    ]
    for model_name, scheme, m in specs:
        cv_rows.append(
            {
                "Model": model_name,
                "CV Scheme": scheme,
                "Notch MAE": round(m["notch_mae"], 3) if "notch_mae" in m else None,
                "Within ±1 Notch (%)": round(m["within_1_notch_pct"], 1) if "within_1_notch_pct" in m else None,
                "Within ±2 Notch (%)": round(m["within_2_notch_pct"], 1) if "within_2_notch_pct" in m else None,
                "Broad-Tier Acc (%)": round(m["broad_tier_acc_pct"], 1) if "broad_tier_acc_pct" in m else None,
                "QWK (18-Notch)": round(m["qwk_18"], 3) if "qwk_18" in m else None,
                "IG ROC-AUC": round(m["ig_roc_auc"], 3),
                "IG PR-AUC": round(m["ig_pr_auc"], 3),
                "IG Brier": round(m["ig_brier"], 4),
            }
        )
    st.dataframe(pd.DataFrame(cv_rows), width="stretch", hide_index=True)

    cv_view = st.radio(
        "Inspect Confusion Matrix & Calibration From",
        ["Stratified 5-Fold OOF", "Out-of-Sector GroupKFold OOF"],
        horizontal=True,
        key="val_cv_view",
    )
    is_strat = cv_view.startswith("Stratified")
    cm = (
        metrics["confusion_matrix_7tier_stratified"]
        if is_strat
        else metrics["confusion_matrix_7tier_group"]
    )
    cal = (
        metrics["calibration_stratified"]
        if is_strat
        else metrics["calibration_group"]
    )

    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(
            plot_confusion_matrix_7tier(
                cm,
                labels=metrics["confusion_matrix_7tier_labels"],
                title=f"7-Tier Confusion Matrix ({cv_view})",
            ),
            width="stretch",
        )
    with v2:
        st.plotly_chart(plot_calibration_curve(cal), width="stretch")

    st.markdown("#### Scorecard Feature Weights Audit")
    weights_df: pd.DataFrame = bundle["scorecard_weights"]
    st.plotly_chart(plot_scorecard_weights(weights_df), width="stretch")
    st.dataframe(
        weights_df[["pillar", "label", "mean_log_odds_weight"]].rename(
            columns={
                "pillar": "Credit Pillar",
                "label": "Feature",
                "mean_log_odds_weight": "Mean Log-Odds Weight (17 Thresholds)",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    """Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Corporate Credit Shadow-Rating & Underwriting Workbench",
        page_icon="🏦",
        layout="wide",
    )

    st.title("Corporate Credit Shadow-Rating & Underwriting Workbench")
    st.caption(
        "Ordinal credit shadow-rating scorecard, strict OOF relative-value screener, "
        "and real-time stress-test simulator across 1,224 listed Indian corporates (FY25)."
    )

    df_clean, df_quarantined, df_oof, metrics = load_data_artifacts()
    bundle = load_model_bundle()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Universe & Credit Ladder",
            "2. Rating-Gap Screener",
            "3. Underwriting & Stress-Test Simulator",
            "4. Model Validation & Scorecard Audit",
        ]
    )
    with tab1:
        _render_tab_universe(df_clean, df_quarantined)
    with tab2:
        _render_tab_screener(df_clean, df_oof)
    with tab3:
        _render_tab_underwriting(df_clean, df_oof, bundle)
    with tab4:
        _render_tab_validation(metrics, bundle)


if __name__ == "__main__":
    main()
