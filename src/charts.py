"""Reusable Plotly figure builders for the 4-tab Credit Rating Workbench."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import (
    BROAD_TIERS,
    COL_MACRO,
    COL_RATING,
    COL_SECTOR,
    IG_MAX_NOTCH,
    RATING_LADDER,
)

PALETTE = {
    "ig": "#1B6B93",
    "hy": "#D9534F",
    "rising": "#1E9E6B",
    "fallen": "#D94343",
    "inline": "#5C768D",
    "accent": "#F0A500",
    "neutral": "#6C757D",
    "navy": "#102A43",
}

PILLAR_COLORS = {
    "Scale": "#1B6B93",
    "Profitability": "#2A9D8F",
    "Solvency & Coverage": "#264653",
    "Distress Flags": "#D94343",
    "Liquidity & Working Capital": "#E9A03B",
    "Growth": "#6C63FF",
    "Sector-Relative": "#8D5B4C",
}


def _apply_clean_layout(fig: go.Figure, height: int = 400, **kwargs) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=45, r=25, t=50, b=45),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        **kwargs,
    )
    return fig


def plot_rating_distribution(df_clean: pd.DataFrame) -> go.Figure:
    """18-notch rating distribution bar chart with IG / HY cliff boundary."""
    counts = (
        df_clean[COL_RATING]
        .value_counts()
        .reindex(RATING_LADDER, fill_value=0)
    )
    colors = [
        PALETTE["ig"] if (i + 1) <= IG_MAX_NOTCH else PALETTE["hy"]
        for i in range(len(RATING_LADDER))
    ]
    fig = go.Figure(
        go.Bar(
            x=list(RATING_LADDER),
            y=counts.values,
            marker_color=colors,
            text=counts.values,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Corporates: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=9.5,
        line_dash="dash",
        line_color=PALETTE["accent"],
        line_width=2,
        annotation_text="IG / HY Cliff (BBB- | BB+)",
        annotation_position="top right",
    )
    return _apply_clean_layout(
        fig,
        height=390,
        title="Corporate Universe Distribution Across 18 Credit Notches (Blue = IG, Red = HY)",
        xaxis_title="Agency Rating",
        yaxis_title="Number of Corporates",
    )


def plot_sector_breakdown(
    df_clean: pd.DataFrame, group_col: str = COL_MACRO
) -> go.Figure:
    """Horizontal stacked IG vs HY corporate count breakdown by Macro Industry or Sector."""
    grouped = (
        df_clean.groupby([group_col, "is_ig"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 0], fill_value=0)
    )
    grouped["total"] = grouped[1] + grouped[0]
    grouped = grouped.sort_values("total", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=grouped.index.tolist(),
            x=grouped[1].values,
            name="Investment Grade (AAA to BBB-)",
            orientation="h",
            marker_color=PALETTE["ig"],
        )
    )
    fig.add_trace(
        go.Bar(
            y=grouped.index.tolist(),
            x=grouped[0].values,
            name="High Yield (BB+ to D)",
            orientation="h",
            marker_color=PALETTE["hy"],
        )
    )
    height = max(380, min(680, 28 * len(grouped) + 100))
    return _apply_clean_layout(
        fig,
        height=height,
        barmode="stack",
        title=f"IG vs. HY Corporate Count by {group_col}",
        xaxis_title="Number of Corporates",
        yaxis_title="",
    )


def plot_credit_ladder_medians(ladder_df: pd.DataFrame) -> go.Figure:
    """2x2 Credit Ladder subplot showing median fundamentals from AAA down to C/D."""
    tiers = ladder_df["Broad Tier"].tolist()
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Median TOI (Rs. Cr)",
            "Median PBILDT Margin (%)",
            "Median GCA / Implied Debt (x)",
            "Median Interest Coverage (x)",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.1,
    )
    specs = [
        (1, 1, "Median TOI (Rs. Cr)", PALETTE["navy"]),
        (1, 2, "Median PBILDT Margin (%)", PALETTE["ig"]),
        (2, 1, "Median GCA / Debt (x)", PALETTE["rising"]),
        (2, 2, "Median Interest Cov. (x)", PALETTE["accent"]),
    ]
    for r, c, col, color in specs:
        fig.add_trace(
            go.Bar(
                x=tiers,
                y=ladder_df[col].values,
                marker_color=color,
                showlegend=False,
                hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.2f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )
    return _apply_clean_layout(
        fig,
        height=480,
        title="Credit Ladder: Median Financial Profile from AAA to C/D",
    )


def plot_rating_gap_scatter(df_screener: pd.DataFrame) -> go.Figure:
    """Strict OOF Agency Notch vs. Shadow Notch scatter with ±2 notch watch bands."""
    fig = go.Figure()

    # Reference parity line and ±2 notch watch corridors
    x_line = np.array([1.0, 18.0])
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=x_line,
            mode="lines",
            line=dict(color="#4A5568", width=1.5, dash="solid"),
            name="Parity (Shadow = Agency)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=np.clip(x_line - 2.0, 1.0, 18.0),
            mode="lines",
            line=dict(color=PALETTE["rising"], width=1.2, dash="dash"),
            name="Rising Star Threshold (+2 Notches)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=np.clip(x_line + 2.0, 1.0, 18.0),
            mode="lines",
            line=dict(color=PALETTE["fallen"], width=1.2, dash="dash"),
            name="Fallen Angel Threshold (-2 Notches)",
            hoverinfo="skip",
        )
    )

    color_map = {
        "Rising Star (Upgrade Watch)": PALETTE["rising"],
        "In-Line": PALETTE["inline"],
        "Fallen Angel (Downgrade Watch)": PALETTE["fallen"],
    }
    for status, color in color_map.items():
        sub = df_screener[df_screener["watch_status"] == status]
        if sub.empty:
            continue
        # Small deterministic x-jitter so overlapping integer agency notches stay readable
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        customdata = np.column_stack(
            [
                sub["Company"],
                sub[COL_SECTOR],
                sub["agency_rating"],
                sub["shadow_rating"],
                sub["rating_gap_notches"],
                sub["p_ig_logistic"] * 100.0,
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=sub["agency_notch"].to_numpy(dtype=float) + jitter,
                y=sub["shadow_notch"].to_numpy(dtype=float),
                mode="markers",
                name=f"{status} ({len(sub)})",
                marker=dict(
                    color=color,
                    size=7,
                    opacity=0.78,
                    line=dict(width=0.5, color="white"),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Sector: %{customdata[1]}<br>"
                    "Agency: %{customdata[2]} | Shadow: %{customdata[3]} (%{y:.2f})<br>"
                    "Gap: %{customdata[4]:+.2f} notches | P(IG): %{customdata[5]:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    tick_vals = list(range(1, 19))
    tick_text = list(RATING_LADDER)
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        range=[0.4, 18.6],
        title="Agency Rating (1 = AAA ... 18 = D)",
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        range=[0.4, 18.6],
        title="Strict OOF Shadow Rating (Lower = Stronger Credit)",
    )
    return _apply_clean_layout(
        fig,
        height=520,
        title="Strict OOF Agency Rating vs. Shadow Rating Screener",
    )


def plot_peer_percentiles(
    percentiles_df: pd.DataFrame, peer_label: str
) -> go.Figure:
    """Horizontal percentile rank bar chart vs. pooled NSE Sector peer group."""
    df_p = percentiles_df.iloc[::-1]
    colors = [
        PALETTE["rising"] if v >= 60 else (PALETTE["accent"] if v >= 35 else PALETTE["fallen"])
        for v in df_p["Percentile"]
    ]
    fig = go.Figure(
        go.Bar(
            y=df_p["Metric"],
            x=df_p["Percentile"],
            orientation="h",
            marker_color=colors,
            text=[f"{p:.0f}th ({val})" for p, val in zip(df_p["Percentile"], df_p["Company Value"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Peer Percentile: %{x:.1f}th<extra></extra>",
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color="#4A5568", line_width=1.2)
    fig.update_xaxes(range=[0, 118], title="Peer Percentile (Higher = Stronger Credit Profile)")
    return _apply_clean_layout(
        fig,
        height=360,
        title=f"Credit Metric Percentiles vs. {peer_label}",
    )


def plot_scorecard_waterfall(
    baseline_notch: float,
    pillar_attributions: dict[str, float],
    expected_notch: float,
    title: str = "Linear Scorecard Notch Driver Waterfall (Negative = Stronger Credit)",
) -> go.Figure:
    """Exact additive OrdinalLogistic notch attribution waterfall across the 7 credit pillars."""
    ordered = sorted(pillar_attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    x_labels = ["Universe Baseline"] + [k for k, _ in ordered] + ["Shadow Notch"]
    measures = ["absolute"] + ["relative"] * len(ordered) + ["total"]
    y_vals = [baseline_notch] + [v for _, v in ordered] + [expected_notch]
    text_labels = (
        [f"{baseline_notch:.2f}"]
        + [f"{v:+.2f}" for _, v in ordered]
        + [f"{expected_notch:.2f}"]
    )

    # Note: decreasing notch_num improves credit (green); increasing notch_num weakens credit (red)
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_vals,
            text=text_labels,
            textposition="outside",
            connector={"line": {"color": "#CBD5E0"}},
            decreasing={"marker": {"color": PALETTE["rising"]}},
            increasing={"marker": {"color": PALETTE["fallen"]}},
            totals={"marker": {"color": PALETTE["navy"]}},
        )
    )
    fig.update_yaxes(title="Expected Notch Score (1 = AAA ... 18 = D)", range=[0, 19])
    return _apply_clean_layout(fig, height=400, title=title)


def plot_ig_cliff_comparison(base_p_ig: float, stressed_p_ig: float) -> go.Figure:
    """Base vs. Stressed P(Investment Grade) bar chart with 50% IG-Cliff threshold."""
    vals = [base_p_ig * 100.0, stressed_p_ig * 100.0]
    colors = [
        PALETTE["ig"] if base_p_ig >= 0.5 else PALETTE["hy"],
        PALETTE["rising"] if stressed_p_ig >= 0.5 else PALETTE["fallen"],
    ]
    fig = go.Figure(
        go.Bar(
            x=["Base Case", "Stressed Case"],
            y=vals,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            width=0.48,
        )
    )
    fig.add_hline(
        y=50.0,
        line_dash="dash",
        line_color=PALETTE["accent"],
        line_width=2,
        annotation_text="IG-Cliff (50%)",
        annotation_position="top right",
    )
    fig.update_yaxes(range=[0, 112], title="Calibrated P(Investment Grade) %")
    return _apply_clean_layout(
        fig, height=340, title="IG-Cliff Migration Risk (Base vs. Stressed)"
    )


def plot_confusion_matrix_7tier(
    cm: list[list[int]], labels: list[str] = list(BROAD_TIERS), title: str = "7-Tier OOF Confusion Matrix"
) -> go.Figure:
    """Annotated 7-broad-tier OOF confusion matrix heatmap."""
    z = np.asarray(cm, dtype=int)
    row_sums = np.maximum(z.sum(axis=1, keepdims=True), 1)
    pct = z / row_sums * 100.0
    text = [
        [f"{z[i, j]}<br>({pct[i, j]:.0f}%)" for j in range(len(labels))]
        for i in range(len(labels))
    ]
    fig = go.Figure(
        go.Heatmap(
            z=pct,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Row Share: %{z:.1f}%<extra></extra>",
            showscale=True,
            colorbar=dict(title="Row %"),
        )
    )
    fig.update_yaxes(autorange="reversed", title="Actual Agency Broad Tier")
    fig.update_xaxes(title="OOF Predicted Broad Tier")
    return _apply_clean_layout(fig, height=440, title=title)


def plot_calibration_curve(cal_data: dict) -> go.Figure:
    """IG-vs-HY reliability / calibration diagram across models."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#4A5568", dash="dash"),
            name="Perfect Calibration",
        )
    )
    series = [
        ("ig_logistic", "Calibrated Logistic (IG-Cliff)", PALETTE["ig"]),
        ("ig_hgb", "Calibrated HistGradBoost (IG-Cliff)", PALETTE["accent"]),
        ("ordinal_implied", "OrdinalLogistic Implied P(IG)", PALETTE["rising"]),
    ]
    for key, label, color in series:
        if key not in cal_data:
            continue
        fig.add_trace(
            go.Scatter(
                x=cal_data[key]["prob_pred"],
                y=cal_data[key]["prob_true"],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.2),
                marker=dict(size=7),
            )
        )
    fig.update_xaxes(range=[0, 1.02], title="Mean Predicted P(Investment Grade)")
    fig.update_yaxes(range=[0, 1.02], title="Observed Investment Grade Frequency")
    return _apply_clean_layout(
        fig, height=420, title="IG-vs-HY OOF Calibration / Reliability Curve"
    )


def plot_scorecard_weights(weights_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of mean OrdinalLogistic log-odds weights across 17 thresholds."""
    df_w = weights_df.sort_values("mean_log_odds_weight", ascending=True)
    colors = [PILLAR_COLORS.get(p, PALETTE["inline"]) for p in df_w["pillar"]]
    fig = go.Figure(
        go.Bar(
            y=df_w["label"],
            x=df_w["mean_log_odds_weight"],
            orientation="h",
            marker_color=colors,
            customdata=df_w["pillar"],
            hovertemplate=(
                "<b>%{y}</b><br>Pillar: %{customdata}<br>"
                "Mean Log-Odds Weight: %{x:+.3f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0.0, line_color="#2D3748", line_width=1.2)
    fig.update_xaxes(
        title="Mean Log-Odds Weight on P(Y >= k)  [Negative = Strengthens Credit | Positive = Weakens Credit]"
    )
    return _apply_clean_layout(
        fig,
        height=560,
        title="OrdinalLogistic Scorecard Feature Weights (Averaged Across 17 Cutpoints)",
    )
