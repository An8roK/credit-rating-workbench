"""Offline training entrypoint: cleans data, runs Dual CV, fits models, writes artifacts/."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.data import RAW_DATA_PATH, load_cleaned_dataset
from src.modeling import fit_production_bundle, run_dual_cv

ARTIFACTS_DIR = Path("artifacts")


def _dir_size_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def train_and_save_artifacts(
    raw_path: Path = RAW_DATA_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict:
    """Execute full offline training pipeline and persist lightweight artifacts."""
    t0 = time.perf_counter()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df_clean, df_quarantined = load_cleaned_dataset(raw_path)
    oof_df, validation_report = run_dual_cv(df_clean)
    bundle = fit_production_bundle(df_clean)

    # Persist artifacts (compact CSV/JSON + compressed joblib bundle for <1s Cloud Run load)
    df_clean.to_csv(artifacts_dir / "cleaned_universe.csv", index=False)
    df_quarantined.to_csv(artifacts_dir / "quarantined_rows.csv", index=False)
    oof_df.to_csv(artifacts_dir / "oof_predictions.csv", index=False)
    joblib.dump(bundle, artifacts_dir / "model_bundle.joblib", compress=3)

    elapsed_sec = time.perf_counter() - t0
    validation_report["training_runtime_sec"] = round(elapsed_sec, 3)
    validation_report["quarantined_count"] = int(len(df_quarantined))

    with open(artifacts_dir / "validation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(validation_report, fh, indent=2)

    size_bytes = _dir_size_bytes(artifacts_dir)
    return {
        "runtime_sec": elapsed_sec,
        "artifacts_size_bytes": size_bytes,
        "validation_report": validation_report,
        "oof_df": oof_df,
        "n_clean": len(df_clean),
        "n_quarantined": len(df_quarantined),
    }


def _print_summary(result: dict) -> None:
    report = result["validation_report"]
    oof_df: pd.DataFrame = result["oof_df"]
    size_kb = result["artifacts_size_bytes"] / 1024.0

    print(
        f"\nTraining complete in {result['runtime_sec']:.2f}s | "
        f"Valid corporates: {result['n_clean']} (quarantined: {result['n_quarantined']}) | "
        f"artifacts/ size: {size_kb:.1f} KB ({size_kb / 1024.0:.2f} MB)"
    )

    print("\n=== Dual CV Summary (Stratified 5-Fold OOF vs. Out-of-Sector GroupKFold) ===")
    header = (
        f"{'Model / CV Scheme':38s} | {'MAE':>5s} | {'±1N%':>6s} | {'±2N%':>6s} | "
        f"{'Tier%':>6s} | {'QWK':>5s} | {'ROC':>5s} | {'PR-IG':>5s} | {'Brier':>6s}"
    )
    print(header)
    print("-" * len(header))

    rows = [
        ("OrdinalLogistic (Stratified 5-Fold)", report["stratified_5fold"]["ordinal_logistic"]),
        ("OrdinalLogistic (GroupKFold Macro)", report["group_kfold_macro"]["ordinal_logistic"]),
        ("HistGradBoost (Stratified 5-Fold)", report["stratified_5fold"]["hist_gradient_boosting"]),
        ("HistGradBoost (GroupKFold Macro)", report["group_kfold_macro"]["hist_gradient_boosting"]),
    ]
    for label, m in rows:
        print(
            f"{label:38s} | {m['notch_mae']:5.2f} | {m['within_1_notch_pct']:5.1f}% | "
            f"{m['within_2_notch_pct']:5.1f}% | {m['broad_tier_acc_pct']:5.1f}% | "
            f"{m['qwk_18']:5.3f} | {m['ig_roc_auc']:5.3f} | {m['ig_pr_auc']:5.3f} | "
            f"{m['ig_brier']:6.4f}"
        )

    for label, key in [
        ("IG-Cliff Cal. Logistic (Stratified)", ("stratified_5fold", "ig_cliff_logistic")),
        ("IG-Cliff Cal. Logistic (GroupKFold)", ("group_kfold_macro", "ig_cliff_logistic")),
        ("IG-Cliff Cal. HGB (Stratified)", ("stratified_5fold", "ig_cliff_hgb")),
        ("IG-Cliff Cal. HGB (GroupKFold)", ("group_kfold_macro", "ig_cliff_hgb")),
    ]:
        m = report[key[0]][key[1]]
        print(
            f"{label:38s} | {'-':>5s} | {'-':>6s} | {'-':>6s} | {'-':>6s} | {'-':>5s} | "
            f"{m['ig_roc_auc']:5.3f} | {m['ig_pr_auc']:5.3f} | {m['ig_brier']:6.4f}"
        )

    print("\n=== Top 5 OOF Rising Stars (Upgrade Watch: Shadow Stronger than Agency) ===")
    rising = oof_df.sort_values("rating_gap_notches", ascending=False).head(5)
    for _, r in rising.iterrows():
        print(
            f"  {r['Company'][:45]:45s} | Agency: {r['agency_rating']:4s} ({int(r['agency_notch']):2d}) | "
            f"Shadow: {r['shadow_rating']:4s} ({r['shadow_notch']:5.2f}) | Gap: {r['rating_gap_notches']:+5.2f}"
        )

    print("\n=== Top 5 OOF Fallen Angels (Downgrade Watch: Shadow Weaker than Agency) ===")
    fallen = oof_df.sort_values("rating_gap_notches", ascending=True).head(5)
    for _, r in fallen.iterrows():
        print(
            f"  {r['Company'][:45]:45s} | Agency: {r['agency_rating']:4s} ({int(r['agency_notch']):2d}) | "
            f"Shadow: {r['shadow_rating']:4s} ({r['shadow_notch']:5.2f}) | Gap: {r['rating_gap_notches']:+5.2f}"
        )


if __name__ == "__main__":
    summary = train_and_save_artifacts()
    _print_summary(summary)
