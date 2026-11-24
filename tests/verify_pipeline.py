"""Verification suite covering all 5 invariants in docs/ARCHITECTURE.md."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    COL_COMPANY,
    COL_MONTHS,
    COL_NETWORTH,
    COL_PBILDT_MARGIN,
    COL_RATING,
    COL_SECTOR,
    COL_STATE,
    RAW_DATA_PATH,
    RATING_LADDER,
    load_cleaned_dataset,
)
from src.features import (
    FEATURE_NAMES,
    PILLAR_FEATURES,
    CreditFeatureTransformer,
    apply_stress_shocks,
)
from src.modeling import (
    OrdinalLogistic,
    fit_production_bundle,
    score_company_record,
)
from src.train import ARTIFACTS_DIR, train_and_save_artifacts


class TestCreditWorkbenchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df_clean, cls.df_quarantined = load_cleaned_dataset(RAW_DATA_PATH)
        cls.bundle = fit_production_bundle(cls.df_clean)

    # ------------------------------------------------------------------
    # Suite 1: Data hygiene & Months != 12 quarantine
    # ------------------------------------------------------------------
    def test_1_months_quarantine_and_whitespace_hygiene(self) -> None:
        self.assertEqual(len(self.df_clean), 1224)
        self.assertEqual(len(self.df_quarantined), 1)
        self.assertEqual(
            self.df_quarantined.iloc[0][COL_COMPANY], "KEYSTONE REALTORS LIMITED"
        )
        self.assertNotEqual(self.df_quarantined.iloc[0][COL_MONTHS], 12)
        self.assertTrue((self.df_clean[COL_MONTHS] == 12).all())

        # Header & string whitespace hygiene
        self.assertIn(COL_RATING, self.df_clean.columns)
        self.assertNotIn("Rating ", self.df_clean.columns)
        for col in [COL_STATE, "Period", COL_RATING]:
            vals = self.df_clean[col].astype(str)
            self.assertTrue((vals == vals.str.strip()).all(), f"Whitespace left in {col}")

        # All 18 rating labels present and mapped cleanly
        self.assertEqual(set(self.df_clean[COL_RATING].unique()), set(RATING_LADDER))
        self.assertEqual(int(self.df_clean["is_ig"].sum()), 1035)

        # Tiny sector pooling: single-firm Financial Services falls back cleanly
        rpsg = self.df_clean[self.df_clean[COL_COMPANY] == "RPSG VENTURES LIMITED"]
        self.assertEqual(len(rpsg), 1)
        self.assertEqual(rpsg.iloc[0]["peer_group"], "Universe (Fallback)")

    # ------------------------------------------------------------------
    # Suite 2: Negative-Networth & edge-case implied debt recovery
    # ------------------------------------------------------------------
    def test_2_implied_debt_recovery_and_negative_networth(self) -> None:
        neg_nw = self.df_clean[self.df_clean[COL_NETWORTH] < 0]
        self.assertEqual(len(neg_nw), 58)
        self.assertTrue((neg_nw["implied_total_debt"] >= 0.0).all())
        # Verify (-gearing)*(-Networth) > 0 on known negative-networth corporates
        parsvnath = self.df_clean[
            self.df_clean[COL_COMPANY] == "Parsvnath Developers Limited"
        ].iloc[0]
        self.assertGreater(parsvnath["implied_total_debt"], 2500.0)

        renuka = self.df_clean[
            self.df_clean[COL_COMPANY] == "Shree Renuka Sugars Limited"
        ].iloc[0]
        self.assertGreater(renuka["implied_total_debt"], 5000.0)

        # Verify |Networth| < 1 fallback recovers positive debt (Konark Synthetic ~ 29.82 Cr)
        konark = self.df_clean[
            self.df_clean[COL_COMPANY] == "Konark Synthetic Limited"
        ].iloc[0]
        self.assertTrue(bool(konark["debt_reconstruction_fallback"]))
        self.assertAlmostEqual(konark["implied_total_debt"], 29.82, delta=0.5)

        # Verify gearing == 0 & Total debt/GCA > 0 fallback recovers positive debt
        nhc = self.df_clean[self.df_clean[COL_COMPANY] == "NHC FOODS LIMITED"].iloc[0]
        self.assertTrue(bool(nhc["debt_reconstruction_fallback"]))
        self.assertAlmostEqual(nhc["implied_total_debt"], 36.0, delta=0.5)

        # Global consistency bounds
        self.assertTrue((self.df_clean["implied_total_debt"] >= 0.0).all())
        self.assertTrue((self.df_clean["implied_term_debt"] >= 0.0).all())
        self.assertTrue(
            (
                self.df_clean["implied_term_debt"]
                <= self.df_clean["implied_total_debt"] + 1e-9
            ).all()
        )

    # ------------------------------------------------------------------
    # Suite 3: Fold-level leakage isolation
    # ------------------------------------------------------------------
    def test_3_fold_level_leakage_isolation(self) -> None:
        train_df = self.df_clean.iloc[:900].copy()
        val_df_a = self.df_clean.iloc[900:].copy()
        val_df_b = val_df_a.copy()
        # Massively perturb validation fold to confirm zero bleed into fitted transformer
        val_df_b[COL_PBILDT_MARGIN] = val_df_b[COL_PBILDT_MARGIN] * 100.0 + 500.0
        val_df_b[COL_NETWORTH] = -999999.0

        tf_a = CreditFeatureTransformer(scale=True)
        X_tr_a = tf_a.fit_transform(train_df)
        _ = tf_a.transform(val_df_a)

        tf_b = CreditFeatureTransformer(scale=True)
        X_tr_b = tf_b.fit_transform(train_df)
        _ = tf_b.transform(val_df_b)

        np.testing.assert_allclose(X_tr_a, X_tr_b, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(tf_a.scaler_.mean_, tf_b.scaler_.mean_, atol=0.0)
        np.testing.assert_allclose(tf_a.scaler_.scale_, tf_b.scaler_.scale_, atol=0.0)
        self.assertEqual(tf_a.sector_stats_, tf_b.sector_stats_)
        self.assertEqual(tf_a.macro_stats_, tf_b.macro_stats_)
        self.assertEqual(tf_a.universe_stats_, tf_b.universe_stats_)
        self.assertEqual(X_tr_a.shape[1], len(FEATURE_NAMES))

        # Ensure unseen sector in validation hits Macro -> Universe fallback cleanly with finite output
        val_unseen = val_df_a.iloc[:5].copy()
        val_unseen[COL_SECTOR] = "Unseen Future Sector"
        X_unseen = tf_a.transform(val_unseen)
        self.assertTrue(np.all(np.isfinite(X_unseen)))

    # ------------------------------------------------------------------
    # Suite 4: Ordinal probability monotonicity & additive attributions
    # ------------------------------------------------------------------
    def test_4_ordinal_monotonicity_and_attribution_exactness(self) -> None:
        tf: CreditFeatureTransformer = self.bundle["transformer_scaled"]
        ord_model: OrdinalLogistic = self.bundle["ordinal_model"]

        X_all = tf.transform(self.df_clean)
        q = ord_model.predict_cumulative_proba(X_all)
        probs = ord_model.predict_proba(X_all)
        notches = ord_model.predict(X_all)

        # Monotonicity: 1 >= q_2 >= q_3 >= ... >= q_18 >= 0
        self.assertTrue(np.all((q >= 0.0) & (q <= 1.0)))
        self.assertTrue(np.all(np.diff(q, axis=1) <= 1e-12))

        # Valid discrete probabilities summing identically to 1
        self.assertTrue(np.all(probs >= 0.0))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(self.df_clean)), atol=1e-9)

        # Expected notch bounds [1, 18]
        self.assertTrue(np.all((notches >= 1.0) & (notches <= 18.0)))

        # Additive exactness of 7-pillar waterfall attributions on 25 spread samples
        sample_indices = np.linspace(0, len(self.df_clean) - 1, 25, dtype=int)
        for idx in sample_indices:
            expl = ord_model.explain_notch_attributions(X_all[idx])
            self.assertEqual(set(expl["pillar_attributions"].keys()), set(PILLAR_FEATURES.keys()))
            reconstructed = expl["baseline_notch"] + sum(expl["pillar_attributions"].values())
            self.assertAlmostEqual(reconstructed, expl["expected_notch"], delta=1e-3)

    # ------------------------------------------------------------------
    # Suite 5: Stress-test directional sanity
    # ------------------------------------------------------------------
    def test_5_stress_test_directional_sanity(self) -> None:
        # Select representative firms across tiers (AAA, AA, A, BBB, BB, negative-NW, debt-free)
        samples = []
        for tier in ["AAA", "AA", "A", "BBB", "BB"]:
            sub = self.df_clean[
                (self.df_clean["broad_tier"] == tier)
                & (self.df_clean["implied_total_debt"] > 50.0)
            ]
            if not sub.empty:
                samples.append(sub.iloc[0])
                if len(sub) > 5:
                    samples.append(sub.iloc[5])
        # Add a negative-Networth firm and a zero-debt firm
        samples.append(self.df_clean[self.df_clean[COL_NETWORTH] < 0].iloc[0])
        samples.append(self.df_clean[self.df_clean["implied_total_debt"] == 0].iloc[0])

        shock_scenarios = [
            {"rate_shock_bps": 300.0},
            {"pbildt_shock_pct": -30.0},
            {"debt_shock_pct": 50.0},
            {"wc_days_shock": 90.0},
            {
                "rate_shock_bps": 250.0,
                "pbildt_shock_pct": -25.0,
                "debt_shock_pct": 40.0,
                "wc_days_shock": 60.0,
            },
        ]

        median_cod = self.bundle["median_cost_of_debt"]
        for row in samples:
            base_score = score_company_record(pd.DataFrame([row]), self.bundle)
            for sc in shock_scenarios:
                stressed_df = apply_stress_shocks(
                    row, fallback_cost_of_debt=median_cod, **sc
                )
                stressed_score = score_company_record(stressed_df, self.bundle)
                self.assertGreaterEqual(
                    stressed_score["expected_notch"],
                    base_score["expected_notch"] - 1e-6,
                    f"Notch improved under adverse shock {sc} for {row[COL_COMPANY]}: "
                    f"{base_score['expected_notch']:.4f} -> {stressed_score['expected_notch']:.4f}",
                )
                self.assertLessEqual(
                    stressed_score["p_ig"],
                    base_score["p_ig"] + 1e-6,
                    f"P(IG) increased under adverse shock {sc} for {row[COL_COMPANY]}: "
                    f"{base_score['p_ig']:.4f} -> {stressed_score['p_ig']:.4f}",
                )

    # ------------------------------------------------------------------
    # Artifact Integrity Check (runs if artifacts/ exists)
    # ------------------------------------------------------------------
    def test_6_artifacts_oof_integrity(self) -> None:
        oof_path = Path(ARTIFACTS_DIR) / "oof_predictions.csv"
        if not oof_path.exists():
            train_and_save_artifacts(RAW_DATA_PATH, ARTIFACTS_DIR)
        oof = pd.read_csv(oof_path)
        self.assertEqual(len(oof), 1224)
        self.assertEqual(oof[COL_COMPANY].nunique(), 1224)
        self.assertFalse(oof["shadow_notch"].isna().any())
        self.assertFalse(oof["p_ig_logistic"].isna().any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
