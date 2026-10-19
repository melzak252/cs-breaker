"""Prevent a challenger from silently changing the fixed development population."""
import unittest

import pandas as pd

from scripts.develop_bayesian_series import join_development_features


class BayesianWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame([
            {"hltv_match_id": "1", "source_date": "2026-05-01", "team1": "A", "team2": "B", "target": 1, "format": "bo3", "old_prob": 0.6},
            {"hltv_match_id": "2", "source_date": "2026-05-02", "team1": "C", "team2": "D", "target": 0, "format": "bo5", "old_prob": 0.4},
        ])
        self.bayesian = self.base.drop(columns="old_prob").copy()
        self.bayesian["bayes_score_s1_rw0025_prob"] = [0.7, 0.3]

    def test_join_uses_native_identity_not_row_position(self):
        merged = join_development_features(self.base, self.bayesian.iloc[::-1])
        by_id = merged.set_index("hltv_match_id")
        self.assertEqual(by_id.loc["1", "bayes_score_s1_rw0025_prob"], 0.7)
        self.assertEqual(by_id.loc["2", "bayes_score_s1_rw0025_prob"], 0.3)
        self.assertEqual(by_id.loc["1", "old_prob"], 0.6)

    def test_missing_or_duplicated_native_rows_cannot_shrink_cohort(self):
        for challenger in (self.bayesian.iloc[:1], pd.concat([self.bayesian, self.bayesian.iloc[:1]])):
            with self.subTest(rows=len(challenger)), self.assertRaises(ValueError):
                join_development_features(self.base, challenger)

    def test_orientation_label_format_and_day_must_agree(self):
        for field, value in (("target", 0), ("team1", "B"), ("team2", "A"),
                             ("source_date", "2026-05-02"), ("format", "bo5")):
            bad = self.bayesian.copy()
            bad.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                join_development_features(self.base, bad)


if __name__ == "__main__":
    unittest.main()
