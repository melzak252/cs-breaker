"""Competitiveness needs paired evidence and bounded uncertainty, not weak tests."""
import copy
import unittest
from datetime import date, timedelta

from pipeline.models.bookmaker_comparison import _paired_difference
from pipeline.models.challenge_evaluation import competitiveness_decision


POLICY = {"minimum_paired_matches": 200, "minimum_source_days": 30,
          "log_loss": {"maximum_point_model_minus_consensus": 0.005,
                       "maximum_one_sided_95_upper": 0.01},
          "brier": {"maximum_point_model_minus_consensus": 0.0025,
                    "maximum_one_sided_95_upper": 0.005},
          "individual_book_minimum_matches_for_rule": 50,
          "individual_book_maximum_point_log_loss_difference": 0.01}


def cases(*, wide=False):
    return [{"hltv_match_id": str(i + 1),
             "source_date": (date(2026, 7, 1) + timedelta(days=i // 10)).isoformat(),
             "target": 1, "p_model": (0.9 if i < 150 else 0.4) if wide else 0.65,
             "p_bookmaker": 0.6} for i in range(300)]


class CompetitivenessTests(unittest.TestCase):
    def test_wide_interval_crossing_zero_is_not_noninferiority(self):
        rows = cases(wide=True)
        result = competitiveness_decision(rows, {}, POLICY, test_start="2026-07-01")
        self.assertFalse(result["qualified"])
        evidence = result["uncertainty"]["source_day"]["log_loss"]
        self.assertAlmostEqual(evidence["estimate"], 0.0)
        self.assertGreater(evidence["upper_confidence_bound_95"], 0.01)
        self.assertLess(evidence["confidence_interval_95"][0], 0)

    def test_identical_populations_and_both_cluster_sensitivities_can_qualify(self):
        result = competitiveness_decision(cases(), {}, POLICY, test_start="2026-07-01")
        self.assertTrue(result["qualified"])
        self.assertLess(result["uncertainty"]["seven_day"]["log_loss"]["upper_confidence_bound_95"], 0)
        self.assertEqual(result["uncertainty"]["seven_day"]["log_loss"]["block_count"], 5)

    def test_sparse_cases_or_a_material_individual_book_loss_cannot_qualify(self):
        self.assertFalse(competitiveness_decision(cases()[:100], {}, POLICY,
                                                 test_start="2026-07-01")["qualified"])
        books = {"pinnacle": {"matches": 100,
                              "paired_log_loss_difference": {"estimate": 0.02}}}
        self.assertFalse(competitiveness_decision(cases(), books, POLICY,
                                                 test_start="2026-07-01")["qualified"])

    def test_repeated_book_rows_cannot_inflate_match_support(self):
        rows = cases()
        with self.assertRaises(ValueError):
            _paired_difference(rows + [copy.deepcopy(rows[0])], block_days=7, start_day="2026-07-01")
