"""Frozen inputs, loaded sources and empty market cohorts are real boundaries."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from functools import partial

import pandas as pd

from pipeline.models import challenge_evaluation
from pipeline.models.series_forecast import SeriesForecaster
from scripts import evaluate_bookmaker_challenge as workflow


class ChallengeWorkflowTests(unittest.TestCase):
    def test_actual_alternate_master_and_results_must_match_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {"data/processed/master_matches_consolidated.csv": b"frozen master",
                     "data/raw/hltv/matches.csv": b"frozen results",
                     "cohorts.json": b'{"cohorts":{"train":[],"validation":[],"test":[]}}'}
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            protocol = {"cohort_manifest": "cohorts.json", "input_sha256": {
                name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            altered = root / "altered.csv"
            altered.write_bytes(b"same IDs can still conceal changed labels or features")
            values = {"protocol": protocol_path, "master": root / "data/processed/master_matches_consolidated.csv",
                      "hltv_results": root / "data/raw/hltv/matches.csv", "output_dir": root / "output"}
            # Isolate the repository root while preserving the real SHA256 checks.
            verify_in_root = partial(workflow.verify_hashes, root=root)
            with patch.object(workflow, "ROOT", root), patch.object(workflow, "verify_hashes", verify_in_root):
                for argument in ("master", "hltv_results"):
                    with self.subTest(argument=argument), self.assertRaises(ValueError):
                        workflow.inputs(SimpleNamespace(**{**values, argument: altered}))

    def test_source_changed_after_training_cannot_publish_a_selection_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "implementation.py"
            source.write_text("original implementation")
            sources = {str(source): workflow.digest(source)}
            source.write_text("changed while training")
            protocol, preparation = root / "protocol.json", root / "prepared.json"
            protocol.write_text("{}")
            preparation.write_text("{}")
            model = SeriesForecaster(
                recipe={"name": "raw_example", "family": "rating", "feature_columns": ["strength"],
                        "parameters": {}, "recency_half_life_days": None},
                mirror_rules={"strength": {"operation": "complement", "source": "strength"}},
                train_start="2026-01-01", train_end="2026-04-30")
            with self.assertRaises(ValueError):
                workflow.freeze_selection(root, sources=sources, protocol_path=protocol,
                                          prepared_path=preparation, model=model,
                                          evidence={"selected_model": "raw_example"},
                                          validation_predictions=pd.DataFrame(),
                                          protocol_sha256=workflow.digest(protocol),
                                          prepared_sha256=workflow.digest(preparation))
            self.assertFalse((root / "selection_freeze.json").exists())
            self.assertFalse((root / "selected_model.pkl").exists())

    def test_no_timed_predictions_produces_empty_inconclusive_market_evidence(self):
        snapshot = {"fixtures": [], "links": [], "quotes_by_horizon": {"one_hour": []}}
        predictions = pd.DataFrame([{"hltv_match_id": "1", "source_date": "2026-07-01",
                                     "date": "2026-07-01", "team1": "Alpha", "team2": "Beta",
                                     "target": 1, "prob_team1_win": 0.7, "prob_team2_win": 0.3}])
        metadata = {"selected_model": "raw_example", "splits": {
            "train": {"end_date": "2026-04-30"}, "validation": {"end_date": "2026-06-30"},
            "test": {"start_date": "2026-07-01", "end_date": "2026-07-01",
                     "dates": ["2026-07-01"], "n_series": 1}}}
        report, rows = workflow.timed_comparison(snapshot, predictions, metadata,
                                                 horizon="one_hour", devig="proportional")
        self.assertEqual(rows, [])
        self.assertEqual(report["available_book_consensus"]["matches"], 0)
        self.assertIsNone(report["available_book_consensus"]["model"]["log_loss"])
        self.assertEqual(report["no_eligible_forecast_origin_native_ids"], ["1"])

    def test_lagged_sensitivity_cannot_skip_individual_book_failure(self):
        policy = {"minimum_paired_matches": 200, "minimum_source_days": 30,
                  "log_loss": {"maximum_point_model_minus_consensus": 0.005, "maximum_one_sided_95_upper": 0.01},
                  "brier": {"maximum_point_model_minus_consensus": 0.0025, "maximum_one_sided_95_upper": 0.005},
                  "individual_book_minimum_matches_for_rule": 50,
                  "individual_book_maximum_point_log_loss_difference": 0.01}
        consensus, pairs, probabilities = [], [], {}
        for i in range(300):
            identity = str(i + 1)
            day = (pd.Timestamp("2026-07-01") + pd.Timedelta(days=i // 10)).strftime("%Y-%m-%d")
            row = {"hltv_match_id": identity, "source_date": day, "target": 1,
                   "p_model": 0.65, "p_bookmaker": 0.6}
            consensus.append(row)
            pairs.extend([{**row, "bookmaker": "pinnacle", "p_bookmaker": 0.99},
                          {**row, "bookmaker": "unibet", "p_bookmaker": 0.21}])
            probabilities[identity] = 0.65
        result = challenge_evaluation.history_lag_sensitivity(
            consensus, pairs, probabilities, policy, test_start="2026-07-01")
        self.assertFalse(result["qualified"])
        self.assertFalse(result["individual_book_checks"]["pinnacle"])
