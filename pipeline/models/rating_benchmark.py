"""Chronological comparison of match-rating systems."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import groupby
from typing import Any, Dict, List, Sequence

from pipeline.contracts import MatchFixture
from pipeline.features.player_ratings import IndividualPlayerRatingEngine
from pipeline.features.ratings_glicko2 import Glicko2RatingEngine
from pipeline.features.ratings_openskill import OpenSkillRatingEngine
from pipeline.features.ratings_trueskill import TrueSkillRatingEngine
from pipeline.models.metrics import evaluate_predictions


class RatingBenchmark:
    """Replay completed fixtures without allowing same-time outcomes to leak."""

    TEAM_ELO = "team_elo"
    PLAYER_LINEUP_ELO = "player_lineup_elo"
    GLICKO2 = "glicko2"
    TRUESKILL = "trueskill"
    OPENSKILL = "openskill"

    def __init__(self) -> None:
        self._team_elo: Dict[str, float] = defaultdict(lambda: 1500.0)
        self._player_elo = IndividualPlayerRatingEngine()
        self._glicko2 = Glicko2RatingEngine()
        self._trueskill = TrueSkillRatingEngine()
        self._openskill = OpenSkillRatingEngine()

    @staticmethod
    def _team_key(name: str) -> str:
        return name.strip().casefold()

    def _predict(self, match: MatchFixture) -> Dict[str, float]:
        team1_key = self._team_key(match.team1)
        team2_key = self._team_key(match.team2)
        elo_difference = self._team_elo[team1_key] - self._team_elo[team2_key]
        team_elo_probability = 1.0 / (1.0 + 10.0 ** (-elo_difference / 400.0))
        lineup_probability = self._player_elo.build_lineup_features(
            match.team1, match.team2, match.team1_players, match.team2_players
        )["player_lineup_prob"]
        return {
            self.TEAM_ELO: team_elo_probability,
            self.PLAYER_LINEUP_ELO: lineup_probability,
            self.GLICKO2: self._glicko2.predict_win(
                match.team1, match.team2, match.team1_players, match.team2_players
            ),
            self.TRUESKILL: self._trueskill.predict_win(
                match.team1, match.team2, match.team1_players, match.team2_players
            ),
            self.OPENSKILL: self._openskill.predict_win(
                match.team1, match.team2, match.team1_players, match.team2_players
            ),
        }

    def _update(self, match: MatchFixture) -> None:
        if match.winner not in ("team1", "team2"):
            return
        team1_key = self._team_key(match.team1)
        team2_key = self._team_key(match.team2)
        rating1 = self._team_elo[team1_key]
        rating2 = self._team_elo[team2_key]
        exponent = (rating2 - rating1) / 400.0
        expected1 = 1.0 / (1.0 + 10.0 ** exponent)
        actual1 = 1.0 if match.winner == "team1" else 0.0
        change = 32.0 * (actual1 - expected1)
        self._team_elo[team1_key] = rating1 + change
        self._team_elo[team2_key] = rating2 - change

        self._player_elo.update_match(
            match.team1, match.team2, match.team1_players, match.team2_players,
            match.winner, date_iso=match.timestamp.isoformat(),
        )
        self._glicko2.update_match(
            match.team1, match.team2, match.team1_players, match.team2_players,
            match.winner, timestamp=match.timestamp,
        )
        self._trueskill.update_match(
            match.team1, match.team2, match.team1_players, match.team2_players,
            match.winner, timestamp=match.timestamp,
        )
        self._openskill.update_match(
            match.team1, match.team2, match.team1_players, match.team2_players,
            match.winner, timestamp=match.timestamp,
        )

    @staticmethod
    def _evaluate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for system in (RatingBenchmark.TEAM_ELO, RatingBenchmark.PLAYER_LINEUP_ELO,
                       RatingBenchmark.GLICKO2, RatingBenchmark.TRUESKILL,
                       RatingBenchmark.OPENSKILL):
            results[system] = evaluate_predictions(
                [row["actual"] for row in rows],
                [row["predictions"][system] for row in rows],
            )
        return results

    def run_benchmark(
        self, matches: List[MatchFixture], train_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """Replay fixtures chronologically and score each system on train/test."""
        if not math.isfinite(train_ratio) or not 0.0 <= train_ratio <= 1.0:
            raise ValueError("train_ratio must be between 0 and 1")

        self.__init__()
        ordered = list(enumerate(sorted(matches, key=lambda match: match.timestamp)))
        split_index = int(len(ordered) * train_ratio)
        periods: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
        for _, grouped in groupby(ordered, key=lambda item: item[1].timestamp):
            batch = list(grouped)
            # All predictions at a shared timestamp use the same prior state.
            for original_index, match in batch:
                predictions = self._predict(match)
                if match.winner in ("team1", "team2"):
                    period = "train" if original_index < split_index else "test"
                    periods[period].append({
                        "match_id": match.match_id,
                        "timestamp": match.timestamp.isoformat(),
                        "actual": int(match.winner == "team1"),
                        "predictions": predictions,
                    })
            for _, match in batch:
                self._update(match)

        train_rows = periods["train"]
        test_rows = periods["test"]
        metrics = {
            "train": self._evaluate(train_rows),
            "test": self._evaluate(test_rows),
        }
        leaderboard = [
            {
                "rank": rank,
                "system": system,
                "log_loss": metrics["test"][system]["log_loss"],
                "brier": metrics["test"][system]["brier"],
                "roc_auc": metrics["test"][system]["roc_auc"],
                "accuracy": metrics["test"][system]["accuracy"],
                "ece": metrics["test"][system]["ece"],
                "n_samples": metrics["test"][system]["n_samples"],
            }
            for rank, (system, _) in enumerate(
                sorted(metrics["test"].items(), key=lambda item: (item[1]["log_loss"], item[0])),
                start=1,
            )
        ]
        return {
            "train_ratio": train_ratio,
            "n_matches": len(ordered),
            "n_scored": len(train_rows) + len(test_rows),
            "split_index": split_index,
            "metrics": metrics,
            "leaderboard": leaderboard,
            "predictions": {
                "train": train_rows,
                "test": test_rows,
            },
        }


    @staticmethod
    def print_summary(benchmark_results: Dict[str, Any]) -> None:
        """Print leaderboard metrics for the held-out test period."""
        leaderboard = benchmark_results.get("leaderboard", [])
        print(f"{'Rank':>4}  {'Rating system':<20} {'Log Loss':>9} {'Brier':>9} "
              f"{'ROC-AUC':>9} {'Accuracy':>9} {'ECE':>9} {'N':>6}")
        print("-" * 84)
        for row in leaderboard:
            print(
                f"{row['rank']:>4}  {row['system']:<20} {row['log_loss']:>9.4f} "
                f"{row['brier']:>9.4f} {row['roc_auc']:>9.4f} "
                f"{row['accuracy']:>9.4f} {row['ece']:>9.4f} {row['n_samples']:>6}"
            )

    @staticmethod
    def save_results(benchmark_results: Dict[str, Any], output_path: str) -> None:
        """Serialize benchmark metrics and predictions to JSON."""
        with open(output_path, "w", encoding="utf-8") as output:
            json.dump(benchmark_results, output, indent=2, allow_nan=False)
            output.write("\n")
