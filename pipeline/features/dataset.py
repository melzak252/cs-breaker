"""Load source data and build leak-free chronological feature datasets."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.contracts import MatchFixture, ParsedMapResult
from pipeline.parser.aggregator import OpenCS2Adapter
from pipeline.features.builder import ChronologicalFeatureBuilder


class DatasetPipeline:
    """Orchestrate source loading, chronological feature generation, and export."""

    def __init__(
        self,
        matches: Optional[List[MatchFixture]] = None,
        demo_map_lookup: Optional[Dict[str, List[ParsedMapResult]]] = None,
    ) -> None:
        self.matches: List[MatchFixture] = list(matches) if matches is not None else []
        self.demo_map_lookup: Dict[str, List[ParsedMapResult]] = (
            dict(demo_map_lookup) if demo_map_lookup is not None else {}
        )
        self.manifest: Dict[str, Any] = {}

    def load_sources(
        self,
        results_csv_path: str = "data/exports/kaggle/results_only.csv",
        opencs2_dir: str = "data/raw/opencs2",
        limit_demos: Optional[int] = None,
        csgo_csv_path: Optional[str] = None,
        limit_csgo: Optional[int] = None,
    ) -> Tuple[List[MatchFixture], Dict[str, List[ParsedMapResult]]]:
        """Instance method to load sources and populate self."""
        matches, demo_lookup = self.load_all_sources(
            results_csv_path=results_csv_path,
            opencs2_dir=opencs2_dir,
            limit_demos=limit_demos,
            csgo_csv_path=csgo_csv_path,
            limit_csgo=limit_csgo,
        )
        self.matches = matches
        self.demo_map_lookup = demo_lookup
        return matches, demo_lookup
    @staticmethod
    def _parse_timestamp(value: str, line: int) -> datetime:
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Row {line}: invalid match date {value!r}") from error
        if timestamp.utcoffset() is None:
            raise ValueError(f"Row {line}: timestamp must include a timezone")
        return timestamp.astimezone(timezone.utc)

    @classmethod
    def load_all_sources(
        cls,
        results_csv_path: str = "data/exports/kaggle/results_only.csv",
        opencs2_dir: str = "data/raw/opencs2",
        limit_demos: Optional[int] = None,
        csgo_csv_path: Optional[str] = None,
        limit_csgo: Optional[int] = None,
    ) -> Tuple[List[MatchFixture], Dict[str, List[ParsedMapResult]]]:
        """Load result fixtures and group all parsed demo maps by match ID."""
        matches: List[MatchFixture] = []
        seen_ids = set()
        required = {
            "hltv_match_id", "date", "tournament", "team1_name", "team2_name",
            "winner", "score_team1", "score_team2", "match_type",
        }
        raw_cs2_lookup: Dict[str, Tuple[List[str], List[str]]] = {}
        raw_cs2_path = Path("data/raw/kaggle/cs2_newestcombinedmatches.csv")
        if raw_cs2_path.exists():
            with raw_cs2_path.open(encoding="utf-8-sig", newline="") as stream:
                raw_reader = csv.DictReader(stream)
                for raw_row in raw_reader:
                    raw_id = raw_row.get("hltv_match_id", "").strip()
                    if raw_id:
                        t1_p = [raw_row.get(f"team1_player_{k}_name", "").strip() for k in range(1, 6) if raw_row.get(f"team1_player_{k}_name", "").strip()]
                        t2_p = [raw_row.get(f"team2_player_{k}_name", "").strip() for k in range(1, 6) if raw_row.get(f"team2_player_{k}_name", "").strip()]
                        raw_cs2_lookup[raw_id] = (t1_p, t2_p)

        with Path(results_csv_path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                missing = sorted(required - set(reader.fieldnames or []))
                raise ValueError(f"Input lacks required result columns: {', '.join(missing)}")
            for line, row in enumerate(reader, start=2):
                match_id = row["hltv_match_id"].strip()
                team1, team2 = row["team1_name"].strip(), row["team2_name"].strip()
                if not match_id or match_id in seen_ids or not team1 or not team2 or team1 == team2:
                    raise ValueError(f"Row {line}: missing/duplicate identity or invalid teams")
                seen_ids.add(match_id)
                winner_value = row["winner"].strip().lower()
                if winner_value not in ("team1", "team2"):
                    raise ValueError(f"Row {line}: invalid winner {row['winner']!r}")
                try:
                    score1, score2 = int(row["score_team1"]), int(row["score_team2"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Row {line}: scores must be integers") from error
                if score1 < 0 or score2 < 0 or score1 == score2:
                    raise ValueError(f"Row {line}: invalid result scores")
                if (winner_value == "team1") != (score1 > score2):
                    raise ValueError(f"Row {line}: score disagrees with winner")

                # Extract players directly or from raw lookup
                t1_players = [row.get(f"team1_player_{k}_name", "").strip() for k in range(1, 6) if row.get(f"team1_player_{k}_name", "").strip()]
                t2_players = [row.get(f"team2_player_{k}_name", "").strip() for k in range(1, 6) if row.get(f"team2_player_{k}_name", "").strip()]
                if not t1_players and match_id in raw_cs2_lookup:
                    t1_players, t2_players = raw_cs2_lookup[match_id]

                matches.append(MatchFixture(
                    match_id=match_id,
                    tournament_name=row["tournament"].strip(),
                    timestamp=cls._parse_timestamp(row["date"].strip(), line),
                    team1=team1,
                    team2=team2,
                    format=row["match_type"].strip().lower(),
                    score1=score1,
                    score2=score2,
                    winner=winner_value,
                    team1_players=t1_players,
                    team2_players=t2_players,
                ))
        if csgo_csv_path and Path(csgo_csv_path).exists():
            from pipeline.ingestion.csgo_archive import CSGOHistoricalAdapter
            csgo_matches = CSGOHistoricalAdapter.load_historical_fixtures(
                csv_path=csgo_csv_path,
                limit=limit_csgo,
            )
            matches.extend(csgo_matches)

        matches.sort(key=lambda match: (match.timestamp, match.match_id))
        parsed_maps = OpenCS2Adapter.batch_load_opencs2(opencs2_dir, limit=limit_demos)
        demo_map_lookup: Dict[str, List[ParsedMapResult]] = {}
        for parsed_map in parsed_maps:
            demo_map_lookup.setdefault(str(parsed_map.match_id), []).append(parsed_map)
        return matches, demo_map_lookup

    def build_chronological_dataset(
        self, train_ratio: float = 0.8
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Build ordered rows and split only at the chronological train boundary."""
        if not 0.0 <= train_ratio <= 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        matches = sorted(self.matches, key=lambda match: (match.timestamp, match.match_id))
        feature_rows = ChronologicalFeatureBuilder().process_matches(
            matches, demo_map_lookup=self.demo_map_lookup
        )
        if len(feature_rows) != len(matches):
            raise ValueError("Feature builder must return one row per match")
        all_rows: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        index = 0
        while index < len(matches):
            end = index + 1
            while end < len(matches) and matches[end].timestamp == matches[index].timestamp:
                end += 1
            for match, features in zip(matches[index:end], feature_rows[index:end]):
                row = dict(features)
                row.update({
                    "match_id": match.match_id,
                    "date": match.timestamp.isoformat(),
                    "team1": match.team1,
                    "team2": match.team2,
                    "prior_matches_team1": counts.get(match.team1.casefold(), 0),
                    "prior_matches_team2": counts.get(match.team2.casefold(), 0),
                })
                all_rows.append(row)
            for match in matches[index:end]:
                team1, team2 = match.team1.casefold(), match.team2.casefold()
                counts[team1] = counts.get(team1, 0) + 1
                counts[team2] = counts.get(team2, 0) + 1
            index = end
        boundary = int(len(all_rows) * train_ratio)
        train_rows, test_rows = all_rows[:boundary], all_rows[boundary:]
        train_dates = [self._row_date(row) for row in train_rows]
        test_dates = [self._row_date(row) for row in test_rows]
        self.manifest = {
            "match_count": len(all_rows),
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "train_ratio": train_ratio,
            "train_start": min(train_dates).isoformat() if train_dates else None,
            "train_end": max(train_dates).isoformat() if train_dates else None,
            "test_start": min(test_dates).isoformat() if test_dates else None,
            "test_end": max(test_dates).isoformat() if test_dates else None,
        }
        self.verify_zero_leakage(train_rows, test_rows)
        return train_rows, test_rows, dict(self.manifest)

    @staticmethod
    def _row_date(row: Dict[str, Any]) -> datetime:
        value = row.get("date", row.get("timestamp"))
        if isinstance(value, datetime):
            parsed = value
        elif value is not None:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        else:
            raise ValueError("Feature row is missing date/timestamp")
        if parsed.utcoffset() is None:
            raise ValueError("Feature row date/timestamp must include a timezone")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def verify_zero_leakage(
        train_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]
    ) -> bool:
        """Assert chronological split, sequential histories, and feature/label separation."""
        rows = train_rows + test_rows
        dates = [DatasetPipeline._row_date(row) for row in rows]
        if train_rows and test_rows:
            train_end = max(DatasetPipeline._row_date(row) for row in train_rows)
            test_start = min(DatasetPipeline._row_date(row) for row in test_rows)
            assert train_end <= test_start, "Train dates must not follow test dates"
        assert dates == sorted(dates), "Rows must be in chronological order"

        counts: Dict[str, int] = {}
        forbidden = {"winner", "score1", "score2"}
        index = 0
        while index < len(rows):
            end = index + 1
            while end < len(rows) and dates[end] == dates[index]:
                end += 1
            for row in rows[index:end]:
                for team_key, count_key in (
                    ("team1", "prior_matches_team1"),
                    ("team2", "prior_matches_team2"),
                ):
                    team = row.get(team_key)
                    if team is None or count_key not in row:
                        raise ValueError(f"Feature row missing {team_key}/{count_key}")
                    assert row[count_key] == counts.get(str(team).casefold(), 0), (
                        f"{count_key} does not match sequential history for {team!r}"
                    )
                features = row.get("features", row)
                if isinstance(features, dict):
                    assert forbidden.isdisjoint(features), "Labels must not appear in feature columns"
            for row in rows[index:end]:
                team1, team2 = str(row["team1"]).casefold(), str(row["team2"]).casefold()
                counts[team1] = counts.get(team1, 0) + 1
                counts[team2] = counts.get(team2, 0) + 1
            index = end
        return True

    def save_dataset(
        self,
        train_rows: List[Dict[str, Any]],
        test_rows: List[Dict[str, Any]],
        output_dir: str = "data/processed",
    ) -> Dict[str, str]:
        """Write CSV splits and a JSON manifest; return their paths."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        train_path = directory / "train_features.csv"
        test_path = directory / "test_features.csv"
        manifest_path = directory / "manifest.json"
        self._write_csv(train_path, train_rows)
        self._write_csv(test_path, test_rows)
        manifest = dict(self.manifest)
        manifest.update({"train_file": train_path.name, "test_file": test_path.name})
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        self.manifest = manifest
        return {
            "train_features": str(train_path),
            "test_features": str(test_path),
            "manifest": str(manifest_path),
        }

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        columns: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
