"""Historical rankings must be available before a forecast, not backfilled."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from pipeline.features.valve_standings import (
    StandingsHistory, VRS_FEATURE_COLUMNS, VRS_MIRROR_RULES,
    load_standings, parse_standings,
)
from scripts.collect_valve_standings import collect_standings


def ranking(team, points, rank, snapshot="2026-01-05", available="2026-01-12T00:00:00Z"):
    return {"team": team, "points": points, "rank": rank,
            "snapshot_date": snapshot, "available_at": available,
            "roster": [team + str(i) for i in range(5)], "commit": "a" * 40,
            "source_path": "invitation/2026/standings_global_" + snapshot.replace("-", "_") + ".md",
            "sha256": "b" * 64}


class StandingsHistoryTests(unittest.TestCase):
    def test_rankings_are_missing_before_availability_and_symmetric_after(self):
        history = StandingsHistory([ranking("Alpha", 1500, 1), ranking("Beta", 1200, 10)])
        self.assertEqual(history.features("Alpha", "Beta", "2026-01-12")["vrs_available"], 0)
        forward = history.features("Alpha", "Beta", "2026-01-13")
        reverse = history.features("Beta", "Alpha", "2026-01-13")
        self.assertEqual(forward["vrs_available"], 1)
        self.assertEqual(forward["vrs_points_diff"], 300)
        self.assertEqual(forward["vrs_points_diff"], -reverse["vrs_points_diff"])
        self.assertAlmostEqual(forward["vrs_log_rank_diff"], -reverse["vrs_log_rank_diff"])

    def test_new_global_snapshot_cannot_backfill_a_missing_team_from_older_rankings(self):
        history = StandingsHistory([
            ranking("Alpha", 1500, 1), ranking("Beta", 1200, 10),
            ranking("Alpha", 1600, 1, "2026-02-02", "2026-02-09T00:00:00Z")])
        features = history.features("Alpha", "Beta", "2026-02-10")
        self.assertEqual(features["vrs_available"], 0)
        self.assertEqual(features["vrs_points_diff"], 0)
        self.assertEqual(history.features("Alpha", "Beta", "2026-01-13")["vrs_points_diff"], 300)

    def test_org_aggregate_uses_max_points_and_counts_rosters_without_aliases(self):
        other = ranking(" ALPHA ", 1600, 3)
        other["roster"] = ["different" + str(i) for i in range(5)]
        history = StandingsHistory([ranking("Alpha", 1500, 1), other,
                                   ranking("Beta", 1200, 10)])
        forward = history.features("alpha", "  BETA ", "2026-01-13")
        self.assertEqual(forward["vrs_points_diff"], 400)
        self.assertEqual(forward["vrs_minimum_roster_count"], 1)
        self.assertEqual(history.features("Alpha Esports", "Beta", "2026-01-13")["vrs_available"], 0)
        self.assertEqual(set(forward), set(VRS_FEATURE_COLUMNS))
        self.assertEqual(set(VRS_MIRROR_RULES), set(VRS_FEATURE_COLUMNS))

    def test_equal_points_tie_and_multiple_roster_support_are_deterministic(self):
        alpha_second = ranking("alpha", 1500, 3)
        alpha_second["roster"] = ["second-a" + str(i) for i in range(5)]
        beta_second = ranking("beta", 1100, 12)
        beta_second["roster"] = ["second-b" + str(i) for i in range(5)]
        rows = [ranking("Alpha", 1500, 1), ranking("Beta", 1200, 10),
                alpha_second, beta_second]
        first = StandingsHistory(rows).features("Alpha", "Beta", "2026-01-13")
        reordered = StandingsHistory(list(reversed(rows))).features("Alpha", "Beta", "2026-01-13")
        reverse = StandingsHistory(rows).features("Beta", "Alpha", "2026-01-13")
        self.assertEqual(first, reordered)
        self.assertEqual(first["vrs_minimum_roster_count"], 2)
        self.assertAlmostEqual(first["vrs_log_rank_diff"], math.log(11) - math.log(2))
        for feature, rule in VRS_MIRROR_RULES.items():
            expected = -reverse[feature] if rule["operation"] == "negate" else reverse[feature]
            self.assertEqual(first[feature], expected)

    def test_missing_global_history_is_explicit_and_invalid_query_rejects(self):
        history = StandingsHistory([])
        self.assertEqual(history.features("Alpha", "Beta", "2026-01-13"), dict.fromkeys(VRS_FEATURE_COLUMNS, 0))
        for team1, team2, day in (("Alpha", "alpha", "2026-01-13"),
                                  ("", "Beta", "2026-01-13"),
                                  ("Alpha", "Beta", "2026-1-13")):
            with self.subTest(day=day, team1=team1), self.assertRaises(ValueError):
                history.features(team1, team2, day)

    def test_unsorted_snapshot_rows_and_late_old_publication_do_not_change_selection(self):
        old = [ranking("Alpha", 900, 4, available="2026-03-01T00:00:00Z"),
               ranking("Beta", 800, 5, available="2026-03-01T00:00:00Z")]
        new = [ranking("Alpha", 1600, 1, "2026-02-02", "2026-02-09T00:00:00Z"),
               ranking("Beta", 1300, 2, "2026-02-02", "2026-02-09T00:00:00Z")]
        history = StandingsHistory(list(reversed(new + old)))
        self.assertEqual(history.features("Alpha", "Beta", "2026-03-02")["vrs_points_diff"], 300)
        before = history.records
        history.features("Unknown", "Beta", "2026-03-03")
        self.assertEqual(history.records, before)
        before[0]["points"] = 0
        self.assertNotEqual(history.records[0]["points"], 0)

    def test_duplicate_rows_conflicting_provenance_and_invalid_measurements_reject(self):
        good = ranking("Alpha", 1500, 1)
        for field, value in (("points", float("nan")), ("rank", 0),
                             ("roster", []), ("available_at", "2026-01-12T00:00:00"),
                             ("available_at", "2026-01-06T00:00:00Z")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                StandingsHistory([{**good, field: value}])
        with self.assertRaises(ValueError):
            StandingsHistory([good, good])
        with self.assertRaises(ValueError):
            StandingsHistory([good, {**ranking("Beta", 1200, 2), "commit": "c" * 40}])


TABLE = """# Global standings

| Standing | Points | Team Name | Roster | Details |
| --- | --- | --- | --- | --- |
| 1 | 1500 | Alpha | a, b, c, d, e | [details](alpha.html) |
| 2 | 1200 | Beta | f, g, h, i, j | [details](beta.html) |
"""


class StandingsParserTests(unittest.TestCase):
    def parse(self, text=TABLE):
        return parse_standings(
            text, snapshot_date="2026-01-05", available_at="2026-01-12T00:00:00Z",
            commit="a" * 40, source_path="invitation/2026/standings_global_2026_01_05.md",
            sha256=hashlib.sha256(text.encode()).hexdigest())

    def test_parse_records_keeps_provenance_and_roster(self):
        rows = self.parse()
        self.assertEqual(rows[0]["roster"], ["a", "b", "c", "d", "e"])
        self.assertEqual(rows[1]["points"], 1200)
        self.assertEqual(rows[0]["snapshot_date"], "2026-01-05")

    def test_official_single_dash_alignment_and_empty_details_header_are_supported(self):
        text = TABLE.replace("| --- | --- | --- | --- | --- |", "| :- | -: | :- | :- | :- |")
        text = text.replace("| Details |", "| |")
        rows = self.parse(text)
        self.assertEqual([(row["team"], row["rank"], row["points"]) for row in rows],
                         [("Alpha", 1, 1500), ("Beta", 2, 1200)])

    def test_malformed_missing_empty_duplicate_and_hash_mismatch_reject(self):
        for text in ("not a table", TABLE.split("| 1 |")[0],
                     TABLE.replace("| 1500 |", "| NaN |"),
                     TABLE.replace("a, b, c, d, e", "a, a"),
                     TABLE + TABLE.splitlines()[-1] + "\n",
                     TABLE.replace("| 2 | 1200", "| 1 | 1200"),
                     TABLE.replace("| 1 | 1500 | Alpha |", "| 1 | 1500 | Alpha")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)
        with self.assertRaises(ValueError):
            parse_standings(TABLE, snapshot_date="2026-01-05",
                            available_at="2026-01-12T00:00:00Z", commit="a" * 40,
                            source_path="invitation/2026/standings_global_2026_01_05.md",
                            sha256="b" * 64)


class StandingsCollectorTests(unittest.TestCase):
    def git(self, repository, *args, timestamp=None):
        environment = os.environ.copy()
        if timestamp:
            environment.update(GIT_AUTHOR_DATE=timestamp, GIT_COMMITTER_DATE=timestamp)
        return subprocess.run(["git", "-C", str(repository), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=environment).stdout.decode().strip()

    def test_first_blob_survives_correction_and_offline_cutoff_is_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            repository.mkdir()
            self.git(repository, "init")
            self.git(repository, "config", "user.email", "fixture@example.invalid")
            self.git(repository, "config", "user.name", "Fixture")
            relative = "invitation/2026/standings_global_2026_01_05.md"
            source = repository / relative
            source.parent.mkdir(parents=True)
            original = TABLE.encode()
            source.write_bytes(original)
            self.git(repository, "add", ".")
            self.git(repository, "commit", "-m", "first publication",
                     timestamp="2026-01-15T12:00:00+00:00")
            first_commit = self.git(repository, "rev-parse", "HEAD")
            source.write_text(TABLE.replace("1500", "1900"))
            self.git(repository, "commit", "-am", "later correction",
                     timestamp="2026-01-20T12:00:00+00:00")
            output = root / "saved"
            manifest = collect_standings(output, repository=repository, years=(2026,))
            snapshot = manifest["snapshots"][0]
            self.assertEqual(snapshot["commit"], first_commit)
            self.assertEqual(snapshot["first_commit_at"], "2026-01-15T12:00:00Z")
            self.assertEqual(snapshot["available_at"], "2026-01-15T12:00:00Z")
            self.assertEqual((output / snapshot["raw_path"]).read_bytes(), original)
            self.assertEqual(snapshot["sha256"], hashlib.sha256(original).hexdigest())
            history = load_standings(output)
            self.assertEqual(history.features("Alpha", "Beta", "2026-01-15")["vrs_available"], 0)
            self.assertEqual(history.features("Alpha", "Beta", "2026-01-16")["vrs_points_diff"], 300)
            self.assertEqual(history.manifest, json.loads((output / "manifest.json").read_text()))
            (output / snapshot["raw_path"]).write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                load_standings(output)

    def test_shallow_repository_and_no_global_snapshots_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repo"
            repository.mkdir()
            self.git(repository, "init")
            with self.assertRaises(ValueError):
                collect_standings(Path(temporary) / "saved", repository=repository, years=(2026,))
            self.git(repository, "config", "user.email", "fixture@example.invalid")
            self.git(repository, "config", "user.name", "Fixture")
            relative = "invitation/2026/standings_global_2026_01_05.md"
            source = repository / relative
            source.parent.mkdir(parents=True)
            source.write_text(TABLE)
            self.git(repository, "add", ".")
            self.git(repository, "commit", "-m", "publication",
                     timestamp="2026-01-06T00:00:00+00:00")
            source.write_text(TABLE.replace("1500", "1900"))
            self.git(repository, "commit", "-am", "correction",
                     timestamp="2026-01-08T00:00:00+00:00")
            full_output = Path(temporary) / "full-saved"
            manifest = collect_standings(full_output, repository=repository, years=(2024, 2026))
            self.assertEqual(manifest["snapshots"][0]["available_at"], "2026-01-12T00:00:00Z")
            self.assertEqual(manifest["years_without_snapshots"], [2024])
            self.assertEqual(load_standings(full_output).features("Alpha", "Beta", "2026-01-13")["vrs_points_diff"], 300)
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                collect_standings(full_output, repository=repository, years=(2026,))
            shallow = Path(temporary) / "shallow"
            self.git(repository, "clone", "--depth=1", repository.as_uri(), str(shallow))
            with self.assertRaisesRegex(ValueError, "shallow"):
                collect_standings(Path(temporary) / "saved", repository=shallow, years=(2026,))
