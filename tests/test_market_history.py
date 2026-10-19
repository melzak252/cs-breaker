"""A fixed forecast origin selects history, never an older valid fallback."""
import unittest
import gzip
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

from pipeline.models.market_history import load_market_history

from scripts import collect_cs_oddspapi as provider


class HistoricalQuoteCutoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture = {"fixtureId": "native-fixture", "participant1Name": "Alpha",
                        "participant2Name": "Beta", "startTime": "2026-04-14T18:00:00Z",
                        "trueStartTime": "2026-04-14T18:05:00Z"}
        self.mapping = {"market_id": "171", "outcome_1": "171", "outcome_2": "172"}
        self.history = {"fixtureId": "native-fixture", "bookmakers": {"pinnacle": {"markets": {
            "171": {"outcomes": {
                "171": {"players": {"0": [
                    {"createdAt": "2026-04-14T16:00:00Z", "price": 2.0, "active": True},
                    {"createdAt": "2026-04-14T17:30:00Z", "price": 1.8, "active": True}]}},
                "172": {"players": {"0": [
                    {"createdAt": "2026-04-14T16:00:00Z", "price": 2.0, "active": True},
                    {"createdAt": "2026-04-14T17:30:00Z", "price": 2.2, "active": True}]}}
            }}
        }}}}

    def test_one_hour_origin_uses_earlier_state_without_falsifying_event_start(self):
        coverage, quote = provider.select_pair(
            self.fixture, "pinnacle", self.history, self.mapping,
            cutoff=provider.utc("2026-04-14T17:00:00Z"))
        self.assertEqual(coverage["status"], "admitted")
        self.assertEqual((quote["price_1"], quote["price_2"]), (2.0, 2.0))
        self.assertEqual(quote["startTime"], "2026-04-14T18:00:00Z")
        self.assertEqual(quote["source_created_at_1"], "2026-04-14T16:00:00Z")

    def test_suspension_before_origin_does_not_revive_earlier_price(self):
        states = self.history["bookmakers"]["pinnacle"]["markets"]["171"]["outcomes"]["171"]["players"]["0"]
        states.append({"createdAt": "2026-04-14T16:30:00Z", "price": 2.0, "active": False})
        coverage, quote = provider.select_pair(
            self.fixture, "pinnacle", self.history, self.mapping,
            cutoff=provider.utc("2026-04-14T17:00:00Z"))
        self.assertIsNone(quote)
        self.assertEqual(coverage["side_1_status"], "latest_inactive")

    def test_invalid_explicit_origins_and_parent_scope_do_not_admit_quotes(self):
        for cutoff in ("2026-04-14T17:00:00", "2026-04-14T18:00:00Z"):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                provider.select_pair(self.fixture, "pinnacle", self.history, self.mapping,
                                     cutoff=cutoff)
        for payload in ({**self.history, "fixtureId": "wrong"},
                        {**self.history, "scope": "map"},
                        {**self.history, "mapId": 1}):
            self.assertIsNone(provider.select_pair(self.fixture, "pinnacle", payload, self.mapping)[1])


class MarketHistoryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root / "app.sqlite3"
        self.fixture = {"fixtureId": "id1", "participant1Name": "Beta", "participant2Name": "Alpha",
                        "startTime": "2026-04-14T18:00:00Z", "trueStartTime": "2026-04-14T19:00:00Z",
                        "sportId": 17, "statusId": 2, "tournamentName": "PGL Bucharest"}
        self.results = [{"match_id": "100", "match_url": "https://www.hltv.org/matches/100/a",
                         "date": "2026-04-14T12:00:00", "team1": "Alpha", "team2": "Beta",
                         "tournament": "PGL Bucharest", "format": "bo3",
                         "score1": "2", "score2": "1", "winner": "team1"}]
        connection = sqlite3.connect(self.database)
        connection.executescript("""
            CREATE TABLE fixtures(fixture_id TEXT,start_time TEXT,metadata_json TEXT);
            CREATE TABLE raw_responses(id INTEGER PRIMARY KEY,endpoint TEXT,parameters_json TEXT,
                received_at TEXT,http_status INTEGER,error_code TEXT,terminal INTEGER,
                raw_path TEXT,file_sha256 TEXT,decoded_sha256 TEXT);
            CREATE TABLE links(fixture_id TEXT,history_key TEXT,status TEXT);
            CREATE TABLE historical_matches(history_key TEXT,source TEXT,source_match_id TEXT,
                score1 INTEGER,score2 INTEGER,winner_1 INTEGER);
        """)
        connection.execute("INSERT INTO fixtures VALUES(?,?,?)",
                           ("id1", self.fixture["startTime"], json.dumps(self.fixture)))
        connection.commit()
        connection.close()

    def payload(self, *, price=3.0):
        return {"fixtureId": "id1", "bookmakers": {"pinnacle": {"markets": {"171": {"outcomes": {
            "171": {"players": {"0": [
                {"createdAt": "2026-04-13T23:00:00Z", "price": price, "active": True},
                {"createdAt": "2026-04-14T16:00:00Z", "price": 4.0, "active": True},
                {"createdAt": "2026-04-14T17:30:00Z", "price": 5.0, "active": True}]}},
            "172": {"players": {"0": [
                {"createdAt": "2026-04-13T23:00:00Z", "price": 2.0, "active": True},
                {"createdAt": "2026-04-14T16:00:00Z", "price": 2.0, "active": True}]}}
        }}}}}}

    def receipt(self, identity, payload, *, books="pinnacle,bet365,unibet",
                received="2026-10-01T00:00:00Z", status=200, terminal=1, **changes):
        body = json.dumps(payload).encode()
        compressed = gzip.compress(body, mtime=0)
        relative = f"raw/{identity}.json.gz"
        (self.root / "raw").mkdir(exist_ok=True)
        (self.root / relative).write_bytes(compressed)
        row = dict(id=identity, endpoint="/historical-odds",
                   parameters_json=json.dumps({"fixtureId": "id1", "bookmakers": books}),
                   received_at=received, http_status=status, error_code=None, terminal=terminal,
                   raw_path=relative, file_sha256=hashlib.sha256(compressed).hexdigest(),
                   decoded_sha256=hashlib.sha256(body).hexdigest())
        row.update(changes)
        connection = sqlite3.connect(self.database)
        fields = ",".join(row)
        connection.execute(f"INSERT INTO raw_responses({fields}) VALUES({','.join('?' for _ in row)})",
                           tuple(row.values()))
        connection.commit()
        connection.close()

    def load(self):
        return load_market_history(self.database, self.results)

    def test_real_starts_origins_orientation_and_derived_provenance_are_preserved(self):
        self.receipt(1, self.payload())
        result = self.load()
        self.assertFalse(result["links"][0]["participant1_is_hltv_team1"])
        expected = {"one_hour": (4.0, "2026-04-14T17:00:00Z"),
                    "six_hour": (3.0, "2026-04-14T12:00:00Z"),
                    "source_day": (3.0, "2026-04-14T00:00:00Z")}
        for horizon, (price, origin) in expected.items():
            row = result["quotes_by_horizon"][horizon][0]
            self.assertEqual(row["status"], "admitted")
            self.assertEqual(row["price1"], price)
            self.assertEqual(row["forecast_origin_utc"], origin)
            self.assertEqual(row["id"], f"asof:{horizon}:1:pinnacle")
            details = json.loads(row["details_json"])
            self.assertEqual(details["quote"]["startTime"], self.fixture["startTime"])
            self.assertEqual(details["derived_selection"]["raw_id"], 1)
            self.assertEqual(row["observed_at"], "2026-10-01T00:00:00Z")
        self.assertTrue(result["audit"]["read_only"])
        self.assertEqual(result["audit"]["normalized_states_loaded"], 0)

    def test_latest_requested_book_receipt_including_absence_masks_older_success(self):
        self.receipt(1, self.payload())
        self.receipt(2, {"error": {"code": "NOT_FOUND", "message": "No historical odds found."}},
                     received="2026-10-02T00:00:00Z", books=["pinnacle"], status=404)
        rows = self.load()["quotes_by_horizon"]["one_hour"]
        self.assertEqual(rows[0]["status"], "no_history")
        self.assertEqual(rows[0]["raw_id"], 2)
        self.assertEqual(rows[1]["raw_id"], 1)
        self.assertEqual(rows[1]["status"], "book_missing")

    def test_missing_book_suspension_and_failed_latest_never_revive_old_response(self):
        for case in ("book_missing", "suspended", "failed"):
            with self.subTest(case=case):
                self.setUp()
                self.receipt(1, self.payload())
                payload = self.payload()
                if case == "book_missing":
                    payload["bookmakers"] = {}
                elif case == "suspended":
                    payload["bookmakers"]["pinnacle"]["markets"]["171"]["outcomes"]["171"]["players"]["0"].append(
                        {"createdAt": "2026-04-14T16:30:00Z", "price": 4.0, "active": False})
                self.receipt(2, payload, received="2026-10-02T00:00:00Z",
                             terminal=0 if case == "failed" else 1)
                row = self.load()["quotes_by_horizon"]["one_hour"][0]
                self.assertNotEqual(row["status"], "admitted")
                self.assertEqual(row["raw_id"], 2)

    def test_conflicting_equal_receipts_are_excluded_not_tiebroken_by_id(self):
        self.receipt(1, self.payload())
        self.receipt(2, self.payload(price=6.0))
        row = self.load()["quotes_by_horizon"]["one_hour"][0]
        self.assertEqual(row["status"], "ambiguous_latest_receipt")

    def test_hashes_path_escape_fixture_identity_and_full_parent_scope_are_rejected(self):
        for changes, payload in (
            ({"file_sha256": "0" * 64}, self.payload()),
            ({"decoded_sha256": "0" * 64}, self.payload()),
            ({"raw_path": "../outside.gz"}, self.payload()),
            ({}, {**self.payload(), "fixtureId": "wrong"}),
            ({}, {**self.payload(), "scope": "map"}),
        ):
            with self.subTest(changes=changes, scope=payload.get("scope")):
                self.setUp()
                self.receipt(1, self.payload(), received="2026-09-01T00:00:00Z")
                self.receipt(2, payload, **changes)
                row = self.load()["quotes_by_horizon"]["one_hour"][0]
                self.assertNotEqual(row["status"], "admitted")
                self.assertEqual(row["raw_id"], 2)

    def test_origin_before_native_day_opening_is_excluded_and_source_boundary_is_strict(self):
        self.fixture["startTime"] = "2026-04-14T00:30:00Z"
        connection = sqlite3.connect(self.database)
        connection.execute("UPDATE fixtures SET metadata_json=?", (json.dumps(self.fixture),))
        connection.commit()
        connection.close()
        payload = self.payload()
        payload["bookmakers"]["pinnacle"]["markets"]["171"]["outcomes"]["171"]["players"]["0"] = [
            {"createdAt": "2026-04-14T00:00:00Z", "price": 3.0, "active": True}]
        self.receipt(1, payload)
        result = self.load()
        self.assertEqual(result["quotes_by_horizon"]["one_hour"][0]["reason"],
                         "forecast_origin_before_model_source_day_opening")
        self.assertNotEqual(result["quotes_by_horizon"]["source_day"][0]["status"], "admitted")

    def test_latest_receipt_for_another_requested_book_does_not_mask_pinnacle(self):
        self.receipt(1, self.payload(), books=["pinnacle"])
        self.receipt(2, {"fixtureId": "id1", "bookmakers": {}}, books="bet365",
                     received="2026-10-02T00:00:00Z")
        rows = self.load()["quotes_by_horizon"]["one_hour"]
        self.assertEqual(rows[0]["raw_id"], 1)
        self.assertEqual(rows[0]["status"], "admitted")
        self.assertEqual(rows[1]["raw_id"], 2)
        self.assertEqual(rows[2]["reason"], "missing_requested_book_receipt")

    def test_each_relevant_parent_scope_and_invalid_latest_state_blocks_the_pair(self):
        for level in ("book", "market", "outcome", "state"):
            with self.subTest(level=level):
                self.setUp()
                payload = self.payload()
                book = payload["bookmakers"]["pinnacle"]
                market = book["markets"]["171"]
                outcome = market["outcomes"]["171"]
                if level == "state":
                    outcome["players"]["0"][1]["price"] = 1.0
                else:
                    {"book": book, "market": market, "outcome": outcome}[level]["playerId"] = 7
                self.receipt(1, payload)
                self.assertNotEqual(self.load()["quotes_by_horizon"]["one_hour"][0]["status"], "admitted")

    def test_raw_corruption_and_symlink_escape_are_excluded_without_database_writes(self):
        self.receipt(1, self.payload())
        before = self.database.read_bytes()
        (self.root / "raw/1.json.gz").write_bytes(b"corrupt")
        self.assertEqual(self.load()["quotes_by_horizon"]["one_hour"][0]["reason"], "invalid_raw_evidence")
        self.assertEqual(self.database.read_bytes(), before)
        with tempfile.TemporaryDirectory() as outside:
            path = Path(outside) / "evidence.gz"
            path.write_bytes(gzip.compress(json.dumps(self.payload()).encode()))
            (self.root / "raw/1.json.gz").unlink()
            (self.root / "raw/1.json.gz").symlink_to(path)
            self.assertEqual(self.load()["quotes_by_horizon"]["one_hour"][0]["reason"], "invalid_raw_evidence")

    def test_unorderable_receipt_and_source_clocks_never_bypass_latest_evidence(self):
        self.receipt(1, self.payload(), received="2026-09-01T00:00:00Z")
        self.receipt(2, self.payload(), received="naive-or-invalid")
        row = self.load()["quotes_by_horizon"]["one_hour"][0]
        self.assertEqual(row["reason"], "unorderable_receipt_clock")
        self.assertEqual(row["raw_id"], 2)
        self.setUp()
        payload = self.payload()
        payload["bookmakers"]["pinnacle"]["markets"]["171"]["outcomes"]["171"]["players"]["0"].append(
            {"createdAt": "2026-04-14T16:30:00", "price": 3.0, "active": True})
        self.receipt(1, payload)
        self.assertNotEqual(self.load()["quotes_by_horizon"]["one_hour"][0]["status"], "admitted")

    def test_native_id_and_provider_result_anchors_must_agree_with_the_crosswalk(self):
        for source, source_id, scores in (("hltv", "101", (2, 1, 1)),
                                          ("oddspapi", "id1", (2, 1, 1))):
            with self.subTest(source=source):
                self.setUp()
                connection = sqlite3.connect(self.database)
                connection.execute("INSERT INTO historical_matches VALUES(?,?,?,?,?,?)",
                                   (f"{source}:{source_id}", source, source_id, *scores))
                if source == "hltv":
                    connection.execute("INSERT INTO links VALUES(?,?,?)",
                                       ("id1", f"{source}:{source_id}", "automatic"))
                connection.commit()
                connection.close()
                with self.assertRaises(ValueError):
                    self.load()

    def test_ambiguous_native_matches_cannot_supply_market_quotes(self):
        self.receipt(1, self.payload())
        self.results.append({**self.results[0], "match_id": "101",
                             "match_url": "https://www.hltv.org/matches/101/b"})
        result = self.load()
        self.assertEqual(result["links"][0]["mapping_status"], "ambiguous_fixture")
        self.assertEqual(result["quotes_by_horizon"]["one_hour"][0]["reason"], "no_audited_native_mapping")


