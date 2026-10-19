"""Research-facing application invariants on isolated SQLite and real imports."""
import csv
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

from cs_app.store import Store
from cs_app.services import rebuild_mappings, decide_mapping, iter_export, compute_metrics, import_history, states_list


class ApplicationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Store(Path(self.temporary.name))
        self.store.upsert_fixture({'fixtureId': 'id1', 'sportId': 17, 'statusId': 2,
            'participant1Id': 1, 'participant2Id': 2, 'participant1Name': 'Beta',
            'participant2Name': 'Alpha', 'startTime': '2026-04-14T10:00:00Z',
            'tournamentName': 'PGL Bucharest'})
        self.store.put_history({'history_key': 'opencs2:100', 'source': 'opencs2',
            'source_match_id': '100', 'start_time': '2026-04-14T12:00:00Z',
            'source_date': '2026-04-14T12:00:00Z', 'timezone_status': 'aware',
            'team1': 'Alpha', 'team2': 'Beta', 'tournament': 'PGL Bucharest 2026',
            'score1': 2, 'score2': 1, 'winner_1': 1, 'format': 'bo3',
            'label_status': 'valid_explicit_bo3_result', 'metadata_json': '{}'})

    def test_mapping_preserves_reversed_orientation_and_rejection_survives_rebuild(self):
        rebuild_mappings(self.store)
        link = self.store.one('SELECT * FROM links WHERE fixture_id=?', ('id1',))
        self.assertEqual(link['history_key'], 'opencs2:100')
        self.assertEqual(link['orientation'], 0)
        decide_mapping(self.store, 'id1', 'reject', reason='Different fixture; audited by researcher')
        rebuild_mappings(self.store)
        self.assertEqual(self.store.one('SELECT status FROM links WHERE fixture_id=?', ('id1',))['status'], 'rejected')
        self.assertEqual(len(self.store.query('SELECT * FROM link_decisions WHERE fixture_id=?', ('id1',))), 1)

    def test_manual_mapping_cannot_assign_one_result_to_two_fixtures(self):
        self.store.upsert_fixture({'fixtureId': 'id2', 'sportId': 17, 'statusId': 2,
            'participant1Id': 1, 'participant2Id': 2, 'participant1Name': 'Alpha',
            'participant2Name': 'Beta', 'startTime': '2026-04-14T11:00:00Z',
            'tournamentName': 'PGL Bucharest'})
        decide_mapping(self.store, 'id1', 'accept', history_key='opencs2:100', orientation=0, reason='Inspected evidence')
        with self.assertRaises(ValueError):
            decide_mapping(self.store, 'id2', 'accept', history_key='opencs2:100', orientation=1, reason='Conflicting reuse')
        self.assertIsNone(self.store.one('SELECT * FROM links WHERE fixture_id=?', ('id2',)))

    def test_export_contains_stable_match_identifiers_without_model_features(self):
        data = ''.join(iter_export(self.store, 'fixtures'))
        self.assertIn('fixture_id', data)
        self.assertIn('id1', data)
        self.assertIn('2026-04-14', data)
        self.assertNotIn('apiKey', data)

    def test_missing_quotes_produce_undefined_loss_not_perfect_zero(self):
        rebuild_mappings(self.store)
        metrics = compute_metrics(self.store)
        self.assertEqual(metrics['available_cohorts']['pinnacle']['matches'], 0)
        self.assertIsNone(metrics['available_cohorts']['pinnacle']['log_loss'])

    @staticmethod
    def kaggle_csv(rows):
        output = io.StringIO(newline='')
        writer = csv.writer(output)
        writer.writerow(['hltv_match_id', 'date', 'tournament', 'team1_name', 'team2_name',
                         'match_type', 'score_team1', 'score_team2', 'winner'])
        writer.writerows(rows)
        return output.getvalue().encode()

    def test_csv_import_retains_original_bytes_and_is_idempotent(self):
        body = self.kaggle_csv([['200', '2026-04-14T12:00:00+00:00', 'PGL', 'Alpha', 'Beta',
                                 'bo3', 2, 1, 'team1']])
        import_history(self.store, 'kaggle', body)
        import_history(self.store, 'kaggle', body)
        row = self.store.one("SELECT * FROM historical_matches WHERE history_key='kaggle:200'")
        self.assertEqual(row['winner_1'], 1)
        self.assertEqual(row['start_time'], '2026-04-14T12:00:00.000000Z')
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM historical_matches WHERE source='kaggle'")['n'], 1)
        receipt = self.store.one("SELECT * FROM raw_responses WHERE endpoint='local://history/kaggle'")
        self.assertEqual(receipt['http_status'], 0)
        self.assertEqual(receipt['billable'], 0)
        self.assertEqual(gzip.decompress((self.store.root / receipt['raw_path']).read_bytes()), body)

    def test_conflicting_csv_identity_rejects_entire_batch(self):
        body = self.kaggle_csv([
            ['200', '2026-04-14T12:00:00Z', 'PGL', 'Alpha', 'Beta', 'bo3', 2, 0, 'team1'],
            ['201', '2026-04-14T13:00:00Z', 'PGL', 'Gamma', 'Delta', 'bo3', 2, 1, 'team1'],
            ['200', '2026-04-14T12:00:00Z', 'PGL', 'Alpha', 'Beta', 'bo3', 0, 2, 'team2']])
        with self.assertRaises(ValueError):
            import_history(self.store, 'kaggle', body)
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM historical_matches WHERE source='kaggle'")['n'], 0)

    def test_invalid_nonfinite_state_remains_evidence_but_api_projection_is_json_safe(self):
        body = b'{"price":1e999}'
        raw_id = self.store.put_raw({'endpoint': '/historical-odds', 'parameters': {'fixtureId': 'id1'},
            'request_started_at': '2026-10-06T12:00:00Z', 'received_at': '2026-10-06T12:00:01Z',
            'http_status': 200, 'terminal': True}, body, origin='import', billable=False)
        self.store.add_states('id1', 'pinnacle', '171', '171', [{
            'createdAt': '2026-04-14T09:00:00Z', 'active': True, 'price': float('inf')}], raw_id)
        response = states_list(self.store, 'id1')
        self.assertIsNone(response['items'][0]['price'])
        self.assertIsNone(response['items'][0]['state']['price'])
        json.dumps(response, allow_nan=False)
        raw = self.store.one('SELECT raw_path FROM raw_responses WHERE id=?', (raw_id,))
        self.assertEqual(gzip.decompress((self.store.root / raw['raw_path']).read_bytes()), body)


if __name__ == '__main__':
    unittest.main()
