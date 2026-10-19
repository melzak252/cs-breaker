"""Durable research evidence and worker lifecycle contracts, using isolated SQLite."""
import gzip
import hashlib
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from cs_app.store import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = Store(self.root)

    def fixture(self, identity='fixture1', status=0):
        return {'fixtureId': identity, 'sportId': 17, 'statusId': status,
                'startTime': '2026-10-06T16:00:00Z', 'trueStartTime': None,
                'participant1Id': 11, 'participant2Id': 22,
                'participant1Name': 'Alpha', 'participant2Name': 'Beta',
                'tournamentName': 'CS2 Cup'}

    def receipt(self, clock='2026-10-06T12:00:00Z', endpoint='/historical-odds'):
        return {'endpoint': endpoint,
                'parameters': {'fixtureId': 'fixture1', 'bookmakers': 'pinnacle'},
                'request_started_at': '2026-10-06T11:59:59Z', 'received_at': clock,
                'http_status': 200, 'error_code': None, 'terminal': True}

    def history(self, identity='match1', source='opencs2', score1=2, score2=1,
                updated_at='2026-10-06T17:00:00Z'):
        metadata = {'scores': {'periods': {'result': {'updatedAt': updated_at}}}}
        return {'history_key': f'{source}:{identity}', 'source': source,
                'source_match_id': identity, 'start_time': '2026-10-06T16:00:00Z',
                'source_date': '2026-10-06T16:00:00Z', 'timezone_status': 'aware',
                'team1': 'Alpha', 'team2': 'Beta', 'tournament': 'CS2 Cup',
                'score1': score1, 'score2': score2, 'winner_1': int(score1 > score2),
                'format': 'bo3', 'label_status': 'observed_result',
                'metadata_json': json.dumps(metadata), 'source_sha256': None}

    def test_reopening_preserves_history_and_raw_state_replay_is_idempotent(self):
        self.store.upsert_fixture(self.fixture())
        self.store.put_history(self.history())
        self.store.put_history(self.history())
        body = b'{"fixtureId":"fixture1","bookmakers":{}}\n'
        raw_id = self.store.put_raw(self.receipt(), body)
        self.assertEqual(self.store.put_raw(self.receipt(), body, origin='import'), raw_id)
        states = [{'createdAt': '2026-10-06T11:00:00Z', 'active': True, 'price': 2.4},
                  {'createdAt': '2026-10-06T11:30:00Z', 'active': False, 'price': 2.2},
                  {'createdAt': 'not-a-clock', 'active': 'unknown', 'price': True}]
        self.store.add_states('fixture1', 'pinnacle', '171', '1711', states, raw_id)
        self.store.add_states('fixture1', 'pinnacle', '171', '1711', states, raw_id)
        reopened = Store(self.root)
        self.assertEqual(reopened.one('SELECT count(*) AS n FROM historical_matches')['n'], 1)
        rows = reopened.query('SELECT * FROM odds_states ORDER BY sequence_no')
        self.assertEqual([row['active'] for row in rows], [1, 0, None])
        self.assertIsNone(rows[2]['source_time'])
        self.assertIsNone(rows[2]['price'])
        self.assertEqual(json.loads(rows[2]['state_json']), states[2])
        raw = reopened.one('SELECT * FROM raw_responses WHERE id=?', (raw_id,))
        saved = (self.root / raw['raw_path']).read_bytes()
        self.assertEqual(gzip.decompress(saved), body)
        self.assertEqual(raw['decoded_sha256'], hashlib.sha256(body).hexdigest())
        self.assertEqual(raw['file_sha256'], hashlib.sha256(saved).hexdigest())
        self.assertEqual(raw['origin'], 'network')

    def test_same_receipt_or_state_identity_cannot_replace_existing_evidence(self):
        self.store.upsert_fixture(self.fixture())
        record = self.receipt()
        raw_id = self.store.put_raw(record, b'{}')
        conflicting = dict(record, terminal=False)
        with self.assertRaises(ValueError):
            self.store.put_raw(conflicting, b'{}')
        first = {'createdAt': '2026-10-06T11:00:00Z', 'active': True, 'price': 2.0}
        self.store.add_states('fixture1', 'pinnacle', '171', '1711', [first], raw_id)
        with self.assertRaises(ValueError):
            self.store.add_states('fixture1', 'pinnacle', '171', '1711',
                                  [dict(first, price=3.0), dict(first, price=4.0)], raw_id)
        self.assertEqual(self.store.query('SELECT price FROM odds_states'), [{'price': 2.0}])

    def test_stale_fixture_import_does_not_regress_fresh_native_metadata(self):
        self.store.upsert_fixture(self.fixture(status=0), '2026-10-05T12:00:00Z')
        finished = dict(self.fixture(status=2), trueStartTime='2026-10-06T15:58:00Z')
        self.store.upsert_fixture(finished, '2026-10-06T18:00:00Z')
        self.store.upsert_fixture(self.fixture(status=0), '2026-10-05T23:00:00Z')
        row = self.store.one('SELECT * FROM fixtures WHERE fixture_id=?', ('fixture1',))
        self.assertEqual(row['status_id'], 2)
        self.assertEqual(json.loads(row['metadata_json']), finished)
        with self.assertRaises(ValueError):
            self.store.upsert_fixture(dict(finished, participant1Name='Wrong'), '2026-10-06T18:00:00Z')
        self.assertEqual(self.store.one('SELECT participant1_name FROM fixtures')['participant1_name'], 'Alpha')

    def test_history_conflict_and_explicit_transaction_roll_back_without_lost_rows(self):
        self.store.put_history(self.history())
        with self.assertRaises(ValueError):
            self.store.put_history(self.history(score1=1, score2=2))
        self.assertEqual(self.store.one('SELECT score1 FROM historical_matches')['score1'], 2)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.transaction() as connection:
                connection.execute('INSERT INTO settings(key,value_json) VALUES(?,?)', ('temporary', '1'))
                connection.execute('INSERT INTO settings(key,value_json) VALUES(?,?)', ('temporary', '2'))
        self.assertIsNone(self.store.one('SELECT * FROM settings WHERE key=?', ('temporary',)))

    def test_native_score_correction_requires_later_clock_and_preserves_revision(self):
        original = self.history(source='oddspapi')
        self.store.put_history(original)
        self.store.put_history(self.history(source='oddspapi', score1=1, score2=2,
                                           updated_at='2026-10-06T17:05:00Z'))
        row = self.store.one('SELECT * FROM historical_matches')
        self.assertEqual((row['score1'], row['score2'], row['winner_1']), (1, 2, 0))
        revision = self.store.one('SELECT * FROM history_revisions')
        self.assertEqual(json.loads(revision['previous_json'])['score1'], 2)
        self.assertEqual(json.loads(revision['replacement_json'])['score1'], 1)
        with self.assertRaises(ValueError):
            self.store.put_history(self.history(source='oddspapi', score1=3, score2=0,
                                               updated_at='2026-10-06T17:05:00Z'))
        self.assertEqual(self.store.one('SELECT count(*) AS n FROM history_revisions')['n'], 1)

    def test_current_withdrawal_is_latest_observation_without_deleting_old_quote(self):
        self.store.upsert_fixture(self.fixture())
        first = self.store.put_raw(self.receipt(endpoint='/odds'), b'{"observation":1}')
        quote = {'price_1': 2.0, 'price_2': 2.5, 'no_vig_probability_1': 5 / 9,
                 'source_created_at_1': '2026-10-06T11:59:00Z',
                 'source_created_at_2': '2026-10-06T11:59:00Z', 'cutoff': '2026-10-06T16:00:00Z'}
        self.store.put_quote('fixture1', 'pinnacle', 'current', quote,
                             {'status': 'admitted', 'reason': 'two_sided_source_quote'}, first)
        self.store.put_quote('fixture1', 'pinnacle', 'current', quote,
                             {'status': 'admitted', 'reason': 'two_sided_source_quote'}, first)
        second = self.store.put_raw(self.receipt('2026-10-06T13:00:00Z', '/odds'), b'{"observation":2}')
        self.store.put_quote('fixture1', 'pinnacle', 'current', None,
                             {'status': 'unusable_state', 'reason': 'latest_inactive'}, second)
        latest = self.store.one('SELECT * FROM quotes ORDER BY observed_at DESC,id DESC LIMIT 1')
        self.assertEqual(latest['status'], 'unusable_state')
        self.assertIsNone(latest['price1'])
        self.assertIsNone(latest['probability1'])
        self.assertEqual(self.store.one('SELECT count(*) AS n FROM quotes')['n'], 2)
        self.assertEqual(self.store.one('SELECT price1 FROM quotes WHERE raw_id=?', (first,))['price1'], 2.0)

    def test_raw_account_body_and_any_digest_are_rejected_without_artifacts(self):
        record = dict(self.receipt(endpoint='/account'), parameters={})
        for body, extra in [(b'{"apiKey":"private-key"}', {}),
                            (None, {'decoded_sha256': 'a' * 64}),
                            (None, {'file_sha256': 'b' * 64}),
                            (None, {'account': {'apiKey': 'private-key'}})]:
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    self.store.put_raw(dict(record, **extra), body)
        self.assertEqual(self.store.query('SELECT * FROM raw_responses'), [])
        self.assertEqual(list(self.store.raw_root.iterdir()), [])
        raw_id = self.store.put_raw(record, None)
        safe = self.store.one('SELECT * FROM raw_responses WHERE id=?', (raw_id,))
        self.assertIsNone(safe['decoded_sha256'])
        self.assertIsNone(safe['raw_path'])
        self.assertNotIn('sha256', safe['receipt_json'])

    def test_secret_guard_and_sensitive_parameters_never_reach_disk(self):
        sentinel = b'private-secret-value'
        record = dict(self.receipt(), secret_guard=lambda value: sentinel in value)
        with self.assertRaises(ValueError):
            self.store.put_raw(record, b'{"echo":"private-secret-value"}')
        with self.assertRaises(ValueError):
            self.store.put_raw(dict(self.receipt(), parameters={'apiKey': sentinel.decode()}), b'{}')
        self.assertEqual(self.store.query('SELECT * FROM raw_responses'), [])
        self.assertEqual(list(self.store.raw_root.iterdir()), [])

    def test_raw_path_input_cannot_escape_content_addressed_store(self):
        record = dict(self.receipt(), raw_path='../../outside.json.gz')
        raw_id = self.store.put_raw(record, b'{}')
        path = self.store.one('SELECT raw_path FROM raw_responses WHERE id=?', (raw_id,))['raw_path']
        self.assertTrue((self.root / path).resolve().is_relative_to(self.store.raw_root.resolve()))
        self.assertFalse((self.root.parent / 'outside.json.gz').exists())
        with self.assertRaises(ValueError):
            self.store.upsert_fixture(self.fixture("fixture'; DROP TABLE fixtures;--"))

    def test_failed_attempt_receipts_deduplicate_without_a_body_and_count_billable(self):
        failed = dict(self.receipt(endpoint='/fixtures'), parameters={}, terminal=False,
                      http_status=500, error_code='PROVIDER_ERROR')
        first = self.store.put_raw(failed, None, billable=True)
        self.assertEqual(self.store.put_raw(failed, None, billable=True), first)
        row = self.store.one('SELECT count(*) AS n,sum(billable) AS billed FROM raw_responses')
        self.assertEqual(row, {'n': 1, 'billed': 1})

    def test_claim_is_atomic_and_excludes_second_worker_until_running_finishes(self):
        jobs = [self.store.create_job('upcoming', {}) for _ in range(2)]
        barrier = Barrier(2)

        def claim(worker):
            barrier.wait()
            return self.store.claim_job(worker)

        with ThreadPoolExecutor(max_workers=2) as workers:
            claimed = list(workers.map(claim, ['worker-a', 'worker-b']))
        winner = next(item for item in claimed if item is not None)
        self.assertEqual(sum(item is not None for item in claimed), 1)
        self.assertIn(winner['id'], jobs)
        self.store.update_job(winner['id'], status='completed', result={'fixtures': 1})
        next_job = self.store.claim_job('worker-c')
        self.assertNotEqual(next_job['id'], winner['id'])
        self.assertEqual(next_job['status'], 'running')

    def test_cancel_retry_and_crash_recovery_keep_resumable_progress(self):
        queued = self.store.create_job('upcoming', {})
        self.assertTrue(self.store.cancel_job(queued))
        self.assertEqual(self.store.get_job(queued)['status'], 'cancelled')
        self.assertIsNone(self.store.claim_job('worker'))
        self.assertTrue(self.store.retry_job(queued))
        running = self.store.claim_job('worker')
        self.store.update_job(running['id'], progress={'done_fixture_ids': ['fixture1']})
        self.assertTrue(self.store.cancel_job(queued))
        self.assertTrue(self.store.cancelled(queued))
        self.store.update_job(queued, status='cancelled', result={'cancelled': True})
        self.assertTrue(self.store.retry_job(queued))
        self.assertFalse(self.store.cancelled(queued))
        self.assertEqual(self.store.claim_job('worker')['progress'], {'done_fixture_ids': ['fixture1']})
        reopened = Store(self.root)
        reopened.recover_jobs()
        recovered = reopened.get_job(queued)
        self.assertEqual(recovered['status'], 'partial')
        self.assertEqual(recovered['progress'], {'done_fixture_ids': ['fixture1']})
        self.assertTrue(reopened.retry_job(queued))
        self.assertEqual(reopened.claim_job('worker')['id'], queued)
        reopened.update_job(queued, status='completed')
        self.assertFalse(reopened.retry_job(queued))

    def test_invalid_job_transition_rolls_back_progress_and_redacts_private_error(self):
        job = self.store.create_job('upcoming', {})
        with self.assertRaises(ValueError):
            self.store.update_job(job, status='completed', progress={'lost': True})
        self.assertEqual(self.store.get_job(job)['progress'], {})
        self.store.claim_job('worker')
        self.store.update_job(job, status='failed', error='HTTP https://provider/?apiKey=private-secret-value')
        self.assertNotIn('private-secret-value', self.store.get_job(job)['error'])

    def test_cancellation_wins_the_concurrent_completion_transition(self):
        job = self.store.create_job('upcoming', {})
        self.store.claim_job('worker')
        self.store.cancel_job(job)
        self.store.update_job(job, status='completed', result={'observed': 2})
        self.assertEqual(self.store.get_job(job)['status'], 'cancelled')
        self.assertEqual(self.store.get_job(job)['result'], {'observed': 2})

    def test_budget_paused_and_failed_jobs_resume_without_erasing_progress(self):
        job = self.store.create_job('finished', {})
        self.store.claim_job('worker')
        self.store.update_job(job, status='budget_paused', progress={'next_fixture': 4},
                              result={'billable_attempts': 2})
        self.assertTrue(self.store.retry_job(job))
        claimed = self.store.claim_job('worker')
        self.assertEqual(claimed['progress'], {'next_fixture': 4})
        self.assertEqual(claimed['result'], {'billable_attempts': 2})
        self.store.update_job(job, status='failed', error='Provider unavailable')
        self.assertTrue(self.store.retry_job(job))
        resumed = self.store.claim_job('worker')
        self.assertEqual(resumed['progress'], {'next_fixture': 4})
        self.assertIsNone(resumed['error'])

    def test_settings_persist_and_reject_invalid_subset_atomically(self):
        saved = self.store.save_settings({'bookmakers': ['pinnacle', 'stake.com'],
                                         'billable_per_run': 0, 'upcoming_days': 7,
                                         'scheduler_enabled': True})
        self.assertEqual(Store(self.root).get_settings(), saved)
        for values in [None, [], '{"max_requests":1}',
                       {'billable_per_run': True}, {'billable_per_day': -1},
                       {'max_requests': 1.5}, {'max_history': '1'}, {'max_fixtures': 10001},
                       {'upcoming_days': 8}, {'finished_days': 0},
                       {'min_snapshot_minutes': 0}, {'scheduler_interval_minutes': float('inf')},
                       {'scheduler_enabled': 1}, {'bookmakers': []},
                       {'bookmakers': ['pinnacle', 'pinnacle']}, {'bookmakers': ['../pinnacle']},
                       {'bookmakers': ['a', 'b', 'c', 'd']}, {'api_key': 'private'},
                       {'max_requests': 5, 'unknown': True}]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.store.save_settings(values)
                self.assertEqual(self.store.get_settings(), saved)

    def test_job_dates_caps_and_books_are_validated_before_persistence(self):
        for payload in [{'from': '2026-10-01', 'to': '2026-10-05T00:00:00Z'},
                        {'from': '2026-10-06T00:00:00Z', 'to': '2026-10-05T00:00:00Z'},
                        {'from': '2026-10-01T00:00:00Z'},
                        {'from': '2026-10-01T00:00:00Z', 'to': '2026-10-05T00:00:00Z', 'max_requests': True},
                        {'from': '2026-10-01T00:00:00Z', 'to': '2026-10-05T00:00:00Z', 'bookmakers': 'pinnacle'}]:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.store.create_job('backfill', payload)
        self.assertEqual(self.store.list_jobs(), [])
        job = self.store.create_job('backfill', {'from': '2026-01-01T00:00:00Z',
                                               'to': '2026-10-05T00:00:00Z',
                                               'max_requests': 0, 'bookmakers': ['stake.com']})
        self.assertEqual(self.store.get_job(job)['payload']['max_requests'], 0)

    def test_foreign_receipt_cannot_be_normalized_into_another_fixture(self):
        self.store.upsert_fixture(self.fixture())
        self.store.upsert_fixture(self.fixture('fixture2'))
        raw_id = self.store.put_raw(self.receipt(), b'{}')
        with self.assertRaises(ValueError):
            self.store.add_states('fixture2', 'pinnacle', '171', '1711', [], raw_id)
        with self.assertRaises(ValueError):
            self.store.put_quote('fixture2', 'pinnacle', 'current', None,
                                 {'status': 'book_missing', 'reason': 'book_missing'}, raw_id)
        self.assertEqual(self.store.query('SELECT * FROM quotes'), [])

    def test_corrupted_raw_file_is_not_overwritten_on_receipt_replay(self):
        record = self.receipt()
        raw_id = self.store.put_raw(record, b'{"immutable":true}')
        raw = self.store.one('SELECT * FROM raw_responses WHERE id=?', (raw_id,))
        path = self.root / raw['raw_path']
        path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            self.store.put_raw(record, b'{"immutable":true}')
        self.assertEqual(path.read_bytes(), b'corrupt')
        self.assertEqual(self.store.one('SELECT count(*) AS n FROM raw_responses')['n'], 1)

    def test_unknown_future_schema_is_rejected_without_deleting_current_data(self):
        self.store.put_history(self.history())
        self.store.execute('PRAGMA user_version=999')
        with self.assertRaises(ValueError):
            Store(self.root)
        self.assertEqual(self.store.one('SELECT score1 FROM historical_matches')['score1'], 2)

    def test_aware_offsets_normalize_but_naive_state_clocks_remain_unorderable(self):
        self.store.upsert_fixture(self.fixture())
        raw_id = self.store.put_raw(self.receipt(), b'{}')
        states = [{'changedAt': '2026-10-06T14:00:00+02:00', 'active': True, 'price': '2.25'},
                  {'createdAt': '2026-10-06T11:30:00', 'active': False, 'price': float('nan')}]
        self.store.add_states('fixture1', 'pinnacle', '171', '1711', states, raw_id)
        rows = self.store.query('SELECT * FROM odds_states ORDER BY sequence_no')
        self.assertEqual(rows[0]['source_time'], '2026-10-06T12:00:00.000000Z')
        self.assertEqual(rows[0]['price'], 2.25)
        self.assertIsNone(rows[1]['source_time'])
        self.assertIsNone(rows[1]['price'])
        self.assertTrue(json.loads(rows[1]['state_json'])['price'] !=
                        json.loads(rows[1]['state_json'])['price'])


if __name__ == '__main__':
    unittest.main()
