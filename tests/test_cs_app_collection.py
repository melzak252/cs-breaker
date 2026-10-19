"""Deterministic v4 transport exercises durable collection, not a fake Store."""
import gzip
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from scripts import collect_cs_oddspapi as provider
from cs_app.collection import Collector
from cs_app.store import Store


NOW = datetime(2026, 10, 6, 12, tzinfo=UTC)
BOOKS = ['pinnacle', 'bet365', 'unibet']


def fixture(identity='id1', status=2):
    return {'fixtureId': identity, 'sportId': 17, 'statusId': status,
            'participant1Id': 1, 'participant2Id': 2,
            'participant1Name': 'Beta', 'participant2Name': 'Alpha',
            'startTime': '2026-10-06T14:00:00Z' if status == 0 else '2026-04-14T10:00:00Z',
            'trueStartTime': None, 'tournamentName': 'PGL'}


def account(count=0, limit=200):
    return {'apiKey': 'private-key', 'current_subscription_id': 1,
            'subscriptions': [{'subscription_id': 1, 'is_active': True,
                'sport_ids': [17], 'bookmakers': {b: {} for b in BOOKS},
                'request_limit': limit, 'request_count': count}]}


def markets():
    return [{'marketId': 171, 'sportId': 17, 'marketName': 'Winner',
             'marketType': 'moneyline', 'period': 'result', 'playerProp': False,
             'marketLength': 2, 'outcomes': [{'outcomeId': 172, 'outcomeName': '2'},
                                          {'outcomeId': 171, 'outcomeName': '1'}]}]


def history(identity='id1'):
    def states(price):
        return [{'createdAt': '2026-04-14T09:00:00Z', 'active': True, 'price': price}]
    return {'fixtureId': identity, 'bookmakers': {'pinnacle': {'markets': {
        '171': {'outcomes': {'171': {'players': {'0': states(2)}},
                             '172': {'players': {'0': states(4)}}}},
        '999': {'unrelated': ['exact raw remains']}}}}}


def current(active=True, missing=False):
    data = fixture(status=0)
    sides = {'171': {'players': {'0': {'active': active, 'price': 2,
                                     'changedAt': '2026-10-06T11:00:00Z'}}}}
    if not missing:
        sides['172'] = {'players': {'0': {'active': True, 'price': 4,
                                        'changedAt': '2026-10-06T11:05:00Z'}}}
    data['bookmakerOdds'] = {'pinnacle': {'bookmakerIsActive': True, 'suspended': False,
        'markets': {'171': {'marketActive': True, 'outcomes': sides}, '999': {'other': 42}}}}
    return data


class Response:
    status = 200
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return self.body


class Transport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.callback = None
    def __call__(self, request, **kwargs):
        url = urlsplit(request.full_url)
        endpoint = url.path.removeprefix('/v4')
        params = {k: v[0] for k, v in parse_qs(url.query).items() if k != 'apiKey'}
        self.calls.append((endpoint, params))
        if self.callback:
            self.callback(endpoint)
        value = self.responses[endpoint]
        if isinstance(value, list) and endpoint not in ('/markets', '/fixtures'):
            value = value.pop(0)
        status, payload = value if isinstance(value, tuple) else (200, value)
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        if status != 200:
            import io
            raise HTTPError(request.full_url, status, 'private URL', {}, io.BytesIO(body))
        return Response(body)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name))
        self.store.save_settings({'billable_per_run': 20, 'billable_per_day': 40,
                                  'max_requests': 100, 'max_history': 100, 'max_fixtures': 20,
                                  'min_snapshot_minutes': 1})
        self.transport = Transport({'/account': account(), '/markets': markets(),
                                    '/fixtures': [fixture()], '/historical-odds': history()})
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(provider, 'urlopen', self.transport).start()
        mock.patch('cs_app.collection.now', return_value=NOW).start()
        mock.patch.object(provider, 'clock', side_effect=lambda: __import__('cs_app.collection', fromlist=['now']).now().isoformat()).start()
        mock.patch('cs_app.collection.ProviderClient.cooldowns',
                   {e: 0 for e in ['/account', '/markets', '/fixtures', '/historical-odds', '/odds', '/scores']}).start()

    def job(self, kind='backfill', **payload):
        if kind == 'backfill':
            payload = {'from': '2026-04-14T00:00:00Z', 'to': '2026-04-15T00:00:00Z', **payload}
        identity = self.store.create_job(kind, payload)
        return self.store.get_job(identity)

    def run_job(self, job, key='private-key'):
        return Collector(self.store, key).run(job)

    def test_per_job_fixture_cap_stops_after_one_of_two_frozen_fixtures(self):
        self.transport.responses['/fixtures'] = [fixture('id1'), fixture('id2')]
        result = self.run_job(self.job(max_fixtures=1))
        self.assertEqual(result['status'], 'budget_paused')
        self.assertEqual(result['progress']['fixture_ids'], ['id1', 'id2'])
        self.assertEqual(result['progress']['done_fixture_ids'], ['id1'])
        self.assertEqual([p['fixtureId'] for e, p in self.transport.calls if e == '/historical-odds'], ['id1'])

    def test_terminal_resume_is_offline_preserves_all_markets_and_source_clocks(self):
        job = self.job()
        result = self.run_job(job)
        self.assertEqual(result['status'], 'completed')
        quotes = self.store.query("SELECT * FROM quotes WHERE bookmaker='pinnacle'")
        self.assertEqual((quotes[0]['price1'], quotes[0]['price2'], quotes[0]['probability1']), (2, 4, 2/3))
        self.assertNotEqual(quotes[0]['source_time_1'], quotes[0]['observed_at'])
        self.assertEqual(len(self.store.query('SELECT * FROM quotes')), 3)
        raw = self.store.one("SELECT * FROM raw_responses WHERE endpoint='/historical-odds'")
        body = gzip.decompress((self.store.root / raw['raw_path']).read_bytes())
        self.assertEqual(json.loads(body)['bookmakers']['pinnacle']['markets']['999'],
                         {'unrelated': ['exact raw remains']})
        self.transport.calls.clear()
        resumed = self.run_job(self.store.get_job(job['id']), key='')
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual(resumed['actual_requests'], 0)
        self.assertEqual(self.transport.calls, [])
        self.assertEqual(len(self.store.query('SELECT * FROM quotes')), 3)
        self.assertTrue(all(r['raw_path'] is None and r['decoded_sha256'] is None
                            for r in self.store.query("SELECT * FROM raw_responses WHERE endpoint='/account'")))
        for path in self.store.root.rglob('*.gz'):
            self.assertNotIn(b'private-key', gzip.decompress(path.read_bytes()))

    def test_exact_absence_is_terminal_but_500_is_partial_and_retried_only_next_run(self):
        self.transport.responses['/fixtures'] = [fixture('id1'), fixture('id2')]
        self.transport.responses['/historical-odds'] = [
            (404, {'error': {'code': 'NOT_FOUND', 'message': 'No historical odds found.'}}),
            (500, {'error': {'code': 'SERVER_ERROR'}})]
        job = self.job()
        self.assertEqual(self.run_job(job)['status'], 'partial')
        rows = self.store.query("SELECT * FROM raw_responses WHERE endpoint='/historical-odds' ORDER BY id")
        self.assertEqual([r['terminal'] for r in rows], [1, 0])
        self.transport.calls.clear()
        self.transport.responses['/historical-odds'] = history('id2')
        self.assertEqual(self.run_job(self.store.get_job(job['id']))['status'], 'completed')
        self.assertEqual([p['fixtureId'] for e, p in self.transport.calls if e == '/historical-odds'], ['id2'])

    def test_failed_billable_attempt_counts_daily_and_pauses_next_job_without_provider(self):
        self.store.save_settings({'billable_per_day': 1})
        self.transport.responses['/markets'] = (500, {'error': {'code': 'SERVER_ERROR'}})
        self.assertEqual(self.run_job(self.job())['status'], 'partial')
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM raw_responses WHERE billable=1")['n'], 1)
        self.transport.calls.clear()
        self.assertEqual(self.run_job(self.job())['status'], 'budget_paused')
        self.assertEqual(self.transport.calls, [])

    def test_positive_headroom_and_hard_access_rate_and_secret_stops(self):
        for response in [(401, {}), (403, {}), (429, {}),
                         (200, {'error': {'code': 'REQUEST_LIMIT_EXCEEDED'}}),
                         (200, b'{"secret":"private-key"}')]:
            with self.subTest(response=response):
                self.transport.responses['/markets'] = response
                self.transport.calls.clear()
                with self.assertRaises(provider.Stop):
                    self.run_job(self.job())
                self.assertEqual([e for e, _ in self.transport.calls], ['/account', '/markets'])
        self.transport.responses['/account'] = account(count=200, limit=200)
        self.transport.calls.clear()
        with self.assertRaises(provider.Stop):
            self.run_job(self.job())
        self.assertEqual([e for e, _ in self.transport.calls], ['/account'])

    def test_latest_upcoming_batch_missing_or_inactive_supersedes_older_pair(self):
        self.transport.responses['/fixtures'] = [fixture(status=0)]
        for data in [current(), current(active=False), current(missing=True)]:
            self.transport.responses['/odds'] = data
            with mock.patch('cs_app.collection.now', return_value=NOW + timedelta(minutes=len(self.transport.calls))):
                self.run_job(self.job('upcoming'))
            latest = self.store.one("SELECT * FROM quotes WHERE bookmaker='pinnacle' ORDER BY id DESC LIMIT 1")
            if data == current():
                self.assertEqual(latest['status'], 'admitted')
                self.assertEqual(provider.utc(latest['source_time_1']), provider.utc('2026-10-06T11:00:00Z'))
            else:
                self.assertNotEqual(latest['status'], 'admitted')
                self.assertIsNone(latest['price1'])

    def test_native_scores_result_only_and_exact_orientation(self):
        self.store.upsert_fixture(fixture())
        self.transport.responses['/scores'] = {'fixtureId': 'id1', 'scores': {'periods': {
            'result': {'participant1Score': 0, 'participant2Score': 2,
                       'updatedAt': '2026-04-14T12:00:00Z'},
            '1': {'participant1Score': 13, 'participant2Score': 4}}}}
        self.assertEqual(self.run_job(self.job('results'))['status'], 'completed')
        row = self.store.one("SELECT * FROM historical_matches WHERE history_key='oddspapi:id1'")
        self.assertEqual((row['team1'], row['team2'], row['score1'], row['score2'], row['winner_1']),
                         ('Beta', 'Alpha', 0, 2, 0))
        self.transport.responses['/scores'] = {'fixtureId': 'id2', 'scores': {}}
        with self.assertRaises(provider.Stop):
            self.run_job(self.job('results', refresh=True))

    def test_cancellation_keeps_frozen_ids_and_resume_does_not_redownload_history(self):
        self.transport.responses['/fixtures'] = [fixture('id1'), fixture('id2')]
        job = self.job(max_history=1)
        result = self.run_job(job)
        self.assertEqual(result['status'], 'budget_paused')
        self.assertEqual(result['progress']['fixture_ids'], ['id1', 'id2'])
        self.transport.responses['/historical-odds'] = history('id2')
        self.transport.calls.clear()
        self.store.cancel_job(job['id'])
        self.assertEqual(self.run_job(self.store.get_job(job['id']))['status'], 'cancelled')
        self.assertEqual(self.transport.calls, [])
        self.store.retry_job(job['id'])
        self.assertEqual(self.run_job(self.store.get_job(job['id']))['status'], 'completed')
        self.assertEqual([p['fixtureId'] for e, p in self.transport.calls if e == '/historical-odds'], ['id2'])

    def test_seven_day_windows_resume_without_discarding_failed_discovery(self):
        self.transport.responses['/fixtures'] = []
        job = self.job(to='2026-05-01T00:00:00Z', billable_per_run=2)
        first = self.run_job(job)
        self.assertEqual(first['status'], 'budget_paused')
        windows = first['progress']['windows']
        self.assertEqual(len(windows), 3)
        self.assertTrue(all(provider.utc(w['to']) - provider.utc(w['from']) <= timedelta(days=7) for w in windows))
        self.assertEqual(sum(w.get('raw_id') is not None for w in windows), 1)
        self.transport.calls.clear()
        second = self.run_job(self.store.get_job(job['id']))
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(len([e for e, _ in self.transport.calls if e == '/fixtures']), 2)

    def test_total_cap_reserves_after_snapshot_and_resumes_pending_fixture(self):
        job = self.job(max_requests=4)
        first = self.run_job(job)
        self.assertEqual(first['status'], 'budget_paused')
        self.assertEqual(first['actual_requests'], 4)
        self.assertEqual([e for e, _ in self.transport.calls], ['/account', '/markets', '/fixtures', '/account'])
        self.transport.calls.clear()
        second = self.run_job(self.store.get_job(job['id']))
        self.assertEqual(second['status'], 'completed')
        self.assertEqual([e for e, _ in self.transport.calls], ['/account', '/historical-odds', '/account'])

    def test_imported_billable_receipt_does_not_consume_daily_application_budget(self):
        self.store.save_settings({'billable_per_day': 2})
        self.store.put_raw({'endpoint': '/scores', 'parameters': {'fixtureId': 'other'},
            'request_started_at': NOW.isoformat(), 'received_at': NOW.isoformat(),
            'http_status': 500, 'error_code': 'PROVIDER_ERROR', 'terminal': False},
            b'{"error":"source failure"}', origin='import', billable=False)
        result = self.run_job(self.job())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['billable_attempts'], 2)

    def test_historical_latest_suspension_retained_and_does_not_revive_older_price(self):
        data = history()
        data['bookmakers']['pinnacle']['markets']['171']['outcomes']['171']['players']['0'].append(
            {'createdAt': '2026-04-14T09:30:00Z', 'active': False, 'price': 2.2})
        self.transport.responses['/historical-odds'] = data
        self.assertEqual(self.run_job(self.job())['status'], 'completed')
        quote = self.store.one("SELECT * FROM quotes WHERE bookmaker='pinnacle'")
        self.assertIsNone(quote['price1'])
        self.assertIn('latest_inactive', quote['reason'])
        self.assertEqual([s['active'] for s in self.store.query(
            "SELECT active FROM odds_states WHERE outcome_id='171' ORDER BY sequence_no")], [1, 0])

    def test_upcoming_aggregate_suspension_and_missing_clocks_block_price(self):
        self.transport.responses['/fixtures'] = [fixture(status=0)]
        cases = []
        for field, value in [('suspended', True), ('bookmakerIsActive', None)]:
            data = current()
            data['bookmakerOdds']['pinnacle'][field] = value
            cases.append(data)
        data = current()
        data['bookmakerOdds']['pinnacle']['markets']['171']['marketActive'] = False
        cases.append(data)
        for clock in [None, '2026-10-06T11:00:00', '2026-10-06T13:00:00Z']:
            data = current()
            data['bookmakerOdds']['pinnacle']['markets']['171']['outcomes']['171']['players']['0']['changedAt'] = clock
            cases.append(data)
        for index, data in enumerate(cases):
            with self.subTest(index=index), mock.patch('cs_app.collection.now', return_value=NOW + timedelta(minutes=index * 2)):
                self.transport.responses['/odds'] = data
                self.run_job(self.job('upcoming'))
                row = self.store.one("SELECT * FROM quotes WHERE bookmaker='pinnacle' ORDER BY id DESC LIMIT 1")
                self.assertIsNone(row['price1'])
                self.assertNotEqual(row['status'], 'admitted')

    def test_failed_current_snapshot_is_partial_and_explicit_resume_reacquires_it(self):
        self.transport.responses['/fixtures'] = [fixture(status=0)]
        self.transport.responses['/odds'] = (500, {'error': {'code': 'SERVER_ERROR'}})
        job = self.job('upcoming')
        self.assertEqual(self.run_job(job)['status'], 'partial')
        self.transport.responses['/odds'] = current()
        self.transport.calls.clear()
        self.assertEqual(self.run_job(self.store.get_job(job['id']))['status'], 'completed')
        self.assertEqual([e for e, _ in self.transport.calls], ['/account', '/odds', '/account'])

    def test_numbered_score_periods_never_fabricate_series_result(self):
        self.store.upsert_fixture(fixture())
        self.transport.responses['/scores'] = {'fixtureId': 'id1', 'scores': {
            '0': {'participant1Score': 2, 'participant2Score': 0}}}
        with self.assertRaises(provider.Stop):
            self.run_job(self.job('results'))
        self.assertEqual(self.store.query("SELECT * FROM historical_matches WHERE source='oddspapi'"), [])

    def test_cancel_requested_during_response_stops_following_requests_and_keeps_raw(self):
        job = self.job()
        self.transport.callback = lambda endpoint: self.store.cancel_job(job['id']) if endpoint == '/fixtures' else None
        result = self.run_job(job)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual([e for e, _ in self.transport.calls], ['/account', '/markets', '/fixtures'])
        self.assertEqual(result['progress']['fixture_ids'], ['id1'])
        self.assertEqual(self.store.one("SELECT COUNT(*) n FROM raw_responses WHERE endpoint='/fixtures'")['n'], 1)

    def test_finished_uses_recent_finished_window_and_retains_existing_fixture(self):
        old = fixture('old')
        self.store.upsert_fixture(old)
        recent = fixture('recent')
        recent['startTime'] = '2026-10-05T10:00:00Z'
        self.transport.responses['/fixtures'] = [recent]
        self.transport.responses['/historical-odds'] = history('recent')
        result = self.run_job(self.job('finished'))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual({r['fixture_id'] for r in self.store.query('SELECT fixture_id FROM fixtures')}, {'old', 'recent'})
        params = next(p for e, p in self.transport.calls if e == '/fixtures')
        self.assertEqual(params['statusId'], '2')
        self.assertNotIn('hasOdds', params)
        self.assertEqual(provider.utc(params['from']), NOW - timedelta(days=self.store.get_settings()['finished_days']))

    def test_resume_retains_frozen_books_after_default_settings_change(self):
        job = self.job(max_requests=4)
        self.assertEqual(self.run_job(job)['status'], 'budget_paused')
        self.store.save_settings({'bookmakers': ['pinnacle']})
        resumed = self.run_job(self.store.get_job(job['id']))
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual(resumed['progress']['requested_books'], BOOKS)
        self.assertEqual({r['bookmaker'] for r in self.store.query('SELECT bookmaker FROM quotes')}, set(BOOKS))

    def test_cached_history_keeps_original_derivation_after_fixture_correction(self):
        self.run_job(self.job())
        original = self.store.one("SELECT details_json FROM quotes WHERE bookmaker='pinnacle'")['details_json']
        corrected = fixture()
        corrected['startTime'] = '2026-04-14T08:00:00Z'
        self.transport.responses['/fixtures'] = [corrected]
        with mock.patch('cs_app.collection.now', return_value=NOW + timedelta(minutes=2)):
            self.assertEqual(self.run_job(self.job())['status'], 'completed')
        self.assertEqual(self.store.one("SELECT details_json FROM quotes WHERE bookmaker='pinnacle'")['details_json'], original)
        self.assertEqual(len([e for e, _ in self.transport.calls if e == '/historical-odds']), 1)

    def test_invalid_current_fixture_clock_masks_prior_pair_and_is_not_terminal(self):
        self.transport.responses['/fixtures'] = [fixture(status=0)]
        self.transport.responses['/odds'] = current()
        self.run_job(self.job('upcoming'))
        invalid = current()
        invalid['startTime'] = None
        self.transport.responses['/odds'] = invalid
        with mock.patch('cs_app.collection.now', return_value=NOW + timedelta(minutes=2)):
            job = self.job('upcoming')
            with self.assertRaises(provider.Stop):
                self.run_job(job)
            last = self.store.one("SELECT * FROM quotes WHERE bookmaker='pinnacle' ORDER BY id DESC LIMIT 1")
            self.assertIsNone(last['price1'])
            self.assertEqual(last['status'], 'unsupported_response')
            self.assertEqual(self.store.one('SELECT terminal FROM raw_responses WHERE id=?', (last['raw_id'],))['terminal'], 0)
            self.transport.responses['/odds'] = current()
            self.assertEqual(self.run_job(self.store.get_job(job['id']))['status'], 'completed')
        self.assertEqual(len([e for e, _ in self.transport.calls if e == '/odds']), 3)

    def test_explicit_public_result_refresh_records_newer_native_correction(self):
        from cs_app.api import validate_payload
        self.store.upsert_fixture(fixture())
        self.transport.responses['/scores'] = {'fixtureId': 'id1', 'scores': {'periods': {'result': {
            'participant1Score': 2, 'participant2Score': 1, 'updatedAt': '2026-04-14T12:00:00Z'}}}}
        self.assertEqual(self.run_job(self.job('results'))['status'], 'completed')
        self.transport.calls.clear()
        self.assertEqual(self.run_job(self.job('results'))['status'], 'completed')
        self.assertEqual(self.transport.calls, [])
        self.transport.responses['/scores']['scores']['periods']['result'].update({
            'participant1Score': 1, 'participant2Score': 2, 'updatedAt': '2026-04-15T12:00:00Z'})
        payload = validate_payload('results', {'refresh': True}, self.store.get_settings())
        self.assertEqual(self.run_job(self.job('results', **payload))['status'], 'completed')
        self.assertEqual(self.store.one("SELECT winner_1 FROM historical_matches WHERE history_key='oddspapi:id1'")['winner_1'], 0)
        self.assertEqual(self.store.one('SELECT COUNT(*) AS n FROM history_revisions')['n'], 1)
        with self.assertRaises(ValueError):
            validate_payload('upcoming', {'refresh': True}, self.store.get_settings())


    def test_newly_finished_results_are_not_starved_by_the_historical_seed(self):
        old, recent = fixture('old'), fixture('recent')
        recent['startTime'] = '2026-10-05T12:00:00Z'
        self.store.upsert_fixture(old)
        self.store.upsert_fixture(recent)
        self.transport.responses['/scores'] = {'fixtureId': 'recent', 'scores': {'periods': {'result': {
            'participant1Score': 2, 'participant2Score': 0, 'updatedAt': '2026-10-05T15:00:00Z'}}}}
        outcome = self.run_job(self.job('results', max_fixtures=1))
        self.assertEqual(outcome['status'], 'budget_paused')
        self.assertEqual(outcome['progress']['done_fixture_ids'], ['recent'])
        self.assertEqual(self.store.one("SELECT winner_1 FROM historical_matches WHERE history_key='oddspapi:recent'")['winner_1'], 1)

if __name__ == '__main__':
    unittest.main()
