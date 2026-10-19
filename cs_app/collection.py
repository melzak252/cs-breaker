"""Explicit, bounded v4 CS2 collection with durable evidence and resumable cursors.

Credentials live only in the transport instance. A provider lock covers account
checks, quota reservations and collection; no requests occur on module import.
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
import fcntl
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace

from scripts import collect_cs_oddspapi as provider

Stop = provider.Stop


def now():
    return datetime.now(UTC)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


class ProviderClient(provider.Client):
    billable_endpoints = {'/markets', '/fixtures', '/odds', '/scores'}
    cooldowns = {'/historical-odds': 6.5, '/fixtures': 3.5, '/scores': 1.5,
                 '/odds': .8, '/account': 1.1, '/markets': 1.1}

    def request(self, endpoint, params):
        record, body, payload = super().request(endpoint, params)
        # v4 pacing is measured from the previous response, not its start.
        self.last_start[endpoint] = record['receipt_monotonic']
        if endpoint == '/odds' and isinstance(payload, dict):
            books = payload.get('bookmakerOdds')
            if isinstance(books, dict) and all(provider.SAFE_BOOK.fullmatch(b) for b in books):
                record['returned_books'] = sorted(books)
        return record, body, payload


class BudgetPause(Exception):
    pass


class Cancelled(Exception):
    pass


class Incomplete(Exception):
    pass


class Collector:
    def __init__(self, store, key):
        self.store = store
        self.key = key
        self.root = Path(store.db_path).parent

    def _setting(self, name, default):
        row = self.store.one('SELECT value_json FROM settings WHERE key=?', (name,))
        return json.loads(row['value_json']) if row else default

    def _save_setting(self, name, value):
        self.store.execute('INSERT INTO settings(key,value_json) VALUES(?,?) '
                           'ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json',
                           (name, encoded(value)))

    def _check_cancel(self):
        if self.store.cancelled(self.job['id']):
            raise Cancelled

    def _progress(self):
        self.progress['current_run'] = {
            'actual_requests': sum(self.client.attempts.values()),
            'billable_attempts': self.client.billable,
            'attempts_by_endpoint': dict(self.client.attempts)}
        self.store.update_job(self.job['id'], progress=self.progress)

    def _daily(self):
        day = now().date().isoformat()
        counter = self._setting('provider_billable_days', {})
        ledger = self.store.one("SELECT COUNT(*) AS n FROM raw_responses WHERE origin='network' "
                                "AND billable=1 AND substr(started_at,1,10)=?", (day,))['n']
        return day, counter, max(counter.get(day, 0), ledger)

    def _wait_cooldown(self, endpoint):
        previous = self._setting('provider_cooldowns', {}).get(endpoint)
        if not previous:
            return
        deadline = provider.utc(previous) + timedelta(seconds=self.client.cooldowns[endpoint])
        while True:
            self._check_cancel()
            delay = (deadline - now()).total_seconds()
            if delay <= 0:
                return
            time.sleep(min(delay, .2))

    def _network(self, endpoint, params, *, after=False):
        self._check_cancel()
        total = sum(self.client.attempts.values())
        # Reserve the last request for an after-account snapshot after any work.
        reserve = 0 if after else 1
        if total >= self.args.max_requests - reserve:
            raise BudgetPause('Local total request cap reached')
        if endpoint == '/historical-odds' and self.client.attempts[endpoint] >= self.args.max_history:
            raise BudgetPause('Local history request cap reached')
        billable = endpoint in self.client.billable_endpoints
        if billable:
            day, counts, used = self._daily()
            if self.client.billable >= self.args.max_billable_requests or used >= self.settings['billable_per_day']:
                raise BudgetPause('Local billable allowance exhausted')
            if self.provider_remaining <= 1:
                raise Stop('Insufficient provider quota headroom')
        self._wait_cooldown(endpoint)
        self._check_cancel()
        if billable:
            # Reserve before send so a crash cannot erase a paid attempt.
            day, counts, used = self._daily()
            counts[day] = used + 1
            self._save_setting('provider_billable_days', counts)
            self.provider_remaining -= 1
        record, body, payload = self.client.request(endpoint, params)
        if record.get('error_code') in {'RATE_LIMIT_EXCEEDED', 'INVALID_API_KEY',
                'UNAUTHORIZED', 'FORBIDDEN', 'SECRET_ECHO', 'INVALID_JSON'}:
            record['stop'] = True
        clocks = self._setting('provider_cooldowns', {})
        clocks[endpoint] = record['received_at']
        self._save_setting('provider_cooldowns', clocks)
        if endpoint == '/account':
            record.pop('decoded_sha256', None)
            record.pop('file_sha256', None)
            body = None
        else:
            record['secret_guard'] = self.client.has_secret
        record['terminal'] = False
        if not record.get('stop') and record['http_status'] == 200 and not record.get('error_code'):
            try:
                self._validate(endpoint, params, payload)
            except Stop:
                raw_id = self.store.put_raw(record, body, job_id=self.job['id'], billable=billable)
                if endpoint == '/odds':
                    for book in self.books:
                        self.store.put_quote(params['fixtureId'], book, 'current', None,
                            {'status': 'unsupported_response', 'reason': 'provider_contract_invalid'}, raw_id)
                self._progress()
                raise
            record['terminal'] = endpoint != '/account'
            if endpoint == '/scores' and parse_result(payload) is None:
                record['terminal'] = False
        elif endpoint == '/historical-odds' and provider.no_history(record['http_status'], payload):
            record['terminal'] = True
        raw_id = self.store.put_raw(record, body, job_id=self.job['id'], billable=billable)
        self._progress()
        if record.get('stop'):
            raise Stop('Provider stopped acquisition: ' + (record.get('error_code') or 'ACCESS_DENIED'))
        return raw_id, record, payload

    def _validate(self, endpoint, params, payload):
        if endpoint == '/markets':
            provider.resolve_market(payload)
        elif endpoint == '/fixtures':
            validate_fixtures(payload, params)
        elif endpoint == '/historical-odds':
            provider.validate_history(payload, params['fixtureId'])
        elif endpoint in ('/odds', '/scores'):
            fixture = self._fixture(params['fixtureId'])
            validate_native(payload, fixture, current=endpoint == '/odds')
            if endpoint == '/odds':
                books = payload.get('bookmakerOdds')
                if not isinstance(books, dict) or not all(provider.SAFE_BOOK.fullmatch(b) for b in books):
                    raise Stop('Unsupported current bookmakerOdds envelope')
                try:
                    provider.fixture_cutoff(payload)
                except (ValueError, OverflowError, Stop):
                    raise Stop('Current fixture requires valid aware scheduled and actual start clocks') from None
            else:
                parse_result(payload)

    def _read_raw(self, row):
        path = self.root / row['raw_path'] if row.get('raw_path') else None
        if path is None or not path.resolve().is_relative_to(self.root.resolve()):
            raise Stop('Terminal cache lacks safe raw evidence')
        try:
            compressed = path.read_bytes()
            body = gzip.decompress(compressed)
            if provider.sha(compressed) != row['file_sha256'] or provider.sha(body) != row['decoded_sha256']:
                raise Stop('Cached raw evidence hash mismatch')
            if self.client.has_secret(body):
                raise Stop('Secret echo in cached evidence')
            payload = json.loads(body)
        except (OSError, ValueError, UnicodeError):
            raise Stop('Unreadable terminal raw evidence') from None
        params = json.loads(row['parameters_json'])
        if row['endpoint'] == '/historical-odds' and row['http_status'] == 404:
            if not provider.no_history(404, payload):
                raise Stop('Cached absence is not explicit no-history')
        else:
            if row['http_status'] != 200 or row['error_code']:
                raise Stop('Invalid terminal cache status')
            self._validate(row['endpoint'], params, payload)
        record = {'endpoint': row['endpoint'], 'parameters': params,
                  'http_status': row['http_status'], 'error_code': row['error_code'],
                  'received_at': row['received_at'], 'terminal': True}
        return row['id'], record, payload

    def _cache(self, endpoint, params, *, job_only=False):
        sql = 'SELECT * FROM raw_responses WHERE endpoint=? AND parameters_json=? AND terminal=1'
        values = [endpoint, encoded(params)]
        if job_only:
            sql += ' AND job_id=?'
            values.append(self.job['id'])
        row = self.store.one(sql + ' ORDER BY id DESC LIMIT 1', tuple(values))
        return self._read_raw(row) if row else None

    def _account(self, phase, *, after=False):
        _, record, payload = self._network('/account', {}, after=after)
        if record['http_status'] != 200 or record.get('error_code'):
            raise Stop('Current account could not be verified')
        snapshot, effective = provider.account_gate(payload, self.books, None)
        snapshot['checked_at'] = record['received_at']
        snapshot['phase'] = phase
        # An account body may echo the key, but allowlisted fields must not.
        if self.client.has_secret(encoded(snapshot).encode()):
            raise Stop('Secret echo in account allowlist')
        self._save_setting('provider_account', snapshot)
        self.progress.setdefault('accounts', []).append(snapshot)
        self._progress()
        if snapshot['request_limit'] - snapshot['request_count'] <= 0:
            raise Stop('Provider quota has no positive headroom')
        return snapshot, effective

    def _gate(self, billable_needed):
        if self.gated:
            return
        self._check_cancel()
        if not self.key or not self.key.strip():
            raise Stop('ODDSPAPI_API_KEY process credential is required for acquisition')
        if billable_needed:
            _, _, used = self._daily()
            if self.args.max_billable_requests == 0 or used >= self.settings['billable_per_day']:
                raise BudgetPause('Local billable allowance exhausted')
        if self.args.max_requests < 2:
            raise BudgetPause('Two account checks require request allowance')
        snapshot, effective = self._account('before')
        self.gated = True
        self.provider_remaining = snapshot['request_limit'] - snapshot['request_count']
        if self.provider_remaining <= (1 if billable_needed else 0):
            raise Stop('Insufficient provider quota headroom')
        frozen = self.progress.get('effective_books')
        if frozen is not None and frozen != effective:
            raise Stop('Current bookmaker entitlement differs from frozen job scope')
        self.progress['effective_books'] = effective
        self._progress()

    def _acquire(self, endpoint, params, *, cache=True, job_only=False):
        cached = self._cache(endpoint, params, job_only=job_only) if cache else None
        if cached:
            self.reused[endpoint] += 1
            return cached
        self._gate(endpoint in self.client.billable_endpoints)
        return self._network(endpoint, params)

    def _fixture(self, identity):
        row = self.store.one('SELECT metadata_json FROM fixtures WHERE fixture_id=?', (identity,))
        if row is None:
            raise Stop('Frozen fixture is missing from durable storage')
        return json.loads(row['metadata_json'])

    def _windows(self):
        if 'windows' in self.progress:
            return
        if self.job['kind'] == 'backfill':
            try:
                start, end = provider.utc(self.payload['from']), provider.utc(self.payload['to'])
            except (KeyError, ValueError, OverflowError):
                raise Stop('Backfill requires aware from/to timestamps') from None
        else:
            end = now()
            if self.job['kind'] == 'upcoming':
                start, end = end, end + timedelta(days=self.settings['upcoming_days'])
            else:
                start = end - timedelta(days=self.settings['finished_days'])
        if end <= start:
            raise Stop('Collection interval must be positive')
        windows = []
        while start < end:
            finish = min(start + timedelta(days=7), end)
            windows.append({'from': provider.iso(start), 'to': provider.iso(finish), 'raw_id': None})
            start = finish
        self.progress['windows'] = windows
        self.progress['fixture_ids'] = []
        self._progress()

    def _discover(self):
        self._windows()
        status = 0 if self.job['kind'] == 'upcoming' else 2
        for window in self.progress['windows']:
            self._check_cancel()
            if window.get('raw_id'):
                continue
            params = {'sportId': 17, 'statusId': status, 'language': 'en',
                      'from': window['from'], 'to': window['to']}
            raw_id, record, payload = self._acquire('/fixtures', params,
                job_only=self.job['kind'] != 'backfill')
            if not record['terminal']:
                self.failures['discovery_failed'] += 1
                # Freeze the failure, not a fabricated empty discovery; retry on next run.
                continue
            fixtures = validate_fixtures(payload, params)
            for fixture in fixtures:
                self.store.upsert_fixture(fixture, observed_at=record['received_at'])
                if fixture['fixtureId'] not in self.progress['fixture_ids']:
                    self.progress['fixture_ids'].append(fixture['fixtureId'])
            window['raw_id'] = raw_id
            self._progress()
        return all(w.get('raw_id') for w in self.progress['windows'])

    def _mapping(self):
        raw_id, record, payload = self._acquire('/markets', {'language': 'en'})
        if not record['terminal']:
            self.failures['catalog_failed'] += 1
            raise Incomplete
        mapping = provider.resolve_market(payload)
        old = self.progress.get('market_mapping')
        if old is not None and old != mapping:
            raise Stop('Frozen Winner mapping differs from catalog evidence')
        self.progress['market_mapping'] = mapping
        self.progress['market_raw_id'] = raw_id
        self._progress()
        return mapping

    def _history(self, fixture, mapping):
        effective = self.progress.get('effective_books')
        if effective is None:
            self._gate(False)
            effective = self.progress['effective_books']
        params = {'fixtureId': fixture['fixtureId'], 'bookmakers': ','.join(sorted(effective))}
        raw_id, record, payload = self._acquire('/historical-odds', params)
        for book in self.books:
            existing = self.store.one("SELECT status FROM quotes WHERE fixture_id=? AND bookmaker=? AND kind='historical' AND raw_id=?",
                                      (fixture['fixtureId'], book, raw_id))
            if existing:
                self.missing[existing['status']] += 1
                continue
            quote = None
            if book not in effective:
                coverage = {'status': 'not_entitled', 'reason': 'not_entitled'}
            elif provider.no_history(record['http_status'], payload):
                coverage = {'status': 'no_history', 'reason': 'explicit_no_history'}
            elif not record['terminal']:
                coverage = {'status': 'failed_request', 'reason': record.get('error_code') or 'http_error'}
            else:
                self._states(fixture['fixtureId'], book, payload.get('bookmakers', {}).get(book), mapping, raw_id)
                coverage, quote = provider.select_pair(fixture, book, payload, mapping)
            self.store.put_quote(fixture['fixtureId'], book, 'historical', quote, coverage, raw_id)
            self.missing[coverage['status']] += 1
        if not record['terminal']:
            self.failures['history_failed'] += 1
        return record['terminal']

    def _states(self, identity, book, bookmaker, mapping, raw_id, *, current=False):
        if not isinstance(bookmaker, dict):
            return
        markets = bookmaker.get('markets', {})
        market = markets.get(mapping['market_id']) if isinstance(markets, dict) else None
        outcomes = market.get('outcomes', {}) if isinstance(market, dict) else {}
        if not isinstance(outcomes, dict):
            return
        for outcome_id, outcome in outcomes.items():
            if not isinstance(outcome, dict):
                continue
            players = outcome.get('players', {})
            states = players.get('0') if isinstance(players, dict) else None
            if current and isinstance(states, dict):
                states = [states]
            if isinstance(states, list):
                self.store.add_states(identity, book, mapping['market_id'], outcome_id, states, raw_id)

    def _current(self, fixture, mapping):
        effective = self.progress.get('effective_books')
        if effective is None:
            self._gate(True)
            effective = self.progress['effective_books']
        params = {'fixtureId': fixture['fixtureId'], 'bookmakers': ','.join(sorted(effective)),
                  'language': 'en', 'oddsFormat': 'decimal', 'verbosity': 3}
        raw_id, record, payload = self._acquire('/odds', params, job_only=True)
        if record['terminal']:
            native = {key: value for key, value in payload.items() if key != 'bookmakerOdds'}
            self.store.upsert_fixture(native, observed_at=record['received_at'])
        for book in self.books:
            quote = None
            if book not in effective:
                coverage = {'status': 'not_entitled', 'reason': 'not_entitled'}
            elif not record['terminal']:
                coverage = {'status': 'failed_request', 'reason': record.get('error_code') or 'http_error'}
            else:
                self._states(fixture['fixtureId'], book, payload['bookmakerOdds'].get(book), mapping,
                             raw_id, current=True)
                coverage, quote = select_current(payload, book, mapping, record['received_at'])
            self.store.put_quote(fixture['fixtureId'], book, 'current', quote, coverage, raw_id)
            self.missing[coverage['status']] += 1
        if not record['terminal']:
            self.failures['odds_failed'] += 1
        return record['terminal']

    def _collect_odds(self):
        mapping = self._mapping()
        discovered = self._discover()
        done = self.progress.setdefault('done_fixture_ids', [])
        count = 0
        for identity in self.progress['fixture_ids']:
            self._check_cancel()
            if identity in done:
                continue
            fixture = self._fixture(identity)
            if self.job['kind'] == 'upcoming':
                latest = self.store.one("SELECT q.observed_at,q.status,r.job_id FROM quotes q "
                                        "JOIN raw_responses r ON r.id=q.raw_id "
                                        "WHERE q.fixture_id=? AND q.kind='current' "
                                        "ORDER BY q.observed_at DESC,q.id DESC LIMIT 1", (identity,))
                if (latest and latest['job_id'] != self.job['id']
                        and latest['status'] != 'failed_request'
                        and now() - provider.utc(latest['observed_at']) < timedelta(minutes=self.settings['min_snapshot_minutes'])):
                    done.append(identity)
                    self._progress()
                    continue
            if count >= self.settings['max_fixtures']:
                raise BudgetPause('Local fixture processing cap reached')
            count += 1
            terminal = (self._current(fixture, mapping) if self.job['kind'] == 'upcoming'
                        else self._history(fixture, mapping))
            if terminal:
                done.append(identity)
            self._progress()
        return discovered and len(done) == len(self.progress['fixture_ids'])

    def _results(self):
        if 'fixture_ids' not in self.progress:
            refresh = self.payload.get('refresh') is True
            sql = "SELECT f.fixture_id FROM fixtures f WHERE f.status_id=2"
            if not refresh:
                sql += " AND NOT EXISTS(SELECT 1 FROM historical_matches h WHERE h.history_key='oddspapi:'||f.fixture_id)"
            sql += " ORDER BY EXISTS(SELECT 1 FROM quotes q WHERE q.fixture_id=f.fixture_id AND q.status='admitted') DESC, f.start_time DESC,f.fixture_id"
            self.progress['fixture_ids'] = [r['fixture_id'] for r in self.store.query(sql)]
            self._progress()
        done = self.progress.setdefault('done_fixture_ids', [])
        count = 0
        for identity in self.progress['fixture_ids']:
            self._check_cancel()
            if identity in done:
                continue
            if count >= self.settings['max_fixtures']:
                raise BudgetPause('Local fixture processing cap reached')
            count += 1
            fixture = self._fixture(identity)
            raw_id, record, payload = self._acquire('/scores', {'fixtureId': identity},
                job_only=self.payload.get('refresh') is True)
            result = parse_result(payload) if record['http_status'] == 200 and not record.get('error_code') else None
            if result is None:
                self.failures['result_missing'] += 1
                continue
            one, two = result
            raw = self.store.one('SELECT decoded_sha256 FROM raw_responses WHERE id=?', (raw_id,))
            self.store.put_history({'history_key': 'oddspapi:' + identity, 'source': 'oddspapi',
                'source_match_id': identity, 'start_time': fixture['startTime'],
                'source_date': payload['scores']['periods']['result']['updatedAt'], 'timezone_status': 'aware',
                'team1': fixture.get('participant1Name', ''), 'team2': fixture.get('participant2Name', ''),
                'tournament': fixture.get('tournamentName', ''), 'score1': one, 'score2': two,
                'winner_1': int(one > two), 'format': None, 'label_status': 'native_series_result',
                'metadata_json': encoded(payload), 'source_sha256': raw['decoded_sha256']})
            done.append(identity)
            self._progress()
        return len(done) == len(self.progress['fixture_ids'])

    def run(self, job):
        self.job = job
        self.payload = job.get('payload') or json.loads(job.get('payload_json') or '{}')
        self.progress = job.get('progress') or json.loads(job.get('progress_json') or '{}')
        self.settings = self.store.get_settings()
        self.settings['max_fixtures'] = min(self.payload.get('max_fixtures', self.settings['max_fixtures']), self.settings['max_fixtures'])
        self.books = self.payload.get('bookmakers', self.progress.get('requested_books', self.settings['bookmakers']))
        if (not isinstance(self.books, list) or not 1 <= len(self.books) <= 3
                or len(set(self.books)) != len(self.books)
                or not all(isinstance(b, str) and provider.SAFE_BOOK.fullmatch(b) for b in self.books)):
            raise Stop('Choose one to three distinct bookmaker slugs')
        if job['kind'] not in ('backfill', 'upcoming', 'finished', 'results'):
            raise Stop('Unsupported collection job kind')
        old_books = self.progress.get('requested_books')
        if old_books is not None and old_books != self.books:
            raise Stop('Requested bookmakers differ from frozen job scope')
        self.progress['requested_books'] = self.books
        limits = {}
        for name in ('max_requests', 'max_history', 'billable_per_run'):
            value = self.payload.get(name, self.settings[name])
            if type(value) is not int or value < 0:
                raise Stop('Collection caps must be nonnegative integers')
            limits[name] = min(value, self.settings[name])
        self.args = SimpleNamespace(max_requests=limits['max_requests'], max_history=limits['max_history'],
            max_billable_requests=limits['billable_per_run'], timeout_seconds=45)
        self.client = ProviderClient(self.args, self.key)
        self.gated = False
        self.provider_remaining = 0
        self.reused, self.failures, self.missing = Counter(), Counter(), Counter()
        status, reason = 'partial', None
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'provider.lock').open('a+b') as lock:
            while True:
                if self.store.cancelled(job['id']):
                    self._progress()
                    return self._result('cancelled', None)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.2)
            try:
                self._check_cancel()
                complete = self._results() if job['kind'] == 'results' else self._collect_odds()
                status = 'completed' if complete else 'partial'
            except BudgetPause as error:
                status, reason = 'budget_paused', str(error)
            except Cancelled:
                status = 'cancelled'
            except Incomplete:
                status = 'partial'
            # No finally account request after an auth/unsupported/secret stop.
            if self.gated and status != 'cancelled':
                try:
                    self._account('after', after=True)
                except BudgetPause:
                    status, reason = 'budget_paused', 'After-account request allowance exhausted'
                except Cancelled:
                    status = 'cancelled'
            self._progress()
        return self._result(status, reason)

    def _result(self, status, reason):
        return {'status': status, 'reason': reason, 'actual_requests': sum(self.client.attempts.values()),
                'billable_attempts': self.client.billable, 'attempts_by_endpoint': dict(self.client.attempts),
                'cache_reused': dict(self.reused),
                'missing': {k: v for k, v in self.missing.items() if k != 'admitted'},
                'coverage': dict(self.missing), 'selected_pairs': self.missing.get('admitted', 0),
                'failures': dict(self.failures), 'progress': self.progress}


def validate_fixtures(payload, params):
    if params['statusId'] == 2:
        return provider.discover_fixtures(payload, SimpleNamespace(start=params['from'], end=params['to']))
    if not isinstance(payload, list):
        raise Stop('Unsupported fixtures envelope: expected list')
    seen = {}
    for fixture in payload:
        if (not isinstance(fixture, dict) or not isinstance(fixture.get('fixtureId'), str)
                or not provider.SAFE_ID.fullmatch(fixture['fixtureId'])
                or fixture.get('sportId') != 17 or fixture.get('statusId') != 0):
            raise Stop('Fixture identity, sport or upcoming status mismatch')
        try:
            start = provider.utc(fixture.get('startTime'))
        except (ValueError, OverflowError):
            raise Stop('Invalid fixture start evidence') from None
        if not provider.utc(params['from']) <= start <= provider.utc(params['to']):
            raise Stop('Fixture outside requested bounds')
        if fixture['fixtureId'] in seen and fixture != seen[fixture['fixtureId']]:
            raise Stop('Conflicting duplicate fixture metadata')
        seen[fixture['fixtureId']] = fixture
    return sorted(seen.values(), key=lambda f: (provider.utc(f['startTime']), f['fixtureId']))


def validate_native(payload, fixture, *, current=False):
    if not isinstance(payload, dict) or payload.get('fixtureId') != fixture['fixtureId']:
        raise Stop('Native response fixture identity mismatch')
    for field in ('sportId', 'participant1Id', 'participant2Id', 'participant1Name', 'participant2Name'):
        if (current or field in payload) and payload.get(field) != fixture.get(field):
            raise Stop('Native response sport or participant orientation mismatch')
    if current and (type(payload.get('statusId')) is not int or payload['statusId'] not in (0, 1, 2, 3)):
        raise Stop('Unsupported current fixture status')


def parse_result(payload):
    """Only the provider-observed named series result is a CS2 outcome label."""
    if not isinstance(payload, dict) or not isinstance(payload.get('scores'), dict):
        raise Stop('Unsupported native scores envelope')
    periods = payload['scores'].get('periods')
    if not isinstance(periods, dict):
        raise Stop('Unsupported scores: named periods.result required')
    result = periods.get('result')
    if result is None:
        return None
    if not isinstance(result, dict):
        raise Stop('Unsupported native series result')
    values = [result.get('participant1Score'), result.get('participant2Score')]
    if any(type(v) is not int or v < 0 for v in values) or values[0] == values[1]:
        return None
    try:
        provider.utc(result.get('updatedAt'))
    except (ValueError, OverflowError):
        return None
    return tuple(values)


def select_current(payload, book, mapping, received_at):
    coverage = {'status': 'unusable_state', 'reason': ''}
    if payload.get('statusId') != 0:
        coverage['reason'] = 'not_pregame'
        return coverage, None
    try:
        cutoff = provider.fixture_cutoff(payload)
        receipt = provider.utc(received_at)
    except (Stop, ValueError, OverflowError):
        coverage['reason'] = 'invalid_start_or_receipt_clock'
        return coverage, None
    if provider.unsupported_scope(payload):
        coverage['reason'] = 'explicit_live_or_map_scope'
        return coverage, None
    if receipt >= cutoff:
        coverage['reason'] = 'observed_at_or_after_start'
        return coverage, None
    bookmaker = payload['bookmakerOdds'].get(book)
    if not isinstance(bookmaker, dict):
        return {'status': 'book_missing', 'reason': 'book_missing'}, None
    markets = bookmaker.get('markets', {})
    market = markets.get(mapping['market_id']) if isinstance(markets, dict) else None
    if not isinstance(market, dict):
        return {'status': 'market_missing', 'reason': 'market_missing'}, None
    if bookmaker.get('bookmakerIsActive') is not True:
        coverage['reason'] = 'bookmaker_inactive_or_unknown'
        return coverage, None
    if bookmaker.get('suspended') is not False:
        coverage['reason'] = 'bookmaker_suspended_or_unknown'
        return coverage, None
    if market.get('marketActive') is not True:
        coverage['reason'] = 'market_inactive_or_unknown'
        return coverage, None
    outcomes = market.get('outcomes', {})
    converted = {}
    for side in ('1', '2'):
        outcome_id = mapping['outcome_' + side]
        outcome = outcomes.get(outcome_id) if isinstance(outcomes, dict) else None
        if not isinstance(outcome, dict):
            continue
        players = outcome.get('players', {})
        state = players.get('0') if isinstance(players, dict) else None
        if not isinstance(state, dict):
            continue
        try:
            source = provider.utc(state.get('changedAt'))
        except (ValueError, OverflowError):
            coverage['reason'] = 'unorderable_changed_at'
            return coverage, None
        if source > receipt or source >= cutoff:
            coverage['reason'] = 'source_clock_after_receipt_or_start'
            return coverage, None
        converted[outcome_id] = dict(outcome, players={'0': [dict(state, createdAt=state.get('changedAt'))]})
    history = {'bookmakers': {book: dict(bookmaker, markets={mapping['market_id']: dict(market, outcomes=converted)})}}
    return provider.select_pair(payload, book, history, mapping)
