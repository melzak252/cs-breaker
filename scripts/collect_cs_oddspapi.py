"""Opt-in, provider-native CS2 v4 history acquisition; no application imports.

Source createdAt is not receipt/publication/execution time. Provider outcome labels
are not independent bookmaker participant binding. Raw account bodies are never
saved; credentials exist only in process memory and the authenticated request.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import zlib
from collections import Counter
from datetime import UTC, datetime, timedelta
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, quote_plus, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

BASE = 'https://api.oddspapi.io/v4'
VERSION = 1
BILLABLE = {'/markets', '/fixtures'}
COOLDOWNS = {'/account': 1.1, '/markets': 1.1, '/fixtures': 2.1, '/historical-odds': 5.1}
SAFE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
SAFE_BOOK = re.compile(r'[a-z0-9][a-z0-9-]{0,63}(?:\.[a-z0-9][a-z0-9-]{0,63})*\Z')
ERROR_CODES = {'NOT_FOUND', 'REQUEST_LIMIT_EXCEEDED', 'RATE_LIMIT_EXCEEDED',
               'UNAUTHORIZED', 'FORBIDDEN', 'INVALID_API_KEY', 'FIXTURE_NOT_FOUND'}


class Stop(Exception):
    """Sanitized acquisition stop; never wrap a secret-bearing exception."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


urlopen = build_opener(NoRedirect()).open


def utc(value):
    if not isinstance(value, str):
        raise ValueError('Timestamp must be an aware ISO string')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('Timestamp has no timezone')
    return parsed.astimezone(UTC)


def clock():
    return datetime.now(UTC).isoformat()


def iso(value):
    return value.isoformat().replace('+00:00', 'Z')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_bytes(data)
    temporary.replace(path)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from', dest='start', required=True)
    parser.add_argument('--to', dest='end', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bookmakers', default='pinnacle,bet365,unibet')
    parser.add_argument('--max-history', type=int, default=10)
    parser.add_argument('--max-billable-requests', type=int, default=2)
    parser.add_argument('--max-requests', type=int, default=14)
    parser.add_argument('--timeout-seconds', type=float, default=45)
    parser.add_argument('--api-key-env', default='ODDSPAPI_API_KEY')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    try:
        start, end = utc(args.start), utc(args.end)
    except (ValueError, OverflowError):
        parser.error('--from and --to require timezone-aware ISO timestamps')
    if not timedelta(0) < end - start < timedelta(days=10):
        parser.error('Request interval must be positive and shorter than ten days')
    books = args.bookmakers.split(',')
    if not 1 <= len(books) <= 3 or len(set(books)) != len(books) or not all(SAFE_BOOK.fullmatch(b) for b in books):
        parser.error('--bookmakers requires one to three distinct safe slugs')
    if min(args.max_history, args.max_billable_requests, args.max_requests) < 0:
        parser.error('Request caps must be nonnegative')
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error('--timeout-seconds must be finite and positive')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.api_key_env):
        parser.error('--api-key-env must name a process environment variable')
    args.start, args.end, args.bookmakers = iso(start), iso(end), books
    return args


class Client:
    cooldowns = COOLDOWNS
    billable_endpoints = BILLABLE

    def __init__(self, args, key):
        self.args = args
        self.key = key
        self.attempts = Counter()
        self.billable = 0
        self.last_start = {}
        # Also detect JSON escaping and both URL encodings, without logging them.
        self.secret_forms = {v.encode() for v in (key, quote(key, safe=''), quote_plus(key),
                                                   json.dumps(key)[1:-1]) if v}

    def has_secret(self, data):
        return any(value in data for value in self.secret_forms)

    def request(self, endpoint, params):
        if sum(self.attempts.values()) >= self.args.max_requests:
            raise Stop('Local total request cap reached')
        if endpoint in self.billable_endpoints and self.billable >= self.args.max_billable_requests:
            raise Stop('Local billable request cap reached')
        if endpoint == '/historical-odds' and self.attempts[endpoint] >= self.args.max_history:
            raise Stop('Local history request cap reached')
        if endpoint == '/account' and self.attempts[endpoint] >= 2:
            raise Stop('Local account-check cap reached')
        delay = self.cooldowns[endpoint] - (time.monotonic() - self.last_start.get(endpoint, -math.inf))
        if delay > 0:
            time.sleep(delay)
        started = time.monotonic()
        self.last_start[endpoint] = started
        self.attempts[endpoint] += 1
        self.billable += endpoint in self.billable_endpoints
        record = {'endpoint': endpoint, 'parameters': params, 'request_started_at': clock(),
                  'request_start_monotonic': started, 'http_status': 0, 'error_code': None,
                  'requested_books': params.get('bookmakers', '').split(',') if params.get('bookmakers') else [],
                  'returned_books': [], 'raw_path': None, 'terminal': False, 'stop': False}
        body = None
        try:
            request = Request(BASE + endpoint + '?' + urlencode({**params, 'apiKey': self.key}),
                              headers={'Accept': 'application/json', 'User-Agent': 'cs-breaker-research/1'})
            with urlopen(request, timeout=self.args.timeout_seconds) as response:
                record['http_status'] = response.status
                body = response.read()
        except HTTPError as error:
            record['http_status'] = error.code
            try:
                body = error.read()
            except (OSError, ValueError, HTTPException):
                record['error_code'] = 'NETWORK_ERROR'
            finally:
                error.close()
        except (URLError, OSError, ValueError, HTTPException):
            record['error_code'] = 'NETWORK_ERROR'
        record['received_at'] = clock()
        record['receipt_monotonic'] = time.monotonic()
        payload = None
        if body is not None:
            record['decoded_sha256'] = sha(body)
            if endpoint != '/account' and self.has_secret(body):
                record.update(error_code='SECRET_ECHO', stop=True)
                return record, None, None
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeError):
                record['error_code'] = 'INVALID_JSON'
            if isinstance(payload, dict):
                error = payload.get('error', {})
                code = payload.get('code') or (error.get('code') if isinstance(error, dict) else None)
                if code:
                    record['error_code'] = code if isinstance(code, str) and code in ERROR_CODES else 'PROVIDER_ERROR'
                books = payload.get('bookmakers')
                if endpoint != '/account' and isinstance(books, dict) and all(SAFE_BOOK.fullmatch(b) for b in books):
                    record['returned_books'] = sorted(books)
        record['stop'] = record['http_status'] in (401, 403, 429) or record['error_code'] == 'REQUEST_LIMIT_EXCEEDED'
        if record['http_status'] == 429 and record['error_code'] != 'REQUEST_LIMIT_EXCEEDED':
            record['error_code'] = 'RATE_LIMIT_EXCEEDED'
        if 300 <= record['http_status'] < 400:
            record.update(stop=True, error_code='UNSUPPORTED_REDIRECT')
        if endpoint == '/account':
            # The allowlisted entitlement snapshot is the only retained account evidence.
            body = None
            record.pop('decoded_sha256', None)
        return record, body, payload


def account_gate(payload, requested, uncached_billable):
    if not isinstance(payload, dict) or not isinstance(payload.get('subscriptions'), list):
        raise Stop('Unsupported account subscription contract')
    current = payload.get('current_subscription_id')
    matches = [s for s in payload['subscriptions'] if isinstance(s, dict)
               and current is not None and s.get('subscription_id') == current]
    if len(matches) != 1 or matches[0].get('is_active') is not True:
        raise Stop('Missing, ambiguous or inactive current subscription')
    subscription = matches[0]
    sports, books = subscription.get('sport_ids'), subscription.get('bookmakers')
    limit, count = subscription.get('request_limit'), subscription.get('request_count')
    if (not isinstance(sports, list) or not all(type(s) is int for s in sports)
            or not isinstance(books, dict) or not all(SAFE_BOOK.fullmatch(b) for b in books)
            or any(type(n) not in (int, float) or isinstance(n, float) and not math.isfinite(n)
                   for n in (limit, count)) or not 0 <= count <= limit):
        raise Stop('Unverifiable account entitlement or numeric quota')
    snapshot = {'checked_at': clock(), 'active': True, 'sport_ids': sorted(sports),
                'bookmaker_slugs': sorted(books), 'request_limit': limit, 'request_count': count}
    effective = [b for b in requested if b in books]
    if 17 not in sports:
        raise Stop('Current subscription does not include sport 17')
    if not effective:
        raise Stop('No requested bookmaker is entitled')
    if uncached_billable is not None and limit - count <= uncached_billable:
        raise Stop('Insufficient quota headroom: remaining must exceed uncached billable calls')
    return snapshot, effective


def catalog_id(value):
    if type(value) is not int or value <= 0:
        raise Stop('Unsupported catalog identifier')
    return str(value)


def resolve_market(payload):
    if not isinstance(payload, list):
        raise Stop('Unsupported markets envelope: expected list')
    matches = [m for m in payload if isinstance(m, dict) and m.get('sportId') == 17
               and m.get('marketName') == 'Winner' and m.get('marketType') == 'moneyline'
               and m.get('period') == 'result' and m.get('playerProp') is False
               and m.get('marketLength') == 2]
    if len(matches) != 1:
        raise Stop('Unsupported or ambiguous sport-17 series Winner catalog')
    market = matches[0]
    outcomes = market.get('outcomes')
    if (not isinstance(outcomes, list) or len(outcomes) != 2
            or not all(isinstance(o, dict) and isinstance(o.get('outcomeName'), str) for o in outcomes)
            or {o.get('outcomeName') for o in outcomes} != {'1', '2'}):
        raise Stop('Unsupported Winner outcome labels')
    ids = {o['outcomeName']: catalog_id(o.get('outcomeId')) for o in outcomes}
    if ids['1'] == ids['2']:
        raise Stop('Ambiguous Winner outcome identifiers')
    return {'market_id': catalog_id(market.get('marketId')), 'outcome_1': ids['1'], 'outcome_2': ids['2']}


def discover_fixtures(payload, args):
    if not isinstance(payload, list):
        raise Stop('Unsupported fixtures envelope: expected list')
    fixtures = {}
    for fixture in payload:
        if not isinstance(fixture, dict):
            raise Stop('Unsupported fixture metadata')
        identity = fixture.get('fixtureId')
        if (not isinstance(identity, str) or not SAFE_ID.fullmatch(identity)
                or fixture.get('sportId') != 17 or fixture.get('statusId') != 2):
            raise Stop('Fixture identity, sport or finished status mismatch')
        try:
            start = utc(fixture.get('startTime'))
        except (ValueError, OverflowError):
            raise Stop('Invalid fixture scheduled start evidence') from None
        if not utc(args.start) <= start <= utc(args.end):
            raise Stop('Fixture outside requested bounds')
        if identity in fixtures and fixture != fixtures[identity]:
            raise Stop('Conflicting duplicate fixture metadata')
        fixtures[identity] = fixture
    return sorted(fixtures.values(), key=lambda f: (utc(f['startTime']), f['fixtureId']))


def no_history(status, payload):
    error = payload.get('error') if isinstance(payload, dict) else None
    return (status == 404 and isinstance(error, dict) and error.get('code') == 'NOT_FOUND'
            and error.get('message') == 'No historical odds found.')


def validate_history(payload, fixture_id):
    if (not isinstance(payload, dict) or payload.get('fixtureId') != fixture_id
            or not isinstance(payload.get('bookmakers'), dict)
            or not all(SAFE_BOOK.fullmatch(b) for b in payload['bookmakers'])):
        raise Stop('Unsupported history envelope or mismatched fixture identity')


def fixture_cutoff(fixture):
    try:
        start = utc(fixture.get('startTime'))
        actual = fixture.get('trueStartTime')
        return min(start, utc(actual)) if actual is not None else start
    except (ValueError, OverflowError):
        raise Stop('Invalid scheduled or actual start evidence') from None


def unsupported_scope(evidence):
    if not isinstance(evidence, dict):
        return True
    return (any(evidence.get(k) is True for k in ('live', 'isLive', 'inPlay', 'is_live', 'playerProp'))
            or ('period' in evidence and evidence['period'] != 'result')
            or ('playerId' in evidence and evidence['playerId'] not in (0, '0', None))
            or ('player_id' in evidence and evidence['player_id'] not in (0, '0', None))
            or ('marketType' in evidence and evidence['marketType'] != 'moneyline')
            or ('marketName' in evidence and evidence['marketName'] != 'Winner')
            or ('scope' in evidence and evidence['scope'] not in ('series', 'result', 'prematch'))
            or evidence.get('isMap') is True
            or any(evidence.get(k) not in (None, '', 0, '0')
                   for k in ('map', 'mapId', 'mapNumber', 'map_id', 'map_number')))


def select_state(states, cutoff):
    if not isinstance(states, list) or not states:
        return {'status': 'no_pre_start_state'}
    eligible = []
    for state in states:
        try:
            when = utc(state.get('createdAt')) if isinstance(state, dict) else utc(None)
        except (ValueError, OverflowError):
            return {'status': 'unorderable_clock'}
        if when < cutoff:
            eligible.append((when, state))
    if not eligible:
        return {'status': 'no_pre_start_state'}
    latest_time = max(when for when, _ in eligible)
    latest = [state for when, state in eligible if when == latest_time]
    # Normalize the clock, not the state. Even unknown fields can expose conflicts.
    signatures = [dict(state, createdAt=iso(latest_time)) for state in latest]
    if any(state != signatures[0] for state in signatures[1:]):
        return {'status': 'conflicting_latest_states'}
    state = latest[0]
    if unsupported_scope(state):
        return {'status': 'explicit_live_or_map_scope'}
    if state.get('active') is False:
        return {'status': 'latest_inactive'}
    if state.get('active') is not True:
        return {'status': 'unknown_active'}
    value = state.get('price')
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError
        price = float(value)
        if not math.isfinite(price) or price <= 1:
            raise ValueError
    except (ValueError, OverflowError):
        return {'status': 'invalid_price'}
    return {'status': 'usable', 'price': price, 'created_at': state['createdAt']}


def select_pair(fixture, book, history, mapping, *, cutoff=None):
    coverage = {'status': 'unusable_state', 'reason': '', 'side_1_status': '', 'side_2_status': ''}
    try:
        event_cutoff = fixture_cutoff(fixture)
    except Stop:
        coverage['reason'] = 'invalid_start_evidence'
        return coverage, None
    if cutoff is None:
        cutoff = event_cutoff
    else:
        if isinstance(cutoff, str):
            cutoff = utc(cutoff)
        if (not isinstance(cutoff, datetime) or cutoff.tzinfo is None
                or cutoff.utcoffset() is None or cutoff >= event_cutoff):
            raise ValueError('Explicit cutoff must be timezone-aware and strictly before fixture start')
        cutoff = cutoff.astimezone(UTC)
        coverage['forecast_origin_utc'] = iso(cutoff)
    if (not isinstance(history, dict)
            or history.get('fixtureId', fixture.get('fixtureId')) != fixture.get('fixtureId')):
        coverage['reason'] = 'history_fixture_identity_mismatch'
        return coverage, None
    if unsupported_scope(fixture) or unsupported_scope(history):
        coverage['reason'] = 'explicit_live_or_map_scope'
        return coverage, None
    bookmakers = history.get('bookmakers', {})
    bookmaker = bookmakers.get(book) if isinstance(bookmakers, dict) else None
    if not isinstance(bookmaker, dict):
        coverage.update(status='book_missing', reason='book_missing')
        return coverage, None
    markets = bookmaker.get('markets', {})
    market = markets.get(mapping['market_id']) if isinstance(markets, dict) else None
    if not isinstance(market, dict):
        coverage.update(status='market_missing', reason='market_missing')
        return coverage, None
    if unsupported_scope(bookmaker) or unsupported_scope(market):
        coverage['reason'] = 'explicit_live_or_map_scope'
        return coverage, None
    outcomes = market.get('outcomes', {})
    sides = []
    for side in ('1', '2'):
        outcome = outcomes.get(mapping['outcome_' + side]) if isinstance(outcomes, dict) else None
        if not isinstance(outcome, dict):
            result = {'status': 'outcome_missing'}
        elif unsupported_scope(outcome):
            result = {'status': 'explicit_live_or_map_scope'}
        else:
            players = outcome.get('players', {})
            result = (select_state(players['0'], cutoff) if isinstance(players, dict) and '0' in players
                      else {'status': 'player_zero_missing'})
        coverage['side_' + side + '_status'] = result['status']
        sides.append(result)
    if any(s['status'] != 'usable' for s in sides):
        coverage['reason'] = ';'.join(f'{i}:{s["status"]}' for i, s in enumerate(sides, 1) if s['status'] != 'usable')
        return coverage, None
    one, two = sides
    inv1, inv2 = 1 / one['price'], 1 / two['price']
    row = {field: fixture.get(field, '') for field in
           ('fixtureId', 'tournamentId', 'tournamentName', 'participant1Id', 'participant2Id',
            'participant1Name', 'participant2Name', 'startTime', 'trueStartTime')}
    row.update(bookmaker=book, cutoff=iso(cutoff), market_id=mapping['market_id'],
               outcome_1=mapping['outcome_1'], outcome_2=mapping['outcome_2'],
               source_created_at_1=one['created_at'], source_created_at_2=two['created_at'],
               price_1=one['price'], price_2=two['price'], no_vig_probability_1=inv1 / (inv1 + inv2),
               margin=inv1 + inv2 - 1, binding_status='provider_catalog_convention_only',
               execution_eligibility='not_certified')
    if 'forecast_origin_utc' in coverage:
        row['forecast_origin_utc'] = coverage['forecast_origin_utc']
        row['selection_method'] = 'latest_side_state_strictly_before_explicit_cutoff_then_validity'
    coverage.update(status='admitted', reason='two_sided_source_quote')
    return coverage, row


COVERAGE_FIELDS = ['fixtureId', 'bookmaker', 'status', 'reason', 'side_1_status', 'side_2_status',
                   'raw_path', 'decoded_sha256', 'received_at', 'origin']
QUOTE_FIELDS = ['fixtureId', 'bookmaker', 'tournamentId', 'tournamentName', 'participant1Id',
                'participant2Id', 'participant1Name', 'participant2Name', 'startTime', 'trueStartTime',
                'cutoff', 'market_id', 'outcome_1', 'outcome_2', 'source_created_at_1',
                'source_created_at_2', 'price_1', 'price_2', 'no_vig_probability_1', 'margin',
                'received_at', 'decoded_sha256', 'raw_path', 'binding_status', 'execution_eligibility']


def csv_bytes(fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class Run:
    def __init__(self, args, key):
        self.args, self.root = args, args.output_dir
        self.client = Client(args, key)
        self.receipts = []
        self.cache = {}
        self.reused = Counter()
        self.fixtures = []
        self.protocol = {'version': VERSION, 'api_base': BASE, 'sport_id': 17,
                         'from': args.start, 'to': args.end, 'requested_books': args.bookmakers,
                         'sample_limit': args.max_history, 'effective_books': None,
                         'selected_fixture_ids': None, 'market_mapping': None,
                         'accounts': []}

    def save_artifact(self, relative, data):
        if self.client.has_secret(data):
            raise Stop('Secret echo in generated artifact; refusing persistence')
        atomic(self.root / relative, data)

    def persist_protocol(self):
        self.save_artifact('protocol.json', json_bytes(self.protocol))

    def parameters(self, endpoint, fixture_id=None):
        if endpoint == '/markets':
            return {'language': 'en'}
        if endpoint == '/fixtures':
            return {'sportId': 17, 'statusId': 2, 'language': 'en',
                    'from': self.args.start, 'to': self.args.end}
        if endpoint == '/historical-odds':
            return {'fixtureId': fixture_id, 'bookmakers': ','.join(sorted(self.protocol['effective_books']))}
        return {}

    @staticmethod
    def query_key(endpoint, params):
        return endpoint, json.dumps(params, sort_keys=True)

    def load_resume(self):
        """Verify every saved receipt before allowing reuse or any mutation."""
        try:
            protocol = json.loads((self.root / 'protocol.json').read_bytes())
            for field in ('version', 'api_base', 'sport_id', 'from', 'to', 'requested_books', 'sample_limit'):
                if protocol[field] != self.protocol[field]:
                    raise Stop('Resume protocol does not match requested scope')
            effective = protocol['effective_books']
            if effective is not None:
                before = next(a for a in protocol['accounts'] if a['phase'] == 'before')
                expected = [b for b in self.args.bookmakers if b in before['bookmaker_slugs']]
                if not effective or effective != expected or 17 not in before['sport_ids']:
                    raise Stop('Resume entitlement freeze is inconsistent')
            self.protocol = protocol
            receipts_path = self.root / 'receipts.jsonl'
            self.receipts = ([json.loads(line) for line in receipts_path.read_bytes().splitlines()]
                             if receipts_path.exists() else [])
            referenced = set()
            for record in self.receipts:
                endpoint, params = record['endpoint'], record['parameters']
                if endpoint not in COOLDOWNS or not isinstance(params, dict):
                    raise Stop('Resume receipt has an unsupported endpoint')
                fixture_id = params.get('fixtureId')
                if endpoint == '/historical-odds' and (
                        fixture_id not in (protocol['selected_fixture_ids'] or []) or effective is None):
                    raise Stop('Resume receipt fixture is outside the frozen sample')
                if params != self.parameters(endpoint, fixture_id):
                    raise Stop('Resume receipt query does not match exact acquisition scope')
                requested = params['bookmakers'].split(',') if params.get('bookmakers') else []
                if (type(record['http_status']) is not int or type(record['terminal']) is not bool
                        or record['requested_books'] != requested):
                    raise Stop('Resume receipt status or requested books is invalid')
                utc(record['request_started_at'])
                utc(record['received_at'])
                relative = record['raw_path']
                payload = None
                if endpoint == '/account' and (relative or record.get('decoded_sha256')):
                    raise Stop('Raw account evidence is forbidden')
                if relative:
                    path = self.root / relative
                    if (Path(relative).is_absolute() or '..' in Path(relative).parts
                            or not path.resolve().is_relative_to(self.root.resolve())):
                        raise Stop('Unsafe resume raw evidence path')
                    encoded = path.read_bytes()
                    body = gzip.decompress(encoded)
                    if sha(encoded) != record['file_sha256'] or sha(body) != record['decoded_sha256']:
                        raise Stop('Resume evidence hash mismatch')
                    if self.client.has_secret(body):
                        raise Stop('Secret echo in existing cache; refusing reuse')
                    if relative in referenced:
                        raise Stop('Duplicate resume raw evidence path')
                    referenced.add(relative)
                    try:
                        payload = json.loads(body)
                    except (ValueError, UnicodeError):
                        if record['terminal']:
                            raise Stop('Terminal cache is not valid JSON') from None
                if record['terminal']:
                    if payload is None:
                        raise Stop('Terminal cache has no raw evidence')
                    if endpoint == '/historical-odds':
                        if record['http_status'] == 200:
                            validate_history(payload, fixture_id)
                        elif not no_history(record['http_status'], payload):
                            raise Stop('Cached absence is not documented no-history')
                        expected_path = 'history/' + fixture_id + '__' + params['bookmakers'].replace(',', '__') + '.json.gz'
                    elif endpoint == '/markets' and record['http_status'] == 200:
                        resolve_market(payload)
                        expected_path = 'markets.json.gz'
                    elif endpoint == '/fixtures' and record['http_status'] == 200:
                        discover_fixtures(payload, self.args)
                        expected_path = 'fixtures.json.gz'
                    else:
                        raise Stop('Unsupported terminal cache material')
                    if relative != expected_path:
                        raise Stop('Terminal cache path does not match its query')
                    query = self.query_key(endpoint, params)
                    if query in self.cache:
                        raise Stop('Duplicate terminal query in receipts')
                    self.cache[query] = record, payload
            saved_raw = {p.relative_to(self.root).as_posix() for p in self.root.rglob('*.json.gz')}
            if saved_raw != referenced or any(self.root.rglob('*.partial')):
                raise Stop('Unreceipted or partial evidence prevents safe resume')
            market = self.cache.get(self.query_key('/markets', self.parameters('/markets')))
            if market:
                mapping = resolve_market(market[1])
                if protocol['market_mapping'] not in (None, mapping):
                    raise Stop('Frozen market mapping does not match raw catalog')
                protocol['market_mapping'] = mapping
            elif protocol['market_mapping'] is not None:
                raise Stop('Frozen market mapping lacks verified evidence')
            fixtures = self.cache.get(self.query_key('/fixtures', self.parameters('/fixtures')))
            if fixtures:
                self.fixtures = discover_fixtures(fixtures[1], self.args)
                selected = [f['fixtureId'] for f in self.fixtures[:self.args.max_history]]
                if protocol['selected_fixture_ids'] not in (None, selected):
                    raise Stop('Frozen sample does not match raw fixture discovery')
                protocol['selected_fixture_ids'] = selected
            elif protocol['selected_fixture_ids'] is not None:
                raise Stop('Frozen sample lacks verified evidence')
        except Stop:
            raise
        except (OSError, ValueError, KeyError, TypeError, StopIteration, EOFError, zlib.error):
            raise Stop('Corrupt or incomplete resume protocol, receipts or evidence') from None

    def obtain(self, endpoint, params, validator):
        cached = self.cache.get(self.query_key(endpoint, params))
        if cached:
            self.reused[endpoint] += 1
            return cached
        return self.fetch(endpoint, params, validator)

    def fetch(self, endpoint, params, validator=None):
        record, body, payload = self.client.request(endpoint, params)
        record['origin'] = 'network'
        if record['http_status'] == 200 and not record['stop'] and record['error_code'] != 'NETWORK_ERROR':
            try:
                if payload is None:
                    raise Stop('Missing JSON response contract')
                if validator:
                    validator(payload)
                record['terminal'] = endpoint != '/account'
            except Stop:
                record['error_code'] = 'UNSUPPORTED_CONTRACT'
                record['stop'] = True
        if endpoint == '/historical-odds' and no_history(record['http_status'], payload):
            record['terminal'] = True
        if body is not None:
            if endpoint == '/historical-odds':
                name = params['fixtureId'] + '__' + params['bookmakers'].replace(',', '__')
                relative = Path('history') / (name + '.json.gz')
            else:
                relative = Path(endpoint[1:] + '.json.gz')
            if not record['terminal']:
                relative = Path('responses') / (endpoint[1:] + f'__attempt{len(self.receipts) + 1}.json.gz')
            encoded = gzip.compress(body, mtime=0)
            atomic(self.root / relative, encoded)
            record.update(raw_path=relative.as_posix(), file_sha256=sha(encoded))
        self.receipts.append(record)
        self.save_artifact('receipts.jsonl', b''.join(json.dumps(r, sort_keys=True).encode() + b'\n'
                                                    for r in self.receipts))
        if record['terminal']:
            self.cache[self.query_key(endpoint, params)] = (record, payload)
        if record['stop']:
            raise Stop(f'Acquisition stopped: {endpoint} HTTP {record["http_status"]}, {record["error_code"]}')
        return record, payload

    def prepare(self):
        if not self.args.resume:
            if self.root.exists() and any(self.root.iterdir()):
                raise Stop('Nonempty output directory requires --resume')
            self.persist_protocol()
        needed = sum(self.query_key(e, self.parameters(e)) not in self.cache for e in BILLABLE)
        if needed > self.args.max_billable_requests:
            raise Stop('Local billable cap cannot cover uncached discovery')
        record, payload = self.fetch('/account', {})
        if record['http_status'] != 200:
            raise Stop(f'Account gate failed: HTTP {record["http_status"]}, {record["error_code"]}')
        snapshot, effective = account_gate(payload, self.args.bookmakers, needed)
        frozen = self.protocol['effective_books']
        if frozen is not None and not set(frozen).issubset(effective):
            raise Stop('Current entitlement no longer covers the frozen bookmaker subset')
        self.protocol['accounts'].append({'phase': 'before', **snapshot})
        if frozen is None:
            self.protocol['effective_books'] = effective
        self.persist_protocol()
        record, payload = self.obtain('/markets', self.parameters('/markets'), resolve_market)
        if not record['terminal']:
            raise Stop('Market catalog acquisition failed')
        self.protocol['market_mapping'] = resolve_market(payload)
        self.persist_protocol()
        record, payload = self.obtain('/fixtures', self.parameters('/fixtures'),
                                      lambda p: discover_fixtures(p, self.args))
        if not record['terminal']:
            raise Stop('Fixture discovery acquisition failed')
        self.fixtures = discover_fixtures(payload, self.args)
        self.protocol['selected_fixture_ids'] = [f['fixtureId'] for f in self.fixtures[:self.args.max_history]]
        self.persist_protocol()

    def complete(self):
        ids = self.protocol['selected_fixture_ids']
        return (ids is not None and self.protocol['market_mapping'] is not None
                and self.protocol['effective_books'] is not None
                and all(self.query_key(e, self.parameters(e)) in self.cache for e in BILLABLE)
                and all(self.query_key('/historical-odds', self.parameters('/historical-odds', f))
                        in self.cache for f in ids))

    def derive(self, stopped=None):
        coverage, rows = [], []
        selected = set(self.protocol['selected_fixture_ids'] or [])
        for fixture in self.fixtures:
            fixture_id = fixture['fixtureId']
            if fixture_id not in selected:
                continue
            params = self.parameters('/historical-odds', fixture_id)
            query = self.query_key('/historical-odds', params)
            terminal = self.cache.get(query)
            attempts = [r for r in self.receipts if self.query_key(r['endpoint'], r['parameters']) == query]
            receipt = terminal[0] if terminal else (attempts[-1] if attempts else {})
            for book in self.args.bookmakers:
                item = {'fixtureId': fixture_id, 'bookmaker': book, 'status': 'not_attempted',
                        'reason': 'acquisition_incomplete', 'side_1_status': '', 'side_2_status': '',
                        'raw_path': receipt.get('raw_path') or '', 'decoded_sha256': receipt.get('decoded_sha256', ''),
                        'received_at': receipt.get('received_at', ''),
                        'origin': receipt.get('origin', 'network') if receipt else 'none'}
                row = None
                if book not in (self.protocol['effective_books'] or []):
                    item.update(status='not_entitled', reason='not_entitled',
                                raw_path='', decoded_sha256='', received_at='', origin='none')
                elif terminal and receipt['http_status'] == 404:
                    item.update(status='no_history', reason='provider_explicit_no_history')
                elif terminal:
                    result, row = select_pair(fixture, book, terminal[1], self.protocol['market_mapping'])
                    item.update(result)
                elif receipt:
                    item.update(status='failed_request',
                                reason=f'HTTP_{receipt["http_status"]}:{receipt["error_code"] or "FAILED"}')
                coverage.append(item)
                if row:
                    row.update({f: item[f] for f in ('received_at', 'decoded_sha256', 'raw_path')})
                    rows.append(row)
        self.save_artifact('coverage.csv', csv_bytes(COVERAGE_FIELDS, coverage))
        self.save_artifact('selected_pre_match_quotes.csv', csv_bytes(QUOTE_FIELDS, rows))
        by_book = {b: dict(Counter(r['status'] for r in coverage if r['bookmaker'] == b)) for b in self.args.bookmakers}
        reasons = {b: dict(Counter(r['reason'] for r in coverage if r['bookmaker'] == b)) for b in self.args.bookmakers}
        summary = {'acquisition_complete': self.complete(), 'stop_reason': stopped,
                   'discovered_fixtures': len(self.fixtures), 'selected_fixtures': len(selected),
                   'selected_pairs': len(rows), 'coverage_rows': len(coverage),
                   'coverage_by_book': by_book, 'reasons_by_book': reasons,
                   'actual_attempts_this_invocation': dict(self.client.attempts),
                   'billable_attempts_this_invocation': self.client.billable,
                   'actual_attempts_all_invocations': dict(Counter(r['endpoint'] for r in self.receipts)),
                   'billable_attempts_all_invocations': sum(r['endpoint'] in BILLABLE for r in self.receipts),
                   'cached_responses_this_invocation': dict(self.reused),
                   'binding_status': 'provider_catalog_convention_only', 'execution_eligibility': 'not_certified'}
        self.save_artifact('summary.json', json_bytes(summary))
        return summary

    def execute(self):
        stopped = None
        offline = self.args.resume and self.complete()
        try:
            if offline:
                for endpoint in ('/markets', '/fixtures'):
                    self.reused[endpoint] += 1
            else:
                self.prepare()
            for fixture_id in self.protocol['selected_fixture_ids']:
                self.obtain('/historical-odds', self.parameters('/historical-odds', fixture_id),
                            lambda p: validate_history(p, fixture_id))
            if not offline and self.complete():
                record, payload = self.fetch('/account', {})
                if record['http_status'] == 200:
                    snapshot, _ = account_gate(payload, self.args.bookmakers, None)
                    self.protocol['accounts'].append({'phase': 'after', **snapshot})
                    self.persist_protocol()
                else:
                    raise Stop('After-account snapshot unavailable')
        except Stop as error:
            stopped = str(error)
        return self.derive(stopped)


def main(argv=None):
    args = parse_args(argv)
    key = os.environ.get(args.api_key_env, '').strip()
    try:
        run = Run(args, key)
        if args.resume:
            run.load_resume()
        if not key and not (args.resume and run.complete()):
            print(f'Prerequisite missing: nonblank {args.api_key_env} in process environment', file=sys.stderr)
            return 2
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
            raise Stop('Nonempty output directory requires --resume')
        summary = run.execute()
        print(json.dumps(summary, indent=2))
        return 0 if summary['acquisition_complete'] and not summary['stop_reason'] else 1
    except Stop as error:
        print(f'Collection failed: {error}', file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print('Collection failed: local artifact I/O or encoding failure', file=sys.stderr)
        return 1




if __name__ == '__main__':
    raise SystemExit(main())
