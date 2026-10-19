"""SQLite research evidence and a durable, single-running-job work queue.

Every public operation opens its own connection. Raw evidence is immutable and
content addressed; mutable provider fixture metadata is ordered by observation,
not by import order. Credentials never belong to this database.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from scripts.collect_cs_oddspapi import SAFE_BOOK, SAFE_ID


SCHEMA_VERSION = 1
DEFAULT_SETTINGS = {
    'bookmakers': ['pinnacle', 'bet365', 'unibet'],
    'billable_per_run': 4, 'billable_per_day': 8, 'max_requests': 100,
    'max_history': 100, 'max_fixtures': 20, 'upcoming_days': 7,
    'finished_days': 2, 'min_snapshot_minutes': 60,
    'scheduler_enabled': False, 'scheduler_interval_minutes': 360,
}
_INT_BOUNDS = {
    'billable_per_run': (0, 10000), 'billable_per_day': (0, 10000),
    'max_requests': (0, 10000), 'max_history': (0, 10000),
    'max_fixtures': (0, 10000), 'upcoming_days': (1, 7),
    'finished_days': (1, 7), 'min_snapshot_minutes': (1, 525600),
    'scheduler_interval_minutes': (1, 525600),
}
_JOB_KINDS = {'backfill', 'upcoming', 'finished', 'results', 'import_existing', 'import_history', 'remap'}
_JOB_STATUSES = {'queued', 'running', 'completed', 'partial', 'failed', 'budget_paused', 'cancelled'}
_RETRY_STATUSES = {'partial', 'failed', 'budget_paused', 'cancelled'}
_HISTORY_COLUMNS = (
    'history_key', 'source', 'source_match_id', 'start_time', 'source_date',
    'timezone_status', 'team1', 'team2', 'tournament', 'score1', 'score2',
    'winner_1', 'format', 'label_status', 'metadata_json', 'source_sha256',
)
_ACCOUNT_RECEIPT_FIELDS = {
    'endpoint', 'parameters', 'request_started_at', 'started_at', 'received_at',
    'http_status', 'error_code', 'terminal', 'stop', 'request_start_monotonic',
    'receipt_monotonic', 'requested_books', 'returned_books', 'origin', 'billable',
    'raw_path', 'decoded_sha256', 'file_sha256',
}
_SENSITIVE_KEYS = {
    'apikey', 'authorization', 'password', 'secret', 'token', 'accesskey',
    'credential', 'accesstoken', 'refreshtoken', 'clientsecret',
}
_SCHEMA = (
    '''CREATE TABLE schema_migrations (
        version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)''',
    '''CREATE TABLE fixtures (
        fixture_id TEXT PRIMARY KEY, start_time TEXT NOT NULL, true_start_time TEXT,
        status_id INTEGER NOT NULL CHECK(status_id IN (0,1,2,3)),
        participant1_id TEXT, participant2_id TEXT, participant1_name TEXT NOT NULL,
        participant2_name TEXT NOT NULL, tournament TEXT NOT NULL,
        metadata_json TEXT NOT NULL, observed_at TEXT NOT NULL)''',
    '''CREATE TABLE historical_matches (
        history_key TEXT PRIMARY KEY, source TEXT NOT NULL, source_match_id TEXT NOT NULL,
        start_time TEXT, source_date TEXT NOT NULL,
        timezone_status TEXT NOT NULL CHECK(timezone_status IN ('aware','naive')),
        team1 TEXT NOT NULL, team2 TEXT NOT NULL, tournament TEXT NOT NULL,
        score1 INTEGER, score2 INTEGER, winner_1 INTEGER CHECK(winner_1 IN (0,1)),
        format TEXT, label_status TEXT NOT NULL, metadata_json TEXT NOT NULL,
        source_sha256 TEXT, UNIQUE(source,source_match_id))''',
    '''CREATE TABLE history_revisions (
        id INTEGER PRIMARY KEY, history_key TEXT NOT NULL REFERENCES historical_matches(history_key),
        previous_json TEXT NOT NULL, replacement_json TEXT NOT NULL, created_at TEXT NOT NULL)''',
    '''CREATE TABLE raw_responses (
        id INTEGER PRIMARY KEY, job_id TEXT, endpoint TEXT NOT NULL,
        parameters_json TEXT NOT NULL, http_status INTEGER NOT NULL,
        error_code TEXT, started_at TEXT NOT NULL, received_at TEXT NOT NULL,
        decoded_sha256 TEXT, file_sha256 TEXT, raw_path TEXT,
        terminal INTEGER NOT NULL CHECK(terminal IN (0,1)),
        origin TEXT NOT NULL CHECK(origin IN ('network','import')),
        billable INTEGER NOT NULL CHECK(billable IN (0,1)), receipt_json TEXT NOT NULL,
        CHECK(origin != 'import' OR billable=0),
        CHECK(endpoint != '/account' OR
              (decoded_sha256 IS NULL AND file_sha256 IS NULL AND raw_path IS NULL)))''',
    '''CREATE UNIQUE INDEX raw_receipt_identity ON raw_responses
        (endpoint,parameters_json,received_at,COALESCE(decoded_sha256,''),http_status)''',
    '''CREATE INDEX raw_query_cache ON raw_responses(endpoint,parameters_json,terminal,received_at)''',
    '''CREATE INDEX raw_billable_clock ON raw_responses(origin,billable,started_at)''',
    '''CREATE TABLE odds_states (
        id INTEGER PRIMARY KEY, fixture_id TEXT NOT NULL REFERENCES fixtures(fixture_id),
        bookmaker TEXT NOT NULL, market_id TEXT NOT NULL, outcome_id TEXT NOT NULL,
        source_time TEXT, active INTEGER CHECK(active IN (0,1)), price REAL,
        raw_id INTEGER NOT NULL REFERENCES raw_responses(id), sequence_no INTEGER NOT NULL,
        state_json TEXT NOT NULL,
        UNIQUE(raw_id,bookmaker,market_id,outcome_id,sequence_no))''',
    '''CREATE INDEX states_fixture ON odds_states(fixture_id,bookmaker,source_time,id)''',
    '''CREATE TABLE quotes (
        id INTEGER PRIMARY KEY, fixture_id TEXT NOT NULL REFERENCES fixtures(fixture_id),
        bookmaker TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('historical','current')),
        observed_at TEXT NOT NULL, source_time_1 TEXT, source_time_2 TEXT,
        price1 REAL, price2 REAL, probability1 REAL, cutoff_at TEXT,
        status TEXT NOT NULL, reason TEXT NOT NULL,
        raw_id INTEGER NOT NULL REFERENCES raw_responses(id), details_json TEXT NOT NULL,
        UNIQUE(fixture_id,bookmaker,kind,raw_id))''',
    '''CREATE INDEX quotes_latest ON quotes(fixture_id,bookmaker,kind,observed_at DESC,id DESC)''',
    '''CREATE TABLE links (
        fixture_id TEXT PRIMARY KEY REFERENCES fixtures(fixture_id),
        history_key TEXT REFERENCES historical_matches(history_key),
        status TEXT NOT NULL CHECK(status IN ('automatic','manual','rejected','unmatched','ambiguous')),
        orientation INTEGER CHECK(orientation IN (0,1)), method TEXT NOT NULL,
        reason TEXT NOT NULL, updated_at TEXT NOT NULL)''',
    '''CREATE UNIQUE INDEX links_accepted_history ON links(history_key)
        WHERE history_key IS NOT NULL AND status IN ('automatic','manual')''',
    '''CREATE TABLE link_decisions (
        id INTEGER PRIMARY KEY, fixture_id TEXT NOT NULL REFERENCES fixtures(fixture_id),
        history_key TEXT REFERENCES historical_matches(history_key), decision TEXT NOT NULL,
        orientation INTEGER CHECK(orientation IN (0,1)), reason TEXT NOT NULL,
        created_at TEXT NOT NULL, previous_json TEXT NOT NULL)''',
    '''CREATE TABLE link_candidates (
        fixture_id TEXT NOT NULL REFERENCES fixtures(fixture_id),
        history_key TEXT NOT NULL REFERENCES historical_matches(history_key),
        status TEXT NOT NULL, reason TEXT NOT NULL, details_json TEXT NOT NULL,
        UNIQUE(fixture_id,history_key))''',
    '''CREATE TABLE jobs (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('queued','running','completed','partial','failed','budget_paused','cancelled')),
        created_at TEXT NOT NULL, start_time TEXT, finished_at TEXT,
        progress_json TEXT NOT NULL, result_json TEXT NOT NULL, error TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)))''',
    '''CREATE UNIQUE INDEX single_running_job ON jobs((1)) WHERE status='running' ''',
    '''CREATE INDEX jobs_queue ON jobs(status,created_at,id)''',
    '''CREATE TABLE settings (key TEXT PRIMARY KEY,value_json TEXT NOT NULL)''',
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _time(value, field='timestamp') -> str:
    if not isinstance(value, str):
        raise ValueError(f'{field} must be a timezone-aware timestamp')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')
    except (ValueError, OverflowError):
        raise ValueError(f'{field} must be a timezone-aware timestamp') from None


def _json(value, *, nonfinite=False) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=nonfinite)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Value must be serializable JSON') from None


def _object(value, field='metadata', *, encoded=False) -> dict:
    if encoded and isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            raise ValueError(f'{field} must be a JSON object') from None
    if not isinstance(value, dict):
        raise ValueError(f'{field} must be a JSON object')
    return value


def _text(value, field, *, nullable=False, nonempty=False):
    if nullable and value is None:
        return None
    if not isinstance(value, str) or '\x00' in value or (nonempty and not value.strip()):
        raise ValueError(f'{field} must be text')
    return value


def _identity(value, field='identifier', *, nullable=False) -> str | None:
    if nullable and value is None:
        return None
    if type(value) is int and value >= 0:
        value = str(value)
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f'{field} is not a safe identifier')
    return value


def _book(value):
    if not isinstance(value, str) or not SAFE_BOOK.fullmatch(value):
        raise ValueError('bookmaker must be a safe provider slug')
    return value


def _books(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError('bookmakers must contain one to three distinct provider slugs')
    books = [_book(book) for book in value]
    if len(set(books)) != len(books):
        raise ValueError('bookmakers must contain one to three distinct provider slugs')
    return books


def _integer(value, field, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{field} must be an integer between {lower} and {upper}')
    return value


def _digest(value):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        raise ValueError('Invalid evidence digest')
    return value.lower()


def _sensitive(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('JSON object keys must be text')
            normalized = re.sub(r'[^a-z]', '', key.lower())
            if normalized in _SENSITIVE_KEYS:
                raise ValueError('Credentials are forbidden in persisted data')
            _sensitive(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _sensitive(item)


def _decimal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 1 else None


def _safe_error(value):
    if value is None:
        return None
    _text(value, 'error')
    if re.search(r'https?://|api[_-]?key|authorization|password|secret|token', value, re.IGNORECASE):
        return 'Sensitive error details withheld'
    return ''.join(character for character in value if character.isprintable())[:1000]


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / 'app.sqlite3'
        self.raw_root = self.root / 'raw'
        if self.raw_root.is_symlink():
            raise ValueError('Raw evidence directory cannot be a symbolic link')
        self.raw_root.mkdir(exist_ok=True)
        if self.db_path.is_symlink():
            raise ValueError('Application database cannot be a symbolic link')
        with self.connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
        with self.transaction() as connection:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError('Application database has an unsupported newer schema')
            if version == 0:
                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.execute('INSERT INTO schema_migrations(version,applied_at) VALUES(?,?)',
                                   (SCHEMA_VERSION, _now()))
                connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA busy_timeout=30000')
            connection.execute('PRAGMA synchronous=FULL')
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def query(self, sql, params=()) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def one(self, sql, params=()) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(sql, params).fetchone()
            return dict(row) if row is not None else None

    def execute(self, sql, params=()) -> int:
        with self.transaction() as connection:
            return connection.execute(sql, params).rowcount

    def upsert_fixture(self, fixture: dict, observed_at: str | None = None) -> None:
        fixture = _object(fixture, 'fixture')
        _sensitive(fixture)
        if type(fixture.get('sportId')) is not int or fixture['sportId'] != 17:
            raise ValueError('Only sport-17 fixtures are supported')
        identity = _identity(fixture.get('fixtureId'), 'fixtureId')
        status = _integer(fixture.get('statusId'), 'statusId', 0, 3)
        start = _time(fixture.get('startTime'), 'startTime')
        actual = fixture.get('trueStartTime')
        actual = _time(actual, 'trueStartTime') if actual is not None else None
        observation = _now() if observed_at is None else _time(observed_at, 'observed_at')
        values = (identity, start, actual, status,
                  _identity(fixture.get('participant1Id'), 'participant1Id', nullable=True),
                  _identity(fixture.get('participant2Id'), 'participant2Id', nullable=True),
                  _text(fixture.get('participant1Name', ''), 'participant1Name'),
                  _text(fixture.get('participant2Name', ''), 'participant2Name'),
                  _text(fixture.get('tournamentName', ''), 'tournamentName'),
                  _json(fixture), observation)
        with self.transaction() as connection:
            previous = connection.execute('SELECT * FROM fixtures WHERE fixture_id=?', (identity,)).fetchone()
            if previous is not None:
                if previous['observed_at'] > observation:
                    return
                if previous['observed_at'] == observation:
                    if previous['metadata_json'] != values[9]:
                        raise ValueError('Conflicting fixture metadata at the same observation time')
                    return
            connection.execute('''INSERT INTO fixtures VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fixture_id) DO UPDATE SET start_time=excluded.start_time,
                true_start_time=excluded.true_start_time,status_id=excluded.status_id,
                participant1_id=excluded.participant1_id,participant2_id=excluded.participant2_id,
                participant1_name=excluded.participant1_name,participant2_name=excluded.participant2_name,
                tournament=excluded.tournament,metadata_json=excluded.metadata_json,
                observed_at=excluded.observed_at''', values)

    @staticmethod
    def normalize_history(match: dict) -> dict:
        match = _object(match, 'historical match')
        source = _text(match.get('source'), 'source', nonempty=True)
        source_id = match.get('source_match_id')
        if type(source_id) is int:
            source_id = str(source_id)
        source_id = _text(source_id, 'source_match_id', nonempty=True)
        key = _text(match.get('history_key'), 'history_key', nonempty=True)
        if key != f'{source}:{source_id}':
            raise ValueError('history_key must be source:source_match_id')
        clock_status = match.get('timezone_status')
        if clock_status not in ('aware', 'naive'):
            raise ValueError('timezone_status must be aware or naive')
        start = match.get('start_time')
        if start is not None:
            if clock_status != 'aware':
                raise ValueError('Naive source dates cannot supply UTC start_time')
            start = _time(start, 'start_time')
        metadata = _object(match.get('metadata_json', {}), encoded=True)
        _sensitive(metadata)
        scores = []
        for field in ('score1', 'score2'):
            value = match.get(field)
            scores.append(_integer(value, field, 0, 100000) if value is not None else None)
        winner = match.get('winner_1')
        if winner is not None:
            winner = _integer(winner, 'winner_1', 0, 1)
            if all(score is not None for score in scores):
                if scores[0] == scores[1] or winner != int(scores[0] > scores[1]):
                    raise ValueError('winner_1 conflicts with the explicit score')
        normalized = dict(zip(_HISTORY_COLUMNS, (
            key, source, source_id, start,
            _text(match.get('source_date', ''), 'source_date'), clock_status,
            _text(match.get('team1'), 'team1'), _text(match.get('team2'), 'team2'),
            _text(match.get('tournament', ''), 'tournament'), scores[0], scores[1], winner,
            _text(match.get('format'), 'format', nullable=True),
            _text(match.get('label_status'), 'label_status'), _json(metadata),
            _digest(match.get('source_sha256')),
        )))
        return normalized

    def put_history(self, match: dict) -> None:
        normalized = self.normalize_history(match)
        key, source = normalized['history_key'], normalized['source']
        metadata = json.loads(normalized['metadata_json'])
        with self.transaction() as connection:
            previous = connection.execute('SELECT * FROM historical_matches WHERE history_key=?', (key,)).fetchone()
            if previous is not None:
                old = dict(previous)
                if old == normalized:
                    return
                if source != 'oddspapi' or old['source'] != source:
                    raise ValueError('Conflicting historical match identity')
                try:
                    old_metadata = json.loads(old['metadata_json'])
                    old_clock = old_metadata['scores']['periods']['result']['updatedAt']
                    new_clock = metadata['scores']['periods']['result']['updatedAt']
                    if _time(new_clock) <= _time(old_clock):
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    raise ValueError('Native score correction requires strictly newer aware result updatedAt') from None
                connection.execute('''INSERT INTO history_revisions
                    (history_key,previous_json,replacement_json,created_at) VALUES(?,?,?,?)''',
                                   (key, _json(old), _json(normalized), _now()))
                assignments = ','.join(f'{field}=?' for field in _HISTORY_COLUMNS[1:])
                connection.execute(f'UPDATE historical_matches SET {assignments} WHERE history_key=?',
                                   tuple(normalized[field] for field in _HISTORY_COLUMNS[1:]) + (key,))
            else:
                columns = ','.join(_HISTORY_COLUMNS)
                placeholders = ','.join('?' for _ in _HISTORY_COLUMNS)
                connection.execute(f'INSERT INTO historical_matches({columns}) VALUES({placeholders})',
                                   tuple(normalized[field] for field in _HISTORY_COLUMNS))

    def _write_raw(self, body: bytes, decoded_sha: str) -> tuple[str, str]:
        if self.raw_root.is_symlink() or self.raw_root.resolve().parent != self.root:
            raise ValueError('Raw evidence directory is outside the store')
        encoded = gzip.compress(body, mtime=0)
        file_sha = hashlib.sha256(encoded).hexdigest()
        target = self.raw_root / f'{decoded_sha}.json.gz'
        if target.is_symlink():
            raise ValueError('Raw evidence file cannot be a symbolic link')
        if target.exists():
            if target.read_bytes() != encoded:
                raise ValueError('Existing immutable raw evidence is corrupt')
            return target.relative_to(self.root).as_posix(), file_sha
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.raw_root, prefix='.raw-', suffix='.partial', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.is_symlink() or target.read_bytes() != encoded:
                    raise ValueError('Existing immutable raw evidence is corrupt') from None
            directory = os.open(self.raw_root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target.relative_to(self.root).as_posix(), file_sha

    def put_raw(self, record: dict, body: bytes | None, *, job_id=None,
                origin='network', billable=False) -> int:
        record = dict(_object(record, 'receipt'))
        guard = record.pop('secret_guard', None)
        if guard is not None and not callable(guard):
            raise ValueError('secret_guard must be callable')
        if body is not None and not isinstance(body, bytes):
            raise ValueError('Raw evidence body must be bytes')
        _sensitive(record)
        endpoint = record.get('endpoint')
        local_history = (isinstance(endpoint, str) and re.fullmatch(r'local://history/(upload|opencs2|kaggle|strandedpond|ektarr|hltv)', endpoint)
                         and origin == 'import' and billable is False and record.get('http_status') == 0)
        if not isinstance(endpoint, str) or not (re.fullmatch(r'/[a-z][a-z0-9-]*', endpoint) or local_history):
            raise ValueError('Receipt endpoint must be a provider path or an unbilled local historical import')
        parameters = _object(record.get('parameters', {}), 'parameters')
        parameters_json = _json(parameters)
        if endpoint == '/account':
            if (body is not None or record.get('decoded_sha256') is not None
                    or record.get('file_sha256') is not None or record.get('raw_path') is not None
                    or set(record) - _ACCOUNT_RECEIPT_FIELDS):
                raise ValueError('Account bodies and digests are forbidden')
        serialized_record = _json(record).encode()
        if guard is not None:
            try:
                detected = guard(serialized_record) or (body is not None and guard(body))
            except Exception:
                raise ValueError('Secret guard could not validate receipt safety') from None
            if detected:
                raise ValueError('Credential-bearing evidence is forbidden')
        if origin not in ('network', 'import') or type(billable) is not bool:
            raise ValueError('Invalid receipt origin or billable flag')
        if origin == 'import' and billable:
            raise ValueError('Imported receipts cannot be billable')
        if job_id is not None:
            _text(job_id, 'job_id', nonempty=True)
        started = _time(record.get('request_started_at', record.get('started_at')), 'request_started_at')
        received = _time(record.get('received_at'), 'received_at')
        if started > received:
            raise ValueError('Receipt time cannot precede request start')
        status = _integer(record.get('http_status'), 'http_status', 0, 599)
        terminal = record.get('terminal', False)
        if type(terminal) is not bool:
            raise ValueError('terminal must be a boolean')
        error = record.get('error_code')
        if error is not None and (not isinstance(error, str) or not re.fullmatch(r'[A-Z0-9_]{1,80}', error)):
            raise ValueError('error_code must be a sanitized provider error code')
        decoded_sha = _digest(record.get('decoded_sha256'))
        if body is not None:
            actual_sha = hashlib.sha256(body).hexdigest()
            if decoded_sha is not None and decoded_sha != actual_sha:
                raise ValueError('Decoded evidence digest mismatch')
            decoded_sha = actual_sha
        elif record.get('file_sha256') is not None:
            raise ValueError('A saved-file digest requires raw response bytes')
        identity = (endpoint, parameters_json, received, decoded_sha or '', status)
        with self.transaction() as connection:
            existing = connection.execute('''SELECT * FROM raw_responses WHERE endpoint=?
                AND parameters_json=? AND received_at=? AND COALESCE(decoded_sha256,'')=?
                AND http_status=?''', identity).fetchone()
            if existing is not None:
                if (existing['started_at'] != started or existing['error_code'] != error
                        or existing['terminal'] != int(terminal)):
                    raise ValueError('Conflicting immutable receipt identity')
                if body is not None:
                    raw_path, file_sha = self._write_raw(body, decoded_sha)
                    if existing['raw_path'] != raw_path or existing['file_sha256'] != file_sha:
                        raise ValueError('Conflicting immutable receipt evidence')
                return existing['id']
            raw_path, file_sha = self._write_raw(body, decoded_sha) if body is not None else (None, None)
            record.update(endpoint=endpoint, parameters=parameters, request_started_at=started,
                          received_at=received, http_status=status, terminal=terminal,
                          error_code=error, origin=origin, billable=billable)
            record.pop('started_at', None)
            if endpoint == '/account':
                for field in ('decoded_sha256', 'file_sha256', 'raw_path'):
                    record.pop(field, None)
            else:
                record.update(decoded_sha256=decoded_sha, file_sha256=file_sha, raw_path=raw_path)
            cursor = connection.execute('''INSERT INTO raw_responses
                (job_id,endpoint,parameters_json,http_status,error_code,started_at,received_at,
                 decoded_sha256,file_sha256,raw_path,terminal,origin,billable,receipt_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (job_id, endpoint, parameters_json, status, error, started, received,
                 decoded_sha, file_sha, raw_path, int(terminal), origin, int(billable), _json(record)))
            return cursor.lastrowid

    @staticmethod
    def _raw_for_fixture(connection, fixture_id, raw_id):
        _integer(raw_id, 'raw_id', 1, 9223372036854775807)
        row = connection.execute('SELECT * FROM raw_responses WHERE id=?', (raw_id,)).fetchone()
        if row is None:
            raise ValueError('Raw receipt does not exist')
        if connection.execute('SELECT 1 FROM fixtures WHERE fixture_id=?', (fixture_id,)).fetchone() is None:
            raise ValueError('Fixture does not exist')
        parameters = json.loads(row['parameters_json'])
        if parameters.get('fixtureId') is not None and str(parameters['fixtureId']) != fixture_id:
            raise ValueError('Raw receipt belongs to a different fixture')
        if row['endpoint'] == '/account':
            raise ValueError('Account receipt cannot supply odds evidence')
        return row

    def add_states(self, fixture_id, bookmaker, market_id, outcome_id,
                   states: list[dict], raw_id: int) -> None:
        fixture_id = _identity(fixture_id, 'fixture_id')
        bookmaker = _book(bookmaker)
        market_id = _identity(market_id, 'market_id')
        outcome_id = _identity(outcome_id, 'outcome_id')
        if not isinstance(states, list):
            raise ValueError('states must be a list')
        normalized = []
        for sequence, state in enumerate(states):
            state = _object(state, 'state')
            _sensitive(state)
            try:
                source_time = _time(state.get('createdAt', state.get('changedAt')))
            except ValueError:
                source_time = None
            active = state.get('active')
            active = int(active) if type(active) is bool else None
            normalized.append((fixture_id, bookmaker, market_id, outcome_id, source_time,
                               active, _decimal(state.get('price')), raw_id, sequence,
                               _json(state, nonfinite=True)))
        with self.transaction() as connection:
            self._raw_for_fixture(connection, fixture_id, raw_id)
            for values in normalized:
                existing = connection.execute('''SELECT fixture_id,state_json FROM odds_states
                    WHERE raw_id=? AND bookmaker=? AND market_id=? AND outcome_id=? AND sequence_no=?''',
                    (raw_id, bookmaker, market_id, outcome_id, values[8])).fetchone()
                if existing is not None:
                    if existing['fixture_id'] != fixture_id or existing['state_json'] != values[9]:
                        raise ValueError('Conflicting immutable odds state identity')
                    continue
                connection.execute('''INSERT INTO odds_states
                    (fixture_id,bookmaker,market_id,outcome_id,source_time,active,price,raw_id,sequence_no,state_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?)''', values)

    def put_quote(self, fixture_id, bookmaker, kind, quote: dict | None,
                  coverage: dict, raw_id: int) -> None:
        fixture_id = _identity(fixture_id, 'fixture_id')
        bookmaker = _book(bookmaker)
        if kind not in ('historical', 'current'):
            raise ValueError('quote kind must be historical or current')
        coverage = _object(coverage, 'coverage')
        _sensitive(coverage)
        status = _text(coverage.get('status'), 'coverage status', nonempty=True)
        reason = _text(coverage.get('reason', ''), 'coverage reason')
        price1 = price2 = probability = source1 = source2 = cutoff = None
        if quote is not None:
            quote = _object(quote, 'quote')
            _sensitive(quote)
            if status != 'admitted':
                raise ValueError('Usable prices require admitted coverage')
            price1, price2 = _decimal(quote.get('price_1')), _decimal(quote.get('price_2'))
            value = quote.get('no_vig_probability_1')
            if (price1 is None or price2 is None or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1):
                raise ValueError('Quote must contain finite two-sided decimal prices and probability')
            probability = float(value)
            expected = (1 / price1) / (1 / price1 + 1 / price2)
            if not math.isclose(probability, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Quote probability disagrees with two-sided no-vig arithmetic')
            clocks = [quote.get(f'source_created_at_{side}', quote.get(f'source_changed_at_{side}'))
                      for side in (1, 2)]
            source1, source2 = [_time(clock) if clock is not None else None for clock in clocks]
            cutoff = _time(quote['cutoff']) if quote.get('cutoff') is not None else None
        elif status == 'admitted':
            raise ValueError('Admitted coverage requires a two-sided quote')
        details = _json({'quote': quote, 'coverage': coverage})
        with self.transaction() as connection:
            raw = self._raw_for_fixture(connection, fixture_id, raw_id)
            values = (fixture_id, bookmaker, kind, raw['received_at'], source1, source2,
                      price1, price2, probability, cutoff, status, reason, raw_id, details)
            existing = connection.execute('''SELECT * FROM quotes WHERE fixture_id=?
                AND bookmaker=? AND kind=? AND raw_id=?''', (fixture_id, bookmaker, kind, raw_id)).fetchone()
            if existing is not None:
                if existing['details_json'] != details:
                    raise ValueError('Conflicting immutable quote observation')
                return
            connection.execute('''INSERT INTO quotes
                (fixture_id,bookmaker,kind,observed_at,source_time_1,source_time_2,price1,price2,
                 probability1,cutoff_at,status,reason,raw_id,details_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', values)

    def create_job(self, kind, payload: dict) -> str:
        if not isinstance(kind, str) or kind not in _JOB_KINDS:
            raise ValueError('Unsupported job kind')
        payload = _object(payload, 'payload')
        _sensitive(payload)
        for key, value in payload.items():
            if key in _INT_BOUNDS:
                _integer(value, key, *_INT_BOUNDS[key])
            elif key == 'max_billable_requests':
                _integer(value, key, 0, 10000)
            elif key == 'bookmakers':
                _books(value)
        if kind == 'backfill' or 'from' in payload or 'to' in payload:
            start = _time(payload.get('from'), 'from')
            end = _time(payload.get('to'), 'to')
            if start >= end:
                raise ValueError('Job from/to interval must be positive')
        identity = uuid.uuid4().hex
        with self.transaction() as connection:
            connection.execute('''INSERT INTO jobs
                (id,kind,payload_json,status,created_at,progress_json,result_json,cancel_requested)
                VALUES(?,?,?,'queued',?,'{}','{}',0)''', (identity, kind, _json(payload), _now()))
        return identity

    @staticmethod
    def _job(row):
        if row is None:
            return None
        job = dict(row)
        for field in ('payload', 'progress', 'result'):
            job[field] = json.loads(job[field + '_json'])
        job['cancel_requested'] = bool(job['cancel_requested'])
        return job

    def get_job(self, identity) -> dict | None:
        return self._job(self.one('SELECT * FROM jobs WHERE id=?', (identity,)))

    def list_jobs(self, limit=50) -> list:
        limit = _integer(limit, 'limit', 1, 1000)
        return [self._job(row) for row in self.query('SELECT * FROM jobs ORDER BY created_at DESC,id DESC LIMIT ?', (limit,))]

    def claim_job(self, worker_id: str) -> dict | None:
        _text(worker_id, 'worker_id', nonempty=True)
        with self.transaction() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone() is not None:
                return None
            row = connection.execute("SELECT * FROM jobs WHERE status='queued' AND cancel_requested=0 ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            connection.execute("UPDATE jobs SET status='running',start_time=?,finished_at=NULL WHERE id=? AND status='queued'", (_now(), row['id']))
            return self._job(connection.execute('SELECT * FROM jobs WHERE id=?', (row['id'],)).fetchone())

    def update_job(self, identity, **fields) -> None:
        if set(fields) - {'status', 'progress', 'result', 'error'}:
            raise ValueError('Unsupported job update field')
        updates = {}
        for field in ('progress', 'result'):
            if field in fields:
                value = _object(fields[field], field)
                _sensitive(value)
                updates[field + '_json'] = _json(value)
        if 'error' in fields:
            updates['error'] = _safe_error(fields['error'])
        with self.transaction() as connection:
            current = connection.execute('SELECT * FROM jobs WHERE id=?', (identity,)).fetchone()
            if current is None:
                raise ValueError('Job does not exist')
            if 'status' in fields:
                status = fields['status']
                if not isinstance(status, str) or status not in _JOB_STATUSES:
                    raise ValueError('Unsupported job status')
                old_status = current['status']
                if status != old_status:
                    allowed = (old_status == 'running' and status in _JOB_STATUSES - {'queued', 'running'})
                    if not allowed:
                        raise ValueError('Invalid job status transition; use claim, cancel, or retry')
                    if current['cancel_requested']:
                        status = 'cancelled'
                    updates['status'] = status
                    updates['finished_at'] = _now()
            if updates:
                assignments = ','.join(f'{field}=?' for field in updates)
                connection.execute(f'UPDATE jobs SET {assignments} WHERE id=?', tuple(updates.values()) + (identity,))

    def cancel_job(self, identity) -> bool:
        with self.transaction() as connection:
            row = connection.execute('SELECT status FROM jobs WHERE id=?', (identity,)).fetchone()
            if row is None or row['status'] not in ('queued', 'running'):
                return False
            if row['status'] == 'queued':
                connection.execute("UPDATE jobs SET status='cancelled',cancel_requested=1,finished_at=? WHERE id=?", (_now(), identity))
            else:
                connection.execute('UPDATE jobs SET cancel_requested=1 WHERE id=?', (identity,))
            return True

    def retry_job(self, identity) -> bool:
        with self.transaction() as connection:
            row = connection.execute('SELECT status FROM jobs WHERE id=?', (identity,)).fetchone()
            if row is None or row['status'] not in _RETRY_STATUSES:
                return False
            connection.execute("UPDATE jobs SET status='queued',cancel_requested=0,start_time=NULL,finished_at=NULL,error=NULL WHERE id=?", (identity,))
            return True

    def cancelled(self, identity) -> bool:
        row = self.one('SELECT status,cancel_requested FROM jobs WHERE id=?', (identity,))
        return row is not None and (bool(row['cancel_requested']) or row['status'] == 'cancelled')

    def recover_jobs(self) -> int:
        """Recover interrupted jobs; the caller must hold the exclusive worker lock."""
        with self.transaction() as connection:
            return connection.execute("""UPDATE jobs SET status='partial',finished_at=?,
                error=COALESCE(error,'Worker interrupted before job completed') WHERE status='running'""", (_now(),)).rowcount

    def get_settings(self) -> dict:
        settings = dict(DEFAULT_SETTINGS, bookmakers=list(DEFAULT_SETTINGS['bookmakers']))
        for row in self.query('SELECT key,value_json FROM settings'):
            if row['key'] not in DEFAULT_SETTINGS:
                continue
            try:
                value = json.loads(row['value_json'])
            except (ValueError, TypeError):
                raise ValueError('Persisted application setting is not valid JSON') from None
            self._validate_setting(row['key'], value)
            settings[row['key']] = value
        return settings

    @staticmethod
    def _validate_setting(key, value):
        if key not in DEFAULT_SETTINGS:
            raise ValueError('Unknown application setting')
        if key == 'bookmakers':
            return _books(value)
        if key == 'scheduler_enabled':
            if type(value) is not bool:
                raise ValueError('scheduler_enabled must be boolean')
            return value
        return _integer(value, key, *_INT_BOUNDS[key])

    def save_settings(self, values: dict) -> dict:
        values = _object(values, 'settings')
        validated = {key: self._validate_setting(key, value) for key, value in values.items()}
        with self.transaction() as connection:
            for key, value in validated.items():
                connection.execute('''INSERT INTO settings(key,value_json) VALUES(?,?)
                    ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json''', (key, _json(value)))
        return self.get_settings()
