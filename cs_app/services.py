"""Historical inputs, audited link decisions and research exports on owned storage."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from scripts import collect_cs_oddspapi as provider
from scripts import map_cs_oddspapi_history as mapper




def now():
    return datetime.now(timezone.utc).isoformat()


def decoded(value, fallback=None):
    return json.loads(value, parse_constant=lambda _: None) if value else ({} if fallback is None else fallback)


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def native_history(fixture, payload, raw_sha256=None):
    if payload.get('fixtureId') != fixture['fixtureId']:
        raise ValueError('Native result fixture ID mismatch')
    result = payload.get('scores', {}).get('periods', {}).get('result')
    if not isinstance(result, dict):
        raise ValueError('No explicit native series-result period')
    one, two = result.get('participant1Score'), result.get('participant2Score')
    if any(type(x) is not int or x < 0 for x in (one, two)) or one == two:
        raise ValueError('Native result needs unequal nonnegative integer series scores')
    provider.utc(result.get('updatedAt'))
    return {'history_key': 'oddspapi:' + fixture['fixtureId'], 'source': 'oddspapi',
            'source_match_id': fixture['fixtureId'], 'start_time': fixture['startTime'],
            'source_date': result['updatedAt'], 'timezone_status': 'aware',
            'team1': fixture['participant1Name'], 'team2': fixture['participant2Name'],
            'tournament': fixture.get('tournamentName', ''), 'score1': one, 'score2': two,
            'winner_1': int(one > two), 'format': None, 'label_status': 'native_series_result',
            'metadata_json': json.dumps(payload, ensure_ascii=False),
            'source_sha256': raw_sha256}


def import_history(store, source, data, job_id=None):
    if source not in ('opencs2', 'kaggle', 'strandedpond', 'ektarr', 'hltv'):
        raise ValueError('Unsupported historical source; choose opencs2, kaggle, strandedpond, ektarr or hltv')
    rows = csv_rows(data)
    if not rows:
        raise ValueError('Historical CSV has no data rows')
    stamp = now()
    sha = hashlib.sha256(data).hexdigest()
    matches = []
    if source == 'opencs2':
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['match_id']].append(row)
        for series in mapper.historical_series(rows):
            identity = series['historical_match_id']
            first = grouped[identity][0]
            matches.append({'history_key': 'opencs2:' + identity, 'source': source,
                'source_match_id': identity, 'start_time': provider.iso(provider.utc(first['match_date'])),
                'source_date': first['match_date'], 'timezone_status': 'aware', 'team1': first['team1'],
                'team2': first['team2'], 'tournament': first['event'],
                'score1': int(first['score1']), 'score2': int(first['score2']),
                'winner_1': int(first['winner'] == 'team1') if series['label_status'] == 'valid_explicit_bo3_result' else None,
                'format': first['format'], 'label_status': series['label_status'],
                'metadata_json': json.dumps({'series': series, 'maps': grouped[identity]}, ensure_ascii=False),
                'source_sha256': sha})
    elif source == 'kaggle':
        for row in rows:
            identity = row['hltv_match_id']
            if not identity.isdigit():
                raise ValueError('Kaggle export must retain native HLTV IDs')
            one, two = int(row['score_team1']), int(row['score_team2'])
            valid = (row['match_type'].casefold() == 'bo3' and max(one, two) == 2
                     and min(one, two) in (0, 1) and row['winner'] == ('team1' if one > two else 'team2'))
            matches.append({'history_key': 'kaggle:' + identity, 'source': source,
                'source_match_id': identity, 'start_time': provider.iso(provider.utc(row['date'])),
                'source_date': row['date'], 'timezone_status': 'aware', 'team1': row['team1_name'],
                'team2': row['team2_name'], 'tournament': row['tournament'], 'score1': one, 'score2': two,
                'winner_1': int(one > two) if valid else None, 'format': row['match_type'],
                'label_status': 'valid_explicit_bo3_result' if valid else 'invalid_series_result',
                'metadata_json': json.dumps(row, ensure_ascii=False), 'source_sha256': sha})
    elif source in ('ektarr', 'hltv'):
        for row in rows:
            identity = row['match_id']
            one, two = int(float(row['score1'])), int(float(row['score2']))
            valid = (one != two)
            dt_str = (row.get('datetime') or row.get('date', '')).strip()
            parsed_dt = datetime.fromisoformat(dt_str)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            start_iso = provider.iso(parsed_dt)
            matches.append({'history_key': source + ':' + identity, 'source': source,
                'source_match_id': identity, 'start_time': start_iso,
                'source_date': start_iso, 'timezone_status': 'aware', 'team1': row['team1'],
                'team2': row['team2'], 'tournament': row['tournament'], 'score1': one, 'score2': two,
                'winner_1': int(one > two) if valid else None, 'format': row.get('format') or 'bo3',
                'label_status': 'valid_explicit_bo3_result' if valid else 'unsettled',
                'metadata_json': json.dumps(row, ensure_ascii=False), 'source_sha256': sha})
    else:
        grouped = defaultdict(list)
        for row in rows:
            identity = mapper.HLTV_URL.fullmatch(row['match_url'])
            if identity is None:
                raise ValueError('StrandedPond row must have native HLTV match URL')
            datetime.fromisoformat(row['date'])
            grouped[identity[1]].append(row)
        for identity, group in grouped.items():
            first = group[0]
            dates = sorted({r['date'] for r in group})
            matches.append({'history_key': 'strandedpond:' + identity, 'source': source,
                'source_match_id': identity, 'start_time': None, 'source_date': ';'.join(dates),
                'timezone_status': 'naive', 'team1': first['team_left'], 'team2': first['team_right'],
                'tournament': first['event'], 'score1': None, 'score2': None, 'winner_1': None,
                'format': None, 'label_status': 'map_only_no_series_format',
                'metadata_json': json.dumps({'maps': group}, ensure_ascii=False), 'source_sha256': sha})
    unique = {}
    for match in matches:
        match = store.normalize_history(match)
        previous = unique.get(match['history_key'])
        if previous is not None and previous != match:
            raise ValueError('CSV contains conflicting rows for the same historical identity')
        unique[match['history_key']] = match
    matches = list(unique.values())
    record = {'endpoint': 'local://history/' + source, 'parameters': {'source': source, 'sha256': sha},
              'request_started_at': stamp, 'received_at': stamp, 'http_status': 0,
              'error_code': None, 'terminal': True}
    store.put_raw(record, data, job_id=job_id, origin='import', billable=False)
    # Entire import must be rejected on conflicting identities, not partly applied.
    with store.transaction() as connection:
        for match in matches:
            existing = connection.execute('SELECT * FROM historical_matches WHERE history_key=?', (match['history_key'],)).fetchone()
            if existing:
                for key in ('source', 'source_match_id', 'start_time', 'source_date', 'timezone_status',
                            'team1', 'team2', 'tournament', 'score1', 'score2', 'winner_1', 'format', 'label_status', 'metadata_json'):
                    if existing[key] != match[key]:
                        raise ValueError('Historical source identity conflicts with an existing retained record')
        for match in matches:
            columns = list(match)
            connection.execute('INSERT OR IGNORE INTO historical_matches (' + ','.join(columns) + ') VALUES ('
                               + ','.join('?' for _ in columns) + ')', tuple(match[k] for k in columns))
    return {'source': source, 'rows': len(rows), 'series': len(matches), 'sha256': sha}


def import_existing(store, repo: Path, job_id=None):
    archive = repo / 'data/raw/oddspapi/cs2_full_2026'
    if not (archive / 'fixture_index.json').is_file():
        raise ValueError('Existing CS2 OddsPapi archive not found; use backfill to collect a new cohort')
    fixtures = json.loads((archive / 'fixture_index.json').read_bytes())['fixtures']
    manifest = json.loads((archive / 'manifest.json').read_bytes())
    history_reports = []
    for source, path in [('kaggle', repo / 'data/exports/kaggle/results_only.csv'),
                         ('opencs2', repo / 'data/exports/opencs2/maps.csv'),
                         ('strandedpond', repo / 'data/raw/strandedpond/matches.csv'),
                         ('ektarr', repo / 'data/exports/ektarr/matches.csv')]:
        if path.is_file():
            history_reports.append(import_history(store, source, path.read_bytes(), job_id))
    fixture_index = {f['fixtureId']: f for f in fixtures}
    receipts = 0
    for number, window in enumerate(manifest['windows']):
        if job_id and store.cancelled(job_id):
            return {'status': 'cancelled', 'imported_receipts': receipts}
        folder = archive / window['directory']
        protocol = json.loads((folder / 'protocol.json').read_bytes())
        mapping = protocol['market_mapping']
        records = [json.loads(line) for line in (folder / 'receipts.jsonl').read_bytes().splitlines()]
        for original in records:
            relative = original.get('raw_path')
            body = None
            if relative:
                path = folder / relative
                if not path.resolve().is_relative_to(folder.resolve()):
                    raise ValueError('Unsafe archived raw evidence path')
                encoded = path.read_bytes()
                body = gzip.decompress(encoded)
                if hashlib.sha256(encoded).hexdigest() != original['file_sha256'] or hashlib.sha256(body).hexdigest() != original['decoded_sha256']:
                    raise ValueError('Archived raw receipt hash mismatch')
            raw_id = store.put_raw(original, body, job_id=job_id, origin='import', billable=False)
            receipts += 1
            if original['endpoint'] == '/fixtures' and original['http_status'] == 200:
                observed = provider.discover_fixtures(json.loads(body), SimpleNamespace(start=window['from'], end=window['to']))
                for fixture in observed:
                    store.upsert_fixture(fixture, observed_at=original['received_at'])
            if original['endpoint'] != '/historical-odds' or not original['terminal']:
                continue
            identity = original['parameters']['fixtureId']
            fixture = fixture_index[identity]
            payload = json.loads(body)
            requested = original['parameters']['bookmakers'].split(',')
            for book in protocol['requested_books']:
                if book not in requested:
                    coverage, quote = {'status': 'not_entitled', 'reason': 'outside_frozen_entitled_subset'}, None
                elif provider.no_history(original['http_status'], payload):
                    coverage, quote = {'status': 'no_history', 'reason': 'provider_explicit_no_history'}, None
                else:
                    provider.validate_history(payload, identity)
                    market = payload.get('bookmakers', {}).get(book, {}).get('markets', {}).get(mapping['market_id'], {})
                    for outcome in (mapping['outcome_1'], mapping['outcome_2']):
                        states = market.get('outcomes', {}).get(outcome, {}).get('players', {}).get('0', [])
                        if isinstance(states, list):
                            store.add_states(identity, book, mapping['market_id'], outcome, states, raw_id)
                    coverage, quote = provider.select_pair(fixture, book, payload, mapping)
                store.put_quote(identity, book, 'historical', quote, coverage, raw_id)
        if job_id:
            store.update_job(job_id, progress={'stage': 'import_archive', 'windows': number + 1,
                                               'total_windows': len(manifest['windows']), 'receipts': receipts})
    native_root = repo / 'data/raw/oddspapi/cs2_20260414_20260420_probe_evaluation'
    if (native_root / 'receipts.jsonl').is_file():
        for record in [json.loads(line) for line in (native_root / 'receipts.jsonl').read_bytes().splitlines()]:
            if record['endpoint'] != '/scores' or record['http_status'] != 200:
                continue
            encoded = (native_root / record['raw_path']).read_bytes()
            body = gzip.decompress(encoded)
            if hashlib.sha256(body).hexdigest() != record['decoded_sha256']:
                raise ValueError('Native score receipt hash mismatch')
            payload = json.loads(body)
            store.put_raw(record, body, job_id=job_id, origin='import', billable=False)
            store.put_history(native_history(fixture_index[payload['fixtureId']], payload, record['decoded_sha256']))
    rebuilt = rebuild_mappings(store)
    return {'status': 'completed', 'fixtures': len(fixtures), 'imported_receipts': receipts,
            'historical_sources': history_reports, 'mapping': rebuilt, 'network_requests': 0}


def rebuild_mappings(store):
    fixture_rows = store.query('SELECT * FROM fixtures ORDER BY start_time,fixture_id')
    fixtures = [decoded(r['metadata_json']) for r in fixture_rows if r['status_id'] == 2]
    history = store.query('SELECT * FROM historical_matches ORDER BY history_key')
    primary, secondary, hltv_series, legacy_rows = [], [], [], []
    native = {}
    for row in history:
        metadata = decoded(row['metadata_json'])
        if row['source'] == 'opencs2':
            primary.append({'historical_source': 'opencs2', 'historical_match_id': row['source_match_id'],
                'historical_url': metadata.get('series', {}).get('historical_url', ''),
                'historical_team1': row['team1'], 'historical_team2': row['team2'],
                'historical_event': row['tournament'], 'historical_source_time': row['source_date'],
                'historical_score1': row['score1'], 'historical_score2': row['score2'],
                'metadata_conflict': metadata.get('series', {}).get('metadata_conflict', False),
                'label_status': row['label_status']})
        elif row['source'] == 'ektarr':
            secondary.append({'historical_source': 'ektarr', 'historical_match_id': row['source_match_id'],
                'historical_url': '',
                'historical_team1': row['team1'], 'historical_team2': row['team2'],
                'historical_event': row['tournament'], 'historical_source_time': row['source_date'],
                'historical_score1': row['score1'], 'historical_score2': row['score2'],
                'metadata_conflict': False,
                'label_status': row['label_status']})
        elif row['source'] == 'hltv':
            hltv_series.append({'historical_source': 'hltv', 'historical_match_id': row['source_match_id'],
                'historical_url': metadata.get('match_url', ''),
                'historical_team1': row['team1'], 'historical_team2': row['team2'],
                'historical_event': row['tournament'], 'historical_source_time': row['source_date'],
                'historical_score1': row['score1'], 'historical_score2': row['score2'],
                'metadata_conflict': False,
                'label_status': row['label_status']})
        elif row['source'] == 'strandedpond':
            legacy_rows.extend(metadata.get('maps', []))
        elif row['source'] == 'oddspapi' and row['winner_1'] is not None:
            native[row['source_match_id']] = row
    auto_oc2, candidates_oc2 = mapper.map_fixtures(fixtures, primary)
    auto_ek, candidates_ek = mapper.map_fixtures(fixtures, secondary)
    auto_hltv, candidates_hltv = mapper.map_fixtures(fixtures, hltv_series)
    legacy, _ = mapper.stranded_candidates(legacy_rows, fixtures, auto_oc2)
    candidates = candidates_oc2 + candidates_ek + candidates_hltv + legacy
    by_oc2 = {r['fixtureId']: r for r in auto_oc2}
    by_ek = {r['fixtureId']: r for r in auto_ek}
    by_hltv = {r['fixtureId']: r for r in auto_hltv}
    timestamp = now()
    candidate_records = []
    for candidate in candidates:
        source = candidate['historical_source']
        key = source + ':' + candidate['historical_match_id']
        candidate_records.append((candidate['fixtureId'], key, candidate['mapping_status'],
                                  candidate['mapping_reason'], json.dumps(candidate, ensure_ascii=False)))
    for identity, row in native.items():
        candidate_records.append((identity, row['history_key'], 'native_identity', 'same_provider_fixture_id', '{}'))
    with store.transaction() as connection:
        connection.execute('DELETE FROM link_candidates')
        connection.executemany('INSERT OR REPLACE INTO link_candidates(fixture_id,history_key,status,reason,details_json) VALUES(?,?,?,?,?)', candidate_records)
        existing = {r['fixture_id']: dict(r) for r in connection.execute('SELECT * FROM links')}
        protected = {r['history_key'] for r in existing.values() if r['status'] == 'manual' and r['history_key']}
        for fixture in fixture_rows:
            identity = fixture['fixture_id']
            if existing.get(identity, {}).get('status') in ('manual', 'rejected'):
                continue
            prop_oc2 = by_oc2.get(identity)
            prop_hltv = by_hltv.get(identity)
            prop_ek = by_ek.get(identity)
            if prop_oc2 and prop_oc2['mapping_status'] == 'matched':
                proposed = prop_oc2
            elif prop_hltv and prop_hltv['mapping_status'] == 'matched':
                proposed = prop_hltv
            elif prop_ek and prop_ek['mapping_status'] == 'matched':
                proposed = prop_ek
            else:
                proposed = prop_oc2 or prop_hltv or prop_ek
            history_key, orientation, method = None, None, 'none'
            status, reason = 'unmatched', 'no_supported_historical_identity'
            if identity in native:
                history_key, orientation, method = native[identity]['history_key'], 1, 'native_provider_fixture_id'
                status, reason = 'automatic', 'exact_native_result_identity'
            elif proposed and proposed['mapping_status'] == 'matched':
                history_key = (proposed.get('historical_source') or 'opencs2') + ':' + proposed['historical_match_id']
                orientation = int(proposed['participant1_is_historical_team1'])
                method = proposed['matching_method']
                status, reason = 'automatic', 'declared_unique_natural_keys_not_certified'
            elif proposed:
                reason = proposed['mapping_status']
                status = 'ambiguous' if reason.startswith('ambiguous') else 'unmatched'
            if history_key in protected:
                status, reason, history_key, orientation = 'ambiguous', 'historical_identity_reserved_by_manual_decision', None, None
            connection.execute('INSERT INTO links(fixture_id,history_key,status,orientation,method,reason,updated_at) VALUES(?,?,?,?,?,?,?) '
                               'ON CONFLICT(fixture_id) DO UPDATE SET history_key=excluded.history_key,status=excluded.status,orientation=excluded.orientation,method=excluded.method,reason=excluded.reason,updated_at=excluded.updated_at',
                               (identity, history_key, status, orientation, method, reason, timestamp))
    return {'statuses': dict(Counter(r['status'] for r in store.query('SELECT status FROM links'))), 'candidates': len(candidate_records)}


def decide_mapping(store, fixture_id, decision, history_key=None, orientation=None, reason=''):
    if decision not in ('accept', 'reject', 'clear') or not reason.strip():
        raise ValueError('Mapping decision requires an accept/reject/clear action and a nonempty audit reason')
    if decision == 'accept' and (not history_key or type(orientation) is not int or orientation not in (0, 1)):
        raise ValueError('Acceptance requires an explicit historical key and participant orientation 0 or 1')
    with store.transaction() as connection:
        if connection.execute('SELECT 1 FROM fixtures WHERE fixture_id=?', (fixture_id,)).fetchone() is None:
            raise ValueError('Provider fixture does not exist')
        previous = connection.execute('SELECT * FROM links WHERE fixture_id=?', (fixture_id,)).fetchone()
        if decision == 'accept':
            if connection.execute('SELECT 1 FROM historical_matches WHERE history_key=?', (history_key,)).fetchone() is None:
                raise ValueError('Historical match does not exist')
            occupied = connection.execute("SELECT fixture_id FROM links WHERE history_key=? AND status IN ('manual','automatic') AND fixture_id<>?", (history_key, fixture_id)).fetchone()
            if occupied:
                raise ValueError('Historical identity is already linked to another provider fixture')
        timestamp = now()
        connection.execute('INSERT INTO link_decisions(fixture_id,history_key,decision,orientation,reason,created_at,previous_json) VALUES(?,?,?,?,?,?,?)',
                           (fixture_id, history_key, decision, orientation, reason.strip(), timestamp, json.dumps(dict(previous) if previous else None)))
        if decision == 'clear':
            connection.execute('DELETE FROM links WHERE fixture_id=?', (fixture_id,))
        else:
            connection.execute('INSERT INTO links(fixture_id,history_key,status,orientation,method,reason,updated_at) VALUES(?,?,?,?,?,?,?) '
                'ON CONFLICT(fixture_id) DO UPDATE SET history_key=excluded.history_key,status=excluded.status,orientation=excluded.orientation,method=excluded.method,reason=excluded.reason,updated_at=excluded.updated_at',
                (fixture_id, history_key if decision == 'accept' else None, 'manual' if decision == 'accept' else 'rejected',
                 orientation if decision == 'accept' else None, 'operator_decision', reason.strip(), timestamp))
    if decision == 'clear':
        rebuild_mappings(store)
    return store.one('SELECT * FROM links WHERE fixture_id=?', (fixture_id,))


def latest_quotes(store, fixture_id=None, bookmaker=None):
    clauses, parameters = [], []
    if fixture_id:
        clauses.append('fixture_id=?')
        parameters.append(fixture_id)
    if bookmaker:
        clauses.append('bookmaker=?')
        parameters.append(bookmaker)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    return store.query('SELECT * FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY fixture_id,bookmaker ORDER BY observed_at DESC,id DESC) AS ranked FROM quotes'
                       + where + ') WHERE ranked=1', parameters)


def mapped_quote_rows(store):
    link_rows = store.query("SELECT l.*,h.team1,h.team2,h.score1,h.score2,h.winner_1,h.label_status,h.source,f.metadata_json AS fixture_metadata "
                            "FROM links l JOIN historical_matches h ON h.history_key=l.history_key JOIN fixtures f ON f.fixture_id=l.fixture_id "
                            "WHERE l.status IN ('automatic','manual') AND h.winner_1 IS NOT NULL AND f.status_id=2")
    links = {r['fixture_id']: r for r in link_rows}
    result = []
    for quote in latest_quotes(store):
        link = links.get(quote['fixture_id'])
        if link is None or quote['status'] != 'admitted' or quote['price1'] is None or quote['price2'] is None:
            continue
        fixture = decoded(link['fixture_metadata'])
        try:
            cutoff = provider.fixture_cutoff(fixture)
            if any(provider.utc(quote[f'source_time_{side}']) >= cutoff for side in (1, 2)):
                continue
            if quote['kind'] == 'current' and provider.utc(quote['observed_at']) >= cutoff:
                continue
        except (provider.Stop, ValueError, TypeError):
            continue
        one, two = quote['price1'], quote['price2']
        if not all(math.isfinite(x) and x > 1 for x in (one, two)):
            continue
        probability = (1 / one) / (1 / one + 1 / two)
        if not 0 < probability < 1:
            continue
        y = int(link['winner_1']) if link['orientation'] == 1 else 1 - int(link['winner_1'])
        details = decoded(quote['details_json']).get('quote') or {}
        result.append({**details, 'fixtureId': quote['fixture_id'], 'bookmaker': quote['bookmaker'],
                       'quote_id': quote['id'], 'raw_id': quote['raw_id'], 'observed_at': quote['observed_at'],
                       'source_created_at_1': quote['source_time_1'], 'source_created_at_2': quote['source_time_2'],
                       'price_1': one, 'price_2': two, 'no_vig_probability_1': probability,
                       'history_key': link['history_key'], 'history_source': link['source'],
                       'mapping_status': link['status'], 'mapping_method': link['method'],
                       'orientation': link['orientation'], 'outcome_1_won': y,
                       'log_loss': -math.log(probability) if y else -math.log1p(-probability),
                       'binding_status': 'provider_catalog_convention_only', 'execution_eligibility': 'not_certified'})
    return result


def compute_metrics(store):
    rows = mapped_quote_rows(store)
    metrics = mapper.bookmaker_metrics(rows, store.get_settings()['bookmakers'])
    metrics['source_composition'] = dict(Counter(r['history_source'] for r in rows))
    metrics['mapped_fixture_count'] = len({r['fixtureId'] for r in rows})
    metrics['scored_quote_pairs'] = len(rows)
    metrics['limits'].append('Manual decisions and native provider IDs are distinct from declared natural-key links.')
    return metrics


def fixture_item(store, row):
    result = {k: row[k] for k in ('fixture_id', 'start_time', 'true_start_time', 'status_id', 'participant1_id',
                                 'participant2_id', 'participant1_name', 'participant2_name', 'tournament', 'observed_at')}
    result['mapping'] = store.one('SELECT * FROM links WHERE fixture_id=?', (row['fixture_id'],))
    result['quotes'] = latest_quotes(store, row['fixture_id'])
    for quote in result['quotes']:
        quote.pop('details_json', None)
        quote.pop('ranked', None)
    result['native_result'] = store.one("SELECT score1,score2,winner_1,source_date,label_status FROM historical_matches WHERE history_key=?", ('oddspapi:' + row['fixture_id'],))
    return result


def list_fixtures(store, status='all', q='', bookmaker='', mapping='all', offset=0, limit=50):
    clauses, parameters = [], []
    if status == 'upcoming':
        clauses.append('f.status_id=0')
    elif status == 'finished':
        clauses.append('f.status_id=2')
    if q:
        clauses.append('(f.participant1_name LIKE ? OR f.participant2_name LIKE ? OR f.tournament LIKE ? OR f.fixture_id LIKE ?)')
        parameters.extend(['%' + q + '%'] * 4)
    if bookmaker:
        clauses.append('EXISTS(SELECT 1 FROM quotes q WHERE q.fixture_id=f.fixture_id AND q.bookmaker=?)')
        parameters.append(bookmaker)
    if mapping == 'mapped':
        clauses.append("EXISTS(SELECT 1 FROM links l WHERE l.fixture_id=f.fixture_id AND l.status IN ('manual','automatic'))")
    elif mapping == 'unmapped':
        clauses.append("NOT EXISTS(SELECT 1 FROM links l WHERE l.fixture_id=f.fixture_id AND l.status IN ('manual','automatic'))")
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    total = store.one('SELECT COUNT(*) AS n FROM fixtures f' + where, parameters)['n']
    found = store.query('SELECT f.* FROM fixtures f' + where + ' ORDER BY f.start_time DESC,f.fixture_id LIMIT ? OFFSET ?', parameters + [limit, offset])
    return {'total': total, 'items': [fixture_item(store, row) for row in found]}


def historical_list(store, q='', source='', offset=0, limit=50):
    clauses, parameters = [], []
    if source:
        clauses.append('source=?')
        parameters.append(source)
    if q:
        clauses.append('(team1 LIKE ? OR team2 LIKE ? OR tournament LIKE ? OR history_key LIKE ?)')
        parameters.extend(['%' + q + '%'] * 4)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    total = store.one('SELECT COUNT(*) AS n FROM historical_matches' + where, parameters)['n']
    found = store.query('SELECT * FROM historical_matches' + where + ' ORDER BY COALESCE(start_time,source_date) DESC,history_key LIMIT ? OFFSET ?', parameters + [limit, offset])
    for row in found:
        row['metadata'] = decoded(row.pop('metadata_json'))
    return {'total': total, 'items': found}


def states_list(store, fixture_id, bookmaker='', offset=0, limit=100):
    where, parameters = ' WHERE s.fixture_id=?', [fixture_id]
    if bookmaker:
        where += ' AND s.bookmaker=?'
        parameters.append(bookmaker)
    total = store.one('SELECT COUNT(*) AS n FROM odds_states s' + where, parameters)['n']
    rows = store.query('SELECT s.*,r.received_at AS observed_at FROM odds_states s JOIN raw_responses r ON r.id=s.raw_id' + where
                       + ' ORDER BY r.received_at DESC,s.source_time DESC,s.id DESC LIMIT ? OFFSET ?', parameters + [limit, offset])
    for row in rows:
        row['state'] = decoded(row.pop('state_json'))
    return {'total': total, 'items': rows}


def fixture_detail(store, identity):
    row = store.one('SELECT * FROM fixtures WHERE fixture_id=?', (identity,))
    if row is None:
        return None
    result = fixture_item(store, row)
    result['metadata'] = decoded(row['metadata_json'])
    result['odds_history'] = states_list(store, identity)
    candidates = store.query('SELECT c.*,h.team1,h.team2,h.tournament,h.score1,h.score2,h.source_date,h.label_status '
                             'FROM link_candidates c JOIN historical_matches h ON h.history_key=c.history_key WHERE c.fixture_id=? ORDER BY c.history_key', (identity,))
    for candidate in candidates:
        candidate['details'] = decoded(candidate.pop('details_json'))
    result['candidates'] = candidates
    result['mapping_decisions'] = store.query('SELECT * FROM link_decisions WHERE fixture_id=? ORDER BY id DESC', (identity,))
    result['raw_evidence'] = store.query('SELECT DISTINCT r.id,r.endpoint,r.http_status,r.received_at,r.decoded_sha256 '
        'FROM raw_responses r LEFT JOIN odds_states s ON s.raw_id=r.id LEFT JOIN quotes q ON q.raw_id=r.id '
        'WHERE s.fixture_id=? OR q.fixture_id=? OR json_extract(r.parameters_json,\'$.fixtureId\')=? ORDER BY r.received_at DESC LIMIT 100', (identity, identity, identity))
    return result


def quota_today(store):
    day = datetime.now(timezone.utc).date().isoformat()
    ledger = store.one("SELECT COUNT(*) AS n FROM raw_responses WHERE origin='network' AND billable=1 AND substr(started_at,1,10)=?", (day,))['n']
    reserved = decoded((store.one("SELECT value_json FROM settings WHERE key='provider_billable_days'") or {}).get('value_json')).get(day, 0)
    return max(ledger, reserved)


def status(store, credential_configured=False):
    counts = {table: store.one('SELECT COUNT(*) AS n FROM ' + table)['n'] for table in
              ('fixtures', 'historical_matches', 'odds_states', 'quotes')}
    counts['mapped_fixtures'] = store.one("SELECT COUNT(*) AS n FROM links WHERE status IN ('manual','automatic')")['n']
    settings = store.get_settings()
    public_settings = {k: v for k, v in settings.items() if k not in ('provider_account', 'provider_billable_days')}
    account_row = store.one("SELECT value_json FROM settings WHERE key='provider_account'")
    return {'counts': counts, 'credential_configured': credential_configured, 'settings': public_settings,
            'provider_account': decoded(account_row['value_json']) if account_row else None,
            'recent_jobs': store.list_jobs(10), 'books': settings['bookmakers'],
            'quota': {'billable_today': quota_today(store), 'billable_per_day': settings['billable_per_day']}}


def _csv_chunks(rows, fields):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
        if output.tell() >= 65536:
            yield output.getvalue()
            output.seek(0)
            output.truncate()
    if output.tell():
        yield output.getvalue()


def iter_export(store, dataset):
    """Stream one stable SQLite read snapshot without loading millions of states."""
    table = {'fixtures': 'fixtures', 'history': 'historical_matches', 'quotes': 'quotes',
             'states': 'odds_states', 'mappings': 'links', 'jobs': 'jobs'}.get(dataset)
    if dataset == 'mapped_quotes':
        rows = mapped_quote_rows(store)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        if not fields:
            fields = ['fixtureId', 'bookmaker', 'history_key', 'outcome_1_won', 'log_loss']
        yield from _csv_chunks(rows, fields)
    elif table:
        with store.connect() as connection:
            cursor = connection.execute('SELECT * FROM ' + table + ' ORDER BY rowid')
            yield from _csv_chunks(cursor, [column[0] for column in cursor.description])
    else:
        raise ValueError('Unknown export dataset')
