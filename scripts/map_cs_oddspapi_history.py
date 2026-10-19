"""Offline provider-fixture linkage to historical series, separate from model intake.

OpenCS2 links require mutually unique participant names, compatible event metadata,
and an aware source timestamp 0–6 hours after the scheduled provider start. This
is a declared natural-key linkage, not independently verified entity binding.
StrandedPond's naive calendar dates and missing series format support candidates
only, unless an accepted OpenCS2 link anchors the same HLTV match ID.

Native HLTV archive links instead use calendar-day evidence (within one UTC
scheduled day), never the archive's synthetic noon as a kickoff. Every plausible
name/day edge counts before event checks; duplicate/conflicting source evidence
is retained separately from the mutually unique identity decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


MAX_DELTA_SECONDS = 6 * 3600
BOOKS = ['pinnacle', 'bet365', 'unibet']
DECORATIONS = {'team', 'esports', 'e-sports', 'gaming', 'clan'}
EVENT_GENERIC = {'league', 'series', 'season', 'cup', 'closed', 'open', 'qualifier',
                 'qualifiers', 'championship', 'tournament', 'counter', 'strike', 'esports'}
HLTV_URL = re.compile(r'https://www\.hltv\.org/matches/(\d+)(?:/.*)?\Z')
LINK_FIELDS = ['fixtureId', 'mapping_status', 'mapping_reason', 'historical_source',
               'historical_match_id', 'historical_url', 'participant1Name', 'participant2Name',
               'historical_team1', 'historical_team2', 'participant1_is_historical_team1',
               'scheduled_start', 'historical_source_time', 'time_delta_seconds',
               'odds_tournament', 'historical_event', 'historical_score1', 'historical_score2',
               'label_status', 'outcome_1_won', 'matching_method']
HLTV_LINK_FIELDS = [
    'fixtureId', 'mapping_status', 'mapping_reason', 'hltv_match_id', 'hltv_match_url',
    'participant1Name', 'participant2Name', 'participant1ShortName', 'participant2ShortName',
    'participant1_is_hltv_team1', 'scheduled_start', 'true_start', 'provider_status_id',
    'provider_sport_id', 'odds_tournament', 'hltv_team1', 'hltv_team2', 'hltv_event',
    'hltv_calendar_day', 'calendar_day_offset', 'hltv_format', 'hltv_score1', 'hltv_score2',
    'hltv_winner', 'hltv_winner_1', 'outcome_1_won', 'label_status', 'matching_method',
    'candidateIDs', 'candidate_count', 'event_check', 'metadata_conflict',
    'source_row_count', 'source_row_index', 'source_date', 'candidate_edge',
    'historical_source', 'provenance', 'certification_limit',
]
HLTV_NATIVE_ID = re.compile(r'[1-9][0-9]*\Z')
HLTV_NATIVE_URL = re.compile(r'https://www\.hltv\.org/matches/([1-9][0-9]*)(?:/[^?#\s]*)?\Z')
HLTV_RAW_FIELDS = (
    'match_id', 'date', 'team1', 'team2', 'tournament', 'score1', 'score2',
    'winner', 'format', 'match_url',
)
HLTV_MATCHING_METHOD = 'mutually_unique_decorated_names_calendar_day_plus_minus_1_compatible_event'
HLTV_CERTIFICATION_LIMIT = (
    'Natural-key identity linkage only; archive date is synthetic-noon calendar evidence, '
    'not a kickoff or proof of pre-match lineup or executable quote availability.'
)


def aware(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.utcoffset() is None:
        raise ValueError('Timezone-unspecified clock cannot support an accepted linkage')
    return parsed.astimezone(timezone.utc)


def normalize_team(value):
    words = unicodedata.normalize('NFKC', value).casefold().split()
    if words and words[0] == 'team':
        words = words[1:]
    if words and words[-1] in DECORATIONS:
        words = words[:-1]
    return ' '.join(words)


def variants(fixture, side):
    return {normalize_team(fixture[key]) for key in
            (f'participant{side}Name', f'participant{side}ShortName') if fixture.get(key)} - {''}


def pair_keys(fixture):
    return {tuple(sorted((one, two))) for one in variants(fixture, 1)
            for two in variants(fixture, 2) if one != two}


def event_tokens(value):
    words = re.findall(r'[a-z0-9]+', unicodedata.normalize('NFKC', value).casefold())
    expanded = []
    for word in words:
        expanded.extend(['intel', 'extreme', 'masters'] if word == 'iem' else
                        ['europe'] if word == 'european' else [word])
    return {w for w in expanded if not w.isdigit() and w not in EVENT_GENERIC}, expanded


def event_check(one, two):
    a, aw = event_tokens(one)
    b, bw = event_tokens(two)
    def regions(words):
        result = set(words) & {'europe', 'oceania', 'asia', 'africa'}
        text = ' '.join(words)
        for name in ('north america', 'south america'):
            if name in text:
                result.add(name)
        return result
    ar, br = regions(aw), regions(bw)
    if ar and br and ar.isdisjoint(br):
        return 'event_conflict'
    if a and b and (a.issubset(b) or b.issubset(a)):
        return 'compatible'
    return 'event_unconfirmed'


def historical_series(rows):
    grouped = defaultdict(list)
    fields = ['team1', 'team2', 'event', 'match_date', 'format', 'score1', 'score2', 'winner', 'match_url']
    for row in rows:
        if not row['match_id'].isdigit():
            raise ValueError('Historical match ID must be an HLTV integer identifier')
        match = HLTV_URL.fullmatch(row['match_url'])
        if not match or match[1] != row['match_id']:
            raise ValueError('Historical match ID and URL disagree')
        aware(row['match_date'])
        grouped[row['match_id']].append(row)
    result = []
    for identity, group in sorted(grouped.items()):
        source = group[0]
        conflict = any(len({r[f] for r in group}) != 1 for f in fields)
        # Explicit top-level series results: never infer them from map row counts.
        try:
            s1, s2 = int(source['score1']), int(source['score2'])
            valid = (source['format'] == 'bo3' and max(s1, s2) == 2 and min(s1, s2) in (0, 1)
                     and source['winner'] == ('team1' if s1 > s2 else 'team2'))
        except ValueError:
            valid = False
        result.append({'historical_source': 'opencs2', 'historical_match_id': identity,
                       'historical_url': source['match_url'], 'historical_team1': source['team1'],
                       'historical_team2': source['team2'], 'historical_event': source['event'],
                       'historical_source_time': source['match_date'],
                       'historical_score1': source['score1'], 'historical_score2': source['score2'],
                       'metadata_conflict': conflict,
                       'label_status': 'valid_explicit_bo3_result' if valid and not conflict else 'invalid_series_result'})
    return result


def _hltv_source_evidence(source, index):
    """Retain a raw row's identity/result evidence without inventing a start."""
    identity = str(source['match_id'])
    url = source['match_url']
    match = HLTV_NATIVE_URL.fullmatch(url)
    if not HLTV_NATIVE_ID.fullmatch(identity) or not match or match[1] != identity:
        raise ValueError('Native HLTV match ID must be canonical ASCII positive and agree with its HLTV URL')
    day = datetime.fromisoformat(source['date']).date().isoformat()
    valid = False
    try:
        one, two = int(source['score1']), int(source['score2'])
        needed = {'bo3': 2, 'bo5': 3}.get(source['format'])
        valid = (needed is not None and max(one, two) == needed
                 and 0 <= min(one, two) < needed
                 and source['winner'] == ('team1' if one > two else 'team2'))
    except (TypeError, ValueError):
        pass
    return {
        'hltv_match_id': identity, 'hltv_match_url': url,
        'hltv_team1': source['team1'], 'hltv_team2': source['team2'],
        'hltv_event': source['tournament'], 'hltv_calendar_day': day,
        'hltv_format': source['format'], 'hltv_score1': source['score1'],
        'hltv_score2': source['score2'], 'hltv_winner': source['winner'],
        'hltv_winner_1': int(source['winner'] == 'team1') if valid else '',
        'label_status': 'valid_explicit_series_result' if valid else 'invalid_series_result',
        'source_row_index': index, 'source_date': source['date'],
        'historical_source': 'hltv',
        'provenance': 'local_raw_hltv_archive',
        'certification_limit': HLTV_CERTIFICATION_LIMIT,
    }


def hltv_calendar_series(raw_rows):
    """Group native IDs while preserving every duplicate and conflicting row.

    Source timestamps contain a synthetic noon. Only their calendar component
    can constrain linkage. Conflicts are never resolved by selecting a row.
    """
    grouped = defaultdict(list)
    signatures = defaultdict(set)
    for index, source in enumerate(raw_rows, 1):
        evidence = _hltv_source_evidence(source, index)
        identity = evidence['hltv_match_id']
        grouped[identity].append(evidence)
        signatures[identity].add(tuple(source.get(field, '') for field in HLTV_RAW_FIELDS))
    series = []
    for identity, evidence in sorted(grouped.items()):
        conflict = len(signatures[identity]) != 1
        series.append({
            **evidence[0], 'source_rows': evidence,
            'source_row_count': len(evidence), 'metadata_conflict': conflict,
            'label_status': 'hltv_metadata_conflict' if conflict else evidence[0]['label_status'],
            'hltv_winner_1': '' if conflict else evidence[0]['hltv_winner_1'],
        })
    return series


def _hltv_fixture_evidence(fixture):
    return {
        'fixtureId': fixture['fixtureId'],
        **{key: fixture.get(key, '') for key in (
            'participant1Name', 'participant2Name', 'participant1ShortName', 'participant2ShortName')},
        'scheduled_start': fixture.get('startTime', ''),
        'true_start': fixture.get('trueStartTime', ''),
        'provider_status_id': fixture.get('statusId', ''),
        'provider_sport_id': fixture.get('sportId', ''),
        'odds_tournament': fixture.get('tournamentName', ''),
        'historical_source': 'hltv', 'provenance': 'local_raw_hltv_archive',
        'certification_limit': HLTV_CERTIFICATION_LIMIT,
        'matching_method': HLTV_MATCHING_METHOD,
    }


def _hltv_edge(fixture, source, start_day):
    h1, h2 = (normalize_team(source[f'hltv_team{side}']) for side in (1, 2))
    forward = h1 in variants(fixture, 1) and h2 in variants(fixture, 2)
    backward = h2 in variants(fixture, 1) and h1 in variants(fixture, 2)
    offset = (datetime.fromisoformat(source['hltv_calendar_day']).date() - start_day).days
    return {
        'candidate_edge': forward != backward and abs(offset) <= 1,
        'participant1_is_hltv_team1': forward if forward != backward else '',
        'calendar_day_offset': offset,
        'event_check': event_check(fixture.get('tournamentName', ''), source['hltv_event']),
    }


def map_hltv_fixtures(fixtures, series):
    """Return the full fixture crosswalk and all raw candidate evidence.

    Uniqueness counts native identities, not duplicate archive rows. Conflicting
    rows still contribute every plausible edge, including incompatible events.
    """
    fixtures = list(fixtures)
    indexed = defaultdict(list)
    for match in series:
        for source in match['source_rows']:
            pair = tuple(sorted(normalize_team(source[f'hltv_team{side}']) for side in (1, 2)))
            if all(pair) and pair[0] != pair[1]:
                indexed[(pair, source['hltv_calendar_day'])].append((match, source))
    proposals = defaultdict(dict)
    reverse = defaultdict(set)
    fixture_errors = {}
    identities = set()
    for fixture in fixtures:
        identity = fixture['fixtureId']
        if identity in identities:
            raise ValueError('Provider fixture IDs must be unique')
        identities.add(identity)
        if fixture.get('sportId') != 17:
            fixture_errors[identity] = 'unsupported_sport'
            continue
        try:
            start_day = aware(fixture['startTime']).date()
        except (KeyError, TypeError, ValueError, AttributeError):
            fixture_errors[identity] = 'invalid_scheduled_start'
            continue
        for pair in sorted(pair_keys(fixture)):
            for offset in (-1, 0, 1):
                day = (start_day + timedelta(days=offset)).isoformat()
                for match, source in indexed.get((pair, day), []):
                    edge = _hltv_edge(fixture, source, start_day)
                    if not edge['candidate_edge']:
                        continue
                    hid = match['hltv_match_id']
                    if hid not in proposals[identity]:
                        proposals[identity][hid] = (match, source, edge, start_day)
                    reverse[hid].add(identity)
    links, candidates = [], []
    for fixture in fixtures:
        identity = fixture['fixtureId']
        options = proposals[identity]
        row = {field: '' for field in HLTV_LINK_FIELDS}
        row.update(_hltv_fixture_evidence(fixture))
        row.update(candidateIDs=';'.join(sorted(options)), candidate_count=len(options),
                   label_status='not_linked')
        if fixture.get('statusId') != 2:
            status, reason = 'not_finished', 'provider_fixture_not_finished'
        elif identity in fixture_errors:
            status = reason = fixture_errors[identity]
        elif not options:
            status, reason = 'unmatched', 'no_unambiguous_name_pair_within_calendar_day_window'
        elif len(options) > 1:
            status, reason = 'ambiguous_fixture', 'multiple_native_hltv_candidates'
        else:
            match, source, edge, _ = next(iter(options.values()))
            row.update({key: source[key] for key in HLTV_LINK_FIELDS if key in source})
            row.update(edge, source_row_count=match['source_row_count'],
                       metadata_conflict=match['metadata_conflict'])
            hid = match['hltv_match_id']
            if len(reverse[hid]) > 1:
                status = 'ambiguous_hltv_match'
            elif match['metadata_conflict']:
                status = 'hltv_metadata_conflict'
            elif edge['event_check'] != 'compatible':
                status = edge['event_check']
            else:
                status = 'matched'
            reason = status
            if status == 'matched' and source['label_status'] == 'valid_explicit_series_result':
                row['outcome_1_won'] = int(bool(source['hltv_winner_1']) == edge['participant1_is_hltv_team1'])
        row.update(mapping_status=status, mapping_reason=reason)
        if status != 'matched':
            row.update(outcome_1_won='', hltv_winner_1='',
                       label_status='blocked_identity_link' if options else 'not_linked')
        links.append(row)
        # Include even non-edge rows of a conflicted candidate ID: a reviewer
        # must see the whole conflict, not just the row favoring this fixture.
        for hid in sorted(options):
            match, _, _, start_day = options[hid]
            for source in match['source_rows']:
                candidate = {field: '' for field in HLTV_LINK_FIELDS}
                candidate.update(_hltv_fixture_evidence(fixture))
                candidate.update(source)
                candidate.update(_hltv_edge(fixture, source, start_day))
                candidate.update(
                    mapping_status=status, mapping_reason=reason,
                    candidateIDs=row['candidateIDs'], candidate_count=len(options),
                    source_row_count=match['source_row_count'],
                    metadata_conflict=match['metadata_conflict'],
                    label_status='candidate_not_an_independent_binding', outcome_1_won='',
                )
                candidates.append(candidate)
    return links, candidates


def map_fixtures(fixtures, series):
    indexed = defaultdict(list)
    for match in series:
        pair = tuple(sorted(normalize_team(match[f'historical_team{side}']) for side in (1, 2)))
        if all(pair) and pair[0] != pair[1]:
            indexed[pair].append(match)
    proposals = defaultdict(dict)
    reverse = defaultdict(set)
    all_candidates = []
    times = [aware(m['historical_source_time']) for m in series]
    ids = set()
    for fixture in fixtures:
        identity = fixture['fixtureId']
        if identity in ids or fixture['sportId'] != 17 or fixture['statusId'] != 2:
            raise ValueError('Provider fixtures must have unique IDs and finished sport-17 scope')
        ids.add(identity)
        start = aware(fixture['startTime'])
        for pair in pair_keys(fixture):
            for match in indexed.get(pair, []):
                delta = (aware(match['historical_source_time']) - start).total_seconds()
                if not 0 <= delta <= MAX_DELTA_SECONDS:
                    continue
                hid = match['historical_match_id']
                h1, h2 = (normalize_team(match[f'historical_team{side}']) for side in (1, 2))
                forward = h1 in variants(fixture, 1) and h2 in variants(fixture, 2)
                backward = h2 in variants(fixture, 1) and h1 in variants(fixture, 2)
                if forward == backward:
                    continue
                if hid in proposals[identity]:
                    continue
                proposal = {**match, 'fixtureId': identity, 'time_delta_seconds': delta,
                            'participant1_is_historical_team1': forward,
                            'event_check': event_check(fixture.get('tournamentName', ''), match['historical_event'])}
                proposals[identity][hid] = proposal
                # All plausible name/time edges count: never select the nearest,
                # use outcomes to break ties, or hide another candidate's event conflict.
                reverse[hid].add(identity)
    links = []
    for fixture in fixtures:
        identity = fixture['fixtureId']
        options = list(proposals[identity].values())
        row = {field: '' for field in LINK_FIELDS}
        row.update(fixtureId=identity, participant1Name=fixture['participant1Name'],
                   participant2Name=fixture['participant2Name'], scheduled_start=fixture['startTime'],
                   odds_tournament=fixture.get('tournamentName', ''), label_status='not_linked')
        if not options:
            start = aware(fixture['startTime'])
            outside = times and (start > max(times) or start + timedelta(seconds=MAX_DELTA_SECONDS) < min(times))
            row.update(mapping_status='outside_historical_coverage' if outside else 'unmatched',
                       mapping_reason='no_name_pair_within_declared_aware_time_window')
        elif len(options) > 1:
            row.update(mapping_status='ambiguous_fixture', mapping_reason='multiple_historical_candidates')
        else:
            selected = options[0]
            row.update({k: selected[k] for k in LINK_FIELDS if k in selected})
            if len(reverse[selected['historical_match_id']]) > 1:
                status = 'ambiguous_historical_match'
            elif selected['metadata_conflict']:
                status = 'historical_metadata_conflict'
            elif selected['event_check'] != 'compatible':
                status = selected['event_check']
            else:
                status = 'matched'
            row.update(mapping_status=status, mapping_reason=status,
                       matching_method='mutually_unique_decorated_names_aware_0_to_6h_compatible_event')
            if status == 'matched' and selected['label_status'] == 'valid_explicit_bo3_result':
                historical_one_won = int(selected['historical_score1']) > int(selected['historical_score2'])
                row['outcome_1_won'] = int(historical_one_won == selected['participant1_is_historical_team1'])
            elif status != 'matched':
                row['label_status'] = 'blocked_identity_link'
        links.append(row)
        for proposal in options:
            all_candidates.append({**{k: proposal.get(k, '') for k in LINK_FIELDS},
                                   'mapping_status': row['mapping_status'], 'mapping_reason': row['mapping_reason'],
                                   'label_status': 'candidate_not_an_independent_binding', 'outcome_1_won': '',
                                   'odds_tournament': fixture.get('tournamentName', ''),
                                   'participant1Name': fixture['participant1Name'], 'participant2Name': fixture['participant2Name'],
                                   'scheduled_start': fixture['startTime']})
    return links, all_candidates


def stranded_candidates(rows, fixtures, links):
    grouped = defaultdict(list)
    for row in rows:
        match = HLTV_URL.fullmatch(row['match_url'])
        if not match:
            raise ValueError('StrandedPond row lacks a valid HLTV match URL')
        grouped[match[1]].append(row)
    indexed = defaultdict(dict)
    for hid, group in grouped.items():
        for row in group:
            pair = tuple(sorted(normalize_team(row[f'team_{side}']) for side in ('left', 'right')))
            if all(pair) and pair[0] != pair[1]:
                indexed[pair][hid] = group
    link_index = {r['fixtureId']: r for r in links}
    result = []
    for fixture in fixtures:
        seen = set()
        day = aware(fixture['startTime']).date()
        for pair in pair_keys(fixture):
            for hid, group in indexed.get(pair, {}).items():
                if hid in seen:
                    continue
                seen.add(hid)
                anchored = (link_index[fixture['fixtureId']]['mapping_status'] == 'matched'
                            and link_index[fixture['fixtureId']]['historical_match_id'] == hid)
                dates = sorted({r['date'] for r in group})
                if not anchored and not any(abs((datetime.fromisoformat(d).date() - day).days) <= 1 for d in dates):
                    continue
                source = group[0]
                result.append({'fixtureId': fixture['fixtureId'], 'historical_source': 'strandedpond',
                               'historical_match_id': hid, 'historical_url': source['match_url'],
                               'participant1Name': fixture['participant1Name'], 'participant2Name': fixture['participant2Name'],
                               'historical_team1': source['team_left'], 'historical_team2': source['team_right'],
                               'scheduled_start': fixture['startTime'], 'historical_source_time': ';'.join(dates),
                               'odds_tournament': fixture.get('tournamentName', ''), 'historical_event': source['event'],
                               'mapping_status': 'anchored_same_hltv_id' if anchored else 'candidate_only',
                               'mapping_reason': 'accepted_opencs2_hltv_anchor' if anchored else 'naive_dates_and_missing_series_format',
                               'label_status': 'map_results_not_used_as_series_labels', 'outcome_1_won': '',
                               'matching_method': 'same_hltv_id' if anchored else 'decorated_names_calendar_day_plus_or_minus_one'})
    return result, len(grouped)


def score_quotes(quotes, links):
    matched = {r['fixtureId']: r for r in links if r['mapping_status'] == 'matched' and r['outcome_1_won'] != ''}
    result, seen = [], set()
    for quote in quotes:
        key = quote['fixtureId'], quote['bookmaker']
        if key in seen:
            raise ValueError('Duplicate fixture/book quote')
        seen.add(key)
        if quote['fixtureId'] not in matched:
            continue
        one, two = float(quote['price_1']), float(quote['price_2'])
        if not all(math.isfinite(x) and x > 1 for x in (one, two)):
            raise ValueError('Selected quote has invalid decimal prices')
        q = (1 / one) / (1 / one + 1 / two)
        if not 0 < q < 1 or not math.isclose(q, float(quote['no_vig_probability_1']), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Selected quote probability does not match two-sided prices')
        link = matched[quote['fixtureId']]
        y = link['outcome_1_won']
        result.append({**quote, **{k: link[k] for k in LINK_FIELDS if k not in quote},
                       'log_loss': -math.log(q) if y else -math.log1p(-q)})
    return result


def bookmaker_metrics(rows, books):
    lookup = {b: {r['fixtureId']: r['log_loss'] for r in rows if r['bookmaker'] == b} for b in books}
    def cohort(names, identities):
        ordered = sorted(identities)
        return {'matches': len(ordered), 'fixture_ids': ordered,
                'log_loss': {b: math.fsum(lookup[b][i] for i in ordered) / len(ordered) if ordered else None for b in names}}
    available = {b: {'matches': len(lookup[b]), 'log_loss': math.fsum(lookup[b].values()) / len(lookup[b]) if lookup[b] else None} for b in books}
    pairs = {f'{a}__{b}': cohort([a, b], set(lookup[a]) & set(lookup[b])) for a, b in itertools.combinations(books, 2)}
    common = set.intersection(*(set(lookup[b]) for b in books)) if books else set()
    return {'available_cohorts': available, 'pairwise_common_cohorts': pairs,
            'common_all_requested_books': cohort(books, common),
            'convention': 'natural-log binary loss of proportional two-sided no-vig pre-start quotes',
            'uninformative_0_5_reference': math.log(2),
            'limits': ['Different available cohorts cannot rank bookmaker quality.',
                       'Natural-key entity linkage and provider outcome convention are not independently certified.',
                       'Coverage is limited to the historical result export, not the full odds cohort.']}


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n', extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odds-dir', type=Path, default=Path('data/raw/oddspapi/cs2_full_2026'))
    parser.add_argument('--opencs2', type=Path, default=Path('data/exports/opencs2/maps.csv'))
    parser.add_argument('--strandedpond', type=Path, default=Path('data/raw/strandedpond/matches.csv'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty; historical mappings never overwrite a previous audit')
    fixture_path = args.odds_dir / 'fixture_index.json'
    quote_path = args.odds_dir / 'selected_pre_match_quotes.csv'
    fixtures = json.loads(fixture_path.read_bytes())['fixtures']
    quotes = read_csv(quote_path)
    fixture_index = {f['fixtureId']: f for f in fixtures}
    for quote in quotes:
        fixture = fixture_index.get(quote['fixtureId'])
        if fixture is None or any(quote[f'participant{n}Name'] != fixture[f'participant{n}Name']
                                  or str(quote[f'participant{n}Id']) != str(fixture[f'participant{n}Id']) for n in (1, 2)):
            raise ValueError('Selected quote participant orientation disagrees with native fixture metadata')
    series = historical_series(read_csv(args.opencs2))
    links, candidates = map_fixtures(fixtures, series)
    legacy, legacy_count = stranded_candidates(read_csv(args.strandedpond), fixtures, links)
    candidates.extend(legacy)
    quote_counts = Counter(r['fixtureId'] for r in quotes)
    for link in links:
        link['usable_quote_pairs'] = quote_counts[link['fixtureId']]
    matched = [r for r in links if r['mapping_status'] == 'matched']
    scored = score_quotes(quotes, links)
    metrics = bookmaker_metrics(scored, BOOKS)
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'network_requests': 0,
               'fixture_count': len(fixtures), 'fixtures_with_usable_quotes': len(quote_counts),
               'opencs2_historical_series': len(series), 'strandedpond_historical_series': legacy_count,
               'mapping_status_counts': dict(Counter(r['mapping_status'] for r in links)),
               'accepted_mapped_fixtures': len(matched),
               'accepted_mapped_fixtures_with_quotes': sum(r['usable_quote_pairs'] > 0 for r in matched),
               'scored_fixture_count': len({r['fixtureId'] for r in scored}), 'scored_quote_pairs': len(scored),
               'strandedpond_candidate_status_counts': dict(Counter(r['mapping_status'] for r in legacy)),
               'bookmaker_metrics': metrics,
               'source_files': {str(p): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                                for p in [fixture_path, quote_path, args.opencs2, args.strandedpond]},
               'matching_policy': {'source_timestamp_after_scheduled_start_seconds': [0, MAX_DELTA_SECONDS],
                                   'normalization': 'Unicode NFKC, case/whitespace; leading Team and trailing esports/e-sports/gaming/clan only',
                                   'aliases': 'Provider full/short names only; no fuzzy aliases or removal of academy/junior/ex qualifiers',
                                   'event_check': 'Distinctive token containment; IEM expansion and European/Europe; explicit region conflicts block',
                                   'ambiguity': 'All name/time edges must be mutually unique before event checking; no closest-match tie breaking',
                                   'strandedpond': 'Naive calendar ±1-day candidates; no inferred BO format or series outcome'},
               'limits': ['Kaggle 2024–2025 series history has no overlap with 2026 odds.',
                          'Source match timestamps are not assumed scheduled start or original result-publication clocks.',
                          'Historical identity links use declared natural keys, not shared provider/HLTV native IDs.',
                          'Existing model inputs, source manifests and baseline artifacts are unchanged.']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'mapping_coverage.csv', LINK_FIELDS + ['usable_quote_pairs'], links)
    write_csv(args.output_dir / 'matched_fixtures.csv', LINK_FIELDS + ['usable_quote_pairs'], matched)
    write_csv(args.output_dir / 'candidate_links.csv', LINK_FIELDS, candidates)
    quote_fields = list(quotes[0]) if quotes else []
    write_csv(args.output_dir / 'matched_quotes.csv', quote_fields + [k for k in LINK_FIELDS if k not in quote_fields] + ['log_loss'], scored)
    write_json(args.output_dir / 'bookmaker_logloss.json', metrics)
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_files', 'bookmaker_metrics')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
