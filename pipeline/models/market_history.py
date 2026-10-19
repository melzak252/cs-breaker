"""Read-only reconstruction of cached historical quotes at explicit origins.

Receipts establish acquisition provenance only. No provider request, application
write, normalized odds-state scan, or historical execution claim is made here.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import zlib
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from pipeline.models.bookmaker_comparison import BOOKS, _quote_evidence, _validate_link
from scripts import collect_cs_oddspapi as provider
from scripts import map_cs_oddspapi_history as mapper

HORIZONS = ("one_hour", "six_hour", "source_day")
SELECTION_METHOD = "latest_requested_book_receipt_then_latest_side_state_strictly_before_origin_then_validity"
MARKET_MAPPING = {"market_id": "171", "outcome_1": "171", "outcome_2": "172"}


def _requested_books(value):
    if isinstance(value, str):
        books = [book.strip() for book in value.split(",")]
    elif isinstance(value, list):
        books = value
    else:
        raise ValueError("Historical receipt requires comma-string or list bookmakers")
    if (not books or any(not isinstance(book, str) or not provider.SAFE_BOOK.fullmatch(book) for book in books)
            or len(set(books)) != len(books)):
        raise ValueError("Invalid historical receipt requested bookmakers")
    return tuple(books)


def _load_payload(root, receipt):
    """Validate both immutable byte layers before interpreting any payload."""
    relative = receipt.get("raw_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("invalid_raw_path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("raw_path_outside_database_root")
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != receipt.get("file_sha256"):
        raise ValueError("compressed_hash_mismatch")
    body = gzip.decompress(compressed)
    if hashlib.sha256(body).hexdigest() != receipt.get("decoded_sha256"):
        raise ValueError("decoded_hash_mismatch")
    payload = json.loads(body)
    if receipt["http_status"] == 404:
        if not provider.no_history(404, payload):
            raise ValueError("unsupported_no_history_response")
        if "fixtureId" in payload and payload["fixtureId"] != receipt["fixture_id"]:
            raise ValueError("history_fixture_identity_mismatch")
    elif receipt["http_status"] == 200 and not receipt.get("error_code"):
        provider.validate_history(payload, receipt["fixture_id"])
    else:
        raise ValueError("invalid_terminal_status")
    return payload


def _latest_receipts(connection, fixture_ids):
    """Stream only receipt metadata; keep at most the latest group per fixture/book."""
    latest = {}
    count = 0
    for raw in connection.execute(
        "SELECT id,parameters_json,received_at,http_status,error_code,terminal,"
        "raw_path,file_sha256,decoded_sha256 FROM raw_responses WHERE endpoint='/historical-odds'"
    ):
        count += 1
        receipt = dict(raw)
        parameters = json.loads(receipt.pop("parameters_json"))
        if not isinstance(parameters, dict):
            raise ValueError("Historical receipt parameters must be an object")
        identity = parameters.get("fixtureId")
        if not isinstance(identity, str) or not provider.SAFE_ID.fullmatch(identity):
            raise ValueError("Historical receipt requires explicit valid fixture identity")
        requested = _requested_books(parameters.get("bookmakers"))
        if identity not in fixture_ids:
            continue
        receipt["fixture_id"] = identity
        receipt["requested_books"] = requested
        try:
            when = provider.utc(receipt["received_at"])
        except (ValueError, OverflowError):
            # An unorderable relevant receipt cannot safely be bypassed.
            when = None
        for book in BOOKS:
            if book not in requested:
                continue
            key = (identity, book)
            previous = latest.get(key)
            if previous is None or when is None or (previous[0] is not None and when > previous[0]):
                latest[key] = (when, [receipt])
            elif previous[0] == when:
                previous[1].append(receipt)
    return latest, count


def _receipt_choice(group):
    if group is None:
        return None, "missing_requested_book_receipt"
    when, receipts = group
    if when is None:
        return receipts[0], "unorderable_receipt_clock"
    # Equal receipts with differing response evidence or status are ambiguous,
    # including success versus absence; neither raw ID nor insertion order wins.
    fields = ("http_status", "error_code", "terminal", "raw_path", "file_sha256", "decoded_sha256")
    signatures = {tuple(row.get(field) for field in fields) for row in receipts}
    if len(signatures) != 1:
        return None, "ambiguous_latest_receipt"
    return min(receipts, key=lambda row: row["id"]), None


def _anchor_checks(links, anchors, native_results):
    matched = {row["fixtureId"]: row for row in links if row["mapping_status"] == "matched"}
    identity_checks = []
    for anchor in anchors:
        row = matched.get(anchor["fixture_id"])
        if row is not None:
            identity_checks.append({**anchor, "hltv_match_id": row["hltv_match_id"],
                                    "agrees": anchor["source_match_id"] == row["hltv_match_id"]})
    result_checks = []
    for native in native_results:
        row = matched.get(native["source_match_id"])
        if row is None or row["label_status"] != "valid_explicit_series_result":
            continue
        scores = (int(row["hltv_score1"]), int(row["hltv_score2"]))
        if not row["participant1_is_hltv_team1"]:
            scores = scores[::-1]
        result_checks.append({"fixtureId": row["fixtureId"], "hltv_match_id": row["hltv_match_id"],
                              "agrees": scores == (native["score1"], native["score2"])
                              and row["outcome_1_won"] == native["winner_1"]})
    if any(not row["agrees"] for row in identity_checks + result_checks):
        raise ValueError("Direct HLTV crosswalk contradicts an existing native-ID or native-result anchor")
    return identity_checks, result_checks


def _origin(fixture, link, horizon):
    if link.get("mapping_status") != "matched":
        return None, "no_audited_native_mapping"
    if link.get("label_status") != "valid_explicit_series_result":
        return None, "no_valid_explicit_series_label"
    event = provider.fixture_cutoff(fixture)
    opening = datetime.combine(datetime.fromisoformat(link["hltv_calendar_day"]).date(),
                               time(), timezone.utc)
    origin = opening if horizon == "source_day" else event - timedelta(
        hours=1 if horizon == "one_hour" else 6)
    if origin < opening:
        return origin, "forecast_origin_before_model_source_day_opening"
    if origin >= event:
        return origin, "forecast_origin_not_before_event"
    return origin, None


def load_market_history(database, hltv_results, *, horizons=HORIZONS):
    """Return fixtures, audited links, normalized per-horizon quotes and provenance.

    ``database`` is the SQLite file; immutable gzip paths resolve only beneath
    its parent. ``hltv_results`` is a raw native-results CSV path or iterable of
    the mapper's raw result dictionaries. Only the three fixed books are used.
    Every fixture/book has one observation per horizon, including exclusions.
    """
    horizons = tuple(horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(h not in HORIZONS for h in horizons):
        raise ValueError("Horizons must be a nonempty unique subset of one_hour, six_hour, source_day")
    database = Path(database).resolve()
    root = database.parent
    results = mapper.read_csv(Path(hltv_results)) if isinstance(hltv_results, (str, Path)) else list(hltv_results)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        fixtures = []
        for stored in connection.execute("SELECT fixture_id,metadata_json FROM fixtures ORDER BY start_time,fixture_id"):
            fixture = json.loads(stored["metadata_json"])
            if not isinstance(fixture, dict) or fixture.get("fixtureId") != stored["fixture_id"]:
                raise ValueError("Stored fixture metadata identity mismatch")
            fixtures.append(fixture)
        latest, receipt_count = _latest_receipts(connection, {f["fixtureId"] for f in fixtures})
        anchors = [dict(row) for row in connection.execute(
            "SELECT l.fixture_id,h.source,h.source_match_id FROM links l "
            "JOIN historical_matches h ON h.history_key=l.history_key "
            "WHERE l.status IN ('automatic','manual') AND h.source IN ('hltv','opencs2','kaggle')")]
        native_results = [dict(row) for row in connection.execute(
            "SELECT source_match_id,score1,score2,winner_1 FROM historical_matches "
            "WHERE source='oddspapi' AND winner_1 IS NOT NULL")]
    finally:
        connection.rollback()
        connection.close()
    links, candidates = mapper.map_hltv_fixtures(fixtures, mapper.hltv_calendar_series(results))
    identity_checks, result_checks = _anchor_checks(links, anchors, native_results)
    links_by_id = {row["fixtureId"]: row for row in links}
    for fixture in fixtures:
        link = links_by_id[fixture["fixtureId"]]
        if link["mapping_status"] == "matched" and link["label_status"] == "valid_explicit_series_result":
            _validate_link(link, fixture, None)
    quotes = {horizon: [] for horizon in horizons}
    reason_counts = {horizon: Counter() for horizon in horizons}
    evidence_receipts = {}
    for fixture in fixtures:
        identity = fixture["fixtureId"]
        link = links_by_id[identity]
        # At most three selected payloads in memory, never the odds_states table.
        payload_cache = {}
        for book in BOOKS:
            group = latest.get((identity, book))
            receipt, receipt_reason = _receipt_choice(group)
            payload = None
            for candidate in group[1] if group else ():
                evidence_receipts[candidate["id"]] = {key: candidate.get(key) for key in (
                    "id", "fixture_id", "received_at", "raw_path", "file_sha256", "decoded_sha256",
                    "http_status", "terminal", "requested_books")}
            if receipt is not None:
                raw_id = receipt["id"]
                if receipt_reason is None:
                    if not receipt["terminal"]:
                        receipt_reason = "latest_receipt_not_terminal"
                    else:
                        if raw_id not in payload_cache:
                            try:
                                payload_cache[raw_id] = (_load_payload(root, receipt), None)
                            except (OSError, EOFError, ValueError, TypeError, zlib.error, provider.Stop):
                                payload_cache[raw_id] = (None, "invalid_raw_evidence")
                        payload, receipt_reason = payload_cache[raw_id]
                        if receipt_reason is None and receipt["http_status"] == 404:
                            receipt_reason = "explicit_no_history"
            for horizon in horizons:
                try:
                    origin, origin_reason = _origin(fixture, link, horizon)
                except (provider.Stop, ValueError, TypeError, OverflowError):
                    origin, origin_reason = None, "invalid_start_or_source_day_evidence"
                raw_id = receipt["id"] if receipt is not None else None
                row = {"id": f"asof:{horizon}:{raw_id if raw_id is not None else 'none'}:{book}",
                       "fixture_id": identity, "bookmaker": book, "kind": "historical", "raw_id": raw_id,
                       "observed_at": receipt["received_at"] if receipt else None,
                       "source_time_1": None, "source_time_2": None, "price1": None, "price2": None,
                       "forecast_origin_utc": provider.iso(origin) if origin is not None else None,
                       **{key: receipt.get(key) if receipt else None for key in
                          ("raw_path", "file_sha256", "decoded_sha256")}}
                reason = receipt_reason or origin_reason
                coverage, pair = {"status": "excluded", "reason": reason or ""}, None
                if reason is None:
                    coverage, pair = provider.select_pair(fixture, book, payload, MARKET_MAPPING, cutoff=origin)
                    reason = coverage["reason"] if pair is None else None
                if pair is not None:
                    row.update(status="admitted", reason=coverage["reason"],
                               price1=pair["price_1"], price2=pair["price_2"],
                               source_time_1=pair["source_created_at_1"], source_time_2=pair["source_created_at_2"])
                else:
                    row.update(status=("no_history" if reason == "explicit_no_history" else
                                       coverage["status"] if coverage["status"] != "excluded" else "excluded"),
                               reason=reason)
                    if reason == "ambiguous_latest_receipt":
                        row["status"] = reason
                selection = {"method": SELECTION_METHOD, "horizon": horizon, "raw_id": raw_id,
                             "forecast_origin_utc": row["forecast_origin_utc"],
                             "derived_observation": True, "stored_original_quote": False,
                             "latest_receipt_ids": [r["id"] for r in group[1]] if group else [],
                             "historical_availability": "uncertified: receipt is acquisition, not price availability"}
                row["details_json"] = json.dumps({"quote": pair, "coverage": coverage,
                                                 "derived_selection": selection}, sort_keys=True)
                if pair is not None:
                    _, guard_reason = _quote_evidence(row, fixture)
                    if guard_reason is not None:
                        row.update(status="excluded", reason=guard_reason)
                reason_counts[horizon][row["reason"]] += 1
                quotes[horizon].append(row)
        payload_cache.clear()
    audit = {
        "database": str(database), "raw_root": str(root), "read_only": True,
        "network_requests": 0, "application_mutations": 0, "normalized_states_loaded": 0,
        "historical_receipts_scanned": receipt_count, "selected_fixture_book_receipts": len(latest),
        "fixtures": len(fixtures), "links": len(links), "candidate_rows": len(candidates),
        "horizons": list(horizons), "books": list(BOOKS), "selection_method": SELECTION_METHOD,
        "quote_reason_counts": {h: dict(sorted(counts.items())) for h, counts in reason_counts.items()},
        "native_id_anchor_checks": identity_checks, "native_provider_result_checks": result_checks,
        "selected_raw_receipts": [evidence_receipts[key] for key in sorted(evidence_receipts)],
        "limits": ["provider source clocks and source-day UTC boundaries are retrospective proxies",
                   "historical download receipts do not certify pre-event availability or execution"],
    }
    return {"fixtures": fixtures, "links": links, "quotes_by_horizon": quotes, "audit": audit}
