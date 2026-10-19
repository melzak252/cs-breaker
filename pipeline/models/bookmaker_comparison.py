"""Offline, held-out HLTV forecast comparisons on identical bookmaker cohorts.

This module consumes audited native-ID links and stored observations, never app
links, a fitted model, or network data. A historical receipt is acquisition
provenance, not the timestamp of the bookmaker price or proof it was executable.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date, datetime, time, timezone

from pipeline.models.metrics import evaluate_predictions, log_loss_score
from scripts import collect_cs_oddspapi as provider
from scripts.map_cs_oddspapi_history import HLTV_URL, normalize_team, variants


BOOKS = ("pinnacle", "bet365", "unibet")
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 42
PAIRED_FIELDS = [
    "hltv_match_id", "fixtureId", "bookmaker", "date", "source_date", "team1", "team2",
    "target", "p_model", "p_bookmaker", "price_team1", "price_team2", "quote_id", "raw_id",
    "devig_method", "devig_power_exponent", "quote_cutoff_policy",
    "source_time_1", "source_time_2", "observed_at", "quote_kind", "quote_status",
    "hltv_match_url", "hltv_calendar_day", "hltv_team1", "hltv_team2", "hltv_format",
    "hltv_score1", "hltv_score2", "hltv_winner_1", "outcome_1_won", "label_status",
    "mapping_status", "mapping_reason", "matching_method", "participant1Name",
    "participant2Name", "participant1_is_hltv_team1", "prediction_team1_is_hltv_team1",
    "provider_participant1_is_prediction_team1", "startTime", "trueStartTime", "cutoff",
    "model_nominal_cutoff_utc", "market_id", "outcome_1", "outcome_2", "player_id",
    "scope", "binding_status", "execution_eligibility", "source_clock_basis",
    "receipt_clock_basis", "historical_availability", "mapping_certification_limit",
    "tournamentName", "hltv_event", "calendar_day_offset", "source_provenance",
    "odds_tournament", "scheduled_start", "true_start", "candidateIDs",
    "source_row_index", "source_row_count", "lineup_source", "identity_namespace",
    "history_source_dates_before", "minimum_prior_matches", "team1_known_players",
    "team2_known_players", "model_selected",
    "forecast_origin_utc", "derived_selection_method", "derived_horizon",
    "raw_path", "file_sha256", "decoded_sha256",
]


def _native_id(value):
    identity = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", identity):
        raise ValueError(f"Invalid canonical native HLTV match ID: {value!r}")
    return identity


def _day(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError(f"{name} must be a source calendar day, not an invented clock")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"Invalid {name}: {value!r}") from None


def _binary(value, name):
    if value not in (0, 1, "0", "1"):
        raise ValueError(f"{name} must be binary")
    return int(value)


def _boolean(value, name):
    if type(value) is bool:
        return value
    if value in ("True", "true", "1", 1):
        return True
    if value in ("False", "false", "0", 0):
        return False
    raise ValueError(f"{name} must have an explicit boolean orientation")


def _probability(value, name):
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Invalid {name}") from None
    if isinstance(value, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"Invalid {name}")
    return probability


def _team_pair(one, two, name):
    if not isinstance(one, str) or not isinstance(two, str):
        raise ValueError(f"Missing {name} team names")
    pair = (normalize_team(one), normalize_team(two))
    if not all(pair) or pair[0] == pair[1]:
        raise ValueError(f"Invalid or ambiguous {name} team names")
    return pair


def _prediction_index(predictions, metadata):
    selected = metadata.get("selected_model")
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("Frozen model metadata requires selected_model")
    selection = metadata.get("selection")
    if selection is not None:
        if not isinstance(selection, dict) or selection.get("partition") != "validation":
            raise ValueError("The frozen model must be selected on validation, never TEST")
        if selection.get("fit_partition", "train") != "train":
            raise ValueError("The frozen model must be fitted on TRAIN only")
        if selection.get("metric", "log_loss") != "log_loss":
            raise ValueError("Unexpected model-selection metric")
        candidates = selection.get("candidate_log_loss")
        if candidates is not None:
            if not isinstance(candidates, dict) or selected not in candidates:
                raise ValueError("Selected model missing from validation selection evidence")
            try:
                losses = {name: float(value) for name, value in candidates.items()}
            except (TypeError, ValueError, OverflowError):
                raise ValueError("Invalid validation model-selection scores") from None
            if any(not math.isfinite(value) or value < 0 for value in losses.values()):
                raise ValueError("Invalid validation model-selection scores")
            tolerance = selection.get("tie_tolerance", 0)
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1e-12:
                raise ValueError("Selection tolerance must be a bounded numerical tie tolerance")
            if losses[selected] > min(losses.values()) + tolerance:
                raise ValueError("Selected model disagrees with frozen validation selection")
            if "selected_log_loss" in selection and not math.isclose(
                float(selection["selected_log_loss"]), losses[selected], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("Selected validation score disagrees with selection evidence")
    splits = metadata.get("splits")
    if not isinstance(splits, dict) or not all(isinstance(splits.get(k), dict)
                                              for k in ("train", "validation", "test")):
        raise ValueError("Frozen model metadata requires TRAIN/VALIDATION/TEST splits")
    train_end = _day(splits["train"].get("end_date"), "TRAIN end_date")
    validation_end = _day(splits["validation"].get("end_date"), "VALIDATION end_date")
    test = splits["test"]
    test_start = _day(test.get("start_date"), "TEST start_date")
    test_end = _day(test.get("end_date"), "TEST end_date")
    if not train_end < validation_end < test_start <= test_end:
        raise ValueError("Frozen split boundaries are not strictly chronological")
    days = test.get("dates")
    if not isinstance(days, (list, tuple)) or not days:
        raise ValueError("Frozen TEST source dates are required")
    test_days = {_day(value, "TEST date") for value in days}
    if len(test_days) != len(days) or any(not test_start <= day <= test_end for day in test_days):
        raise ValueError("Invalid frozen TEST date membership")
    result = {}
    for raw in predictions:
        row = dict(raw)
        identity = _native_id(row.get("hltv_match_id"))
        if identity in result:
            raise ValueError(f"Duplicate TEST prediction for native HLTV ID {identity}")
        source_day = _day(row.get("source_date", row.get("date")), "prediction source_date")
        if "date" in row and _day(row["date"], "prediction date") != source_day:
            raise ValueError("Prediction date disagrees with source calendar day")
        if source_day not in test_days or source_day <= validation_end:
            raise ValueError(f"Prediction {identity} is not in the frozen held-out TEST partition")
        if row.get("partition", "test").casefold() != "test":
            raise ValueError("Non-TEST prediction entered the held-out benchmark")
        if row.get("selected_model", selected) != selected:
            raise ValueError("Prediction model disagrees with frozen selected_model")
        target = _binary(row.get("target"), "prediction target")
        one = _probability(row.get("prob_team1_win"), "prob_team1_win")
        two = _probability(row.get("prob_team2_win"), "prob_team2_win")
        if not math.isclose(one + two, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Prediction sides must complement")
        _team_pair(row.get("team1"), row.get("team2"), "prediction")
        row.update(hltv_match_id=identity, source_date=source_day.isoformat(),
                   date=source_day.isoformat(), target=target, p_model=one)
        result[identity] = row
    if "n_series" in test and test["n_series"] != len(result):
        raise ValueError("Predictions do not contain the frozen TEST n_series")
    return result


def _validate_link(row, fixture, prediction):
    identity = _native_id(row.get("hltv_match_id"))
    url = row.get("hltv_match_url")
    match = HLTV_URL.fullmatch(url) if isinstance(url, str) else None
    if not match or match[1] != identity:
        raise ValueError(f"Audited native ID and HLTV URL disagree for {identity}")
    native_pair = _team_pair(row.get("hltv_team1"), row.get("hltv_team2"), "native HLTV")
    source_day = _day(row.get("hltv_calendar_day"), "HLTV source day")
    _team_pair(fixture.get("participant1Name"), fixture.get("participant2Name"), "provider")
    for side in (1, 2):
        short_name = fixture.get(f"participant{side}ShortName")
        if short_name is not None and not isinstance(short_name, str):
            raise ValueError("Provider short team name must be text")
    direct = native_pair[0] in variants(fixture, 1) and native_pair[1] in variants(fixture, 2)
    reverse = native_pair[1] in variants(fixture, 1) and native_pair[0] in variants(fixture, 2)
    if direct == reverse:
        raise ValueError(f"Provider/native team names are inconsistent or ambiguous for {identity}")
    provider_direct = _boolean(row.get("participant1_is_hltv_team1"), "provider/native orientation")
    if provider_direct != direct:
        raise ValueError("Audited provider orientation disagrees with native team names")
    fmt = str(row.get("hltv_format", "")).casefold()
    if fmt not in ("bo3", "bo5"):
        raise ValueError("Valid explicit series label requires genuine BO3 or BO5 format")
    scores = []
    for field in ("hltv_score1", "hltv_score2"):
        value = str(row.get(field))
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError("Explicit series result needs nonnegative integer scores")
        scores.append(int(value))
    required_wins = (int(fmt[-1]) + 1) // 2
    if max(scores) != required_wins or min(scores) >= required_wins:
        raise ValueError("Audited explicit series score is incompatible with format")
    winner = _binary(row.get("hltv_winner_1"), "native series winner")
    if winner != int(scores[0] > scores[1]):
        raise ValueError("Native explicit series winner disagrees with scores")
    provider_target = _binary(row.get("outcome_1_won"), "provider outcome")
    if provider_target != (winner if direct else 1 - winner):
        raise ValueError("Provider outcome disagrees with native explicit series result")
    prediction_direct = None
    if prediction is not None:
        prediction_pair = _team_pair(prediction["team1"], prediction["team2"], "prediction")
        if prediction_pair == native_pair:
            prediction_direct = True
        elif prediction_pair == native_pair[::-1]:
            prediction_direct = False
        else:
            raise ValueError(f"Prediction/native team identities disagree for {identity}")
        if prediction["source_date"] != source_day.isoformat():
            raise ValueError("Prediction source day disagrees with native archive day")
        if prediction["target"] != (winner if prediction_direct else 1 - winner):
            raise ValueError(f"Prediction target disagrees with native explicit result for {identity}")
    return provider_direct, prediction_direct


def _metrics(rows, probability_field):
    labels = [row["target"] for row in rows]
    metrics = evaluate_predictions(labels, [row[probability_field] for row in rows])
    if not rows:
        for key in ("brier", "log_loss", "roc_auc", "accuracy", "ece"):
            metrics[key] = None
    elif len(set(labels)) == 1:
        metrics["roc_auc"] = None
    return metrics


def _percentile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _paired_difference(rows, metric="log_loss", *, block_days=1, start_day=None):
    if metric not in ("log_loss", "brier"):
        raise ValueError("Unsupported paired scoring metric")
    if type(block_days) is not int or block_days < 1:
        raise ValueError("Bootstrap block length must be a positive integer")
    source_days = {_day(row["source_date"], "source day") for row in rows}
    anchor = _day(start_day, "block origin") if start_day is not None else min(source_days, default=None)
    if source_days and min(source_days) < anchor:
        raise ValueError("Paired cases precede the declared bootstrap origin")
    blocks, identities = defaultdict(list), set()
    for row in rows:
        identity = _native_id(row["hltv_match_id"])
        if identity in identities:
            raise ValueError("Paired uncertainty requires one row per native match")
        identities.add(identity)
        target = _binary(row["target"], "paired target")
        model = _probability(row["p_model"], "model probability")
        bookmaker = _probability(row["p_bookmaker"], "bookmaker probability")
        if metric == "log_loss":
            model_loss = log_loss_score([target], [model])
            book_loss = log_loss_score([target], [bookmaker])
        else:
            model_loss, book_loss = (model - target) ** 2, (bookmaker - target) ** 2
        block = (_day(row["source_date"], "source day") - anchor).days // block_days
        blocks[block].append(model_loss - book_loss)
    block_totals = [(sum(blocks[day]), len(blocks[day])) for day in sorted(blocks)]
    result = {"estimate": None, "confidence_interval_95": None, "upper_confidence_bound_95": None,
              "source_day_count": len(source_days), "block_count": len(block_totals),
              "block_days": block_days, "block_origin": anchor.isoformat() if anchor else None,
              "bootstrap_replicates": BOOTSTRAP_REPLICATES,
              "seed": BOOTSTRAP_SEED,
              "method": ("source-calendar-day-block percentile bootstrap" if block_days == 1
                         else f"fixed {block_days}-calendar-day-block percentile bootstrap"),
              "direction": f"model minus bookmaker {'log loss' if metric == 'log_loss' else 'Brier'}; negative favors model",
              "confidence_level": 0.95, "interval_crosses_zero": None,
              "statistical_significance_claim": False,
              "single_source_day_degenerate": len(source_days) == 1,
              "single_block_degenerate": len(block_totals) == 1}
    if not rows:
        return result
    result["estimate"] = sum(total for total, _ in block_totals) / len(rows)
    if len(block_totals) == 1:
        interval = [result["estimate"], result["estimate"]]
        upper = result["estimate"]
    else:
        rng = random.Random(BOOTSTRAP_SEED)
        draws = []
        for _ in range(BOOTSTRAP_REPLICATES):
            total = count = 0
            for _ in block_totals:
                block_total, block_count = block_totals[rng.randrange(len(block_totals))]
                total += block_total
                count += block_count
            draws.append(total / count)
        draws.sort()
        interval = [_percentile(draws, 0.025), _percentile(draws, 0.975)]
        upper = _percentile(draws, 0.95)
    result["confidence_interval_95"] = interval
    result["upper_confidence_bound_95"] = upper
    result["interval_crosses_zero"] = interval[0] <= 0 <= interval[1]
    return result


def _devig(prices, method):
    """Remove margin using prices alone, with a bounded monotone power solve."""
    inverse_one, inverse_two = (1 / price for price in prices)
    if method == "proportional":
        return inverse_one / (inverse_one + inverse_two), ""
    # For positive decimal-price logs, f(a) decreases from 1 at a=0
    # to at most 0 at log(2)/min(log(price)). This also bounds underround
    # roots below one without assuming quotes contain a positive margin.
    logs = [math.log(price) for price in prices]
    small_log, large_log = sorted(logs)
    low, high = 0.0, math.log(2) / small_log
    for _ in range(100):
        exponent = (low + high) / 2
        # expm1 preserves the residual when one implied probability is near 1.
        residual = math.expm1(-exponent * small_log) + math.exp(-exponent * large_log)
        if residual > 0:
            low = exponent
        else:
            high = exponent
    exponent = (low + high) / 2
    return math.exp(-exponent * logs[0]), exponent


def _scope_invalid(evidence):
    """Reuse provider scope gates, including any nested stored scope evidence."""
    if provider.unsupported_scope(evidence):
        return True
    if evidence.get("player_id", 0) not in (0, "0", None):
        return True
    for key in ("map", "mapId", "mapNumber", "map_id", "map_number"):
        if evidence.get(key) not in (None, "", 0, "0"):
            return True
    return any(_scope_invalid(value) for value in evidence.values() if isinstance(value, dict))


def _quote_evidence(quote, fixture):
    if quote.get("status") != "admitted":
        return None, "quote_not_admitted"
    try:
        details = quote.get("details_json")
        details = json.loads(details) if isinstance(details, str) else details
    except (ValueError, TypeError):
        return None, "invalid_quote_evidence"
    if not isinstance(details, dict) or not isinstance(details.get("quote"), dict):
        return None, "invalid_quote_evidence"
    evidence = details["quote"]
    if str(evidence.get("market_id")) != "171" or str(evidence.get("outcome_1")) != "171" or str(evidence.get("outcome_2")) != "172":
        return None, "unsupported_market_or_outcomes"
    if _scope_invalid(evidence) or provider.unsupported_scope(fixture):
        return None, "unsupported_series_scope"
    for key, expected in (("fixtureId", quote["fixture_id"]), ("bookmaker", quote["bookmaker"])):
        if key in evidence and evidence[key] != expected:
            raise ValueError(f"Stored quote {key} disagrees with observation identity")
    for side in (1, 2):
        key = f"participant{side}Name"
        if evidence.get(key) and (
            not isinstance(evidence[key], str)
            or normalize_team(evidence[key]) not in variants(fixture, side)
        ):
            raise ValueError("Stored quote participant name disagrees with provider orientation")
    prices = []
    for field in ("price1", "price2"):
        value = quote.get(field)
        try:
            price = float(value)
        except (ValueError, TypeError, OverflowError):
            return None, "invalid_decimal_price"
        if isinstance(value, bool) or not math.isfinite(price) or price <= 1:
            return None, "invalid_decimal_price"
        prices.append(price)
    try:
        cutoff = provider.fixture_cutoff(fixture)
        sources = [provider.utc(quote.get(f"source_time_{side}")) for side in (1, 2)]
        receipt = provider.utc(quote.get("observed_at"))
    except (provider.Stop, ValueError, TypeError, OverflowError):
        return None, "invalid_or_naive_quote_or_start_clock"
    if any(source >= cutoff for source in sources):
        return None, "source_price_at_or_after_cutoff"
    kind = quote.get("kind")
    if kind not in ("historical", "current"):
        return None, "unsupported_quote_kind"
    if kind == "current" and receipt >= cutoff:
        return None, "current_receipt_at_or_after_cutoff"
    return (evidence, prices, cutoff), None


def _available_consensus(paired_rows, books):
    """Aggregate validated observations, never treating books as independent matches."""
    groups = defaultdict(list)
    for row in paired_rows:
        groups[row["hltv_match_id"]].append(row)
    rows = []
    patterns = Counter()
    identity_fields = ("fixtureId", "source_date", "date", "team1", "team2", "target",
                       "p_model", "model_selected", "forecast_origin_utc")
    for identity in sorted(groups, key=int):
        observations = groups[identity]
        first = observations[0]
        if any(any(row[field] != first[field] for field in identity_fields)
               for row in observations[1:]):
            raise ValueError(f"Cross-book model, label, day or identity conflict for {identity}")
        by_book = {row["bookmaker"]: row for row in observations}
        if len(by_book) != len(observations):
            raise ValueError(f"Duplicate consensus book for {identity}")
        available = [book for book in books if book in by_book]
        pattern = "+".join(available)
        patterns[pattern] += 1
        rows.append({
            "hltv_match_id": identity, **{field: first[field] for field in identity_fields},
            "bookmakers": available, "book_count": len(available),
            "p_bookmaker": sum(by_book[book]["p_bookmaker"] for book in available) / len(available),
            "quote_ids": {book: by_book[book]["quote_id"] for book in available},
        })
    return {
        "matches": len(rows), "hltv_match_ids": [row["hltv_match_id"] for row in rows],
        "rows": rows, "coverage_pattern_counts": dict(sorted(patterns.items())),
        "model": _metrics(rows, "p_model"), "bookmaker": _metrics(rows, "p_bookmaker"),
        "paired_log_loss_difference": _paired_difference(rows),
        "paired_brier_difference": _paired_difference(rows, "brier"),
        "method": "equal-weight mean of available already-validated book no-vig probabilities; once per native match",
        "execution_eligibility": "not_an_executable_quote",
    }


def compare_hltv_bookmakers(fixtures, quotes, mapping_rows, predictions, model_metadata, *,
                           books=BOOKS, devig_method="proportional", quote_cutoff="pre_start"):
    """Return ``(report, paired_rows)`` for frozen native-ID TEST predictions.

    Corrupt identities, duplicate input keys, orientation and label conflicts are
    errors. Missing mappings/predictions and unusable prices/scope/clocks are
    reported exclusions, never fabricated forecasts. Each paired row is one
    match/book; all score comparisons use identical native-ID populations.
    """
    if devig_method not in ("proportional", "power"):
        raise ValueError(f"Unsupported de-vig method: {devig_method!r}")
    if quote_cutoff not in ("pre_start", "source_day", "forecast_origin"):
        raise ValueError(f"Unsupported quote cutoff policy: {quote_cutoff!r}")
    books = tuple(books)
    if not books or len(set(books)) != len(books) or any(not isinstance(book, str) or not book for book in books):
        raise ValueError("Bookmakers must be a nonempty unique sequence")
    model_metadata = dict(model_metadata)
    prediction_by_id = _prediction_index(predictions, model_metadata)
    fixture_by_id = {}
    for raw in fixtures:
        fixture = dict(raw)
        identity = fixture.get("fixtureId")
        if not isinstance(identity, str) or not identity or identity in fixture_by_id:
            raise ValueError("Missing or duplicate provider fixture ID")
        fixture_by_id[identity] = fixture
    mappings = {}
    mapping_statuses = Counter()
    mapping_labels = Counter()
    native_to_fixture = {}
    mapped_predictions = {}
    for raw in mapping_rows:
        row = dict(raw)
        identity = row.get("fixtureId")
        if not isinstance(identity, str) or not identity or identity in mappings:
            raise ValueError("Missing or duplicate mapping fixture ID")
        mappings[identity] = row
        mapping_statuses[row.get("mapping_status", "missing")] += 1
        mapping_labels[row.get("label_status") or "missing"] += 1
        if row.get("mapping_status") != "matched":
            continue
        native_id = _native_id(row.get("hltv_match_id"))
        if native_id in native_to_fixture:
            raise ValueError(f"Duplicate provider fixtures mapped to native HLTV ID {native_id}")
        native_to_fixture[native_id] = identity
        if row.get("label_status") != "valid_explicit_series_result":
            continue
        if identity not in fixture_by_id:
            raise ValueError("Matched mapping has no provider fixture metadata")
        prediction = prediction_by_id.get(native_id)
        orientation = _validate_link(row, fixture_by_id[identity], prediction)
        if prediction is not None:
            mapped_predictions[native_id] = orientation
    forecast_origins = {}
    if quote_cutoff == "forecast_origin":
        for native_id, prediction in prediction_by_id.items():
            try:
                origin = provider.utc(prediction.get("forecast_origin_utc"))
            except (ValueError, TypeError, OverflowError):
                raise ValueError("forecast_origin requires explicit aware forecast_origin_utc on every prediction") from None
            opening = datetime.combine(_day(prediction["source_date"], "source day"), time(), timezone.utc)
            if origin < opening:
                raise ValueError("Forecast origin is before model source-day opening")
            fixture_id = native_to_fixture.get(native_id)
            fixture = fixture_by_id.get(fixture_id)
            event = None
            if fixture is not None:
                try:
                    event = provider.fixture_cutoff(fixture)
                except provider.Stop:
                    pass  # Invalid fixture clocks remain quote exclusions, as in existing policies.
            if event is not None and origin >= event:
                raise ValueError("Forecast origin must be strictly before provider event start")
            forecast_origins[native_id] = origin
    exclusions = Counter()
    excluded_rows = []
    paired_rows = []
    seen_quotes = set()
    stored_quote_count = 0
    selected_quote_count = 0
    for raw in quotes:
        quote = dict(raw)
        stored_quote_count += 1
        identity, book = quote.get("fixture_id"), quote.get("bookmaker")
        if not isinstance(identity, str) or not identity or not isinstance(book, str) or not book:
            raise ValueError("Quote requires fixture and bookmaker identities")
        key = (identity, book)
        if key in seen_quotes:
            raise ValueError(f"Duplicate latest fixture/book quote: {key}")
        seen_quotes.add(key)
        reason = None
        row = mappings.get(identity)
        if book not in books:
            reason = "bookmaker_not_requested"
        else:
            selected_quote_count += 1
        if reason is None and identity not in fixture_by_id:
            reason = "missing_provider_fixture"
        if reason is None and (row is None or row.get("mapping_status") != "matched"):
            reason = "no_audited_native_mapping"
        if reason is None and row.get("label_status") != "valid_explicit_series_result":
            reason = "no_valid_explicit_series_label"
        native_id = row.get("hltv_match_id") if row is not None else None
        prediction = prediction_by_id.get(str(native_id))
        if reason is None and prediction is None:
            reason = "mapped_match_without_frozen_test_prediction"
        accepted = None
        if reason is None:
            fixture = fixture_by_id[identity]
            if fixture.get("statusId") not in (2, "2"):
                reason = "provider_fixture_not_finished"
            else:
                accepted, reason = _quote_evidence(quote, fixture)
        if reason is None:
            evidence, prices, cutoff = accepted
            nominal_cutoff = datetime.combine(_day(prediction["source_date"], "source day"), time(), timezone.utc)
            if cutoff < nominal_cutoff:
                reason = "provider_start_before_nominal_source_day_cutoff"
            elif quote_cutoff == "source_day":
                if any(provider.utc(quote[f"source_time_{side}"]) >= nominal_cutoff for side in (1, 2)):
                    reason = "source_price_at_or_after_nominal_source_day_cutoff"
                elif quote["kind"] == "current" and provider.utc(quote["observed_at"]) >= nominal_cutoff:
                    reason = "current_receipt_at_or_after_nominal_source_day_cutoff"
            elif quote_cutoff == "forecast_origin":
                origin = forecast_origins[str(native_id)]
                if any(provider.utc(quote[f"source_time_{side}"]) >= origin for side in (1, 2)):
                    reason = "source_price_at_or_after_forecast_origin"
                elif quote["kind"] == "current" and provider.utc(quote["observed_at"]) >= origin:
                    reason = "current_receipt_at_or_after_forecast_origin"
                elif quote.get("forecast_origin_utc") is not None:
                    try:
                        if provider.utc(quote["forecast_origin_utc"]) != origin:
                            reason = "derived_quote_forecast_origin_mismatch"
                    except (ValueError, TypeError, OverflowError):
                        reason = "invalid_derived_quote_forecast_origin"
        if reason is not None:
            exclusions[reason] += 1
            excluded_rows.append({"fixtureId": identity, "bookmaker": book, "hltv_match_id": native_id,
                                  "quote_id": quote.get("id"), "raw_id": quote.get("raw_id"), "reason": reason})
            continue
        provider_direct, prediction_direct = mapped_predictions[str(native_id)]
        aligned = provider_direct == prediction_direct
        price_one, price_two = prices if aligned else prices[::-1]
        probability, exponent = _devig((price_one, price_two), devig_method)
        if not math.isfinite(probability) or not 0 < probability < 1:
            exclusions["invalid_no_vig_probability"] += 1
            excluded_rows.append({"fixtureId": identity, "bookmaker": book, "hltv_match_id": native_id,
                                  "quote_id": quote.get("id"), "raw_id": quote.get("raw_id"),
                                  "reason": "invalid_no_vig_probability"})
            continue
        paired = {field: "" for field in PAIRED_FIELDS}
        paired.update({field: row[field] for field in PAIRED_FIELDS if field in row})
        paired.update({field: fixture[field] for field in ("participant1Name", "participant2Name",
                      "startTime", "trueStartTime", "tournamentName") if field in fixture})
        paired.update({field: prediction[field] for field in (
            "lineup_source", "identity_namespace", "history_source_dates_before",
            "minimum_prior_matches", "team1_known_players", "team2_known_players"
        ) if field in prediction})
        paired["model_selected"] = model_metadata["selected_model"]
        paired.update(hltv_match_id=str(native_id), fixtureId=identity, bookmaker=book,
                      date=prediction["date"], source_date=prediction["source_date"],
                      team1=prediction["team1"], team2=prediction["team2"], target=prediction["target"],
                      p_model=prediction["p_model"], p_bookmaker=probability,
                      price_team1=price_one, price_team2=price_two, quote_id=quote.get("id"),
                      devig_method=devig_method, devig_power_exponent=exponent,
                      quote_cutoff_policy=quote_cutoff,
                      raw_id=quote.get("raw_id"), source_time_1=quote["source_time_1"],
                      source_time_2=quote["source_time_2"], observed_at=quote["observed_at"],
                      quote_kind=quote["kind"], quote_status=quote["status"],
                      participant1_is_hltv_team1=provider_direct,
                      prediction_team1_is_hltv_team1=prediction_direct,
                      provider_participant1_is_prediction_team1=aligned,
                      cutoff=provider.iso(cutoff), model_nominal_cutoff_utc=provider.iso(nominal_cutoff),
                      market_id="171", outcome_1="171", outcome_2="172", player_id="0",
                      scope="series Winner (provider catalog convention)",
                      binding_status="provider_catalog_convention_only", execution_eligibility="not_certified",
                      source_clock_basis="provider price side clocks; source_time_1/2 retain provider orientation",
                      receipt_clock_basis="stored observation acquisition time, not historical price time",
                      historical_availability="uncertified: historical retrieval does not prove pre-event availability",
                      mapping_certification_limit=row.get("certification_limit", row.get("provenance_limit",
                          "audited calendar/name/event natural keys, not independent entity certification")))
        details = quote.get("details_json")
        details = json.loads(details) if isinstance(details, str) else details
        selection = details.get("derived_selection", {})
        paired.update(
            forecast_origin_utc=(provider.iso(forecast_origins[str(native_id)])
                                 if quote_cutoff == "forecast_origin" else quote.get("forecast_origin_utc", "")),
            derived_selection_method=selection.get("method", ""),
            derived_horizon=selection.get("horizon", ""),
            raw_path=quote.get("raw_path", ""), file_sha256=quote.get("file_sha256", ""),
            decoded_sha256=quote.get("decoded_sha256", ""),
        )
        paired_rows.append(paired)
    paired_rows.sort(key=lambda row: (int(row["hltv_match_id"]), row["bookmaker"]))
    by_book = {book: {row["hltv_match_id"]: row for row in paired_rows if row["bookmaker"] == book}
               for book in books}
    common_ids = sorted(set.intersection(*(set(by_book[book]) for book in books)), key=int)
    common_model_rows = [by_book[books[0]][identity] for identity in common_ids]
    per_book = {}
    for book in books:
        ids = sorted(by_book[book], key=int)
        rows = [by_book[book][identity] for identity in ids]
        per_book[book] = {"matches": len(rows), "hltv_match_ids": ids, "model": _metrics(rows, "p_model"),
                          "bookmaker": _metrics(rows, "p_bookmaker"),
                          "paired_log_loss_difference": _paired_difference(rows),
                          "paired_brier_difference": _paired_difference(rows, "brier")}
    matched_any = set().union(*(set(rows) for rows in by_book.values()))
    mapped_without_prediction = sorted(set(native_to_fixture) - set(prediction_by_id), key=int)
    report = {
        "model_selected": model_metadata["selected_model"], "model_test_matches": len(prediction_by_id),
        "matched_test_matches_any_book": len(matched_any), "quote_pairs": len(paired_rows),
        "common_all_books": {"matches": len(common_ids), "hltv_match_ids": common_ids,
            "model": _metrics(common_model_rows, "p_model"),
            "bookmakers": {book: _metrics([by_book[book][identity] for identity in common_ids], "p_bookmaker")
                           for book in books},
            "paired_differences": {
                book: {
                    "paired_log_loss_difference": _paired_difference([by_book[book][identity] for identity in common_ids]),
                    "paired_brier_difference": _paired_difference([by_book[book][identity] for identity in common_ids], "brier"),
                } for book in books}},
        "per_book_paired_cohorts": per_book,
        "available_book_consensus": _available_consensus(paired_rows, books),
        "exclusions": {"quote_reason_counts": dict(sorted(exclusions.items())), "quote_rows": excluded_rows},
        "coverage": {
            "provider_fixtures": len(fixture_by_id), "mapping_rows": len(mappings),
            "mapping_status_counts": dict(sorted(mapping_statuses.items())),
            "mapping_label_status_counts": dict(sorted(mapping_labels.items())),
            "matched_native_ids": len(native_to_fixture), "stored_quote_observations": stored_quote_count,
            "requested_book_quote_observations": selected_quote_count,
            "mapped_without_test_prediction_ids": mapped_without_prediction,
            "mapped_without_test_prediction_matches": len(mapped_without_prediction),
            "test_predictions_without_usable_book_ids": sorted(set(prediction_by_id) - matched_any, key=int),
            "test_predictions_without_valid_audited_mapping_ids": sorted(set(prediction_by_id) - set(mapped_predictions), key=int),
            "missing_paired_test_ids_by_book": {book: sorted(set(prediction_by_id) - set(by_book[book]), key=int)
                                                for book in books},
        },
        "protocol": {
            "books": list(books), "prediction_partition": "TEST only; strictly after VALIDATION cutoff",
            "selected_model": model_metadata["selected_model"],
            "frozen_selection_evidence": model_metadata.get("selection"), "frozen_splits": model_metadata["splits"],
            "model_refit": False, "test_model_selection": False,
            "cohort_rule": "all-book common native IDs; per-book model and bookmaker identical native IDs",
            "devig_method": devig_method, "quote_cutoff_policy": quote_cutoff,
            "book_probability": (
                "two-sided proportional no-vig: (1 / price_team1) / (1 / price_team1 + 1 / price_team2)"
                if devig_method == "proportional" else
                "two-sided power no-vig: solve q1^a + q2^a = 1 from prices alone; p_team1 = q1^a"
            ),
            "accuracy_threshold": "probability >= 0.5", "quote_selection": "latest stored fixture/book observation supplied by caller; no fallback to older rows",
            "quote_cutoff": (
                "both aware side clocks strictly before min(scheduled start, true start); current receipt also strictly before cutoff"
                + ("; additionally both side clocks and current receipt strictly before nominal source-day 00:00 UTC"
                   if quote_cutoff == "source_day" else "")
                + ("; additionally both side clocks and current receipt strictly before explicit prediction forecast_origin_utc, bounded by source-day opening and event start"
                   if quote_cutoff == "forecast_origin" else "")
            ),
            "forecast_origins_utc": {identity: provider.iso(origin) for identity, origin in forecast_origins.items()},
            "derived_selection_method_counts": dict(sorted(Counter(
                row["derived_selection_method"] for row in paired_rows if row["derived_selection_method"]
            ).items())),
            "sensitivity_interpretation": "source-day cutoff is a nominal UTC proxy applied to stored latest observations, not reconstructed snapshots or certified equal-information/closing prices; no execution or ROI claim",
            "model_cutoff": (
                "explicit declared forecast_origin_utc, no earlier than nominal source-day opening; not a certified feature release clock"
                if quote_cutoff == "forecast_origin" else
                "nominal source-calendar-day 00:00 UTC proxy; not a certified original feature release clock"
            ),
            "market": "CS series Winner market 171, outcomes 171/172, player 0 catalog convention",
            "paired_interval": "model-minus-book log loss and Brier; source-day blocks sampled with replacement; match-weighted mean; 5000 draws seed 42; percentile 95% interval",
            "undefined_scores": "empty scalar metrics null; empty calibration bins []; single-class ROC AUC null",
        },
        "limits": [
            "The broader downloaded/mapped pool is not the frozen player-model TEST population; missing predictions are not extrapolated or fitted.",
            "No invented player identities or pre-match lineups. Existing predictions are conditional on their saved lineup/source evidence; missing prediction reasons and original lineup availability are not independently certified.",
            "HLTV source dates are calendar days with uncertified original timezone; synthetic noon is never treated as an actual kickoff or information cutoff.",
            "Stored bookmaker prices are latest pre-start observations, not equal information cutoffs to the day-frozen model; a source-day UTC proxy does not certify original feature release clocks.",
            "Historical observed_at is download receipt time, not source price time and not proof the quote was obtainable before the match.",
            "Provider Winner/outcome/player-0 bindings follow catalog conventions; executable entity, market and price availability remain not certified.",
            "Bookmaker availability differs. Comparisons on differing sample populations cannot establish a model advantage; only identical-ID paired cohorts are scored.",
            "Day-block intervals are descriptive sampling uncertainty, not profitability evidence. Zero-crossing intervals do not support a significance claim; a single source day yields a degenerate interval with no between-day uncertainty information.",
            "No actionable betting, execution, profitability or statistical significance claim is made, even for intervals excluding zero.",
        ],
    }
    return report, paired_rows
