"""Prepare, develop, and confirm a frozen sports-only bookmaker challenge.

The prepare/develop commands never generate reserved-TEST rating features or
model forecasts. Confirm verifies the saved recipe/artifact/source freeze first.
All bookmaker acquisition is offline from immutable cached historical receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from pipeline.features.series_history import (
    FEATURE_COLUMNS, FEATURE_GROUPS, HISTORY_COLUMNS, MIRROR_RULES,
    build_series_features, load_history,
)
from pipeline.features.valve_standings import VRS_FEATURE_COLUMNS, VRS_MIRROR_RULES, load_standings
from pipeline.models.bookmaker_comparison import (
    BOOKS, PAIRED_FIELDS, _devig, _quote_evidence, _validate_link, compare_hltv_bookmakers,
)
from pipeline.models.challenge_evaluation import competitiveness_decision, history_lag_sensitivity
from pipeline.models.market_history import _origin, load_market_history
from pipeline.models.series_forecast import (
    candidate_recipes, fit_candidate, score_predictions, select_development_model,
)
from scripts import collect_cs_oddspapi as provider


SOURCE_PATHS = (
    "scripts/evaluate_bookmaker_challenge.py", "scripts/collect_valve_standings.py",
    "scripts/collect_cs_oddspapi.py", "scripts/map_cs_oddspapi_history.py",
    "pipeline/features/series_history.py", "pipeline/features/valve_standings.py",
    "pipeline/features/unified_player_ratings.py", "pipeline/ratings/player_elo.py",
    "pipeline/models/series_forecast.py", "pipeline/models/player_baseline.py",
    "pipeline/models/metrics.py", "pipeline/models/bookmaker_comparison.py",
    "pipeline/models/market_history.py", "pipeline/models/challenge_evaluation.py",
    "pipeline/cli.py", "requirements-modeling.txt",
)
RULES = MIRROR_RULES | VRS_MIRROR_RULES
ALL_FEATURES = FEATURE_COLUMNS + VRS_FEATURE_COLUMNS


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def source_manifest():
    return {name: digest(ROOT / name) for name in SOURCE_PATHS}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_pickle(path, value):
    with Path(path).open("xb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    # Only artifacts generated locally by this trusted workflow are supported.
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def verify_hashes(hashes, root=ROOT):
    for name, expected in hashes.items():
        if digest(Path(root) / name) != expected:
            raise ValueError(f"Frozen input/artifact changed: {name}")


def validate_inputs(args, protocol):
    verify_hashes(protocol["input_sha256"])
    for actual, frozen in (
        (args.master, "data/processed/master_matches_consolidated.csv"),
        (args.hltv_results, "data/raw/hltv/matches.csv"),
    ):
        if digest(actual) != protocol["input_sha256"][frozen]:
            raise ValueError(f"Consumed input differs from its frozen bytes: {actual}")


def inputs(args):
    protocol = read_json(args.protocol)
    validate_inputs(args, protocol)
    cohorts = read_json(ROOT / protocol["cohort_manifest"])["cohorts"]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return protocol, cohorts, output


def history(args, protocol, cohorts):
    frame = pd.read_csv(args.master, dtype=str, keep_default_na=False,
                        usecols=lambda name: name in HISTORY_COLUMNS)
    games, audit = load_history(frame, minimum_day=protocol["partitions"]["history_start"])
    expected = {row["hltv_match_id"]: row["source_date"] for rows in cohorts.values() for row in rows}
    observed = {game["hltv_match_id"]: game["source_date"] for game in games if game["kind"] == "series"}
    if expected != observed:
        raise ValueError("Broad series targets or source dates differ from the frozen native cohort")
    return games, audit


def add_standings(frame, standings):
    values = [standings.features(row.team1, row.team2, row.source_date)
              for row in frame.itertuples(index=False)]
    ranked = pd.DataFrame(values, index=frame.index, columns=VRS_FEATURE_COLUMNS)
    return pd.concat((frame, ranked), axis=1)


def market_cases(snapshot, games, *, horizon="one_hour", devig="proportional"):
    """Validate native outcomes/orientations without labeling DEV forecasts TEST."""
    fixture_by_id = {row["fixtureId"]: row for row in snapshot["fixtures"]}
    links = {row["fixtureId"]: row for row in snapshot["links"]}
    native = {game["hltv_match_id"]: game for game in games if game["kind"] == "series"}
    rows, seen = [], set()
    for quote in snapshot["quotes_by_horizon"][horizon]:
        if quote["status"] != "admitted":
            continue
        fixture, link = fixture_by_id[quote["fixture_id"]], links[quote["fixture_id"]]
        identity = str(link.get("hltv_match_id"))
        if identity not in native or fixture.get("statusId") not in (2, "2"):
            continue
        game = native[identity]
        direct, prediction_direct = _validate_link(link, fixture, game)
        accepted, reason = _quote_evidence(quote, fixture)
        if reason:
            raise ValueError(f"Prepared admitted quote failed strict guard: {reason}")
        _, prices, event = accepted
        origin = provider.utc(quote["forecast_origin_utc"])
        opening = provider.utc(game["source_date"] + "T00:00:00Z")
        if not opening <= origin < event or any(provider.utc(quote[f"source_time_{side}"]) >= origin for side in (1, 2)):
            raise ValueError("Prepared quote violates the fixed forecast origin")
        if direct != prediction_direct:
            prices = prices[::-1]
        probability, _ = _devig(prices, devig)
        key = identity, quote["bookmaker"]
        if key in seen:
            raise ValueError("Duplicate native/book observation")
        seen.add(key)
        rows.append({"hltv_match_id": identity, "fixtureId": quote["fixture_id"],
                     "source_date": game["source_date"], "target": game["target"],
                     "team1": game["team1"], "team2": game["team2"],
                     "bookmaker": quote["bookmaker"], "p_bookmaker": probability,
                     "forecast_origin_utc": quote["forecast_origin_utc"], "quote_id": quote["id"]})
    return rows


def consensus_cases(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["hltv_match_id"]].append(row)
    result = []
    for identity in sorted(grouped, key=int):
        observations = grouped[identity]
        if len({row["bookmaker"] for row in observations}) != len(observations):
            raise ValueError("Duplicate book in development consensus")
        fields = ("source_date", "target", "team1", "team2", "forecast_origin_utc")
        first = observations[0]
        if any(any(row[field] != first[field] for field in fields) for row in observations):
            raise ValueError("Cross-book native label, orientation or origin conflict")
        result.append({"hltv_match_id": identity, **{name: first[name] for name in fields},
                       "p_bookmaker": sum(row["p_bookmaker"] for row in observations) / len(observations),
                       "books": sorted(row["bookmaker"] for row in observations)})
    return result


def prepare(args):
    protocol, cohorts, output = inputs(args)
    sources = source_manifest()
    protocol_sha256 = digest(args.protocol)
    standings_sha256 = digest(Path(args.standings) / "manifest.json")
    if (output / "prepared_manifest.json").exists():
        raise ValueError("Preparation is already frozen; use its saved artifacts")
    games, audit = history(args, protocol, cohorts)
    standings = load_standings(args.standings)
    development_games = [game for game in games if game["source_date"] <= protocol["partitions"]["validation_end"]]
    features, state = build_series_features(development_games)
    features = add_standings(features, standings)
    expected = {row["hltv_match_id"] for part in ("train", "validation") for row in cohorts[part]}
    if set(features.hltv_match_id) != expected:
        raise ValueError("Development replay changed frozen native target membership")
    snapshot = load_market_history(args.database, args.hltv_results)
    market = {horizon: market_cases(snapshot, games, horizon=horizon)
              for horizon in snapshot["quotes_by_horizon"]}
    partition_ids = {part: {row["hltv_match_id"] for row in rows} for part, rows in cohorts.items()}
    coverage = {horizon: {part: {
        "unique_matches": len({row["hltv_match_id"] for row in rows if row["hltv_match_id"] in ids}),
        "per_book": dict(Counter(row["bookmaker"] for row in rows if row["hltv_match_id"] in ids)),
    } for part, ids in partition_ids.items()} for horizon, rows in market.items()}
    verify_hashes(sources)
    validate_inputs(args, protocol)
    if digest(args.protocol) != protocol_sha256 or digest(Path(args.standings) / "manifest.json") != standings_sha256:
        raise ValueError("Protocol or standings changed during preparation")
    feature_path = output / "development_features.csv"
    features.to_csv(feature_path, index=False, mode="x")
    save_pickle(output / "development_state.pkl", state)
    raw_snapshot = json.dumps(snapshot, allow_nan=False, separators=(",", ":")).encode()
    with (output / "market_snapshot.json.gz").open("xb") as stream:
        stream.write(gzip.compress(raw_snapshot, mtime=0))
    save_json(output / "market_cases.json", market)
    save_json(output / "market_audit.json", snapshot["audit"])
    save_json(output / "history_audit.json", audit)
    files = ("development_features.csv", "development_state.pkl", "market_snapshot.json.gz",
             "market_cases.json", "market_audit.json", "history_audit.json")
    manifest = {
        "protocol_sha256": protocol_sha256, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {name: digest(output / name) for name in files}, "sources": sources,
        "standings_directory": str(Path(args.standings).resolve()),
        "standings_manifest_sha256": standings_sha256,
        "master": str(Path(args.master).resolve()),
        "master_sha256": protocol["input_sha256"]["data/processed/master_matches_consolidated.csv"],
        "native_results": str(Path(args.hltv_results).resolve()),
        "native_results_sha256": protocol["input_sha256"]["data/raw/hltv/matches.csv"],
        "feature_columns": ALL_FEATURES, "mirror_rules": RULES,
        "development_history_through": state.history_through,
        "development_series": len(features), "test_features_or_model_forecasts_generated": False,
        "coverage_only": coverage, "valve_snapshots": standings.manifest["snapshot_count"],
        "feature_support": {part: {
            name: int(features.loc[features.hltv_match_id.isin(ids), name].eq(1).sum())
            for name in ("performance_available", "vrs_available", "either_team_cold")
        } for part, ids in partition_ids.items() if part in ("train", "validation")},
    }
    save_json(output / "prepared_manifest.json", manifest)
    print(json.dumps({"phase": "prepared", "series": len(features), "history_games": len(games),
                      "coverage": coverage, "feature_support": manifest["feature_support"]}, indent=2))
    return 0


def prepared(args):
    protocol, cohorts, output = inputs(args)
    manifest = read_json(output / "prepared_manifest.json")
    if manifest["protocol_sha256"] != digest(args.protocol):
        raise ValueError("Frozen preparation protocol changed")
    verify_hashes(manifest["artifacts"], output)
    if digest(Path(args.standings) / "manifest.json") != manifest["standings_manifest_sha256"]:
        raise ValueError("Historical standings snapshot changed")
    return protocol, cohorts, output, manifest


def selected_fold(recipe, recipes, train, cutoff):
    dependencies = {}
    if recipe["family"] == "blend":
        for name in (recipe["parameters"]["left"], recipe["parameters"]["right"]):
            dependencies[name] = selected_fold(recipes[name], recipes, train, cutoff)
    return fit_candidate(recipe, train, RULES, train_end=cutoff, fitted_candidates=dependencies)


def freeze_selection(output, *, sources, protocol_path, prepared_path, model, evidence,
                     validation_predictions, protocol_sha256, prepared_sha256):
    """Publish only the input/code versions checked before fitting began."""
    verify_hashes(sources)
    if digest(protocol_path) != protocol_sha256 or digest(prepared_path) != prepared_sha256:
        raise ValueError("Protocol or preparation manifest changed while fitting")
    validation_predictions.to_csv(output / "validation_predictions.csv", index=False, mode="x")
    save_json(output / "selection.json", evidence)
    save_pickle(output / "selected_model.pkl", model)
    save_json(output / "selection_freeze.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_sha256,
        "prepared_manifest_sha256": prepared_sha256,
        "artifacts": {name: digest(output / name) for name in
                      ("selected_model.pkl", "selection.json", "validation_predictions.csv")},
        "sources": sources, "selected_recipe": model.recipe,
        "test_forecasts_or_losses_observed": False,
        "environment": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")},
        "python": sys.version,
    })


def develop(args):
    protocol, cohorts, output, manifest = prepared(args)
    if (output / "selection_freeze.json").exists() or (output / "confirmation.json").exists():
        raise ValueError("This experiment already has a frozen selection; no reselection")
    verify_hashes(manifest["sources"])
    sources = dict(manifest["sources"])
    protocol_sha256 = manifest["protocol_sha256"]
    prepared_sha256 = digest(output / "prepared_manifest.json")
    features = pd.read_csv(output / "development_features.csv", dtype={"hltv_match_id": str, "source_date": str})
    parts = protocol["partitions"]
    train = features.loc[features.source_date.between(parts["training_fit_start"], parts["train_end"])].copy()
    validation = features.loc[features.source_date.between(parts["validation_start"], parts["validation_end"])].copy()
    validation_ids = {row["hltv_match_id"] for row in cohorts["validation"]}
    market = read_json(output / "market_cases.json")["one_hour"]
    validation_market = [row for row in market if row["hltv_match_id"] in validation_ids]
    consensus = consensus_cases(validation_market)
    market_ids = {row["hltv_match_id"] for row in consensus}
    recipes = candidate_recipes(FEATURE_GROUPS, RULES, VRS_FEATURE_COLUMNS)

    def progress(index, total, name, scores):
        if index % 20 == 0 or index == total:
            print(json.dumps({"candidates_finished": index, "total": total, "last_recipe": name}), flush=True)

    model, evidence, forecasts = select_development_model(
        train, validation, market_ids, recipes, RULES, train_end=parts["train_end"],
        validation_end=parts["validation_end"], progress=progress)
    selected = evidence["selected_model"]
    probabilities = dict(zip(validation.hltv_match_id, forecasts[selected]))
    evidence["market_validation_benchmark"] = score_predictions(
        [row["target"] for row in consensus], [row["p_bookmaker"] for row in consensus])
    evidence["market_validation_model"] = score_predictions(
        [row["target"] for row in consensus], [probabilities[row["hltv_match_id"]] for row in consensus])
    evidence["market_validation_per_book"] = {
        book: {"model": score_predictions([row["target"] for row in validation_market if row["bookmaker"] == book],
                                           [probabilities[row["hltv_match_id"]] for row in validation_market if row["bookmaker"] == book]),
               "bookmaker": score_predictions([row["target"] for row in validation_market if row["bookmaker"] == book],
                                               [row["p_bookmaker"] for row in validation_market if row["bookmaker"] == book])}
        for book in BOOKS
    }
    recipes_by_name = {recipe["name"]: recipe for recipe in recipes}
    folds = []
    for cutoff in protocol["candidate_space"]["train_only_diagnostic_cutoffs"]:
        end = min(parts["train_end"], (date.fromisoformat(cutoff) + timedelta(days=60)).isoformat())
        fit_rows = train.loc[train.source_date <= cutoff]
        score_rows = train.loc[(train.source_date > cutoff) & (train.source_date <= end)]
        if not len(fit_rows) or not len(score_rows):
            raise ValueError("Frozen TRAIN-only diagnostic lacks temporal support")
        folded = selected_fold(model.recipe, recipes_by_name, fit_rows, cutoff)
        folds.append({"train_end": cutoff, "evaluation_end": end, "train_matches": len(fit_rows),
                      "evaluation_matches": len(score_rows), "scores": score_predictions(score_rows.target, folded.predict(score_rows))})
    evidence["train_only_expanding_diagnostics"] = folds
    validation_predictions = validation.loc[:, ["hltv_match_id", "source_date", "team1", "team2", "target", "format"]].copy()
    validation_predictions["partition"] = "validation"
    validation_predictions["model_selected"] = selected
    validation_predictions["prob_team1_win"] = forecasts[selected]
    validation_predictions["prob_team2_win"] = 1 - forecasts[selected]
    verify_hashes(manifest["artifacts"], output)
    freeze_selection(output, sources=sources, protocol_path=args.protocol,
                     prepared_path=output / "prepared_manifest.json", model=model, evidence=evidence,
                     validation_predictions=validation_predictions, protocol_sha256=protocol_sha256,
                     prepared_sha256=prepared_sha256)
    leaderboard = sorted(evidence["candidate_scores"].items(), key=lambda item: (item[1]["market_validation"]["log_loss"], item[0]))
    print(json.dumps({"phase": "selected_and_frozen", "selected_model": selected,
                      "model_validation": evidence["market_validation_model"],
                      "market_validation": evidence["market_validation_benchmark"],
                      "top_candidates": [{"name": name, "market_log_loss": score["market_validation"]["log_loss"],
                                          "full_log_loss": score["full_validation"]["log_loss"]}
                                         for name, score in leaderboard[:12]]}, indent=2))
    return 0


def metadata_for_test(features, evidence, protocol):
    parts = protocol["partitions"]
    days = sorted(set(features.source_date))
    return {"selected_model": evidence["selected_model"], "selection": evidence["selection"],
            "splits": {"train": {"end_date": parts["train_end"]},
                       "validation": {"end_date": parts["validation_end"]},
                       "test": {"start_date": days[0], "end_date": days[-1], "dates": days,
                                "n_series": len(features)}},
            "candidate_log_loss": evidence["selection"]["candidate_log_loss"]}


def forecast_origins(snapshot, horizon, frame):
    links = {str(row.get("hltv_match_id")): row for row in snapshot["links"]
             if row.get("mapping_status") == "matched"}
    fixtures = {row["fixtureId"]: row for row in snapshot["fixtures"]}
    origins = []
    for row in frame.itertuples(index=False):
        link = links.get(row.hltv_match_id)
        if link is None:
            origins.append(None)
            continue
        try:
            origin, reason = _origin(fixtures[link["fixtureId"]], link, horizon)
        except (provider.Stop, ValueError, TypeError, OverflowError):
            origin, reason = None, "invalid_start_or_source_day_evidence"
        origins.append(provider.iso(origin) if origin is not None and reason is None else None)
    return origins


def timed_comparison(snapshot, predictions, metadata, *, horizon, devig):
    """Keep frozen TEST date eligibility even when no timed cases survive."""
    scoped = predictions.copy()
    scoped["forecast_origin_utc"] = forecast_origins(snapshot, horizon, scoped)
    unsupported_ids = scoped.loc[scoped.forecast_origin_utc.isna(), "hltv_match_id"].tolist()
    scoped = scoped.loc[scoped.forecast_origin_utc.notna()].copy()
    splits = {**metadata["splits"], "test": {**metadata["splits"]["test"], "n_series": len(scoped)}}
    scoped_metadata = {**metadata, "splits": splits}
    report, rows = compare_hltv_bookmakers(
        snapshot["fixtures"], snapshot["quotes_by_horizon"][horizon], snapshot["links"],
        scoped.to_dict("records"), scoped_metadata, devig_method=devig, quote_cutoff="forecast_origin")
    report["full_reserved_test_matches"] = len(predictions)
    report["no_eligible_forecast_origin_native_ids"] = unsupported_ids
    report["timed_prediction_cohort_rule"] = "frozen TEST intersect valid native/provider forecast-origin evidence; no score-based exclusions"
    return report, rows


def confirm(args):
    protocol, cohorts, output, manifest = prepared(args)
    if (output / "confirmation.json").exists():
        raise ValueError("Reserved TEST was already evaluated; do not reuse it for selection")
    freeze = read_json(output / "selection_freeze.json")
    if freeze["protocol_sha256"] != digest(args.protocol) or freeze["prepared_manifest_sha256"] != digest(output / "prepared_manifest.json"):
        raise ValueError("Selection protocol/preparation freeze changed")
    verify_hashes(freeze["artifacts"], output)
    verify_hashes(freeze["sources"])
    model, evidence = load_pickle(output / "selected_model.pkl"), read_json(output / "selection.json")
    if model.recipe != freeze["selected_recipe"] or model.selection != evidence["selection"]:
        raise ValueError("Persisted forecaster and selected recipe/evidence disagree")
    games, _ = history(args, protocol, cohorts)
    standings = load_standings(args.standings)
    features, state = build_series_features(games)
    features = add_standings(features, standings)
    test_ids = {row["hltv_match_id"] for row in cohorts["test"]}
    test = features.loc[features.hltv_match_id.isin(test_ids)].copy()
    if set(test.hltv_match_id) != test_ids:
        raise ValueError("Reserved native TEST membership changed")
    probability = model.predict(test)
    predictions = test.loc[:, ["hltv_match_id", "source_date", "team1", "team2", "target", "format"]].copy()
    predictions["date"] = predictions.source_date
    predictions["partition"] = "test"
    predictions["model_selected"] = evidence["selected_model"]
    predictions["prob_team1_win"], predictions["prob_team2_win"] = probability, 1 - probability
    snapshot = json.loads(gzip.decompress((output / "market_snapshot.json.gz").read_bytes()))
    metadata = metadata_for_test(test, evidence, protocol)
    comparisons, primary, primary_rows = {}, None, None
    comparison_dir = output / "comparisons"
    comparison_dir.mkdir(exist_ok=True)
    for horizon in ("one_hour", "six_hour", "source_day"):
        for devig in ("proportional", "power"):
            report, rows = timed_comparison(snapshot, predictions, metadata, horizon=horizon, devig=devig)
            name = horizon + "_" + devig
            save_json(comparison_dir / (name + ".json"), report)
            pd.DataFrame(rows, columns=PAIRED_FIELDS).to_csv(comparison_dir / (name + ".csv"), index=False, mode="x")
            comparisons[name] = {"matches": report["available_book_consensus"]["matches"],
                                 "model": report["available_book_consensus"]["model"],
                                 "bookmaker": report["available_book_consensus"]["bookmaker"],
                                 "paired_log_loss_difference": report["available_book_consensus"]["paired_log_loss_difference"]}
            if name == "one_hour_proportional":
                primary, primary_rows = report, rows
    decision = competitiveness_decision(primary["available_book_consensus"]["rows"],
                                        primary["per_book_paired_cohorts"], protocol["acceptance"],
                                        test_start=protocol["partitions"]["test_start"])
    lagged, _ = build_series_features(games, history_lag_days=2)
    lagged = add_standings(lagged, standings)
    lagged = lagged.loc[lagged.hltv_match_id.isin(test_ids)].copy()
    lagged_probability = dict(zip(lagged.hltv_match_id, model.predict(lagged)))
    lagged_decision = history_lag_sensitivity(
        primary["available_book_consensus"]["rows"], primary_rows, lagged_probability,
        protocol["acceptance"], test_start=protocol["partitions"]["test_start"])
    predictions["forecast_origin_utc"] = forecast_origins(snapshot, "one_hour", predictions)
    predictions.to_csv(output / "test_predictions.csv", index=False, mode="x")
    save_json(output / "test_metrics.json", metadata | {"test": score_predictions(test.target, probability)})
    save_pickle(output / "series_inference_state.pkl", {
        "schema_version": 2, "engine": state, "standings": standings, "bayesian_engine": None,
        "model_sha256": digest(output / "selected_model.pkl"), "model_selected": evidence["selected_model"],
        "protocol_sha256": digest(args.protocol), "history_through": state.history_through,
        "competitiveness_qualified": decision["qualified"], "feature_columns": ALL_FEATURES,
        "availability_limits": protocol["limitations"],
    })
    confirmation = {"selected_model": evidence["selected_model"], "test": score_predictions(test.target, probability),
                    "decision": decision, "comparisons": comparisons,
                    "two_day_result_history_sensitivity": lagged_decision,
                    "two_day_sensitivity_vrs_availability": "unchanged conservative original-publication plus7day policy",
                    "test_native_ids": sorted(test_ids, key=int), "history_through": state.history_through,
                    "selection_freeze_sha256": digest(output / "selection_freeze.json"),
                    "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "artifact_sha256": {name: digest(output / name) for name in
                                        ("test_predictions.csv", "test_metrics.json", "series_inference_state.pkl")}}
    save_json(output / "confirmation.json", confirmation)
    print(json.dumps({"phase": "confirmed", "selected_model": evidence["selected_model"],
                      "test": confirmation["test"], "decision": decision,
                      "market_comparisons": comparisons}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "develop", "confirm"))
    parser.add_argument("--protocol", default="reports/cs2-bookmaker-challenge-protocol-2026-10-09.json")
    parser.add_argument("--master", default="data/processed/master_matches_consolidated.csv")
    parser.add_argument("--hltv-results", default="data/raw/hltv/matches.csv")
    parser.add_argument("--database", default="data/app/app.sqlite3")
    parser.add_argument("--standings", default="data/raw/valve_standings_20261009")
    parser.add_argument("--output-dir", default="artifacts/bookmaker_challenge_20261009/verified")
    args = parser.parse_args(argv)
    try:
        return {"prepare": prepare, "develop": develop, "confirm": confirm}[args.phase](args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
