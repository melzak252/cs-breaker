"""Develop and publish the frozen, development-only Bayesian series challenger.

Neither phase forecasts or scores the previously exposed TEST population. Publish
only advances completed histories; the TRAIN-fitted raw model is never refitted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from pipeline.features.bayesian_series import (
    BAYES_FEATURE_COLUMNS, BAYES_MIRROR_RULES, build_bayesian_features,
)
from pipeline.features.series_history import (
    FEATURE_COLUMNS, FEATURE_GROUPS, HISTORY_COLUMNS, MIRROR_RULES,
    SeriesHistoryEngine, load_history,
)
from pipeline.features.valve_standings import VRS_FEATURE_COLUMNS, VRS_MIRROR_RULES, load_standings
from pipeline.models.bookmaker_comparison import BOOKS, _paired_difference
from pipeline.models.series_forecast import fit_candidate, score_predictions, select_development_model
from scripts.evaluate_bookmaker_challenge import (
    consensus_cases, digest, load_pickle, read_json, save_json, save_pickle, verify_hashes,
)

RULES = MIRROR_RULES | VRS_MIRROR_RULES | BAYES_MIRROR_RULES
ALL_FEATURES = FEATURE_COLUMNS + VRS_FEATURE_COLUMNS + BAYES_FEATURE_COLUMNS
IDENTITY_COLUMNS = ["hltv_match_id", "source_date", "team1", "team2", "target", "format"]
INCUMBENT = "ridge_C1_ratings_form_hnone"
INPUT_OPTIONS = {
    "master": "master", "development_features": "development_features",
    "market_cases": "market_cases", "cohorts": "cohorts",
    "baseline_selection": "baseline_selection", "standings": "standings_directory",
}
DEVELOPMENT_ARTIFACTS = (
    "development_started_manifest.json", "development_features.csv", "bayesian_features.csv",
    "development_native_ids.json", "development_history_state.pkl", "selected_model.pkl",
    "selection.json", "development_comparison.json", "validation_predictions.csv",
    "candidate_validation_predictions.csv", "development_manifest.json",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _environment():
    return {"python": sys.version, "packages": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "scipy")
    }}


def _validate_identity_frame(frame, name):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f"{name} needs a DataFrame with unique columns")
    if not set(IDENTITY_COLUMNS).issubset(frame.columns):
        raise ValueError(f"{name} is missing native identity metadata")
    if frame[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError(f"{name} contains missing native identity metadata")
    identities = frame.hltv_match_id
    if not identities.map(lambda value: isinstance(value, str) and
                          re.fullmatch(r"[1-9][0-9]*", value) is not None).all():
        raise ValueError(f"{name} requires canonical native string IDs; no coercion")
    if identities.duplicated().any():
        raise ValueError(f"{name} contains duplicate native IDs")
    for value in frame.source_date:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError(f"{name} contains a noncanonical source date")
    for column in ("team1", "team2"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name} contains an invalid literal team")
    if (frame.team1 == frame.team2).any() or not frame.format.isin(("bo3", "bo5")).all():
        raise ValueError(f"{name} contains invalid opponents or series formats")
    if not frame.target.map(lambda value: not isinstance(value, (bool, np.bool_)) and
                            isinstance(value, (int, np.integer)) and value in (0, 1)).all():
        raise ValueError(f"{name} requires binary integer outcomes; no coercion")


def join_development_features(base, bayesian):
    """One-to-one, full-population native join, preserving every old covariate.

    Only declared Bayesian probability columns are added. Engine history metadata
    may be present, but never overwrites the baseline. Subsets of the declared
    channel bank are accepted here for independently testable join integrity;
    the actual development phase separately requires all sixteen channels.
    """
    _validate_identity_frame(base, "baseline")
    _validate_identity_frame(bayesian, "Bayesian features")
    if set(base.hltv_match_id) != set(bayesian.hltv_match_id):
        raise ValueError("Baseline and Bayesian native populations differ")
    aligned = bayesian.set_index("hltv_match_id").loc[base.hltv_match_id]
    for column in IDENTITY_COLUMNS[1:]:
        if not np.array_equal(base[column].to_numpy(), aligned[column].to_numpy()):
            raise ValueError(f"Native development {column} differs between feature banks")
    columns = [name for name in BAYES_FEATURE_COLUMNS if name in bayesian.columns]
    undeclared = [name for name in bayesian.columns if name.startswith("bayes_") and
                  name.endswith("_prob") and name not in BAYES_FEATURE_COLUMNS]
    if not columns or undeclared:
        raise ValueError("Bayesian features require declared probability channels only")
    if set(columns).intersection(base.columns):
        raise ValueError("Bayesian join would overwrite baseline predictors")
    for column in columns:
        values = aligned[column]
        if (not pd.api.types.is_numeric_dtype(values.dtype) or
                pd.api.types.is_bool_dtype(values.dtype) or
                not np.isfinite(values.to_numpy()).all() or
                ((values < 0) | (values > 1)).any()):
            raise ValueError(f"Invalid Bayesian probability channel: {column}")
    result = base.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    return result


def _paths(args, protocol):
    paths = {}
    for option, key in INPUT_OPTIONS.items():
        supplied = getattr(args, option, None)
        paths[key] = Path(supplied).resolve() if supplied else (ROOT / protocol["paths"][key]).resolve()
    return paths


def _standings_hashes(directory):
    """Freeze every byte consumed by load_standings, not only its manifest."""
    root = Path(directory).resolve()
    manifest_sha256 = digest(root / "manifest.json")
    manifest = read_json(root / "manifest.json")
    expected = {"manifest.json": manifest_sha256,
                manifest["records_path"]: manifest["records_sha256"]}
    expected.update({snapshot["raw_path"]: snapshot["sha256"] for snapshot in manifest["snapshots"]})
    hashes = {}
    for name, checksum in expected.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Standings evidence escapes its frozen directory")
        hashes[str(path)] = digest(path)
        if hashes[str(path)] != checksum:
            raise ValueError(f"Consumed standings bytes differ from manifest: {path}")
    return hashes


def _validate_inputs(protocol, paths):
    verify_hashes(protocol["input_sha256"])
    expected_consumed = {}
    for key, path in paths.items():
        frozen = protocol["paths"][key]
        if key == "standings_directory":
            frozen += "/manifest.json"
            path = path / "manifest.json"
        expected_consumed[str(path)] = protocol["input_sha256"][frozen]
    verify_hashes(expected_consumed)
    # The frozen manifest itself anchors the CSV/raw hashes, and the established
    # loader validates provenance, every blob and the exact freshly parsed CSV.
    standings = load_standings(paths["standings_directory"])
    consumed = {str(path): digest(path) for key, path in paths.items() if key != "standings_directory"}
    consumed.update(_standings_hashes(paths["standings_directory"]))
    if any(consumed[name] != checksum for name, checksum in expected_consumed.items()):
        raise ValueError("Consumed inputs changed while establishing the opening freeze")
    verify_hashes(consumed)
    return standings, consumed


def _source_hashes(protocol):
    return {name: digest(ROOT / name) for name in protocol["source_paths"]}


def _check_frozen(started, protocol_path):
    verify_hashes(started["sources"])
    verify_hashes(started["consumed_inputs"])
    if digest(protocol_path) != started["protocol_sha256"]:
        raise ValueError("Frozen professional protocol changed during the phase")


def _cohort_dates(cohorts, partitions):
    result = {}
    for partition in partitions:
        for row in cohorts[partition]:
            identity = row["hltv_match_id"]
            if identity in result:
                raise ValueError("Frozen native cohorts overlap or duplicate an ID")
            result[identity] = row["source_date"]
    return result


def _history(paths, protocol, cohorts, *, development_only):
    frame = pd.read_csv(paths["master"], dtype=str, keep_default_na=False,
                        usecols=lambda name: name in HISTORY_COLUMNS)
    if development_only:
        # Crucially the raw source prefix is taken BEFORE the historical loader,
        # including before statistics validation or duplicate-ID accounting.
        frame = frame.loc[frame["date"].str[:10] <= protocol["development_end"]].copy()
    games, audit = load_history(frame, minimum_day=protocol["partitions"]["history_start"])
    partitions = ("train", "validation") if development_only else tuple(cohorts)
    expected = _cohort_dates(cohorts, partitions)
    observed = {game["hltv_match_id"]: game["source_date"] for game in games if game["kind"] == "series"}
    if expected != observed:
        raise ValueError("Native history targets/dates differ from the full frozen phase population")
    if development_only and any(game["source_date"] > protocol["development_end"] for game in games):
        raise ValueError("Development history contains protected TEST rows")
    return games, audit


def _advance(engine, games):
    by_day = defaultdict(list)
    for game in games:
        by_day[game["source_date"]].append(game)
    for day in sorted(by_day):
        engine.update_day(day, sorted(by_day[day], key=lambda game: int(game["hltv_match_id"])))
    return engine


def _recipes(protocol, baseline):
    space = protocol["candidate_space"]
    if baseline["selected_model"] != INCUMBENT or space["incumbent"] != INCUMBENT:
        raise ValueError("The frozen incumbent differs from the original selected baseline")
    originals = [recipe for recipe in baseline["candidate_recipes"] if recipe["name"] == INCUMBENT]
    expected = {"name": INCUMBENT, "family": "ridge",
                "feature_columns": FEATURE_GROUPS["ratings"] + FEATURE_GROUPS["form"],
                "parameters": {"C": 1.0}, "recency_half_life_days": None}
    if originals != [expected]:
        raise ValueError("Incumbent recipe no longer matches its original frozen definition")
    recipes = [originals[0]]

    def add(name, family, columns, parameters=None):
        recipes.append({"name": name, "family": family, "feature_columns": list(columns),
                        "parameters": parameters or {}, "recency_half_life_days": None})

    for column in BAYES_FEATURE_COLUMNS:
        stem = column.removesuffix("_prob")
        add("raw_" + stem, "rating", [column])
        add("calibrated_" + stem, "calibrated_rating", [column])
    groups = {"bayesian": BAYES_FEATURE_COLUMNS,
              "ratings_form_bayesian": expected["feature_columns"] + BAYES_FEATURE_COLUMNS}
    for group in space["ridge_groups"]:
        for strength in space["ridge_C"]:
            add(f"ridge_C{strength:g}_{group}_hnone", "ridge", groups[group], {"C": strength})
    by_name = {recipe["name"]: recipe for recipe in recipes}
    for channel in space["blend_right_channels"]:
        right = "calibrated_" + channel
        columns = list(dict.fromkeys(expected["feature_columns"] + by_name[right]["feature_columns"]))
        for weight in space["blend_right_weights"]:
            add(f"blend_{INCUMBENT}__{right}_w{weight:g}", "blend", columns,
                {"left": INCUMBENT, "right": right, "right_weight": weight})
    if len(recipes) != 49 or len(recipes) != space["total_candidates"] or len({r["name"] for r in recipes}) != 49:
        raise ValueError("The candidate ladder is not the frozen 49 recipes")
    return recipes


def _fit_fold(recipe, recipes, train, cutoff, fitted=None):
    fitted = {} if fitted is None else fitted
    name = recipe["name"]
    if name not in fitted:
        if recipe["family"] == "blend":
            for dependency in (recipe["parameters"]["left"], recipe["parameters"]["right"]):
                _fit_fold(recipes[dependency], recipes, train, cutoff, fitted)
        fitted[name] = fit_candidate(recipe, train, RULES, train_end=cutoff, fitted_candidates=fitted)
    return fitted[name]


def _rolling_diagnostics(model, recipes, train, protocol):
    by_name = {recipe["name"]: recipe for recipe in recipes}
    folds = []
    for cutoff in protocol["candidate_space"]["train_only_diagnostic_cutoffs"]:
        end = min(protocol["partitions"]["train_end"],
                  (date.fromisoformat(cutoff) + timedelta(days=60)).isoformat())
        fit_rows = train.loc[train.source_date <= cutoff]
        score_rows = train.loc[(train.source_date > cutoff) & (train.source_date <= end)]
        if not len(fit_rows) or not len(score_rows):
            raise ValueError("TRAIN-only rolling diagnostic lacks temporal support")
        fitted = {}
        selected = _fit_fold(model.recipe, by_name, fit_rows, cutoff, fitted)
        incumbent = _fit_fold(by_name[INCUMBENT], by_name, fit_rows, cutoff, fitted)
        folds.append({"train_end": cutoff, "evaluation_start": (date.fromisoformat(cutoff) + timedelta(days=1)).isoformat(),
                      "evaluation_end": end, "train_matches": len(fit_rows), "evaluation_matches": len(score_rows),
                      "selected_model": model.recipe["name"],
                      "selected": score_predictions(score_rows.target, selected.predict(score_rows)),
                      "incumbent": score_predictions(score_rows.target, incumbent.predict(score_rows)),
                      "evaluation_native_ids": sorted(score_rows.hltv_match_id, key=int),
                      "scope": "TRAIN-only diagnostic of a recipe subsequently selected on validation; not independent confirmation"})
    return folds


def _validation_consensus(paths, validation, baseline):
    identities = set(validation.hltv_match_id)
    market = [row for row in read_json(paths["market_cases"])["one_hour"] if row["hltv_match_id"] in identities]
    native = validation.set_index("hltv_match_id")
    for row in market:
        reference = native.loc[row["hltv_match_id"]]
        if any(row[name] != reference[name] for name in ("source_date", "team1", "team2", "target")):
            raise ValueError("Frozen validation market and native label/orientation disagree")
        if row["bookmaker"] not in BOOKS or not isinstance(row["p_bookmaker"], (float, int)) or not 0 <= row["p_bookmaker"] <= 1:
            raise ValueError("Invalid frozen development market observation")
    consensus = consensus_cases(market)
    market_ids = {row["hltv_match_id"] for row in consensus}
    if market_ids != set(baseline["selection"]["market_native_ids"]) or len(market_ids) != 545 or len(validation) != 1487:
        raise ValueError("Validation must retain the original 545 market / 1487 all-native populations")
    return consensus, market


def _baseline_regression(evidence, baseline):
    original = baseline["candidate_scores"][INCUMBENT]
    rebuilt = evidence["candidate_scores"][INCUMBENT]
    deltas = {}
    for cohort in ("market_validation", "full_validation"):
        deltas[cohort] = {}
        if original[cohort]["n_samples"] != rebuilt[cohort]["n_samples"]:
            raise ValueError("Rebuilt incumbent has changed validation support")
        for metric in ("log_loss", "brier", "accuracy", "roc_auc", "ece"):
            delta = rebuilt[cohort][metric] - original[cohort][metric]
            deltas[cohort][metric] = delta
            if abs(delta) > 1e-12:
                raise ValueError(f"Original selected baseline regression: {cohort}/{metric}")
    return {"original_selected_model": baseline["selected_model"], "original_scores": original,
            "rebuilt_scores": rebuilt, "rebuilt_minus_original": deltas,
            "maximum_absolute_tolerance": 1e-12, "identical_within_tolerance": True}


def _comparison(validation, consensus, market, forecasts, evidence, baseline, protocol):
    selected = evidence["selected_model"]
    probabilities = {name: dict(zip(validation.hltv_match_id, forecasts[name])) for name in (INCUMBENT, selected)}
    rows = [{**row, "p_incumbent": float(probabilities[INCUMBENT][row["hltv_match_id"]]),
             "p_selected": float(probabilities[selected][row["hltv_match_id"]])} for row in consensus]
    pairs = {}
    for label, left, right in (("selected_minus_incumbent", "p_selected", "p_incumbent"),
                               ("selected_minus_consensus", "p_selected", "p_bookmaker"),
                               ("incumbent_minus_consensus", "p_incumbent", "p_bookmaker")):
        paired = [{**row, "p_model": row[left], "p_bookmaker": row[right]} for row in rows]
        pairs[label] = {cluster: {metric: _paired_difference(paired, metric, block_days=days,
                                                           start_day=protocol["partitions"]["validation_start"])
                                 for metric in ("log_loss", "brier")}
                        for cluster, days in (("source_day", 1), ("seven_day", 7))}
        for cluster in pairs[label].values():
            for metric, result in cluster.items():
                result["direction"] = f"{label} {metric}; negative favors the left forecast"
    return {"partition": "validation", "scope": "development selection evidence, not fresh confirmation",
            "qualified": False, "competitiveness_qualified": False,
            "test_forecasts_or_losses_observed": False, "selected_model": selected, "incumbent": INCUMBENT,
            "market_validation_matches": len(rows), "full_validation_matches": len(validation),
            "market_native_ids": sorted((row["hltv_match_id"] for row in rows), key=int),
            "scores": {"selected": evidence["candidate_scores"][selected],
                       "incumbent": evidence["candidate_scores"][INCUMBENT],
                       "consensus_market_validation": score_predictions([row["target"] for row in rows],
                                                                        [row["p_bookmaker"] for row in rows])},
            "paired_differences": pairs, "paired_rows": rows,
            "original_selected_baseline_regression": _baseline_regression(evidence, baseline),
            "per_book": {book: {"matches": len(subset),
                                "selected": score_predictions([row["target"] for row in subset],
                                    [probabilities[selected][row["hltv_match_id"]] for row in subset]),
                                "incumbent": score_predictions([row["target"] for row in subset],
                                    [probabilities[INCUMBENT][row["hltv_match_id"]] for row in subset]),
                                "bookmaker": score_predictions([row["target"] for row in subset],
                                    [row["p_bookmaker"] for row in subset])}
                         for book in BOOKS for subset in [[row for row in market if row["bookmaker"] == book]]},
            "availability_limits": protocol["limitations"]}


def develop(args):
    output = Path(args.output_dir).resolve()
    # Reject any partial/completed experiment before replay, fitting or scores.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output already contains experiment artifacts; no reruns or overwrites")
    protocol_sha256 = digest(args.protocol)
    protocol = read_json(args.protocol)
    paths = _paths(args, protocol)
    sources = _source_hashes(protocol)
    standings, consumed = _validate_inputs(protocol, paths)
    started = {"phase": "development_started", "started_at_utc": _now(),
               "protocol_sha256": protocol_sha256, "sources": sources, "consumed_inputs": consumed,
               "environment": _environment(), "partitions": protocol["partitions"],
               "development_end": protocol["development_end"],
               "test_feature_replay_allowed": False, "test_forecasts_or_losses_observed": False}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "development_started_manifest.json", started)
    started_sha256 = digest(output / "development_started_manifest.json")
    _check_frozen(started, args.protocol)
    cohorts = read_json(paths["cohorts"])["cohorts"]
    baseline = read_json(paths["baseline_selection"])
    recipes = _recipes(protocol, baseline)
    games, audit = _history(paths, protocol, cohorts, development_only=True)
    base = pd.read_csv(paths["development_features"], keep_default_na=False,
                       dtype={"hltv_match_id": str, "source_date": str})
    _validate_identity_frame(base, "frozen baseline")
    if dict(zip(base.hltv_match_id, base.source_date)) != _cohort_dates(cohorts, ("train", "validation")):
        raise ValueError("Frozen baseline differs from the complete development native population")
    bayesian, bayesian_engine = build_bayesian_features(games, history_lag_days=protocol["history"]["source_day_lag"])
    if not set(BAYES_FEATURE_COLUMNS).issubset(bayesian.columns) or len(BAYES_FEATURE_COLUMNS) != 16:
        raise ValueError("Development requires all sixteen frozen Bayesian channels")
    features = join_development_features(base, bayesian)
    base_engine = _advance(SeriesHistoryEngine(), games)
    if base_engine.history_through != bayesian_engine.history_through:
        raise ValueError("Development history banks have different chronological boundaries")
    parts = protocol["partitions"]
    train = features.loc[features.source_date.between(parts["training_fit_start"], parts["train_end"])].copy()
    validation = features.loc[features.source_date.between(parts["validation_start"], parts["validation_end"])].copy()
    if len(train) != baseline["selection"]["training_matches"] or set(validation.hltv_match_id) != set(_cohort_dates(cohorts, ("validation",))):
        raise ValueError("Coefficient TRAIN or validation population differs from the original experiment")
    consensus, market = _validation_consensus(paths, validation, baseline)
    model, evidence, forecasts = select_development_model(
        train, validation, {row["hltv_match_id"] for row in consensus}, recipes, RULES,
        train_end=parts["train_end"], validation_end=parts["validation_end"])
    comparison = _comparison(validation, consensus, market, forecasts, evidence, baseline, protocol)
    evidence["train_only_expanding_diagnostics"] = _rolling_diagnostics(model, recipes, train, protocol)
    evidence.update({"competitiveness_qualified": False, "qualified": False,
                     "test_forecasts_or_losses_observed": False, "availability_limits": protocol["limitations"],
                     "comparison_artifact": "development_comparison.json"})
    predictions = validation[IDENTITY_COLUMNS].copy()
    predictions["partition"] = "validation"
    predictions["model_selected"] = evidence["selected_model"]
    predictions["prob_team1_win"] = forecasts[evidence["selected_model"]]
    predictions["prob_team2_win"] = 1 - forecasts[evidence["selected_model"]]
    predictions["incumbent_prob_team1_win"] = forecasts[INCUMBENT]
    consensus_probability = {row["hltv_match_id"]: row["p_bookmaker"] for row in consensus}
    predictions["in_fixed_market_cohort"] = predictions.hltv_match_id.isin(consensus_probability)
    predictions["consensus_prob_team1_win"] = predictions.hltv_match_id.map(consensus_probability)
    candidate_predictions = validation[IDENTITY_COLUMNS].copy()
    for name, values in forecasts.items():
        candidate_predictions[name] = values
    # Recheck the opening freeze before publishing any selected artifact.
    _check_frozen(started, args.protocol)
    if digest(output / "development_started_manifest.json") != started_sha256:
        raise ValueError("Opening phase manifest changed during development")
    features.to_csv(output / "development_features.csv", index=False, mode="x")
    bayesian.to_csv(output / "bayesian_features.csv", index=False, mode="x")
    save_json(output / "development_native_ids.json", {
        "history_native_ids": [game["hltv_match_id"] for game in games],
        "train_fit": sorted(train.hltv_match_id, key=int), "validation": sorted(validation.hltv_match_id, key=int),
        "market_validation": sorted(consensus_probability, key=int), "test_included": False})
    save_pickle(output / "selected_model.pkl", model)
    save_pickle(output / "development_history_state.pkl", {
        "schema_version": 2, "engine": base_engine, "bayesian_engine": bayesian_engine,
        "standings": standings, "history_through": base_engine.history_through,
        "model_sha256": digest(output / "selected_model.pkl"), "model_selected": model.recipe["name"],
        "protocol_sha256": started["protocol_sha256"], "feature_columns": ALL_FEATURES,
        "competitiveness_qualified": False, "availability_limits": protocol["limitations"]})
    predictions.to_csv(output / "validation_predictions.csv", index=False, mode="x")
    candidate_predictions.to_csv(output / "candidate_validation_predictions.csv", index=False, mode="x")
    save_json(output / "selection.json", evidence)
    save_json(output / "development_comparison.json", comparison)
    save_json(output / "development_manifest.json", {
        "phase": "development_completed", "completed_at_utc": _now(), "qualified": False,
        "development_started_manifest_sha256": started_sha256, "protocol_sha256": started["protocol_sha256"],
        "history_audit": audit, "history_through": base_engine.history_through,
        "selected_model": model.recipe["name"], "candidate_count": len(recipes),
        "sources": sources, "consumed_inputs": consumed, "environment": started["environment"],
        "test_feature_replay_performed": False, "test_forecasts_or_losses_observed": False,
        "artifacts": {name: digest(output / name) for name in DEVELOPMENT_ARTIFACTS if name != "development_manifest.json"}})
    _check_frozen(started, args.protocol)
    save_json(output / "selection_freeze.json", {
        "phase": "selection_frozen", "frozen_at_utc": _now(), "protocol_sha256": started["protocol_sha256"],
        "sources": sources, "consumed_inputs": consumed, "environment": started["environment"],
        "selected_recipe": model.recipe, "model_sha256": digest(output / "selected_model.pkl"),
        "artifacts": {name: digest(output / name) for name in DEVELOPMENT_ARTIFACTS},
        "test_forecasts_or_losses_observed": False, "competitiveness_qualified": False,
        "train_plus_validation_refit": False})
    print(json.dumps({"phase": "selected_and_frozen", "selected_model": model.recipe["name"],
                      "qualified": False, "development_comparison": str(output / "development_comparison.json")}))
    return 0


def publish(args):
    output = Path(args.output_dir).resolve()
    if any((output / name).exists() for name in
           ("publish_started_manifest.json", "publish_manifest.json", "series_inference_state.pkl")):
        raise ValueError("This experiment was already published or publication started; no overwrites")
    freeze_path = output / "selection_freeze.json"
    freeze_sha256 = digest(freeze_path)
    freeze = read_json(freeze_path)
    _check_frozen(freeze, args.protocol)
    verify_hashes(freeze["artifacts"], output)
    protocol = read_json(args.protocol)
    paths = _paths(args, protocol)
    standings, consumed = _validate_inputs(protocol, paths)
    if consumed != freeze["consumed_inputs"]:
        raise ValueError("Publish inputs/paths differ from the pre-replay development freeze")
    if digest(output / "selected_model.pkl") != freeze["model_sha256"]:
        raise ValueError("Raw selected model differs from frozen pairing")
    model = load_pickle(output / "selected_model.pkl")
    history = load_pickle(output / "development_history_state.pkl")
    if (model.recipe != freeze["selected_recipe"] or model.train_end != protocol["partitions"]["train_end"] or
            history["model_sha256"] != freeze["model_sha256"] or
            history["model_selected"] != model.recipe["name"] or
            history["protocol_sha256"] != freeze["protocol_sha256"] or
            history["schema_version"] != 2):
        raise ValueError("Selected TRAIN-only model and saved development state are not paired")
    engine, bayesian_engine = history["engine"], history["bayesian_engine"]
    if (engine.history_through != history["history_through"] or
            engine.history_through != bayesian_engine.history_through or
            not engine.history_through or engine.history_through > protocol["development_end"]):
        raise ValueError("Saved histories violate the frozen development boundary")
    save_json(output / "publish_started_manifest.json", {
        "phase": "publish_started", "started_at_utc": _now(), "selection_freeze_sha256": freeze_sha256,
        "protocol_sha256": freeze["protocol_sha256"], "sources": freeze["sources"],
        "consumed_inputs": consumed, "history_through_before": engine.history_through,
        "test_feature_replay_allowed": False, "test_forecasts_or_losses_observed": False})
    cohorts = read_json(paths["cohorts"])["cohorts"]
    games, audit = _history(paths, protocol, cohorts, development_only=False)
    trailing = [game for game in games if game["source_date"] > engine.history_through]
    # No feature builder, feature query, classifier forecast or TEST quality
    # computation occurs here: completed games only enter update_day.
    _advance(engine, trailing)
    _advance(bayesian_engine, trailing)
    if engine.history_through != bayesian_engine.history_through or engine.history_through != max(game["source_date"] for game in games):
        raise ValueError("Published history banks do not cover the entire frozen archive")
    _check_frozen(freeze, args.protocol)
    verify_hashes(freeze["artifacts"], output)
    if digest(freeze_path) != freeze_sha256:
        raise ValueError("Selection freeze changed during publication")
    save_pickle(output / "series_inference_state.pkl", {
        "schema_version": 2, "engine": engine, "standings": standings, "bayesian_engine": bayesian_engine,
        "model_sha256": freeze["model_sha256"], "model_selected": model.recipe["name"],
        "history_through": engine.history_through, "feature_columns": ALL_FEATURES,
        "protocol_sha256": freeze["protocol_sha256"], "competitiveness_qualified": False,
        "availability_limits": protocol["limitations"]})
    save_json(output / "publish_manifest.json", {
        "phase": "published", "published_at_utc": _now(), "schema_version": 2,
        "selected_model": model.recipe["name"], "history_through": engine.history_through,
        "model_sha256": freeze["model_sha256"], "selection_freeze_sha256": freeze_sha256,
        "protocol_sha256": freeze["protocol_sha256"], "sources": freeze["sources"], "consumed_inputs": consumed,
        "environment": _environment(), "qualified": False, "competitiveness_qualified": False,
        "state_update_only": True, "test_feature_replay_performed": False,
        "test_forecasts_or_losses_observed": False, "new_confirmatory_quality": None,
        "archive_games": len(games), "newly_updated_games": len(trailing), "history_audit": audit,
        "availability_limits": protocol["limitations"],
        "artifacts": {name: digest(output / name) for name in
                      ("series_inference_state.pkl", "publish_started_manifest.json")}})
    print(json.dumps({"phase": "published", "selected_model": model.recipe["name"],
                      "history_through": engine.history_through, "qualified": False}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "publish"))
    parser.add_argument("--protocol", default=str(ROOT / "reports/cs2-professional-model-protocol-2026-10-10.json"))
    for option in INPUT_OPTIONS:
        parser.add_argument("--" + option.replace("_", "-"), default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/professional_model_20261010"))
    args = parser.parse_args(argv)
    try:
        return {"develop": develop, "publish": publish}[args.phase](args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
