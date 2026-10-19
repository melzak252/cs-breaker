"""Freeze same-cohort exploratory before/after evidence without fitting or unpickling.

Run from the repository root: python scripts/evaluate_model_research.py.
Negative paired differences favor the after model. TEST has already been observed;
these descriptive intervals are not fresh confirmation, significance or ROI evidence.
Original and regenerated feature snapshots certify every TRAIN/VALIDATION/TEST
identity and original player predictor; augmented archive columns may differ.
Declared full-schema recipes are checked against the frozen protocol and baseline
schema; opaque pickles are not evidence of fitted-model provenance. All output
paths are checked for input and destination aliases, including existing hard links.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
from collections import defaultdict
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.models.metrics import brier_score, evaluate_predictions, log_loss_score

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 42


def iso_day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError(f"Invalid ISO source day: {value!r}")
    return value


def load_predictions(content):
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    required = {"hltv_match_id", "date", "source_date", "team1", "team2", "target",
                "prob_team1_win", "prob_team2_win"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Prediction CSV is missing required cohort/probability columns")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("Prediction CSV has duplicate column names")
    rows = []
    for raw in reader:
        match_id = raw["hltv_match_id"]
        if not re.fullmatch(r"[1-9][0-9]*", match_id or ""):
            raise ValueError(f"Non-native match ID: {match_id!r}")
        day = iso_day(raw["date"])
        if iso_day(raw["source_date"]) != day:
            raise ValueError(f"Date/source-date disagreement for {match_id}")
        if not raw["team1"] or not raw["team2"] or raw["team1"] == raw["team2"]:
            raise ValueError(f"Invalid team orientation for {match_id}")
        if raw["target"] not in ("0", "1"):
            raise ValueError(f"Nonbinary target for {match_id}")
        p1, p2 = float(raw["prob_team1_win"]), float(raw["prob_team2_win"])
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in (p1, p2)):
            raise ValueError(f"Invalid probability for {match_id}")
        if not math.isclose(p1 + p2, 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Noncomplementary probabilities for {match_id}")
        rows.append({"hltv_match_id": match_id, "date": day, "source_date": day,
                     "team1": raw["team1"], "team2": raw["team2"], "target": int(raw["target"]),
                     "prob_team1_win": p1, "prob_team2_win": p2})
    if not rows:
        raise ValueError("TEST prediction cohort is empty")
    _unique_rows(rows)
    return rows


def _unique_rows(rows):
    indexed = {}
    for row in rows:
        key = row["hltv_match_id"]
        if key in indexed:
            raise ValueError(f"Duplicate native match ID: {key}")
        indexed[key] = row
    return indexed


def pair_predictions(before, after):
    left, right = _unique_rows(before), _unique_rows(after)
    if set(left) != set(right):
        raise ValueError(f"Changed TEST cohort: missing={sorted(set(left)-set(right))}, "
                         f"extra={sorted(set(right)-set(left))}")
    pairs = []
    for key in sorted(left, key=lambda k: (left[k]["source_date"], int(k))):
        old, new = left[key], right[key]
        for field in ("date", "source_date", "team1", "team2", "target"):
            if old[field] != new[field]:
                raise ValueError(f"Changed {field} for native match ID {key}")
        pairs.append({field: old[field] for field in
                      ("hltv_match_id", "date", "source_date", "team1", "team2", "target")}
                     | {"p_before": old["prob_team1_win"], "p_after": new["prob_team1_win"]})
    return pairs


def validate_original_feature_hash(content, protocol, reproduction):
    digest = hashlib.sha256(content).hexdigest()
    if digest != protocol["baseline"]["feature_sha256"] or digest != reproduction["feature_sha256"]:
        raise ValueError("Original feature snapshot hash disagrees with frozen protocol/reproduction")


def load_features(content, player_columns):
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    required = {"hltv_match_id", "date", "team1", "team2", "target", *player_columns}
    if not required.issubset(fields) or len(fields) != len(set(fields)):
        raise ValueError("Feature CSV has missing required or duplicate columns")
    rows = []
    for raw in reader:
        match_id = raw["hltv_match_id"]
        if not re.fullmatch(r"[1-9][0-9]*", match_id or ""):
            raise ValueError(f"Non-native feature match ID: {match_id!r}")
        day = iso_day(raw["date"])
        source_day = iso_day(raw["source_date"]) if "source_date" in fields else day
        if source_day != day:
            raise ValueError(f"Feature date/source-date disagreement for {match_id}")
        if (not raw["team1"].strip() or not raw["team2"].strip()
                or raw["team1"] == raw["team2"]):
            raise ValueError(f"Invalid feature team orientation for {match_id}")
        if raw["target"] not in ("0", "1"):
            raise ValueError(f"Nonbinary feature target for {match_id}")
        values = {name: float(raw[name]) for name in player_columns}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Nonfinite original player features for {match_id}")
        rows.append({"hltv_match_id": match_id, "date": day, "source_date": source_day,
                     "team1": raw["team1"], "team2": raw["team2"],
                     "target": int(raw["target"]), **values})
    _unique_rows(rows)
    return rows


def validate_feature_cohort(before_content, after_content, before, after, protocol, test_rows):
    columns = before["feature_columns"]
    identity_fields = ("date", "source_date", "team1", "team2", "target")
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(name, str) or not name.strip() for name in columns)
            or len(set(columns)) != len(columns)
            or set(columns).intersection({"hltv_match_id", *identity_fields})):
        raise ValueError("Invalid original player feature column declaration")
    old = _unique_rows(load_features(before_content, columns))
    new = _unique_rows(load_features(after_content, columns))
    if len(old) != protocol["cohort"]["eligible_series"]:
        raise ValueError("Original feature snapshot differs from frozen eligible cohort count")
    if set(old) != set(new):
        raise ValueError(f"Changed full feature cohort: missing={sorted(set(old)-set(new))}, "
                         f"extra={sorted(set(new)-set(old))}")
    for key, original in old.items():
        regenerated = new[key]
        for field in identity_fields:
            if original[field] != regenerated[field]:
                raise ValueError(f"Changed full-cohort {field} for native match ID {key}")
        for name in columns:
            # No tolerance: serialization formatting may change, numeric values may not.
            if original[name] != regenerated[name]:
                raise ValueError(f"Changed original player feature {name} for native match ID {key}")
    memberships = {}
    for phase, indexed, metrics in (("before", old, before), ("after", new, after)):
        partition_ids, assigned = {}, set()
        for partition in ("train", "validation", "test"):
            split = metrics["splits"][partition]
            days = set(split["dates"])
            members = sorted((row for row in indexed.values() if row["source_date"] in days),
                             key=lambda row: (row["source_date"], int(row["hltv_match_id"])))
            ids = [row["hltv_match_id"] for row in members]
            if assigned.intersection(ids):
                raise ValueError(f"Overlapping {phase} feature partition IDs")
            assigned.update(ids)
            counts = {str(label): sum(row["target"] == label for row in members) for label in (0, 1)}
            if (len(ids) != split["n_series"] or counts != split["class_counts"]
                    or sorted({row["source_date"] for row in members}) != sorted(days)):
                raise ValueError(f"{phase} feature {partition} membership differs from metrics")
            partition_ids[partition] = ids
        if assigned != set(indexed):
            raise ValueError(f"Unassigned {phase} feature cohort IDs outside metadata dates")
        memberships[phase] = partition_ids
    if memberships["before"] != memberships["after"]:
        raise ValueError("Changed exact per-partition feature IDs")
    predictions = _unique_rows(test_rows)
    if set(predictions) != set(memberships["before"]["test"]):
        raise ValueError("Stored TEST predictions differ from full feature cohort TEST IDs")
    for key, prediction in predictions.items():
        for field in identity_fields:
            if prediction[field] != old[key][field]:
                raise ValueError(f"Stored TEST prediction {field} differs from feature snapshot for {key}")
    return {"n_series": len(old), "partition_ids": memberships["before"],
            "original_player_feature_columns": columns,
            "numeric_comparison": "Exact equality of finite parsed Python floats; no tolerance; textual formatting and column/row order may differ.",
            "source_day_policy": "Explicit source_date must equal date; absent source_date uses date.",
            "unchanged_all_ids_dates_orientation_targets": True,
            "unchanged_original_player_features": True,
            "exact_partition_membership": True, "stored_test_matches_feature_cohort": True}


def score_rows(rows, probability):
    labels = [row["target"] for row in rows]
    result = evaluate_predictions(labels, [row[probability] for row in rows])
    if not rows:
        for field in ("log_loss", "brier", "accuracy", "roc_auc", "ece"):
            result[field] = None
    elif len(set(labels)) == 1:
        result["roc_auc"] = None
    return result


def _percentile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_differences(rows):
    result = {}
    for name, scorer in (("log_loss", log_loss_score), ("brier", brier_score)):
        blocks = defaultdict(list)
        for row in rows:
            blocks[row["source_date"]].append(
                scorer([row["target"]], [row["p_after"]])
                - scorer([row["target"]], [row["p_before"]]))
        totals = [(sum(blocks[day]), len(blocks[day])) for day in sorted(blocks)]
        item = {"estimate": None, "confidence_interval_95": None,
                "source_day_count": len(totals), "n_series": len(rows),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                "method": "source-calendar-day-block percentile bootstrap; match-weighted",
                "direction": f"after minus before {name}; negative favors after",
                "confidence_level": 0.95, "interval_crosses_zero": None,
                "singleton_day_degeneracy": len(totals) == 1,
                "statistical_significance_claim": False}
        if rows:
            item["estimate"] = sum(total for total, _ in totals) / len(rows)
            rng = random.Random(BOOTSTRAP_SEED)
            draws = []
            for _ in range(BOOTSTRAP_REPLICATES):
                total = count = 0
                for _ in totals:
                    block_sum, block_count = totals[rng.randrange(len(totals))]
                    total += block_sum
                    count += block_count
                draws.append(total / count)
            draws.sort()
            interval = [_percentile(draws, 0.025), _percentile(draws, 0.975)]
            item["confidence_interval_95"] = interval
            item["interval_crosses_zero"] = interval[0] <= 0 <= interval[1]
        result[name] = item
    return result


def validate_recipe_schema(before_metrics, after_metrics, protocol):
    """Certify declared recipes, not the provenance of opaque fitted artifacts."""
    try:
        original = before_metrics["feature_columns"]
        archive = protocol["archive_features"]["features"]
        if (not isinstance(original, list) or not original
                or any(not isinstance(column, str) or not column for column in original)
                or len(set(original)) != len(original)
                or not isinstance(archive, list)
                or any(not isinstance(column, str) or not column for column in archive)
                or len(set(original + archive)) != len(original + archive)):
            raise ValueError("Invalid frozen original/archive feature schema")
        if after_metrics["feature_columns"] != original + archive:
            raise ValueError("After feature_columns differ from the frozen ordered full schema")

        frozen = protocol["candidate_set"]
        order = (frozen["original_candidates"] + frozen["new_raw_candidates"]
                 + frozen["calibrated_candidates"]
                 + [item["name"] for item in frozen["compact_candidates"]]
                 + [item["name"] for item in frozen["fixed_blends"]])
        recipes = after_metrics["candidate_parameters"]
        if (not isinstance(recipes, dict) or set(recipes) != set(order)
                or len(order) != frozen["maximum_full_schema_candidates"]):
            raise ValueError("Candidate recipe metadata differs from the frozen candidate set")
        if after_metrics["selection"]["candidate_order"] != order:
            raise ValueError("Candidate recipe order differs from the frozen declared order")
        if after_metrics["selection"]["fit_partition"] != "train":
            raise ValueError("Candidate recipes and scalers must declare TRAIN-only fitting")

        expected = {}

        def declare(name, columns, family, transform, parameters, **fields):
            expected[name] = {"name": name, "feature_columns": columns, "family": family,
                              "transform": transform, "parameters": parameters, **fields}

        for name in frozen["original_candidates"] + frozen["new_raw_candidates"]:
            if name.startswith("pure_"):
                column = name.removeprefix("pure_") + "_prob"
                declare(name, [column], "rating_channel", "identity", {},
                        probability_column=column)
        # Baseline v3 did not serialize recipes. These are its frozen source
        # hyperparameters, with its explicit original predictor schema.
        declare("logistic_regression", original, "classifier", "identity",
                {"C": 1.0, "fit_intercept": True, "with_mean": True,
                 "max_iter": 1000, "random_state": 42})
        declare("gradient_boosting", original, "classifier", "identity",
                {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 15,
                 "min_samples_leaf": 20, "early_stopping": False,
                 "validation_fraction": None, "random_state": 42})
        epsilon = frozen["numeric_logit_epsilon"]
        for name in frozen["calibrated_candidates"]:
            column = name.removeprefix("calibrated_") + "_prob"
            declare(name, [column], "calibrated_rating", "clipped_logit",
                    {"slope_bounds": [0.0, 3.0], "fit_intercept": False,
                     "epsilon": epsilon}, probability_column=column)
        for item in frozen["compact_candidates"]:
            name = item["name"]
            transform = {"ridge_rating_logits": "clipped_logit",
                         "ridge_strength_form": "identity"}[name]
            parameters = {"C": 0.1, "fit_intercept": False, "with_mean": False,
                          "random_state": 42, "max_iter": 1000}
            if transform == "clipped_logit":
                parameters["epsilon"] = epsilon
            declare(name, item["features"], "compact_ridge", transform, parameters)
        for item in frozen["fixed_blends"]:
            declare(item["name"], item["probabilities"], "probability_blend",
                    "identity", {"weights": item["weights"]}, weights=item["weights"])
        if set(expected) != set(order):
            raise ValueError("Frozen candidate declarations are inconsistent")

        def finite_vector(values, size, *, positive=False):
            return (isinstance(values, list) and len(values) == size
                    and all(not isinstance(value, bool) and isinstance(value, (int, float))
                            and math.isfinite(value) and (not positive or value > 0)
                            for value in values))

        for name in order:
            recipe, declaration = recipes[name], expected[name]
            if not isinstance(recipe, dict):
                raise ValueError(f"Missing candidate recipe metadata: {name}")
            for field, value in declaration.items():
                if field not in recipe or recipe[field] != value:
                    raise ValueError(f"Frozen candidate recipe disagreement: {name}.{field}")
            # Python equates True with 1; recipe flags/numeric parameters must
            # still have their declared meanings.
            for key, value in declaration["parameters"].items():
                actual = recipe["parameters"][key]
                if ((isinstance(value, bool) and type(actual) is not bool)
                        or (isinstance(value, (int, float)) and not isinstance(value, bool)
                            and (isinstance(actual, bool) or not isinstance(actual, (int, float))))):
                    raise ValueError(f"Invalid candidate parameter type: {name}.{key}")
                if isinstance(value, list) and not finite_vector(actual, len(value)):
                    raise ValueError(f"Invalid candidate numeric parameter array: {name}.{key}")
            if any(column not in original + archive for column in recipe["feature_columns"]):
                raise ValueError(f"Candidate uses undeclared features: {name}")
            fitted_fields = set()
            if declaration["family"] == "calibrated_rating":
                fitted_fields = {"slope"}
                slope = recipe.get("slope")
                if (isinstance(slope, bool) or not isinstance(slope, (int, float))
                        or not math.isfinite(slope) or not 0 <= slope <= 3):
                    raise ValueError(f"Invalid fitted calibration slope: {name}")
            elif name == "logistic_regression" or declaration["family"] == "compact_ridge":
                fitted_fields = {"coefficients", "intercept", "scaler"}
                size = len(declaration["feature_columns"])
                coefficients, scaler = recipe.get("coefficients"), recipe.get("scaler")
                if (not isinstance(coefficients, list) or len(coefficients) != 1
                        or not finite_vector(coefficients[0], size)
                        or not finite_vector(recipe.get("intercept"), 1)
                        or not isinstance(scaler, dict) or set(scaler) != {"mean", "scale"}
                        or not finite_vector(scaler["mean"], size)
                        or not finite_vector(scaler["scale"], size, positive=True)):
                    raise ValueError(f"Missing or inconsistent fitted classifier metadata: {name}")
                if declaration["family"] == "compact_ridge" and recipe["intercept"] != [0.0]:
                    raise ValueError(f"Compact recipe has a nonzero fitted intercept: {name}")
            if set(recipe) != set(declaration) | fitted_fields:
                raise ValueError(f"Missing or unexpected candidate recipe fields: {name}")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or invalid frozen recipe metadata: {error}") from error


def validate_output_paths(input_paths, destination_paths):
    """Reject input/output and output/output path, symlink and hard-link aliases."""
    inputs, destinations = list(input_paths), list(destination_paths)

    def aliases(left, right):
        if left.resolve() == right.resolve():
            return True
        return left.exists() and right.exists() and left.samefile(right)

    for index, destination in enumerate(destinations):
        for source in inputs:
            if aliases(destination, source):
                raise ValueError(f"Output would overwrite an input alias: {destination} -> {source}")
        for previous in destinations[:index]:
            if aliases(destination, previous):
                raise ValueError(f"Output destinations alias each other: {destination} -> {previous}")


def validate_selection(metrics, protocol, *, after):
    selection = metrics["selection"]
    if (selection["partition"], selection["metric"], selection["fit_partition"]) != (
            "validation", "log_loss", "train"):
        raise ValueError("Selection must be VALIDATION log-loss argmin with TRAIN-only fit")
    candidates = selection["candidate_log_loss"]
    if not candidates:
        raise ValueError("No validation selection candidates")
    for name, value in candidates.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Invalid validation score: {name}")
        if not math.isclose(value, metrics["models"][name]["validation"]["log_loss"],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Candidate validation score disagreement: {name}")
        for partition in ("validation", "test"):
            if metrics["models"][name][partition]["n_samples"] != metrics["splits"][partition]["n_series"]:
                raise ValueError(f"Candidate scored a narrowed {partition} sample: {name}")
    selected = metrics["selected_model"]
    minimum = min(candidates.values())
    if selected not in candidates or candidates[selected] != minimum:
        raise ValueError("Selected model is not the VALIDATION argmin")
    if selection["selected_log_loss"] != candidates[selected]:
        raise ValueError("Selected validation score disagreement")
    if after:
        recipes = protocol["candidate_set"]
        order = (recipes["original_candidates"] + recipes["new_raw_candidates"]
                 + recipes["calibrated_candidates"] + [r["name"] for r in recipes["compact_candidates"]]
                 + [r["name"] for r in recipes["fixed_blends"]])
        if set(candidates) != set(order):
            raise ValueError("After candidates differ from the frozen full-schema candidate set")
        tied = [name for name in order if candidates[name] == minimum]
        expected = "pure_elo" if "pure_elo" in tied else tied[0]
        if selected != expected:
            raise ValueError("Selected model violates the frozen exact-tie policy")


def validate_splits(before, after, protocol, rows):
    prior_end = None
    all_days = set()
    for name in ("train", "validation", "test"):
        old, new = before["splits"][name], after["splits"][name]
        for field in ("n_series", "n_days", "dates", "start_date", "end_date", "class_counts"):
            if old[field] != new[field]:
                raise ValueError(f"Changed {name} split {field}")
        days = [iso_day(day) for day in old["dates"]]
        if not days or days != sorted(set(days)) or len(days) != old["n_days"]:
            raise ValueError(f"Invalid source-day membership for {name}")
        if (days[0], days[-1]) != (old["start_date"], old["end_date"]):
            raise ValueError(f"Invalid split date boundaries: {name}")
        if prior_end is not None and days[0] <= prior_end:
            raise ValueError("Splits are not strictly chronological whole-day partitions")
        if all_days.intersection(days):
            raise ValueError("Source-day membership overlaps")
        prior_end = days[-1]
        all_days.update(days)
        cohort = protocol["cohort"]
        if old["n_series"] != cohort[f"{name}_rows"] or [days[0], days[-1]] != cohort[f"{name}_dates"]:
            raise ValueError(f"Split differs from frozen protocol: {name}")
    test = before["splits"]["test"]
    counts = {str(label): sum(row["target"] == label for row in rows) for label in (0, 1)}
    if len(rows) != test["n_series"] or sorted({row["source_date"] for row in rows}) != test["dates"]:
        raise ValueError("CSV TEST membership differs from metrics")
    if counts != test["class_counts"]:
        raise ValueError("CSV TEST class counts differ from metrics")
    for metrics in (before, after):
        if metrics["support"]["n_series"] != protocol["cohort"]["eligible_series"]:
            raise ValueError("Changed total eligible cohort count")


def test_blocks(rows):
    days = sorted({row["source_date"] for row in rows})
    quotient, remainder = divmod(len(days), 3)
    result, offset = [], 0
    for index in range(3):
        count = quotient + (index < remainder)
        block_days = days[offset:offset + count]
        offset += count
        members = set(block_days)
        block = [row for row in rows if row["source_date"] in members]
        before, after = score_rows(block, "p_before"), score_rows(block, "p_after")
        result.append({"block": index + 1, "descriptive_only": True,
                       "source_days": block_days, "n_series": len(block), "before": before, "after": after,
                       "after_minus_before": {key: after[key] - before[key] if block else None
                                              for key in ("log_loss", "brier")}})
    return result


def _json(content):
    return json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON number: {value}")))


def summarize(args):
    before_dir = Path(args.before_dir)
    paths = {"protocol": Path(args.protocol), "before_reproduction": before_dir / "reproduction.json",
             "before_metrics": before_dir / "metrics.json", "before_predictions": before_dir / "test_predictions.csv",
             "before_features": before_dir / "features.csv", "after_features": Path(args.after_features),
             "before_model": before_dir / "model.pkl", "after_metrics": Path(args.after_metrics),
             "after_predictions": Path(args.after_predictions), "after_model": Path(args.after_model),
             "after_state": Path(args.after_state)}
    output = Path(args.output_dir)
    destinations = {name: output / filename for name, filename in (("after_metrics", "metrics.json"), ("after_predictions", "test_predictions.csv"), ("after_model", "model.pkl"), ("after_state", "state.pkl"), ("after_features", "features.csv"))}
    generated = [output / name for name in ("summary.json", "paired_model_predictions.csv", "manifest.json")]
    validate_output_paths(paths.values(), list(destinations.values()) + generated)
    # Snapshot bytes once: validation, hashes and copies describe exactly the same inputs.
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    protocol = _json(inputs["protocol"])
    before, after = _json(inputs["before_metrics"]), _json(inputs["after_metrics"])
    validate_recipe_schema(before, after, protocol)
    reproduction = _json(inputs["before_reproduction"])
    old_rows, new_rows = load_predictions(inputs["before_predictions"]), load_predictions(inputs["after_predictions"])
    pairs = pair_predictions(old_rows, new_rows)
    validate_splits(before, after, protocol, old_rows)
    validate_selection(before, protocol, after=False)
    validate_selection(after, protocol, after=True)
    if before["selected_model"] != reproduction["selected_model"] or before["selected_model"] != protocol["baseline"]["selected_model"]:
        raise ValueError("Frozen baseline selected model disagreement")
    if reproduction["split_counts"] != {name: before["splits"][name]["n_series"] for name in ("train", "validation", "test")}:
        raise ValueError("Baseline reproduction split counts disagree")
    if reproduction["feature_sha256"] != protocol["baseline"]["feature_sha256"] or reproduction["source_module_sha256"] != protocol["baseline"]["player_baseline_source_sha256"]:
        raise ValueError("Baseline reproduction hashes disagree with frozen protocol")
    validate_original_feature_hash(inputs["before_features"], protocol, reproduction)
    cohort_validation = validate_feature_cohort(
        inputs["before_features"], inputs["after_features"], before, after, protocol, old_rows)
    cohort_validation["original_feature_hash_matches_frozen_records"] = True
    scores = {"before": score_rows(pairs, "p_before"), "after": score_rows(pairs, "p_after")}
    for phase, metrics in (("before", before), ("after", after)):
        reported = metrics["models"][metrics["selected_model"]]["test"]
        for key in ("log_loss", "brier", "accuracy", "ece", "roc_auc", "n_samples"):
            if not math.isclose(scores[phase][key], reported[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{phase} selected CSV/metrics disagreement: {key}")
    for key in ("log_loss", "brier", "accuracy", "ece", "roc_auc", "n_samples"):
        if not math.isclose(scores["before"][key], reproduction["selected_test"][key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Baseline reproduction scoring disagreement: {key}")
    for value in (reproduction["selected_validation"], protocol["baseline"]["validation_log_loss"]):
        if value != before["selection"]["selected_log_loss"]:
            raise ValueError("Baseline validation reproduction disagreement")
    if not math.isclose(scores["before"]["log_loss"], protocol["baseline"]["test_log_loss"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Baseline TEST protocol score disagreement")
    differences = paired_differences(pairs)
    summary = {"schema_version": 1, "selected_models": {phase: metrics["selected_model"] for phase, metrics in (("before", before), ("after", after))},
               "selected_model_test_metrics": scores,
               "selected_model_validation_metrics": {phase: metrics["models"][metrics["selected_model"]]["validation"] for phase, metrics in (("before", before), ("after", after))},
               "paired_after_minus_before": differences,
               "observed_log_loss_improvement": differences["log_loss"]["estimate"] < 0,
               "observed_brier_improvement": differences["brier"]["estimate"] < 0,
               "chronological_test_blocks": test_blocks(pairs),
               "split_membership": {name: {key: before["splits"][name][key] for key in ("n_series", "n_days", "dates", "class_counts")}
                                    | {"hltv_match_ids": cohort_validation["partition_ids"][name]}
                                    for name in ("train", "validation", "test")},
               "full_cohort_validation": cohort_validation,
               "validation_checks": {"exact_unique_native_id_test_cohort": True, "unchanged_test_labels_dates_orientation": True,
                                     "unchanged_chronological_split_days_counts": True, "selected_validation_argmin": True,
                                     "original_feature_hash_matches_frozen_records": True,
                                     "exact_full_cohort_and_partition_ids": True,
                                     "frozen_full_feature_schema": True,
                                     "frozen_candidate_recipe_metadata": True,
                                     "input_output_and_output_output_paths_disjoint": True,
                                     "unchanged_original_player_features": True},
               "limits": {"old_test_already_seen": True, "fresh_confirmation_claim": False, "statistical_significance_claim": False,
                          "roi_claim": False, "forecast_kind": protocol["cohort"]["forecast_kind"],
                          "availability_limit": protocol["cohort"]["availability_limit"],
                          "training_selection": "TRAIN-only fit and VALIDATION-only argmin; no refit or selection performed by this summarizer.",
                          "membership_evidence": "Exact full-cohort and per-partition IDs, source dates, orientation, targets and original player predictors from frozen-hash original and regenerated feature snapshots.",
                          "pickle_policy": "Model/state archived and hashed as opaque bytes; never deserialized.",
                          "recipe_evidence": "Full ordered feature schema and all frozen candidate recipes are validated from declared metrics metadata, including TRAIN-only fit/scaler policy and fitted-array consistency. This does not independently prove that the opaque model/state pickles were fitted using those recipes or contain the declared coefficients, nor establish their training provenance.",
                          "interpretation": "Frozen exploratory comparison on previously observed TEST; descriptive uncertainty, not independent confirmation or promotion advice."}}
    sources = sorted(set(ROOT.joinpath("pipeline").rglob("*.py")) | {Path(__file__).resolve()})
    manifest = {"algorithm": "SHA256", "inputs": {name: {"path": str(paths[name]), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in inputs.items()},
                "source_code": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                "full_cohort_validation": cohort_validation}
    output.mkdir(parents=True, exist_ok=True)
    for name, destination in destinations.items():
        destination.write_bytes(inputs[name])
    paired_path = output / "paired_model_predictions.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["hltv_match_id", "date", "source_date", "team1", "team2", "target", "p_before", "p_after", "log_loss_after_minus_before", "brier_after_minus_before"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pairs:
            writer.writerow(row | {f"{name}_after_minus_before": scorer([row["target"]], [row["p_after"]]) - scorer([row["target"]], [row["p_before"]]) for name, scorer in (("log_loss", log_loss_score), ("brier", brier_score))})
    summary["manifest"] = "manifest.json"
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["outputs"] = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in list(destinations.values()) + [paired_path, output / "summary.json"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"selected_models": summary["selected_models"], "n_test_series": len(pairs),
            "selected_model_test_metrics": {phase: {key: scores[phase][key] for key in ("log_loss", "brier", "accuracy", "roc_auc")} for phase in scores},
            "paired_after_minus_before": differences, "old_test_already_seen": True,
            "fresh_confirmation_claim": False, "statistical_significance_claim": False, "roi_claim": False,
            "output_dir": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", default="artifacts/model_research_20261009/before")
    parser.add_argument("--after-metrics", default="artifacts/baseline_model_metrics.json")
    parser.add_argument("--after-predictions", default="artifacts/baseline_test_predictions.csv")
    parser.add_argument("--after-features", default="data/processed/match_features_with_ratings.csv")
    parser.add_argument("--after-model", default="artifacts/baseline_model.pkl")
    parser.add_argument("--after-state", default="artifacts/player_rating_state.pkl")
    parser.add_argument("--protocol", default="reports/cs2-model-experiment-protocol-2026-10-09.json")
    parser.add_argument("--output-dir", default="artifacts/model_research_20261009/after")
    args = parser.parse_args(argv)
    try:
        evidence = summarize(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"Evidence validation failed: {error}\n")
    print(json.dumps(evidence, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
