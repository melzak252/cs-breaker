"""Behavioral boundaries for offline same-cohort research evidence."""
import csv
import io
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_model_research import (
    load_predictions, pair_predictions, paired_differences, validate_feature_cohort,
    validate_original_feature_hash, validate_recipe_schema, validate_output_paths,
    summarize,
)
from pipeline.models.metrics import evaluate_predictions
from pipeline.models.player_baseline import _recipe


def row(match_id="2390001", **changes):
    value = {"hltv_match_id": match_id, "date": "2026-08-14", "source_date": "2026-08-14",
             "team1": "A", "team2": "B", "target": 1,
             "prob_team1_win": 0.75, "prob_team2_win": 0.25}
    value.update(changes)
    return value


@pytest.mark.parametrize("changes", [
    {"hltv_match_id": "2390002"}, {"target": 0}, {"date": "2026-08-15", "source_date": "2026-08-15"},
    {"team1": "B", "team2": "A"},
])
def test_pairing_rejects_changed_id_label_date_or_orientation(changes):
    with pytest.raises(ValueError):
        pair_predictions([row()], [row(**changes)])


def test_pairing_rejects_silent_intersection_and_duplicate_ids():
    with pytest.raises(ValueError):
        pair_predictions([row(), row("2390002")], [row()])
    with pytest.raises(ValueError):
        pair_predictions([row()], [row(), row()])


@pytest.mark.parametrize("changes", [
    {"target": 2}, {"prob_team1_win": float("nan")},
    {"prob_team1_win": 1.1, "prob_team2_win": -0.1}, {"prob_team2_win": 0.3},
    {"hltv_match_id": "synthetic:1"}, {"source_date": "2026-08-15"},
])
def test_csv_rejects_invalid_labels_probabilities_ids_and_dates(tmp_path, changes):
    path = tmp_path / "predictions.csv"
    value = row(**changes)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(value))
        writer.writeheader()
        writer.writerow(value)
    with pytest.raises(ValueError):
        load_predictions(path.read_bytes())


def test_paired_scores_keep_boundary_probabilities_finite_and_single_day_degeneracy():
    pairs = pair_predictions([row(prob_team1_win=0.0, prob_team2_win=1.0)],
                             [row(prob_team1_win=1.0, prob_team2_win=0.0)])
    scores = paired_differences(pairs)
    assert scores["log_loss"]["estimate"] < -30
    assert scores["brier"]["estimate"] == -1
    for score in scores.values():
        assert score["confidence_interval_95"] == [score["estimate"], score["estimate"]]
        assert score["singleton_day_degeneracy"] is True


def test_paired_difference_is_match_weighted_not_day_weighted():
    before = [row(), row("2390002"), row("2390003", date="2026-08-15", source_date="2026-08-15")]
    after = [row(prob_team1_win=1, prob_team2_win=0),
             row("2390002", prob_team1_win=1, prob_team2_win=0), before[2]]
    scores = paired_differences(pair_predictions(before, after))
    assert scores["brier"]["estimate"] == pytest.approx(-2 * 0.0625 / 3)
    assert scores["brier"]["source_day_count"] == 2


def feature_csv(rows):
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def feature_fixture():
    rows = [
        {"hltv_match_id": str(2390001 + index), "date": day, "team1": "A", "team2": "B",
         "target": index % 2, "elo_prob": 0.5}
        for index, day in enumerate(("2026-06-07", "2026-08-13", "2026-08-14"))
    ]
    metrics = {"feature_columns": ["elo_prob"], "splits": {
        name: {"dates": [member["date"]], "n_series": 1,
               "class_counts": {"0": int(member["target"] == 0), "1": int(member["target"] == 1)}}
        for name, member in zip(("train", "validation", "test"), rows)
    }}
    return rows, metrics, {"cohort": {"eligible_series": 3}}


@pytest.mark.parametrize("partition", [0, 1])
@pytest.mark.parametrize("changes", [
    {"hltv_match_id": "2390099"}, {"target": 1},
    {"team1": "B", "team2": "A"}, {"date": "2026-06-08"},
])
def test_full_cohort_rejects_training_and_validation_identity_changes(partition, changes):
    rows, metrics, protocol = feature_fixture()
    changed = [dict(member) for member in rows]
    if "target" in changes:
        changes = {"target": 1 - rows[partition]["target"]}
    changed[partition].update(changes)
    with pytest.raises(ValueError):
        validate_feature_cohort(feature_csv(rows), feature_csv(changed), metrics, metrics,
                                protocol, [row(target=0, date="2026-08-14") | {"hltv_match_id": "2390003"}])


@pytest.mark.parametrize("partition", [0, 1, 2])
def test_full_cohort_rejects_changed_original_player_values(partition):
    rows, metrics, protocol = feature_fixture()
    changed = [dict(member) for member in rows]
    changed[partition]["elo_prob"] = 0.5000000000000001
    with pytest.raises(ValueError, match="Changed original player feature"):
        validate_feature_cohort(feature_csv(rows), feature_csv(changed), metrics, metrics,
                                protocol, [row("2390003", target=0)])


def test_full_cohort_records_all_ids_accepting_new_schema_and_numeric_formatting():
    rows, metrics, protocol = feature_fixture()
    changed = [dict(member, elo_prob="0.500", archive_elo_prob=0.6) for member in reversed(rows)]
    evidence = validate_feature_cohort(feature_csv(rows), feature_csv(changed), metrics, metrics,
                                       protocol, [row("2390003", target=0)])
    assert evidence["n_series"] == 3
    assert evidence["partition_ids"] == {
        "train": ["2390001"], "validation": ["2390002"], "test": ["2390003"]}
    assert evidence["original_player_feature_columns"] == ["elo_prob"]


def test_full_cohort_rejects_silent_intersection_extra_rows_and_duplicate_ids():
    rows, metrics, protocol = feature_fixture()
    for changed in (rows[1:], rows + [dict(rows[0], hltv_match_id="2390099")], rows + [rows[0]]):
        with pytest.raises(ValueError):
            validate_feature_cohort(feature_csv(rows), feature_csv(changed), metrics, metrics,
                                    protocol, [row("2390003", target=0)])


def test_original_feature_bytes_must_match_both_frozen_hash_records():
    import hashlib

    content = b"exact original feature snapshot\n"
    digest = hashlib.sha256(content).hexdigest()
    protocol = {"baseline": {"feature_sha256": digest}}
    reproduction = {"feature_sha256": digest}
    validate_original_feature_hash(content, protocol, reproduction)
    with pytest.raises(ValueError, match="Original feature snapshot"):
        validate_original_feature_hash(content + b"\n", protocol, reproduction)
    with pytest.raises(ValueError, match="Original feature snapshot"):
        validate_original_feature_hash(content, protocol, {"feature_sha256": "0" * 64})


@pytest.mark.parametrize("changes", [
    {"source_date": "2026-06-08"}, {"elo_prob": float("nan")},
    {"elo_prob": float("inf")}, {"team1": "   "},
])
def test_full_cohort_rejects_invalid_regenerated_source_days_and_player_values(changes):
    rows, metrics, protocol = feature_fixture()
    changed = [dict(member, source_date=member["date"]) for member in rows]
    changed[0].update(changes)
    with pytest.raises(ValueError):
        validate_feature_cohort(feature_csv(rows), feature_csv(changed), metrics, metrics,
                                protocol, [row("2390003", target=0)])


def test_full_cohort_rejects_missing_original_predictor_column():
    rows, metrics, protocol = feature_fixture()
    changed = [{key: value for key, value in member.items() if key != "elo_prob"} for member in rows]
    with pytest.raises(ValueError, match="missing required"):
        validate_feature_cohort(feature_csv(rows), feature_csv(changed), metrics, metrics,
                                protocol, [row("2390003", target=0)])


def test_full_cohort_rejects_partition_membership_not_covering_original_ids():
    rows, metrics, protocol = feature_fixture()
    metrics["splits"]["train"]["dates"] = ["2026-06-08"]
    with pytest.raises(ValueError, match="membership differs"):
        validate_feature_cohort(feature_csv(rows), feature_csv(rows), metrics, metrics,
                                protocol, [row("2390003", target=0)])


def test_full_cohort_requires_stored_test_targets_to_match_feature_targets():
    rows, metrics, protocol = feature_fixture()
    with pytest.raises(ValueError, match="Stored TEST prediction target"):
        validate_feature_cohort(feature_csv(rows), feature_csv(rows), metrics, metrics,
                                protocol, [row("2390003", target=1)])


def test_full_cohort_requires_original_count_to_match_frozen_protocol():
    rows, metrics, protocol = feature_fixture()
    protocol["cohort"]["eligible_series"] = 4
    with pytest.raises(ValueError, match="frozen eligible cohort count"):
        validate_feature_cohort(feature_csv(rows), feature_csv(rows), metrics, metrics,
                                protocol, [row("2390003", target=0)])


@pytest.fixture
def recipe_evidence():
    """Canonical producer metadata, independent of mutable training artifacts."""
    protocol_path = Path(__file__).resolve().parents[1] / "reports" / "cs2-model-experiment-protocol-2026-10-09.json"
    protocol = json.loads(protocol_path.read_text())
    original = [
        "elo_prob", "glicko_prob", "trueskill_prob", "openskill_prob", "elo_diff",
        "lineup_win_rate_diff", "lineup_last10_win_rate_diff", "player_kpr_diff",
        "player_dpr_diff", "player_adr_diff", "player_kast_diff",
    ]
    candidates = protocol["candidate_set"]
    full = original + protocol["archive_features"]["features"]
    names = (candidates["original_candidates"] + candidates["new_raw_candidates"]
             + candidates["calibrated_candidates"]
             + [item["name"] for item in candidates["compact_candidates"]]
             + [item["name"] for item in candidates["fixed_blends"]])
    compact = {item["name"]: item["features"] for item in candidates["compact_candidates"]}
    blends = {item["name"]: item["probabilities"] for item in candidates["fixed_blends"]}
    parameters = {}
    for name in names:
        if name.startswith("pure_"):
            columns = [name.removeprefix("pure_") + "_prob"]
        elif name.startswith("calibrated_"):
            columns = [name.removeprefix("calibrated_") + "_prob"]
        elif name in compact:
            columns = compact[name]
        elif name in blends:
            columns = blends[name]
        else:
            columns = original
        metadata = _recipe(name, columns)
        if name.startswith(("pure_", "calibrated_")):
            metadata["probability_column"] = columns[0]
        if name.startswith("calibrated_"):
            metadata["slope"] = 1.25
        if name in blends:
            metadata["weights"] = [0.5, 0.5]
        if name in compact or name == "logistic_regression":
            metadata.update(coefficients=[[0.2] * len(columns)],
                            intercept=[0.0 if name in compact else 0.15],
                            scaler={"mean": [0.3] * len(columns), "scale": [1.2] * len(columns)})
        parameters[name] = metadata
    before = {"feature_columns": original,
              "candidate_parameters": {name: deepcopy(parameters[name])
                                       for name in candidates["original_candidates"]}}
    after = {"feature_columns": full, "candidate_parameters": parameters,
             "selection": {"candidate_order": names, "fit_partition": "train"}}
    return before, after, protocol


def test_complete_frozen_recipes_accept_learned_arrays_without_pinning_values(recipe_evidence):
    before, after, protocol = recipe_evidence
    validate_recipe_schema(before, after, protocol)
    for name in ("ridge_rating_logits", "ridge_strength_form"):
        metadata = after["candidate_parameters"][name]
        metadata["coefficients"] = [[-0.7] * len(metadata["feature_columns"])]
        metadata["scaler"]["mean"] = [2.5] * len(metadata["feature_columns"])
        metadata["scaler"]["scale"] = [3.0] * len(metadata["feature_columns"])
    after["candidate_parameters"]["calibrated_elo"]["slope"] = 0.0
    after["candidate_parameters"]["calibrated_glicko"]["slope"] = 3.0
    validate_recipe_schema(before, after, protocol)


@pytest.mark.parametrize("schema_change", ["reduced", "reordered", "missing_archive", "extra"])
def test_all_sixteen_names_do_not_certify_an_incomplete_or_changed_schema(recipe_evidence, schema_change):
    before, after, protocol = recipe_evidence
    assert len(after["candidate_parameters"]) == 16
    if schema_change == "reduced":
        after["feature_columns"] = after["feature_columns"][:7]
        after["candidate_parameters"]["ridge_strength_form"]["feature_columns"] = ["elo_diff"]
    elif schema_change == "reordered":
        after["feature_columns"] = list(reversed(after["feature_columns"]))
    elif schema_change == "missing_archive":
        after["feature_columns"].remove("archive_minimum_prior_matches")
    else:
        after["feature_columns"].append("unfrozen_predictor")
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


@pytest.mark.parametrize("name", ["ridge_rating_logits", "ridge_strength_form"])
@pytest.mark.parametrize("change", ["omitted", "reordered", "C", "intercept", "transform", "scaler"])
def test_compact_recipe_cannot_be_narrowed_or_retuned(recipe_evidence, name, change):
    before, after, protocol = recipe_evidence
    metadata = after["candidate_parameters"][name]
    if change == "omitted":
        metadata["feature_columns"].pop()
    elif change == "reordered":
        metadata["feature_columns"].reverse()
    elif change == "C":
        metadata["parameters"]["C"] = 1.0
    elif change == "intercept":
        metadata["parameters"]["fit_intercept"] = True
    elif change == "transform":
        metadata["transform"] = "identity" if name == "ridge_rating_logits" else "clipped_logit"
    else:
        metadata["parameters"]["with_mean"] = True
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


@pytest.mark.parametrize("name", ["logistic_regression", "gradient_boosting"])
@pytest.mark.parametrize("change", ["archive_contamination", "parameters"])
def test_original_classifiers_retain_their_frozen_subsets_and_parameters(recipe_evidence, name, change):
    before, after, protocol = recipe_evidence
    metadata = after["candidate_parameters"][name]
    if change == "archive_contamination":
        metadata["feature_columns"].append("archive_elo_prob")
    else:
        key = "C" if name == "logistic_regression" else "learning_rate"
        metadata["parameters"][key] = 0.25
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


@pytest.mark.parametrize("change", [
    "bounds", "channel", "features", "intercept", "epsilon", "negative_slope",
    "excess_slope", "nan_slope", "infinite_slope", "missing_slope",
])
def test_calibration_requires_frozen_channel_and_finite_bounded_zero_intercept_slope(recipe_evidence, change):
    before, after, protocol = recipe_evidence
    metadata = after["candidate_parameters"]["calibrated_elo"]
    if change == "bounds":
        metadata["parameters"]["slope_bounds"] = [0.0, 4.0]
    elif change == "channel":
        metadata["probability_column"] = "archive_elo_prob"
    elif change == "features":
        metadata["feature_columns"] = ["glicko_prob"]
    elif change == "intercept":
        metadata["parameters"]["fit_intercept"] = True
    elif change == "epsilon":
        metadata["parameters"]["epsilon"] = 1e-6
    elif change == "missing_slope":
        del metadata["slope"]
    else:
        metadata["slope"] = {"negative_slope": -0.1, "excess_slope": 3.1,
                             "nan_slope": float("nan"), "infinite_slope": float("inf")}[change]
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


@pytest.mark.parametrize("name", ["pure_elo", "pure_archive_elo", "pure_archive_decay"])
def test_raw_recipe_cannot_claim_a_different_probability_channel(recipe_evidence, name):
    before, after, protocol = recipe_evidence
    after["candidate_parameters"][name]["probability_column"] = "glicko_prob"
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


@pytest.mark.parametrize("change", ["channels", "declared_weights", "fitted_weights"])
def test_fixed_blend_channels_and_both_weight_declarations_are_frozen(recipe_evidence, change):
    before, after, protocol = recipe_evidence
    metadata = after["candidate_parameters"]["blend_player_archive_elo"]
    if change == "channels":
        metadata["feature_columns"] = ["elo_prob", "archive_decay_prob"]
    elif change == "declared_weights":
        metadata["parameters"]["weights"] = [0.6, 0.4]
    else:
        metadata["weights"] = [0.6, 0.4]
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


@pytest.mark.parametrize("field", ["name", "family", "transform", "parameters", "feature_columns"])
def test_absent_required_recipe_metadata_is_not_certifiable(recipe_evidence, field):
    before, after, protocol = recipe_evidence
    del after["candidate_parameters"]["ridge_strength_form"][field]
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


@pytest.mark.parametrize("change", ["missing_recipe", "missing_array", "short_array", "nonzero_intercept", "nonfinite_scale"])
def test_compact_requires_complete_consistent_fitted_metadata(recipe_evidence, change):
    before, after, protocol = recipe_evidence
    metadata = after["candidate_parameters"]["ridge_strength_form"]
    if change == "missing_recipe":
        del after["candidate_parameters"]["ridge_strength_form"]
    elif change == "missing_array":
        del metadata["coefficients"]
    elif change == "short_array":
        metadata["coefficients"][0].pop()
    elif change == "nonzero_intercept":
        metadata["intercept"] = [0.2]
    else:
        metadata["scaler"]["scale"][0] = float("inf")
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


def test_recipe_metadata_cannot_claim_train_validation_scaler_fitting(recipe_evidence):
    before, after, protocol = recipe_evidence
    after["selection"]["fit_partition"] = "train+validation"
    with pytest.raises(ValueError):
        validate_recipe_schema(before, after, protocol)


def test_output_paths_allow_independent_existing_and_new_destinations(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    existing = tmp_path / "existing"
    existing.write_bytes(b"old output")
    validate_output_paths(iter([source]), iter([existing, tmp_path / "new" / "output"]))
    assert source.read_bytes() == b"source"
    assert existing.read_bytes() == b"old output"
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink"])
def test_output_paths_reject_input_aliases_without_mutating_bytes(tmp_path, alias):
    source = tmp_path / "source"
    source.write_bytes(b"frozen source")
    destination = tmp_path / "destination"
    if alias == "same_path":
        destination = source
    elif alias == "symlink":
        destination.symlink_to(source)
    else:
        os.link(source, destination)
    with pytest.raises(ValueError):
        validate_output_paths([source], [tmp_path / "new_output", destination])
    assert source.read_bytes() == b"frozen source"
    assert not (tmp_path / "new_output").exists()


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink"])
def test_output_paths_reject_aliases_between_destinations(tmp_path, alias):
    first = tmp_path / "first"
    first.write_bytes(b"untouched output")
    second = tmp_path / "second"
    if alias == "same_path":
        second = first
    elif alias == "symlink":
        second.symlink_to(first)
    else:
        os.link(first, second)
    with pytest.raises(ValueError):
        validate_output_paths([], [first, second])
    assert first.read_bytes() == b"untouched output"


@pytest.fixture
def summary_inputs(tmp_path, recipe_evidence):
    """A complete six-row local evidence package; no repository artifact reads."""
    before, after, protocol = recipe_evidence
    days = ("2026-06-07", "2026-08-13", "2026-08-14")
    features = []
    for index in range(6):
        features.append({
            "hltv_match_id": str(2390001 + index), "date": days[index // 2],
            "source_date": days[index // 2], "team1": "A", "team2": "B", "target": index % 2,
            **{column: 0.5 if column.endswith("_prob") else 0.0
               for column in before["feature_columns"]},
        })
    augmented = [member | {column: 0.5 if column.endswith("_prob") else 0.0
                           for column in protocol["archive_features"]["features"]}
                 for member in features]
    predictions = [row(member["hltv_match_id"], target=member["target"],
                       date=member["date"], source_date=member["source_date"])
                   for member in features[-2:]]
    test_score = evaluate_predictions([0, 1], [0.75, 0.75])
    splits = {name: {"n_series": 2, "n_days": 1, "dates": [day],
                     "start_date": day, "end_date": day, "class_counts": {"0": 1, "1": 1}}
              for name, day in zip(("train", "validation", "test"), days)}
    for metadata in (before, after):
        names = list(metadata["candidate_parameters"])
        losses = {name: 0.5 if name == "pure_elo" else 0.8 for name in names}
        metadata.update(
            selected_model="pure_elo", splits=deepcopy(splits), support={"n_series": 6},
            models={name: {"validation": {"log_loss": losses[name], "n_samples": 2},
                           "test": deepcopy(test_score)} for name in names},
            selection={"partition": "validation", "metric": "log_loss", "fit_partition": "train",
                       "candidate_log_loss": losses, "selected_log_loss": 0.5,
                       "candidate_order": names},
        )
    original_bytes = feature_csv(features)
    digest = hashlib.sha256(original_bytes).hexdigest()
    protocol["cohort"]["eligible_series"] = 6
    for name, day in zip(("train", "validation", "test"), days):
        protocol["cohort"][f"{name}_rows"] = 2
        protocol["cohort"][f"{name}_dates"] = [day, day]
    protocol["baseline"].update(feature_sha256=digest, selected_model="pure_elo",
                                validation_log_loss=0.5, test_log_loss=test_score["log_loss"])
    reproduction = {
        "selected_model": "pure_elo", "split_counts": {name: 2 for name in splits},
        "feature_sha256": digest,
        "source_module_sha256": protocol["baseline"]["player_baseline_source_sha256"],
        "selected_test": test_score, "selected_validation": 0.5,
    }
    before_dir = tmp_path / "before"
    before_dir.mkdir()
    contents = {
        before_dir / "reproduction.json": json.dumps(reproduction).encode(),
        before_dir / "metrics.json": json.dumps(before).encode(),
        before_dir / "test_predictions.csv": feature_csv(predictions),
        before_dir / "features.csv": original_bytes,
        before_dir / "model.pkl": b"opaque frozen original model",
        tmp_path / "protocol.json": json.dumps(protocol).encode(),
        tmp_path / "after_metrics.json": json.dumps(after).encode(),
        tmp_path / "after_predictions.csv": feature_csv(predictions),
        tmp_path / "after_features.csv": feature_csv(augmented),
        tmp_path / "after_model.pkl": b"opaque different after model",
        tmp_path / "after_state.pkl": b"opaque after state",
    }
    for path, content in contents.items():
        path.write_bytes(content)
    args = SimpleNamespace(
        before_dir=before_dir, protocol=tmp_path / "protocol.json",
        after_metrics=tmp_path / "after_metrics.json",
        after_predictions=tmp_path / "after_predictions.csv",
        after_features=tmp_path / "after_features.csv",
        after_model=tmp_path / "after_model.pkl", after_state=tmp_path / "after_state.pkl",
        output_dir=tmp_path / "output",
    )
    return args, contents


@pytest.mark.parametrize("alias", ["copied_input", "generated_input", "output_output"])
def test_summarize_rejects_hardlinks_before_any_output_or_source_mutation(summary_inputs, alias):
    args, source_bytes = summary_inputs
    args.output_dir.mkdir()
    if alias == "copied_input":
        os.link(args.before_dir / "model.pkl", args.output_dir / "model.pkl")
    elif alias == "generated_input":
        os.link(args.before_dir / "model.pkl", args.output_dir / "summary.json")
    else:
        (args.output_dir / "model.pkl").write_bytes(b"existing independent output")
        os.link(args.output_dir / "model.pkl", args.output_dir / "manifest.json")
    existing_outputs = {path: path.read_bytes() for path in args.output_dir.iterdir()}
    with pytest.raises(ValueError):
        summarize(args)
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    assert {path: path.read_bytes() for path in args.output_dir.iterdir()} == existing_outputs


def test_summarize_rejects_sixteen_names_with_reduced_schema_before_writing(summary_inputs):
    args, source_bytes = summary_inputs
    metadata = json.loads(args.after_metrics.read_text())
    assert len(metadata["selection"]["candidate_log_loss"]) == 16
    metadata["feature_columns"] = metadata["feature_columns"][:7]
    metadata["candidate_parameters"]["ridge_strength_form"]["feature_columns"] = ["elo_diff"]
    args.after_metrics.write_text(json.dumps(metadata))
    source_bytes[args.after_metrics] = args.after_metrics.read_bytes()
    with pytest.raises(ValueError):
        summarize(args)
    assert not args.output_dir.exists()
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes


def test_summarize_accepts_complete_local_evidence_and_independent_outputs(summary_inputs):
    args, source_bytes = summary_inputs
    result = summarize(args)
    assert result["n_test_series"] == 2
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    assert (args.output_dir / "model.pkl").read_bytes() == source_bytes[args.after_model]
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["full_cohort_validation"]["n_series"] == 6
    assert (args.output_dir / "manifest.json").is_file()
