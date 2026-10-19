"""Day-grouped retrospective player baseline selection, scoring, and inference.

The caller must replay verified played series, freezing every source-calendar
day's features before that day's rating updates. Earlier evaluation-day results
may update subsequent days' ratings; the classifier itself stays TRAIN-fitted.
Actual historical lineups are conditional inputs, not evidence of their prematch
availability. This module cannot establish feature provenance from numbers alone.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.models.metrics import evaluate_predictions


_VERSION = 4
_EPSILON = 1e-12
_RAW_PROBABILITIES = ("elo_prob", "glicko_prob", "trueskill_prob", "openskill_prob")
_ARCHIVE_PROBABILITIES = ("archive_elo_prob", "archive_decay_prob")
_COMPACT_FEATURES = {
    "ridge_rating_logits": (*_RAW_PROBABILITIES, *_ARCHIVE_PROBABILITIES),
    "ridge_strength_form": (
        "elo_diff", "archive_elo_diff", "archive_decay_diff", "lineup_win_rate_diff",
        "lineup_last10_win_rate_diff", "player_kpr_diff", "player_dpr_diff",
        "player_adr_diff", "player_kast_diff",
    ),
}
_CANDIDATE_ORDER = (
    "pure_elo", "pure_glicko", "pure_trueskill", "pure_openskill",
    "logistic_regression", "gradient_boosting", "pure_archive_elo", "pure_archive_decay",
    "calibrated_elo", "calibrated_glicko", "calibrated_trueskill", "calibrated_openskill",
    "ridge_rating_logits", "ridge_strength_form",
    "blend_player_archive_elo", "blend_player_archive_decay",
)
# Feature semantics are explicit: an unfamiliar suffix is not provenance or a
# license to guess how a predictor transforms when the two teams are reversed.
_MIRROR_OPERATIONS = {
    **{name: {"operation": "negate", "source": name} for name in (
        "elo_diff", "glicko_diff", "trueskill_mu_diff", "openskill_ordinal_diff",
        "lineup_exp_diff", "lineup_win_rate_diff", "glicko_rd_diff",
        "trueskill_sigma_diff", "openskill_sigma_diff", "lineup_last10_win_rate_diff",
        "player_kpr_diff", "player_dpr_diff", "player_assists_per_round_diff",
        "player_adr_diff", "player_kast_diff", "player_last10_kpr_diff",
        "player_last10_dpr_diff", "player_last10_adr_diff", "player_last10_kast_diff",
        "archive_elo_diff", "archive_decay_diff",
    )},
    **{name: {"operation": "complement", "source": name}
       for name in (*_RAW_PROBABILITIES, *_ARCHIVE_PROBABILITIES)},
    **{name: {"operation": "identity", "source": name} for name in (
        "glicko_joint_rd", "trueskill_joint_sigma", "openskill_joint_sigma",
        "minimum_prior_matches", "performance_available", "minimum_prior_measured_maps",
        "archive_minimum_prior_matches",
    )},
    "team1_known_players": {"operation": "swap", "source": "team2_known_players"},
    "team2_known_players": {"operation": "swap", "source": "team1_known_players"},
    "archive_team1_prior_matches": {"operation": "swap", "source": "archive_team2_prior_matches"},
    "archive_team2_prior_matches": {"operation": "swap", "source": "archive_team1_prior_matches"},
}
_NON_FEATURES = {
    "hltv_match_id", "match_id", "date", "source_date", "timestamp", "team1", "team2",
    "target", "y", "label", "team1_win", "team1_won", "winner", "outcome", "score1",
    "score2", "score_left", "score_right", "predicted_winner", "correct_prediction",
    "prob_team1_win", "prob_team2_win",
}
_ASSUMPTIONS = {
    "evaluation": "Retrospective actual-lineup-conditional series evaluation; not proof of actionable prematch lineup availability.",
    "rating_replay": "Sequential source-calendar-day replay: earlier evaluation days may update ratings; same-day and future outcomes must not influence a day's features.",
    "classifier_fit": "TRAIN only; no TRAIN+VALIDATION refit or test-label fitting.",
    "selection": "Frozen 16-recipe ladder selected solely by VALIDATION log loss; exact ties favor pure_elo, then declared candidate order. No TEST selection or refit.",
    "feature_provenance": "Caller supplies match-specific identity evidence and day-frozen historical features; numeric validation alone cannot prove original release clocks or pre-event availability.",
    "unit": "One score-compatible played series; map-derived lineups require complete unchanged 5v5, while match-page captures remain conditional lineup evidence. No duplicated map labels or forfeits.",
    "side_symmetry": "Classifier and calibrated forecasts are projected as 0.5 + 0.5 * (raw(X) - raw(mirror(X))); compact zero-intercept strength models and fixed probability blends also complement on reversal.",
    "orientation_controls": "The literal TRAIN class prior is an orientation-asymmetric diagnostic control, not a deployable side-symmetric forecast; fixed 0.5 is an orientation-neutral control.",
    "rating_forecasters": "Fixed-parameter channels consume preceding-day outcomes through the separately persisted rating state; raw channel recipes have no fitted classifier.",
    "development_diagnostics": "Three expanding-origin folds entirely inside TRAIN days; initial half-day prefix followed by three consecutive blocks. Single-class prefixes are skipped without borrowing future labels.",
}


def _feature_frame(features_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not isinstance(features_df, pd.DataFrame) or not features_df.columns.is_unique:
        raise ValueError("Features must be a DataFrame with unique column names")
    if (not isinstance(feature_columns, list) or not feature_columns
            or any(not isinstance(name, str) or not name.strip() for name in feature_columns)
            or len(set(feature_columns)) != len(feature_columns)):
        raise ValueError("feature_columns must be a nonempty ordered list of distinct feature names")
    if any(name.casefold() in _NON_FEATURES for name in feature_columns):
        raise ValueError("Only explicitly selected pre-result predictors may be feature columns")
    missing = [name for name in feature_columns if name not in features_df]
    if missing:
        raise ValueError(f"Missing selected feature columns: {missing}")
    try:
        matrix = features_df.loc[:, feature_columns].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected feature columns must contain finite numeric values") from exc
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Selected feature columns must contain finite numeric values")
    return matrix


def _mirror_rules(feature_columns: list[str]) -> dict:
    unknown = [name for name in feature_columns if name not in _MIRROR_OPERATIONS]
    if unknown:
        raise ValueError(f"Unknown feature mirror semantics: {unknown}")
    for pair in (("team1_known_players", "team2_known_players"),
                 ("archive_team1_prior_matches", "archive_team2_prior_matches")):
        if any(name in feature_columns for name in pair) and not all(name in feature_columns for name in pair):
            raise ValueError(f"Feature schema must contain both {pair[0]} and {pair[1]}")
    return {name: dict(_MIRROR_OPERATIONS[name]) for name in feature_columns}


def _mirror_frame(matrix: pd.DataFrame, rules: dict) -> pd.DataFrame:
    mirrored = matrix.copy()
    for column, rule in rules.items():
        values = matrix[rule["source"]]
        if rule["operation"] == "negate":
            mirrored[column] = -values
        elif rule["operation"] == "complement":
            mirrored[column] = 1 - values
        elif rule["operation"] == "swap":
            mirrored[column] = values
        # Identity columns already have their original values in the copy.
    return mirrored


def _symmetric_probability(
    estimator: Any, matrix: pd.DataFrame, mirrored: pd.DataFrame
) -> np.ndarray:
    positive_index = list(estimator.classes_).index(1)
    forward = estimator.predict_proba(matrix)[:, positive_index]
    reverse = estimator.predict_proba(mirrored)[:, positive_index]
    for probability in (forward, reverse):
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Fitted estimator returned invalid probabilities")
    # Equivalent to 0.5 * (p(X) + 1 - p(mirror(X))), written to preserve exactly
    # 0.5 when both inputs coincide, regardless of the classifier's intercept.
    return 0.5 + 0.5 * (forward - reverse)


def _calendar_day(value: Any) -> str:
    # Do not convert timezone-bearing timestamps to UTC: grouping is by the
    # source's calendar, not an invented globally ordered kickoff timestamp.
    if not isinstance(value, (str, date, datetime, pd.Timestamp, np.datetime64)):
        raise ValueError("Each source-calendar date must be a valid date or datetime")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid source-calendar date: {value!r}") from exc
    if pd.isna(timestamp):
        raise ValueError("Each source-calendar date must be nonmissing")
    return timestamp.date().isoformat()


def _score(target: np.ndarray, probability: np.ndarray) -> dict:
    result = evaluate_predictions(target.tolist(), probability.tolist())
    # AUC has no mathematical meaning for a single-class cohort. Do not inherit
    # the generic metrics module's neutral 0.5 fallback for this audit.
    if np.unique(target).size < 2:
        result["roc_auc"] = None
    return result


def _support(frame: pd.DataFrame) -> dict:
    target = frame["target"].to_numpy(dtype=int)
    summary = {
        "n_series": int(len(frame)),
        "n_days": int(frame["source_date"].nunique()),
        "class_counts": {"0": int(np.sum(target == 0)), "1": int(np.sum(target == 1))},
    }
    for column in ("lineup_source", "identity_namespace"):
        if column in frame:
            summary[column] = {
                str(value): int(count)
                for value, count in frame[column].fillna("missing").astype(str).value_counts().sort_index().items()
            }
    support_columns = [column for column in frame.columns if (
        column in {"minimum_prior_matches", "minimum_prior_player_matches", "minimum_prior_player_series"}
        or "cold_player" in column
    )]
    archive_columns = [column for column in (
        "archive_team1_prior_matches", "archive_team2_prior_matches",
        "archive_minimum_prior_matches",
    ) if column in frame]
    for group, columns in (("player_history", support_columns), ("archive_history", archive_columns)):
        if not columns:
            continue
        summary[group] = {}
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            values = values[np.isfinite(values)]
            summary[group][column] = {
                "n_observed": int(len(values)),
                "min": float(values.min()) if len(values) else None,
                "mean": float(values.mean()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "n_zero": int((values == 0).sum()),
            }
    return summary


def _logits(matrix: pd.DataFrame) -> pd.DataFrame:
    # Center before clipping so the two floating-point endpoint representations
    # use the same magnitude. This is clipped logit algebraically, but avoids
    # cancellation making epsilon and 1-epsilon have different strengths.
    centered = matrix - 0.5
    distance = centered.abs().clip(upper=0.5 - _EPSILON)
    magnitude = np.log((0.5 + distance) / (0.5 - distance))
    return np.sign(centered) * magnitude


def _calibration_slope(probability: np.ndarray, target: np.ndarray) -> float:
    logits = _logits(pd.DataFrame({"probability": probability})).iloc[:, 0].to_numpy()
    if np.all(logits == 0):
        return 1.0

    def derivative(slope: float) -> float:
        return float(np.mean(logits * (expit(slope * logits) - target)))

    # The convex objective can attain its minimum at either bound; an interior
    # optimizer alone would incorrectly replace these exact optima by near-bounds.
    if derivative(0) >= 0:
        return 0.0
    if derivative(3) <= 0:
        return 3.0
    lower, upper = 0.0, 3.0
    for _ in range(64):
        middle = (lower + upper) / 2
        if derivative(middle) < 0:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def _recipe(name: str, columns: list[str]) -> dict:
    """Canonical recipe metadata, also used to reject corrupt saved recipes."""
    if name not in _CANDIDATE_ORDER:
        raise ValueError("Unknown selected forecast recipe")
    recipe = {"name": name, "feature_columns": list(columns)}
    if name.startswith("pure_"):
        expected = [name.removeprefix("pure_") + "_prob"]
        family, transform, parameters = "rating_channel", "identity", {}
    elif name.startswith("calibrated_"):
        expected = [name.removeprefix("calibrated_") + "_prob"]
        family, transform = "calibrated_rating", "clipped_logit"
        parameters = {"slope_bounds": [0.0, 3.0], "fit_intercept": False, "epsilon": _EPSILON}
    elif name in _COMPACT_FEATURES:
        expected = [column for column in _COMPACT_FEATURES[name] if column in columns]
        family = "compact_ridge"
        transform = "clipped_logit" if name == "ridge_rating_logits" else "identity"
        parameters = {"C": 0.1, "fit_intercept": False, "with_mean": False,
                      "random_state": 42, "max_iter": 1000}
        if transform == "clipped_logit":
            parameters["epsilon"] = _EPSILON
    elif name.startswith("blend_"):
        expected = ["elo_prob", name.removeprefix("blend_player_") + "_prob"]
        family, transform, parameters = "probability_blend", "identity", {"weights": [0.5, 0.5]}
    else:
        expected = [column for column in columns if not column.startswith("archive_")]
        family, transform = "classifier", "identity"
        if name == "logistic_regression":
            parameters = {"C": 1.0, "fit_intercept": True, "with_mean": True,
                          "max_iter": 1000, "random_state": 42}
        else:
            parameters = {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 15,
                          "min_samples_leaf": 20, "early_stopping": False,
                          "validation_fraction": None, "random_state": 42}
    if not columns or columns != expected:
        raise ValueError("Artifact ordered feature schema does not match its forecast recipe")
    return dict(recipe, family=family, transform=transform, parameters=parameters)


def _estimator_for(recipe: dict) -> Any:
    parameters = recipe["parameters"]
    if recipe["name"] == "gradient_boosting":
        return HistGradientBoostingClassifier(**parameters)
    classifier_parameters = {key: value for key, value in parameters.items()
                             if key not in {"with_mean", "epsilon"}}
    return Pipeline([
        ("scaler", StandardScaler(with_mean=parameters["with_mean"])),
        ("classifier", LogisticRegression(**classifier_parameters)),
    ])


def _finite_parameters(arrays: list) -> list[np.ndarray]:
    checked = [np.asarray(array) for array in arrays]
    if any(not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all()
           for array in checked):
        raise ValueError("Artifact fitted parameters must be finite numeric arrays")
    return checked


def _validate_estimator(estimator: Any, recipe: dict) -> None:
    columns = recipe["feature_columns"]
    if list(getattr(estimator, "feature_names_in_", [])) != columns:
        raise ValueError("Artifact ordered feature schema does not match its fitted estimator")
    parameters = recipe["parameters"]
    expected_estimator = _estimator_for(recipe)
    if recipe["name"] == "gradient_boosting":
        if not isinstance(estimator, HistGradientBoostingClassifier):
            raise ValueError("Artifact fitted estimator does not match its recipe")
        if estimator.get_params() != expected_estimator.get_params():
            raise ValueError("Artifact fitted estimator parameters do not match its recipe")
        trees_by_iteration = getattr(estimator, "_predictors", [])
        if len(trees_by_iteration) != parameters["max_iter"]:
            raise ValueError("Artifact fitted boosting state is incomplete")
        arrays = [getattr(estimator, "_baseline_prediction", np.array([np.nan]))]
        for trees in trees_by_iteration:
            if len(trees) != 1:
                raise ValueError("Artifact fitted boosting state is invalid")
            arrays.extend(tree.nodes["value"] for tree in trees)
    else:
        if (not isinstance(estimator, Pipeline)
                or list(estimator.named_steps) != ["scaler", "classifier"]):
            raise ValueError("Artifact fitted estimator does not match its recipe")
        scaler, classifier = estimator.named_steps["scaler"], estimator.named_steps["classifier"]
        if not isinstance(scaler, StandardScaler) or not isinstance(classifier, LogisticRegression):
            raise ValueError("Artifact fitted estimator does not match its recipe")
        if scaler.get_params() != expected_estimator.named_steps["scaler"].get_params():
            raise ValueError("Artifact fitted scaler does not match its recipe")
        if classifier.get_params() != expected_estimator.named_steps["classifier"].get_params():
            raise ValueError("Artifact fitted estimator parameters do not match its recipe")
        arrays = _finite_parameters([
            getattr(classifier, "coef_", np.array([np.nan])),
            getattr(classifier, "intercept_", np.array([np.nan])),
            getattr(scaler, "scale_", np.array([np.nan])),
            getattr(scaler, "mean_", np.array([np.nan])),
            getattr(scaler, "var_", np.array([np.nan])),
        ])
        coefficients, intercept, scale, mean, variance = arrays
        if (np.shape(coefficients) != (1, len(columns))
                or np.shape(intercept) != (1,)
                or any(np.shape(array) != (len(columns),) for array in (scale, mean, variance))
                or np.any(scale <= 0) or np.any(variance < 0)
                or (not parameters["fit_intercept"] and np.any(intercept != 0))):
            raise ValueError("Artifact fitted parameter dimensions or intercept are invalid")
    if list(getattr(estimator, "classes_", [])) != [0, 1]:
        raise ValueError("Artifact fitted estimator must contain both target classes")
    _finite_parameters(arrays)


def predict_player_baseline(bundle: dict, features_df: pd.DataFrame) -> np.ndarray:
    """Validate a version-4 recipe and reproduce its unrounded symmetric forecast."""
    if not isinstance(bundle, dict) or bundle.get("version") != _VERSION:
        raise ValueError("Unsupported player baseline artifact version")
    columns = bundle.get("feature_columns")
    matrix = _feature_frame(features_df, columns)
    rules = _mirror_rules(columns)
    if bundle.get("mirror_rules") != rules or list(bundle["mirror_rules"]) != columns:
        raise ValueError("Artifact ordered mirror rules do not match its feature schema")
    recipe = _recipe(bundle.get("selected_model"), columns)
    if bundle.get("recipe") != recipe or bundle.get("family") != recipe["family"]:
        raise ValueError("Artifact forecast recipe or family is invalid")
    for column in columns:
        if column in (*_RAW_PROBABILITIES, *_ARCHIVE_PROBABILITIES):
            if np.any((matrix[column] < 0) | (matrix[column] > 1)):
                raise ValueError("Rating probabilities must be between 0 and 1")
    family = recipe["family"]
    if family in {"rating_channel", "calibrated_rating"}:
        if bundle.get("probability_column") != columns[0]:
            raise ValueError("Artifact probability channel does not match its recipe")
    if family == "calibrated_rating":
        slope = bundle.get("slope")
        if (isinstance(slope, (bool, str)) or not isinstance(slope, (int, float, np.number))
                or not np.isfinite(slope) or not 0 <= slope <= 3):
            raise ValueError("Artifact calibrated slope must be finite and bounded from 0 to 3")
    if family == "probability_blend" and bundle.get("weights") != [0.5, 0.5]:
        raise ValueError("Artifact blend weights must be the frozen equal-probability weights")
    if family in {"classifier", "compact_ridge"}:
        _validate_estimator(bundle.get("estimator"), recipe)
    if len(matrix) == 0:
        return np.empty(0, dtype=float)
    if family == "rating_channel":
        return matrix.iloc[:, 0].to_numpy()
    if family == "probability_blend":
        return 0.5 * matrix.iloc[:, 0].to_numpy() + 0.5 * matrix.iloc[:, 1].to_numpy()
    mirrored = _mirror_frame(matrix, rules)
    if family == "calibrated_rating":
        forward = expit(slope * _logits(matrix).iloc[:, 0].to_numpy())
        reverse = expit(slope * _logits(mirrored).iloc[:, 0].to_numpy())
        return 0.5 + 0.5 * (forward - reverse)
    if recipe["transform"] == "clipped_logit":
        matrix, mirrored = _logits(matrix), _logits(mirrored)
    return _symmetric_probability(bundle["estimator"], matrix, mirrored)


def _fit_candidates(rows: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Fit the frozen ladder once on the supplied chronological TRAIN prefix."""
    target = rows["target"].to_numpy(dtype=int)
    bundles = {}
    for name in _CANDIDATE_ORDER:
        if name.startswith("pure_"):
            columns = [name.removeprefix("pure_") + "_prob"]
            if columns[0] in _ARCHIVE_PROBABILITIES and columns[0] not in feature_columns:
                continue
        elif name.startswith("calibrated_"):
            columns = [name.removeprefix("calibrated_") + "_prob"]
        elif name in _COMPACT_FEATURES:
            columns = [column for column in _COMPACT_FEATURES[name] if column in feature_columns]
            if not columns:
                continue
        elif name.startswith("blend_"):
            columns = ["elo_prob", name.removeprefix("blend_player_") + "_prob"]
            if columns[1] not in feature_columns:
                continue
        else:
            columns = [column for column in feature_columns if not column.startswith("archive_")]
        recipe = _recipe(name, columns)
        bundle = {
            "version": _VERSION, "selected_model": name, "family": recipe["family"],
            "recipe": recipe, "feature_columns": columns, "mirror_rules": _mirror_rules(columns),
            "training_cutoff": max(rows["source_date"]), "training_rows": int(len(rows)),
            "evaluation_assumptions": dict(_ASSUMPTIONS),
        }
        family = recipe["family"]
        if family in {"rating_channel", "calibrated_rating"}:
            bundle["probability_column"] = columns[0]
        if family == "calibrated_rating":
            bundle["slope"] = _calibration_slope(rows[columns[0]].to_numpy(dtype=float), target)
        elif family == "probability_blend":
            bundle["weights"] = [0.5, 0.5]
        elif family in {"classifier", "compact_ridge"}:
            matrix = _feature_frame(rows, columns)
            if recipe["transform"] == "clipped_logit":
                matrix = _logits(matrix)
            bundle["estimator"] = _estimator_for(recipe).fit(matrix, target)
        bundles[name] = bundle
    return bundles


def _candidate_parameters(bundle: dict) -> dict:
    result = dict(bundle["recipe"])
    for field in ("slope", "weights", "probability_column"):
        if field in bundle:
            result[field] = bundle[field]
    estimator = bundle.get("estimator")
    if isinstance(estimator, Pipeline):
        result["coefficients"] = estimator.named_steps["classifier"].coef_.tolist()
        result["intercept"] = estimator.named_steps["classifier"].intercept_.tolist()
        scaler = estimator.named_steps["scaler"]
        result["scaler"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    return result


def _candidate_predictions(bundles: dict, rows: pd.DataFrame, class_prior: float) -> dict:
    return {
        **{name: predict_player_baseline(bundle, rows) for name, bundle in bundles.items()},
        "constant_0_5": np.full(len(rows), 0.5),
        "class_prior": np.full(len(rows), class_prior),
    }


def _development_folds(train: pd.DataFrame, feature_columns: list[str]) -> dict:
    days = sorted(train["source_date"].unique())
    prefix = len(days) // 2
    folds = []
    for block in np.array_split(days[prefix:], 3):
        validation_dates = block.tolist()
        train_dates = days[:prefix]
        fold = {"train_dates": train_dates, "validation_dates": validation_dates, "models": {}}
        fit_rows = train[train["source_date"].isin(train_dates)]
        if np.unique(fit_rows["target"]).size < 2:
            fold["status"] = "skipped_single_class"
        elif not validation_dates:
            fold["status"] = "skipped_empty_validation"
        else:
            fold["status"] = "evaluated"
            bundles = _fit_candidates(fit_rows, feature_columns)
            validation_rows = train[train["source_date"].isin(validation_dates)]
            probabilities = _candidate_predictions(bundles, validation_rows, float(fit_rows["target"].mean()))
            fold["models"] = {name: _score(validation_rows["target"].to_numpy(dtype=int), probability)
                              for name, probability in probabilities.items()}
            fold["candidate_parameters"] = {name: _candidate_parameters(bundle)
                                            for name, bundle in bundles.items()}
        folds.append(fold)
        prefix += len(block)
    return {"fit_partition": "train", "initial_train_dates": days[:len(days) // 2], "folds": folds}


def train_player_baseline(
    features_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[dict, pd.DataFrame, dict]:
    """Return (metrics, TEST predictions, serializable selected forecast bundle).

    Input requires unique nonempty ``hltv_match_id``, binary ``target``, ``date``
    (or explicit ``source_date``), the four raw rating probability columns, and
    finite explicitly selected predictors. At least five distinct source days
    and both TRAIN classes are required. Whole chronological days split as
    floor(60%)/floor(20%)/remainder; no day is shared. All fitted recipes and
    the class-prior control fit TRAIN only. Selection sees VALIDATION only;
    TEST is scored once per baseline on exactly the same rows.
    Validation, TEST, and saved-artifact inference all project classifier
    probabilities through explicit team-reversal rules. Unknown semantics and
    incomplete team-specific pairs are rejected rather than guessed.

    ``metrics['models'][name]['validation'|'test']`` contains accuracy, log_loss,
    brier, roc_auc (None for one class), ece, calibration_bins and n_samples.
    Splits report all dates, boundaries, day/series counts and class support.
    Probabilities are never rounded. This function never writes files or mutates
    input data; callers persist the returned real estimator bundle with joblib.
    """
    _feature_frame(features_df, feature_columns)
    mirror_rules = _mirror_rules(feature_columns)
    required = ["hltv_match_id", "target", *_RAW_PROBABILITIES]
    date_column = "source_date" if "source_date" in features_df else "date"
    required.append(date_column)
    missing = [column for column in required if column not in features_df]
    if missing:
        raise ValueError(f"Missing required baseline columns: {missing}")
    if features_df.empty:
        raise ValueError("At least five distinct source-calendar days are required")
    ids = features_df["hltv_match_id"]
    canonical_ids = ids.astype(str).str.strip()
    if ids.isna().any() or canonical_ids.eq("").any() or canonical_ids.duplicated().any():
        raise ValueError("hltv_match_id must be unique and nonempty for every series")
    target = pd.to_numeric(features_df["target"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("target must contain only binary 0 or 1 values")
    for column in (*_RAW_PROBABILITIES,
                   *(name for name in _ARCHIVE_PROBABILITIES if name in feature_columns)):
        probability = pd.to_numeric(features_df[column], errors="coerce")
        if not np.isfinite(probability).all() or not probability.between(0, 1).all():
            raise ValueError(f"{column} probabilities must be finite and between 0 and 1")

    frame = features_df.copy()
    frame["source_date"] = frame[date_column].map(_calendar_day)
    frame["target"] = target.astype(int)
    # Explicit deterministic within-day ordering is only for output. It is not
    # a claim about kickoff order and never controls rating updates here.
    order = pd.DataFrame({"day": frame["source_date"].to_numpy(),
                          "id": canonical_ids.to_numpy()}).sort_values(["day", "id"]).index
    frame = frame.iloc[order].reset_index(drop=True)
    days = sorted(frame["source_date"].unique())
    if len(days) < 5:
        raise ValueError("At least five distinct source-calendar days are required for TRAIN/VALIDATION/TEST")
    train_end = int(len(days) * 0.6)
    validation_end = train_end + int(len(days) * 0.2)
    partitions = {
        "train": frame[frame["source_date"].isin(days[:train_end])],
        "validation": frame[frame["source_date"].isin(days[train_end:validation_end])],
        "test": frame[frame["source_date"].isin(days[validation_end:])],
    }
    y_train = partitions["train"]["target"].to_numpy(dtype=int)
    if np.unique(y_train).size != 2:
        raise ValueError("TRAIN must contain both binary target classes")

    candidates = _fit_candidates(partitions["train"], feature_columns)
    class_prior = float(y_train.mean())
    model_scores: dict[str, dict] = {}
    prediction_by_partition = {}
    forecast_candidates = list(candidates)
    for partition in ("validation", "test"):
        rows = partitions[partition]
        probabilities = _candidate_predictions(candidates, rows, class_prior)
        prediction_by_partition[partition] = probabilities
        labels = rows["target"].to_numpy(dtype=int)
        for name, probability in probabilities.items():
            model_scores.setdefault(name, {})[partition] = _score(labels, probability)
        if partition == "validation":
            # Commit the choice before computing any TEST predictions or scores.
            validation_log_loss = {
                name: model_scores[name]["validation"]["log_loss"] for name in forecast_candidates
            }
            selected_model = min(validation_log_loss, key=validation_log_loss.get)

    predictions = partitions["test"].copy().reset_index(drop=True)
    for name, probability in prediction_by_partition["test"].items():
        column = f"{name.removeprefix('pure_')}_prob"
        predictions[column] = probability
    predictions["prob_team1_win"] = prediction_by_partition["test"][selected_model]
    predictions["prob_team2_win"] = 1 - predictions["prob_team1_win"]
    team1 = predictions["team1"] if "team1" in predictions else "team1"
    team2 = predictions["team2"] if "team2" in predictions else "team2"
    predictions["predicted_winner"] = np.where(predictions["prob_team1_win"] >= 0.5, team1, team2)
    predictions["correct_prediction"] = (
        (predictions["prob_team1_win"] >= 0.5).astype(int) == predictions["target"]
    )
    splits = {}
    for name, rows in partitions.items():
        split_days = sorted(rows["source_date"].unique())
        splits[name] = {
            **_support(rows), "dates": split_days,
            "start_date": split_days[0], "end_date": split_days[-1],
        }
    metrics = {
        "version": _VERSION,
        "selected_model": selected_model,
        "feature_columns": list(feature_columns),
        "mirror_rules": mirror_rules,
        "controls": {
            "constant_0_5": "Literal 0.5, orientation-neutral chance control",
            "class_prior": "Literal TRAIN class prior, orientation-asymmetric diagnostic control",
        },
        "models": model_scores,
        "candidate_parameters": {name: _candidate_parameters(candidate)
                                 for name, candidate in candidates.items()},
        "development_walk_forward": _development_folds(partitions["train"], feature_columns),
        "selection": {
            "partition": "validation", "metric": "log_loss", "fit_partition": "train",
            "candidate_log_loss": validation_log_loss,
            "selected_log_loss": validation_log_loss[selected_model],
            "class_prior": class_prior,
            "candidate_order": forecast_candidates,
            "tie_break": "pure_elo first, then declared candidate order on exact ties",
        },
        "splits": splits,
        "support": _support(frame),
        "assumptions": dict(_ASSUMPTIONS),
    }
    bundle = candidates[selected_model]
    return metrics, predictions, bundle
