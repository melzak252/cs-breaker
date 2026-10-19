"""TRAIN-fitted, side-symmetric learners for the broader historical series cohort.

Feature provenance and online replay belong to series_history/valve_standings.
This module accepts only declared predictors, fits no evaluation outcomes, and
selects on a separately bounded development cohort. Odds never enter a learner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.models.metrics import evaluate_predictions
from pipeline.models.player_baseline import _calibration_slope


VERSION = 1
EPSILON = 1e-12
_FORBIDDEN = {
    "target", "y", "winner", "label", "outcome", "score1", "score2", "result",
    "hltv_match_id", "match_id", "date", "source_date", "team1", "team2",
    "p_bookmaker", "p_model", "prob_team1_win", "prob_team2_win", "odds",
    "price1", "price2", "price_team1", "price_team2", "bookmaker",
}


def _calendar(value: str) -> str:
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Expected a canonical ISO source day")
    return value


def _schema(columns, rules):
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("A forecast needs unique ordered predictor columns")
    if any(not isinstance(name, str) or name.casefold() in _FORBIDDEN for name in columns):
        raise ValueError("Results, identities and prices cannot be predictor columns")
    for name in columns:
        rule = rules.get(name, {})
        operation, source = rule.get("operation"), rule.get("source")
        if operation not in {"negate", "complement", "identity", "swap"} or source not in columns:
            raise ValueError(f"Missing or invalid mirror semantics for {name}")
        if operation == "swap":
            if rules.get(source) != {"operation": "swap", "source": name}:
                raise ValueError("Mirror swaps must be reciprocal")
        elif source != name:
            raise ValueError("Only a declared swap can read another feature")


def _matrix(frame, columns, rules):
    _schema(columns, rules)
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Predictors must be a DataFrame with unique columns")
    if any(name not in frame for name in columns):
        raise ValueError("Missing declared predictor columns")
    try:
        values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictors must be finite numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("Predictors must be finite numeric values")
    for index, name in enumerate(columns):
        if rules[name]["operation"] == "complement" and np.any((values[:, index] < 0) | (values[:, index] > 1)):
            raise ValueError("Probability-channel features must lie in [0,1]")
    return values


def _mirror_matrix(values, columns, rules):
    mirrored = values.copy()
    positions = {name: i for i, name in enumerate(columns)}
    for i, name in enumerate(columns):
        operation = rules[name]["operation"]
        if operation == "negate":
            mirrored[:, i] = -values[:, i]
        elif operation == "complement":
            mirrored[:, i] = 1 - values[:, i]
        elif operation == "swap":
            mirrored[:, i] = values[:, positions[rules[name]["source"]]]
    return mirrored


def mirror_features(frame: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Reverse declared predictors; metadata and labels are not inferred or changed."""
    columns = list(rules)
    values = _matrix(frame, columns, rules)
    result = frame.copy()
    result.loc[:, columns] = _mirror_matrix(values, columns, rules)
    return result


def _logits(values):
    centered = values - 0.5
    distance = np.minimum(np.abs(centered), 0.5 - EPSILON)
    return np.sign(centered) * np.log((0.5 + distance) / (0.5 - distance))


def _inputs(values, columns, rules):
    # Probability semantics, not a suffix heuristic, determine this transform.
    result = values.copy()
    for i, name in enumerate(columns):
        if rules[name]["operation"] == "complement":
            result[:, i] = _logits(values[:, i])
    return result


def score_predictions(target, probability):
    labels = np.asarray(target).tolist()
    values = np.asarray(probability).tolist()
    metrics = evaluate_predictions(labels, values)
    if not labels:
        for name in ("log_loss", "brier", "accuracy", "roc_auc", "ece"):
            metrics[name] = None
    elif len(set(labels)) < 2:
        metrics["roc_auc"] = None
    return metrics


@dataclass
class SeriesForecaster:
    recipe: dict
    mirror_rules: dict
    train_start: str
    train_end: str
    estimator: Any = None
    slope: float = 1.0
    components: tuple["SeriesForecaster", ...] = ()
    schema_version: int = VERSION
    selection: dict = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.schema_version != VERSION:
            raise ValueError("Unsupported series forecaster schema")
        columns = self.recipe["feature_columns"]
        values = _matrix(frame, columns, self.mirror_rules)
        family = self.recipe["family"]
        if family == "blend":
            if len(self.components) != 2:
                raise ValueError("Incomplete persisted blend")
            weight = self.recipe["parameters"]["right_weight"]
            if not 0 <= weight <= 1:
                raise ValueError("Invalid blend weight")
            probability = ((1 - weight) * self.components[0].predict(frame)
                           + weight * self.components[1].predict(frame))
        elif family in {"rating", "calibrated_rating"}:
            if len(columns) != 1 or self.mirror_rules[columns[0]]["operation"] != "complement":
                raise ValueError("Rating recipe needs one declared probability channel")
            if family == "rating":
                probability = values[:, 0]
            else:
                if not np.isfinite(self.slope) or not 0 <= self.slope <= 3:
                    raise ValueError("Invalid persisted calibration slope")
                probability = expit(self.slope * _logits(values[:, 0]))
        elif family in {"ridge", "hgb"}:
            if self.estimator is None or getattr(self.estimator, "n_features_in_", None) != len(columns):
                raise ValueError("Incomplete fitted classifier or feature schema mismatch")
            reverse = _mirror_matrix(values, columns, self.mirror_rules)
            forward_p = self.estimator.predict_proba(_inputs(values, columns, self.mirror_rules))[:, 1]
            reverse_p = self.estimator.predict_proba(_inputs(reverse, columns, self.mirror_rules))[:, 1]
            probability = 0.5 + 0.5 * (forward_p - reverse_p)
        else:
            raise ValueError("Unsupported series forecast family")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Forecaster produced invalid probabilities")
        return probability


def _training_rows(frame, train_end):
    cutoff = _calendar(train_end)
    if not len(frame) or not {"source_date", "target"}.issubset(frame):
        raise ValueError("Nonempty dated TRAIN outcomes are required")
    days = frame["source_date"].map(_calendar)
    if (days > cutoff).any():
        raise ValueError("Coefficient fitting cannot consume outcomes after TRAIN")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError("TRAIN labels must be binary")
    if "hltv_match_id" in frame and frame.hltv_match_id.astype(str).duplicated().any():
        raise ValueError("Duplicate native TRAIN outcomes")
    return days, frame["target"].to_numpy(dtype=int)


def fit_candidate(recipe: dict, train: pd.DataFrame, rules: dict, *, train_end: str,
                  fitted_candidates: dict | None = None) -> SeriesForecaster:
    days, target = _training_rows(train, train_end)
    columns = list(recipe["feature_columns"])
    selected_rules = {name: dict(rules[name]) for name in columns if name in rules}
    values = _matrix(train, columns, selected_rules)
    model = SeriesForecaster(recipe=dict(recipe, feature_columns=columns), mirror_rules=selected_rules,
                             train_start=min(days), train_end=train_end)
    family = recipe["family"]
    if family == "blend":
        candidates = fitted_candidates or {}
        params = recipe["parameters"]
        model.components = tuple(candidates[name] for name in (params["left"], params["right"]))
        if any(component.train_end != train_end for component in model.components):
            raise ValueError("Blend components must share the frozen training boundary")
    elif family == "calibrated_rating":
        if len(columns) != 1 or selected_rules[columns[0]]["operation"] != "complement":
            raise ValueError("Calibration needs a single probability channel")
        model.slope = _calibration_slope(values[:, 0], target)
    elif family == "rating":
        if len(columns) != 1 or selected_rules[columns[0]]["operation"] != "complement":
            raise ValueError("A raw rating needs a single probability channel")
    elif family in {"ridge", "hgb"}:
        reverse = _mirror_matrix(values, columns, selected_rules)
        inputs = np.concatenate((_inputs(values, columns, selected_rules),
                                 _inputs(reverse, columns, selected_rules)))
        labels = np.concatenate((target, 1 - target))
        half_life = recipe.get("recency_half_life_days")
        weights = np.ones(len(train), dtype=float)
        if half_life is not None:
            if not np.isfinite(half_life) or half_life <= 0:
                raise ValueError("Recency half-life must be positive")
            origin = date.fromisoformat(train_end).toordinal()
            elapsed = np.array([origin - date.fromisoformat(day).toordinal() for day in days])
            weights = np.exp2(-elapsed / half_life)
            weights /= weights.mean()
        weights = np.concatenate((weights, weights))
        if family == "ridge":
            model.estimator = Pipeline([
                ("scale", StandardScaler(with_mean=False)),
                ("classifier", LogisticRegression(C=recipe["parameters"]["C"], fit_intercept=False,
                                                   max_iter=1000, random_state=42)),
            ])
            model.estimator.fit(inputs, labels, scale__sample_weight=weights,
                                classifier__sample_weight=weights)
        else:
            model.estimator = HistGradientBoostingClassifier(
                **recipe["parameters"], early_stopping=False, random_state=42)
            model.estimator.fit(inputs, labels, sample_weight=weights)
    else:
        raise ValueError("Unsupported forecast family")
    return model


def candidate_recipes(feature_groups: dict, rules: dict, vrs_columns: list[str]) -> list[dict]:
    """The bounded ladder in the frozen challenge protocol; no data-dependent grid."""
    ratings = list(feature_groups["ratings"])
    form = list(feature_groups["form"])
    performance = list(feature_groups["performance"])
    groups = {"ratings": ratings, "ratings_form": ratings + form,
              "all": ratings + form + performance,
              "ratings_vrs": ratings + vrs_columns,
              "ratings_form_vrs": ratings + form + vrs_columns,
              "all_vrs": ratings + form + performance + vrs_columns}
    recipes = []

    def add(name, family, columns, parameters=None, half_life=None):
        _schema(columns, rules)
        recipes.append({"name": name, "family": family, "feature_columns": list(columns),
                        "parameters": parameters or {}, "recency_half_life_days": half_life})

    for column in feature_groups["probabilities"]:
        stem = column.removesuffix("_prob")
        add("raw_" + stem, "rating", [column])
        add("calibrated_" + stem, "calibrated_rating", [column])
    for group in ("ratings", "ratings_form", "all", "ratings_vrs", "all_vrs"):
        for strength in (0.001, 0.01, 0.1, 1.0):
            for half_life in (180, 365, None):
                suffix = str(half_life) if half_life is not None else "none"
                add(f"ridge_C{strength:g}_{group}_h{suffix}", "ridge", groups[group],
                    {"C": strength}, half_life)
    for group in ("ratings_form_vrs", "all_vrs"):
        for leaves in (3, 7, 15):
            for regularization in (1, 10):
                for minimum in (50, 150):
                    add(f"hgb_leaf{leaves}_l2_{regularization}_min{minimum}_{group}", "hgb", groups[group],
                        {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": leaves,
                         "l2_regularization": regularization, "min_samples_leaf": minimum}, 365)
    blends = [("calibrated_map_elo_32_180", "ridge_C0.1_all_vrs_h365", weight)
              for weight in (0.25, 0.5, 0.75)]
    blends.append(("hgb_leaf7_l2_10_min150_all_vrs", "ridge_C0.1_all_vrs_h365", 0.5))
    by_name = {recipe["name"]: recipe for recipe in recipes}
    for left, right, weight in blends:
        columns = list(dict.fromkeys(by_name[left]["feature_columns"] + by_name[right]["feature_columns"]))
        add(f"blend_{left}__{right}_w{weight:g}", "blend", columns,
            {"left": left, "right": right, "right_weight": weight})
    if len(recipes) > 256 or len({recipe["name"] for recipe in recipes}) != len(recipes):
        raise ValueError("Candidate ladder exceeds frozen bounds or repeats a recipe")
    return recipes


def select_development_model(train, validation, market_ids, recipes, rules, *, train_end,
                             validation_end, progress: Callable | None = None):
    """Return selected model, development evidence and VAL-only candidate forecasts."""
    _training_rows(train, train_end)
    end = _calendar(validation_end)
    days = validation["source_date"].map(_calendar)
    if not len(validation) or (days <= train_end).any() or (days > end).any():
        raise ValueError("Selection requires only the declared post-TRAIN development days")
    if not validation["target"].isin([0, 1]).all():
        raise ValueError("Development labels must be binary")
    identities = validation.hltv_match_id.astype(str)
    if identities.duplicated().any():
        raise ValueError("Duplicate native development outcomes")
    market_ids = {str(identity) for identity in market_ids}
    if not market_ids or not market_ids.issubset(set(identities)):
        raise ValueError("Fixed development market cohort is empty or outside development")
    mask = identities.isin(market_ids).to_numpy()
    fitted, scores, forecasts = {}, {}, {}
    for index, recipe in enumerate(recipes):
        name = recipe["name"]
        if name in fitted:
            raise ValueError("Duplicate candidate recipe")
        model = fit_candidate(recipe, train, rules, train_end=train_end, fitted_candidates=fitted)
        probability = model.predict(validation)
        fitted[name] = model
        forecasts[name] = probability
        scores[name] = {
            "market_validation": score_predictions(validation.target.to_numpy()[mask], probability[mask]),
            "full_validation": score_predictions(validation.target, probability),
        }
        if progress is not None:
            progress(index + 1, len(recipes), name, scores[name])
    if not scores:
        raise ValueError("No development candidates supplied")
    minimum = min(score["market_validation"]["log_loss"] for score in scores.values())
    selected = min(name for name, score in scores.items()
                   if score["market_validation"]["log_loss"] <= minimum + 1e-12)
    selection = {"selected_model": selected, "partition": "validation", "fit_partition": "train",
                 "metric": "log_loss", "cohort": "fixed_one_hour_available_book_union", "tie_tolerance": 1e-12,
                 "candidate_log_loss": {name: score["market_validation"]["log_loss"] for name, score in scores.items()},
                 "market_native_ids": sorted(market_ids, key=int), "training_matches": len(train),
                 "validation_matches": len(validation), "market_validation_matches": len(market_ids),
                 "train_end": train_end, "validation_end": validation_end,
                 "train_plus_validation_refit": False, "test_used_for_selection": False}
    fitted[selected].selection = selection
    evidence = {"selected_model": selected, "selection": selection, "candidate_scores": scores,
                "candidate_recipes": recipes, "fitted_training_start": min(train.source_date),
                "fitted_training_end": max(train.source_date)}
    return fitted[selected], evidence, forecasts


def predict_series_scenario(model, state, team1, team2, day, format="bo3"):
    """Score a future literal-organization scenario, not a verified live lineup."""
    from pipeline.features.bayesian_series import BAYES_FEATURE_COLUMNS, BayesianSeriesEngine
    from pipeline.features.series_history import SeriesHistoryEngine
    from pipeline.features.valve_standings import StandingsHistory

    day = _calendar(day)
    if not isinstance(model, SeriesForecaster) or not isinstance(state, dict) or state.get("schema_version") != 2:
        raise ValueError("Unsupported persisted series model/state; publish a schema-2 inference state")
    engine, standings = state.get("engine"), state.get("standings")
    if not isinstance(engine, SeriesHistoryEngine) or not isinstance(standings, StandingsHistory):
        raise ValueError("Missing typed historical feature state")
    if state.get("model_selected") != model.recipe["name"] or state.get("history_through") != engine.history_through:
        raise ValueError("Persisted model and historical state disagree")
    bayesian = state.get("bayesian_engine")
    bayesian_columns = sorted(set(model.recipe["feature_columns"]) & set(BAYES_FEATURE_COLUMNS))
    if bayesian is not None:
        if not isinstance(bayesian, BayesianSeriesEngine) or bayesian.history_through != engine.history_through:
            raise ValueError("Bayesian and base feature histories must agree")
    elif bayesian_columns:
        raise ValueError("Selected model requires its persisted Bayesian history")
    if day <= model.train_end or (engine.history_through is not None and day <= engine.history_through):
        raise ValueError("Scenario must be after both coefficient fitting and every saved history day")
    features = engine.features(team1, team2, day, format=format)
    features.update(standings.features(team1, team2, day))
    if bayesian_columns:
        features.update(bayesian.features(team1, team2, day, format=format))
    probability = float(model.predict(pd.DataFrame([features]))[0])
    support_fields = (
        "team1_prior_series", "team2_prior_series", "minimum_prior_series", "either_team_cold",
        "performance_available", "minimum_prior_measured_maps", "performance_age_days",
        "vrs_available", "vrs_age_days",
    )
    return {
        "team1": team1, "team2": team2, "source_date": day, "format": format,
        "prob_team1_win": probability, "prob_team2_win": 1 - probability,
        "model_selected": model.recipe["name"], "fit_through": model.train_end,
        "history_through": engine.history_through,
        "competitiveness_qualified": bool(state.get("competitiveness_qualified", False)),
        "scenario": "literal-organization historical-strength forecast; current starters not verified",
        "support": {name: features[name] for name in support_fields},
        "bayesian_channels_used": bayesian_columns,
        "availability_limits": state.get("availability_limits", []),
        "execution_or_profitability_claim": False,
    }
