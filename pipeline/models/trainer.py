"""Train, evaluate, persist, and serve CS2 match-win probability models."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.models.metrics import evaluate_predictions


RowsOrPath = Union[str, Path, Sequence[Mapping[str, Any]]]
_LABEL_COLUMNS = ("y", "label", "target", "team1_win", "team1_won")
_NON_FEATURE_COLUMNS = {
    "y", "label", "target", "team1_win", "team1_won", "winner", "outcome",
    "match_id", "date", "timestamp", "team1", "team2", "score1", "score2",
}


class TemperatureCalibrator:
    """Shrink a probability toward 0.5 by dividing its logit by temperature."""

    def __init__(self, temperature: float = 1.5) -> None:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")
        self.temperature = float(temperature)

    def transform(self, probability: float) -> float:
        epsilon = 1e-12
        probability = min(max(float(probability), epsilon), 1.0 - epsilon)
        logit = math.log(probability / (1.0 - probability)) / self.temperature
        if logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    shrink_probability = transform

    def __call__(self, probability: float) -> float:
        return self.transform(probability)

def _inverted_features(feature_dict: Mapping[str, float]) -> Dict[str, float]:
    """Invert team-oriented predictors while retaining shared match context."""
    inverted = dict(feature_dict)
    paired = {
        key: value
        for key, value in feature_dict.items()
        if key.endswith("_team1_rd") or key.endswith("_team2_rd")
        or key.endswith("_team1_conservative") or key.endswith("_team2_conservative")
    }
    for key, value in feature_dict.items():
        if key.endswith("_diff"):
            inverted[key] = -value
        elif key == "h2h_win_rate_team1" or key.endswith("_prob"):
            inverted[key] = 1.0 - value
    for key, value in paired.items():
        if "_team1_" in key:
            inverted[key.replace("_team1_", "_team2_")] = value
        elif "_team2_" in key:
            inverted[key.replace("_team2_", "_team1_")] = value
    return inverted


class ModelTrainer:
    """Fit comparative binary classifiers using train-only feature preprocessing."""

    def __init__(self) -> None:
        self.models: Dict[str, Dict[str, Any]] = {}
        self.feature_columns: List[str] = []
        self.evaluation_summary: Dict[str, Any] = {}
        self.best_model_name: str | None = None

    @staticmethod
    def _load_rows(data: RowsOrPath) -> List[Dict[str, Any]]:
        if isinstance(data, (str, Path)):
            with Path(data).open("r", encoding="utf-8-sig", newline="") as stream:
                return [dict(row) for row in csv.DictReader(stream)]
        return [dict(row) for row in data]

    @staticmethod
    def _number(value: Any) -> float:
        if value is None or value == "":
            return float("nan")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return number if math.isfinite(number) else float("nan")

    @classmethod
    def _labels(cls, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not rows:
            raise ValueError("Training and evaluation data must contain rows")
        key = next((name for name in _LABEL_COLUMNS if name in rows[0]), None)
        if key is None:
            raise ValueError("Rows must contain a binary target column (for example, 'y')")
        labels: List[int] = []
        for index, row in enumerate(rows):
            value = cls._number(row.get(key))
            if not math.isfinite(value) or value not in (0.0, 1.0):
                raise ValueError(f"Target at row {index} must be 0 or 1")
            labels.append(int(value))
        return np.asarray(labels, dtype=np.int64)

    @classmethod
    def _feature_names(cls, rows: Sequence[Mapping[str, Any]]) -> List[str]:
        names: List[str] = []
        for key in rows[0]:
            if str(key).casefold() in _NON_FEATURE_COLUMNS:
                continue
            values = [row.get(key) for row in rows]
            if any(value not in (None, "") and not math.isfinite(cls._number(value)) for value in values):
                continue
            if any(value not in (None, "") for value in values):
                names.append(str(key))
        if not names:
            raise ValueError("Training rows contain no numeric predictor columns")
        return names

    @classmethod
    def _matrix(cls, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
        return np.asarray([[cls._number(row.get(name)) for name in columns] for row in rows], dtype=float)

    @classmethod
    def _elo_probabilities(cls, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        result = []
        for row in rows:
            probability = cls._number(row.get("elo_prob"))
            if not math.isfinite(probability):
                elo_diff = cls._number(row.get("elo_diff"))
                probability = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0)) if math.isfinite(elo_diff) else 0.5
            result.append(float(np.clip(probability, 0.0, 1.0)))
        return np.asarray(result, dtype=float)

    @staticmethod
    def _probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
        positive_index = list(estimator.classes_).index(1)
        return estimator.predict_proba(features)[:, positive_index]

    def train_and_evaluate(self, train_data: RowsOrPath, test_data: RowsOrPath) -> Dict[str, Any]:
        """Train Elo, logistic, boosted, and calibrated models and compare scores."""
        train_rows = self._load_rows(train_data)
        test_rows = self._load_rows(test_data)
        y_train = self._labels(train_rows)
        y_test = self._labels(test_rows)
        if len(np.unique(y_train)) != 2:
            raise ValueError("Training targets must contain both classes")

        self.feature_columns = self._feature_names(train_rows)
        x_train = self._matrix(train_rows, self.feature_columns)
        x_test = self._matrix(test_rows, self.feature_columns)

        logistic = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        gradient_boosting = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(max_iter=100, random_state=42)),
        ])
        min_class_size = int(np.bincount(y_train).min())
        calibrated_cv = min(5, min_class_size)
        if calibrated_cv < 2:
            raise ValueError("At least two training examples per class are required for calibration")
        calibrated = CalibratedClassifierCV(
            estimator=Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", HistGradientBoostingClassifier(max_iter=100, random_state=42)),
            ]),
            method="sigmoid",
            cv=calibrated_cv,
        )
        fitted: Dict[str, Any] = {
            "logistic_regression": logistic,
            "gradient_boosting": gradient_boosting,
            "calibrated_gradient_boosting": calibrated,
        }
        self.models = {"elo": {"kind": "elo"}}
        for name, estimator in fitted.items():
            estimator.fit(x_train, y_train)
            self.models[name] = {"kind": "estimator", "estimator": estimator}
        self.models["sota_ensemble"] = {"kind": "sota_ensemble"}

        probabilities: Dict[str, Dict[str, np.ndarray]] = {
            "elo": {"train": self._elo_probabilities(train_rows), "test": self._elo_probabilities(test_rows)}
        }
        for name, estimator in fitted.items():
            probabilities[name] = {
                "train": self._probabilities(estimator, x_train),
                "test": self._probabilities(estimator, x_test),
            }
        summary: Dict[str, Any] = {}
        for name, prediction in probabilities.items():
            summary[name] = {
                "train": evaluate_predictions(y_train.tolist(), prediction["train"].tolist()),
                "test": evaluate_predictions(y_test.tolist(), prediction["test"].tolist()),
            }
        self.evaluation_summary = summary
        self.best_model_name = min(summary, key=lambda name: summary[name]["test"]["log_loss"])

        artifact_dir = Path("artifacts/models")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.save_model(self.best_model_name, str(artifact_dir / "best_model.joblib"))
        with (artifact_dir / "evaluation_summary.json").open("w", encoding="utf-8") as stream:
            json.dump({"best_model": self.best_model_name, "feature_columns": self.feature_columns, "models": summary}, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return {
            "best_model": self.best_model_name,
            "feature_columns": self.feature_columns,
            "models": summary,
        }

    def save_model(self, model_name: str, output_path: str) -> str:
        """Persist a selected trained model and its feature-column contract."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model {model_name!r}; train models before saving")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_name": model_name, "model": self.models[model_name], "feature_columns": self.feature_columns}, path)
        return str(path)

    @staticmethod
    def load_model(model_path: str) -> Any:
        """Load a serialized model bundle from disk."""
        return joblib.load(model_path)

    def predict_proba(self, model_name: str, feature_dict: Dict[str, float]) -> float:
        """Return the probability that team1 wins the supplied fixture."""
        if model_name in ("best", "best_model"):
            if not self.best_model_name or self.best_model_name not in self.models:
                raise ValueError("No best model determined yet; train models before predicting")
            model_name = self.best_model_name
        elif model_name not in self.models:
            raise ValueError(f"Unknown model {model_name!r}; train models before predicting")

        model = self.models[model_name]
        if model["kind"] == "elo":
            return float(self._elo_probabilities([feature_dict])[0])
        if model["kind"] == "sota_ensemble":
            calibrator = TemperatureCalibrator()
            p_gbdt = calibrator(self.predict_proba_symmetric(
                "calibrated_gradient_boosting", feature_dict
            ))
            p_lr = calibrator(self.predict_proba_symmetric(
                "logistic_regression", feature_dict
            ))
            return 0.60 * p_gbdt + 0.40 * p_lr
        features = self._matrix([feature_dict], self.feature_columns)
        return float(self._probabilities(model["estimator"], features)[0])

    def predict_proba_symmetric(
        self,
        model_name: str,
        feature_dict: Dict[str, float],
        inverted_feature_dict: Optional[Dict[str, float]] = None,
        temperature: Optional[float] = None,
    ) -> float:
        """Average forward and reverse predictions to enforce order symmetry."""
        inverted = (
            inverted_feature_dict
            if inverted_feature_dict is not None
            else _inverted_features(feature_dict)
        )
        forward_probability = self.predict_proba(model_name, feature_dict)
        reverse_probability = self.predict_proba(model_name, inverted)
        sym_prob = (forward_probability + (1.0 - reverse_probability)) / 2.0
        if temperature is not None and temperature > 0.0:
            sym_prob = TemperatureCalibrator(temperature).transform(sym_prob)
        return sym_prob
