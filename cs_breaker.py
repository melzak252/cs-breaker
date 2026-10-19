"""Retrospective, UTC-day-batched CS2 winner baseline; Python standard library only."""

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import groupby
from pathlib import Path

K_VALUES = (0, 8, 16, 24, 32, 48, 64)
BO3_SCORES = {(2, 0), (2, 1), (0, 2), (1, 2)}
DEFAULT_DATA = Path("data/exports/kaggle/results_only.csv")
DEFAULT_OUTPUT = Path("artifacts/baseline")


@dataclass(frozen=True)
class Match:
    match_id: str
    timestamp: datetime
    team1: str
    team2: str
    y: int


def load_results(path):
    """Keep original identities; do not merge teams or infer formats from scores."""
    matches, excluded, seen = [], [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"hltv_match_id", "date", "team1_name", "team2_name", "winner",
                    "score_team1", "score_team2", "match_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input lacks required result columns")
        for line, row in enumerate(reader, 2):
            key = row["hltv_match_id"]
            a, b = row["team1_name"], row["team2_name"]
            if not key or key in seen or not a.strip() or not b.strip() or a == b:
                raise ValueError(f"Row {line}: missing/duplicate identity or invalid teams")
            seen.add(key)
            timestamp = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
            if timestamp.utcoffset() is None:
                raise ValueError(f"Row {line}: timestamp must include a timezone")
            timestamp = timestamp.astimezone(timezone.utc)
            s1, s2 = int(row["score_team1"]), int(row["score_team2"])
            if min(s1, s2) < 0 or s1 == s2 or row["winner"] not in ("team1", "team2"):
                raise ValueError(f"Row {line}: invalid result")
            y = int(row["winner"] == "team1")
            if y != int(s1 > s2):
                raise ValueError(f"Row {line}: score disagrees with winner")
            if row["match_type"].lower() != "bo3" or (s1, s2) not in BO3_SCORES:
                excluded.append({**row, "exclusion_reason": "not_bo3_score_compatible"})
                continue
            matches.append(Match(key, timestamp, a, b, y))
    matches.sort(key=lambda m: (m.timestamp, m.match_id))
    return matches, excluded


def win_probability(rating1, rating2):
    # This is directly a series-winner probability, not a map-win probability.
    z = (rating1 - rating2) * math.log(10) / 400
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1 + exp_z)


class Elo:
    def __init__(self, k):
        if not math.isfinite(k) or k < 0:
            raise ValueError("K must be finite and non-negative")
        self.k = k
        self.ratings = {}
        self.counts = Counter()
        self.last_played = {}
        self.last_day = None

    def process_day(self, matches):
        """Predict the entire UTC date before applying any of its outcomes."""
        if not matches:
            return []
        day = matches[0].timestamp.date()
        if any(m.timestamp.date() != day for m in matches):
            raise ValueError("A rating update batch must contain exactly one UTC date")
        if self.last_day is not None and day <= self.last_day:
            raise ValueError("Rating batches must advance strictly in date order")
        rows, deltas = [], defaultdict(list)
        for m in sorted(matches, key=lambda m: (m.timestamp, m.match_id)):
            p = win_probability(self.ratings.get(m.team1, 1500), self.ratings.get(m.team2, 1500))
            rows.append({"match_id": m.match_id, "date": m.timestamp.isoformat(),
                         "team1": m.team1, "team2": m.team2, "y": m.y, "p_elo": p,
                         "prior_matches_team1": self.counts[m.team1],
                         "prior_matches_team2": self.counts[m.team2],
                         "history_dates_before": day.isoformat()})
            delta = self.k * (m.y - p)
            deltas[m.team1].append(delta)
            deltas[m.team2].append(-delta)
        for team, changes in deltas.items():
            self.ratings[team] = self.ratings.get(team, 1500) + math.fsum(changes)
            self.counts[team] += len(changes)
            self.last_played[team] = day.isoformat()
        self.last_day = day
        return rows


def replay(matches, k):
    model, rows = Elo(k), []
    for _, batch in groupby(matches, key=lambda m: m.timestamp.date()):
        rows.extend(model.process_day(list(batch)))
    return rows, model


def score(labels, probabilities):
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Metrics require non-empty, aligned labels and probabilities")
    if any(y not in (0, 1) for y in labels) or any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid binary labels or probabilities")
    n = len(labels)
    clipped = [min(1 - 1e-15, max(1e-15, p)) for p in probabilities]
    loss = -math.fsum(y * math.log(p) + (1 - y) * math.log1p(-p)
                      for y, p in zip(labels, clipped)) / n
    positives = sum(labels)
    negatives = n - positives
    auc = None
    if positives and negatives:
        below, wins = 0, 0.0
        ranked = sorted(zip(probabilities, labels))
        for _, tied in groupby(ranked, key=lambda pair: pair[0]):
            group = list(tied)
            pos = sum(y for _, y in group)
            neg = len(group) - pos
            wins += pos * (below + neg / 2)
            below += neg
        auc = wins / (positives * negatives)
    bins = [[] for _ in range(10)]
    for y, p in zip(labels, probabilities):
        bins[min(9, int(p * 10))].append((y, p))
    calibration = []
    ece = 0.0
    for index, items in enumerate(bins):
        count = len(items)
        observed = sum(y for y, _ in items) / count if count else None
        predicted = math.fsum(p for _, p in items) / count if count else None
        if count:
            ece += count / n * abs(observed - predicted)
        calibration.append({"lower": index / 10, "upper": (index + 1) / 10,
                            "n": count, "mean_probability": predicted, "win_rate": observed})
    return {"n": n, "log_loss": loss,
            "brier": math.fsum((p - y) ** 2 for y, p in zip(labels, probabilities)) / n,
            "accuracy": sum(int(p >= 0.5) == y for y, p in zip(labels, probabilities)) / n,
            "roc_auc": auc, "ece_10_bins": ece, "calibration_bins": calibration}


def evaluate(matches, train_end, validation_end):
    if train_end >= validation_end:
        raise ValueError("Training boundary must precede validation boundary")
    matches = sorted(matches, key=lambda m: (m.timestamp, m.match_id))

    def phase(day):
        return "train" if day < train_end else "validation" if day < validation_end else "test"

    cohorts = {name: [m for m in matches if phase(m.timestamp.date()) == name]
               for name in ("train", "validation", "test")}
    if any(not group for group in cohorts.values()):
        raise ValueError("Training, validation and test cohorts must all be non-empty")
    development = [m for m in matches if m.timestamp.date() < validation_end]
    candidates = []
    for k in K_VALUES:
        rows, _ = replay(development, k)
        validation = [r for r in rows if date.fromisoformat(r["date"][:10]) >= train_end]
        metrics = score([r["y"] for r in validation], [r["p_elo"] for r in validation])
        candidates.append({"k": k, "log_loss": metrics["log_loss"], "brier": metrics["brier"]})
    selected = min(candidates, key=lambda c: (c["log_loss"], c["k"]))["k"]
    rows, fitted = replay(matches, selected)
    prior = sum(m.y for m in cohorts["train"]) / len(cohorts["train"])
    for row in rows:
        row.update(phase=phase(date.fromisoformat(row["date"][:10])), p_coin_flip=0.5,
                   p_training_prior=prior)
    metrics = {}
    for name in ("validation", "test"):
        group = [r for r in rows if r["phase"] == name]
        labels = [r["y"] for r in group]
        metrics[name] = {model: score(labels, [r[column] for r in group])
                         for model, column in (("elo", "p_elo"), ("coin_flip", "p_coin_flip"),
                                               ("training_prior", "p_training_prior"))}
    test = [r for r in rows if r["phase"] == "test"]
    support = {}
    for unseen in (0, 1, 2):
        group = [r for r in test if sum(r[k] == 0 for k in
                                      ("prior_matches_team1", "prior_matches_team2")) == unseen]
        support[str(unseen)] = {"n": len(group), "elo": score(
            [r["y"] for r in group], [r["p_elo"] for r in group]) if group else None}
    monthly = {}
    for month, group in groupby(test, key=lambda r: r["date"][:7]):
        group = list(group)
        monthly[month] = {model: score([r["y"] for r in group], [r[column] for r in group])
                          for model, column in (("elo", "p_elo"), ("coin_flip", "p_coin_flip"))}
    summary = {"schema_version": 1, "selected_k": selected, "k_candidates": list(K_VALUES),
               "selection": "minimum validation log loss; ties use smaller K",
               "train_end_exclusive": train_end.isoformat(),
               "validation_end_exclusive": validation_end.isoformat(),
               "cohorts": {name: {"n": len(group), "first": group[0].timestamp.isoformat(),
                                  "last": group[-1].timestamp.isoformat()}
                           for name, group in cohorts.items()},
               "validation_candidates": candidates, "training_team1_win_rate": prior,
               "metrics": metrics, "test_unseen_team_counts": support, "test_monthly": monthly,
               "protocol": "UTC-day batches; all predictions precede all updates for that date",
               "accuracy_tie_rule": "p >= 0.5 predicts team1",
               "limitations": ["Historical result-release timestamps are unavailable; daily batching is an assumption, not verified publication-time safety.",
                               "Eligibility uses BO3-compatible scores, not independently verified format labels.",
                               "Literal team-name identities; no roster continuity or alias reconciliation.",
                               "No odds, profit evaluation, map predictions, or live data."]}
    model = {"schema_version": 1, "selected_k": selected,
             "history_through": fitted.last_day.isoformat(), "ratings": fitted.ratings,
             "counts": dict(fitted.counts), "last_played": fitted.last_played,
             "target": "BO3-score-compatible series winner", "protocol": summary["protocol"]}
    return summary, rows, model


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def backtest(data, output, train_end, validation_end):
    data, output = Path(data), Path(output)
    matches, excluded = load_results(data)
    summary, predictions, model = evaluate(matches, train_end, validation_end)
    summary["input"] = {"path": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                        "total_rows": len(matches) + len(excluded), "eligible_rows": len(matches),
                        "excluded_rows": len(excluded), "identity": "hltv_match_id"}
    summary["excluded_score_pairs"] = dict(sorted(Counter(
        f'{r["score_team1"]}-{r["score_team2"]}' for r in excluded).items()))
    model["source_sha256"] = summary["input"]["sha256"]
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", summary), ("model.json", model)):
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                   encoding="utf-8")
    write_csv(output / "predictions.csv", predictions, list(predictions[0]))
    excluded_fields = list(excluded[0]) if excluded else ["hltv_match_id", "exclusion_reason"]
    write_csv(output / "excluded.csv", excluded, excluded_fields)
    return summary


def name_key(name):
    return unicodedata.normalize("NFC", name).strip().casefold()


def predict_matchup(model, team1, team2):
    if model.get("schema_version") != 1:
        raise ValueError("Unsupported model schema")
    ratings = model["ratings"]

    def resolve(name):
        if name in ratings:
            return name
        candidates = [known for known in ratings if name_key(known) == name_key(name)]
        if len(candidates) != 1:
            kind = "Ambiguous" if candidates else "Unknown"
            raise ValueError(f"{kind} team {name!r}; use an exact name from the saved model")
        return candidates[0]

    a, b = resolve(team1), resolve(team2)
    if a == b:
        raise ValueError("A matchup requires two different teams")
    p = win_probability(ratings[a], ratings[b])
    return {"team1": a, "team2": b, "p_team1": p, "p_team2": 1 - p,
            "forecast_kind": "historical_snapshot", "history_through": model["history_through"],
            "selected_k": model["selected_k"],
            "prior_matches": {a: model["counts"][a], b: model["counts"][b]},
            "last_played": {a: model["last_played"][a], b: model["last_played"][b]},
            "warning": "Historical name-based ratings, not a current/live forecast or roster-aware model."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("backtest", help="Select K on validation, then score the final test")
    run.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--train-end", type=date.fromisoformat, default=date(2025, 1, 1))
    run.add_argument("--validation-end", type=date.fromisoformat, default=date(2025, 7, 1))
    predict = commands.add_parser("predict", help="Predict from a historical saved rating snapshot")
    predict.add_argument("team1")
    predict.add_argument("team2")
    predict.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.json")
    args = parser.parse_args()
    try:
        if args.command == "backtest":
            report = backtest(args.data, args.output, args.train_end, args.validation_end)
            result = {"input": report["input"], "cohorts": report["cohorts"],
                      "selected_k": report["selected_k"], "output": str(args.output),
                      "test_metrics": {name: {k: v for k, v in metrics.items() if k != "calibration_bins"}
                                       for name, metrics in report["metrics"]["test"].items()}}
        else:
            model = json.loads(args.model.read_text(encoding="utf-8"))
            result = predict_matchup(model, args.team1, args.team2)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
