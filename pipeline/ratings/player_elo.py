"""Retrospective map-winner player Elo on explicitly joined source-calendar rosters."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from itertools import groupby
from pathlib import Path

from cs_breaker import K_VALUES, win_probability, write_csv
from pipeline.models.metrics import evaluate_predictions

NAMESPACE = "strandedpond:nickname:"
DEFAULT_MAPS = Path("data/raw/strandedpond/matches.csv")
DEFAULT_PLAYERS = Path("data/raw/strandedpond/player_stats.csv")
DEFAULT_OUTPUT = Path("artifacts/player_ratings")


@dataclass(frozen=True)
class MapObservation:
    map_id: str
    series_id: str
    timestamp: datetime
    map_name: str
    team1: str
    team2: str
    roster1: tuple[str, ...]
    roster2: tuple[str, ...]
    y: int


def unique_rows(rows):
    seen, unique = set(), []
    for row in rows:
        key = tuple(row.items())
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique, len(rows) - len(unique)

def load_maps(map_path, player_path):
    with Path(map_path).open(encoding="utf-8-sig", newline="") as stream:
        maps = list(csv.DictReader(stream))
    with Path(player_path).open(encoding="utf-8-sig", newline="") as stream:
        players = list(csv.DictReader(stream))
    return load_map_rows(maps, players)



def load_map_rows(map_rows, player_rows):
    """Validate exact historical map/roster joins without performing file I/O."""
    maps, map_duplicates = unique_rows(map_rows)
    players, player_duplicates = unique_rows(player_rows)
    by_map, series_days, join_targets = defaultdict(list), defaultdict(set), defaultdict(set)
    rosters = defaultdict(set)
    for row in maps:
        key = (row["match_url"], row["map"])
        by_map[key].append(row)
        series_days[key[0]].add(row["date"][:10])
        for side in ("left", "right"):
            join_targets[(row["event"], row["date"], row["map"], row[f"team_{side}"])].add(key)
    for row in players:
        rosters[(row["event"], row["date"], row["map"], row["team"])].add(row["player_name"])
    observations, excluded, reasons_count = [], [], Counter()
    for (url, map_name), variants in sorted(by_map.items()):
        row = variants[0]
        reasons = set()
        if len(variants) != 1:
            reasons.add("conflicting_map_representations")
        if len(series_days[url]) > 1:
            reasons.add("series_calendar_date_conflict")
        lineups = []
        for side in ("left", "right"):
            join = (row["event"], row["date"], map_name, row[f"team_{side}"])
            names = rosters[join]
            if len(join_targets[join]) != 1:
                reasons.add("ambiguous_roster_join")
            if len(names) != 5 or any(not name.strip() for name in names):
                reasons.add("not_five_named_players")
            lineups.append(tuple(NAMESPACE + name for name in sorted(names)))
        if set(lineups[0]) & set(lineups[1]):
            reasons.add("overlapping_rosters")
        identity = re.search(r"/matches/(\d+)(?:/|$)", url)
        try:
            timestamp = datetime.strptime(row["date"], "%Y-%m-%d %H:%M")
            s1, s2, rounds = int(row["score_left"]), int(row["score_right"]), int(row["total_rounds"])
            valid = (identity is not None and row["team_left"] != row["team_right"]
                     and min(s1, s2) >= 0 and s1 != s2 and s1 + s2 == rounds
                     and row["winner"] == row["team_left" if s1 > s2 else "team_right"])
            if not valid:
                reasons.add("invalid_map_outcome_or_identity")
        except ValueError:
            reasons.add("invalid_map_outcome_or_identity")
        if reasons:
            reasons_count.update(reasons)
            excluded.append({**row, "reasons": ";".join(sorted(reasons)),
                             "source_dates": json.dumps(sorted({v["date"] for v in variants})),
                             "source_row_variants": len(variants)})
            continue
        series_id = identity.group(1)
        observations.append(MapObservation(f"{series_id}:{map_name}", series_id, timestamp,
                                           map_name, row["team_left"], row["team_right"],
                                           lineups[0], lineups[1], int(s1 > s2)))
    observations.sort(key=lambda m: (m.timestamp, m.map_id))
    audit = {"raw_map_rows": len(maps) + map_duplicates,
             "exact_duplicate_map_rows": map_duplicates,
             "raw_player_rows": len(players) + player_duplicates,
             "exact_duplicate_player_rows": player_duplicates,
             "unique_map_keys": len(by_map), "eligible_maps": len(observations),
             "eligible_series": len({m.series_id for m in observations}),
             "excluded_map_keys": len(excluded), "exclusion_reasons_overlap": dict(reasons_count),
             "series_with_calendar_date_conflict": sum(len(days) > 1 for days in series_days.values()),
             "identity_namespace": NAMESPACE,
             "time_semantics": "Source calendar dates; timezone and original result release are unknown"}
    return observations, audit, excluded


def validate_rosters(roster1, roster2):
    if any(len(r) != 5 or len(set(r)) != 5 or any(not p for p in r) for r in (roster1, roster2)):
        raise ValueError("Each roster must contain exactly five unique player identities")
    if set(roster1) & set(roster2):
        raise ValueError("Opposing rosters must not overlap")


class PlayerElo:
    def __init__(self, k_player, k_team):
        if any(not math.isfinite(k) or k < 0 for k in (k_player, k_team)):
            raise ValueError("Elo K values must be finite and non-negative")
        self.k_player, self.k_team = k_player, k_team
        self.players, self.teams = {}, {}
        self.map_counts = Counter()
        self.series = defaultdict(set)
        self.organizations = defaultdict(set)
        self.last_played = {}
        self.last_day = None

    def process_day(self, observations):
        if not observations:
            return []
        day = observations[0].timestamp.date()
        if self.last_day is not None and day <= self.last_day:
            raise ValueError("Source-date rating batches must advance strictly")
        for m in observations:
            validate_rosters(m.roster1, m.roster2)
            if m.timestamp.date() != day or m.y not in (0, 1) or m.team1 == m.team2:
                raise ValueError("A batch requires one source date and valid map outcomes/teams")
        rows, player_delta, team_delta = [], defaultdict(list), defaultdict(list)
        for m in sorted(observations, key=lambda m: (m.timestamp, m.map_id)):
            means = [math.fsum(self.players.get(p, 1500) for p in roster) / 5
                     for roster in (m.roster1, m.roster2)]
            p_player = win_probability(*means)
            p_team = win_probability(self.teams.get(m.team1, 1500), self.teams.get(m.team2, 1500))
            counts = [self.map_counts[p] for p in m.roster1 + m.roster2]
            rows.append({"map_id": m.map_id, "series_id": m.series_id,
                         "date": m.timestamp.isoformat(), "map": m.map_name,
                         "team1": m.team1, "team2": m.team2, "y": m.y,
                         "roster1": json.dumps(m.roster1), "roster2": json.dumps(m.roster2),
                         "p_player_elo": p_player, "p_team_elo": p_team,
                         "prior_player_maps": json.dumps(counts),
                         "minimum_prior_player_maps": min(counts),
                         "history_source_dates_before": day.isoformat()})
            for roster, sign in ((m.roster1, 1), (m.roster2, -1)):
                for player in roster:
                    player_delta[player].append(sign * self.k_player * (m.y - p_player))
            team_delta[m.team1].append(self.k_team * (m.y - p_team))
            team_delta[m.team2].append(-self.k_team * (m.y - p_team))
        for player, deltas in player_delta.items():
            self.players[player] = self.players.get(player, 1500) + math.fsum(deltas)
            self.map_counts[player] += len(deltas)
            self.last_played[player] = day.isoformat()
        for team, deltas in team_delta.items():
            self.teams[team] = self.teams.get(team, 1500) + math.fsum(deltas)
        for m in observations:
            for team, roster in ((m.team1, m.roster1), (m.team2, m.roster2)):
                for player in roster:
                    self.series[player].add(m.series_id)
                    self.organizations[player].add(team)
        self.last_day = day
        return rows


def replay(observations, k_player, k_team):
    model, rows = PlayerElo(k_player, k_team), []
    for _, batch in groupby(observations, key=lambda m: m.timestamp.date()):
        rows.extend(model.process_day(list(batch)))
    return rows, model


def evaluate(observations, train_end, validation_end, *, identity_namespace=NAMESPACE,
             comparison_entity_kind="organization_name",
             time_semantics="Source calendar-day batching; timezone/result-release clocks unavailable"):
    if train_end >= validation_end:
        raise ValueError("Training must precede validation")
    observations = sorted(observations, key=lambda m: (m.timestamp, m.map_id))

    def phase(day):
        return "train" if day < train_end else "validation" if day < validation_end else "test"

    cohorts = {name: [m for m in observations if phase(m.timestamp.date()) == name]
               for name in ("train", "validation", "test")}
    if any(not c for c in cohorts.values()):
        raise ValueError("Training, validation and test cohorts must all be non-empty")
    development = [m for m in observations if m.timestamp.date() < validation_end]
    candidates = []
    for k in K_VALUES:
        rows, _ = replay(development, k, k)
        rows = [r for r in rows if r["date"][:10] >= train_end.isoformat()]
        labels = [r["y"] for r in rows]
        candidates.append({"k": k, "player_log_loss": evaluate_predictions(
            labels, [r["p_player_elo"] for r in rows])["log_loss"],
            "team_log_loss": evaluate_predictions(labels, [r["p_team_elo"] for r in rows])["log_loss"]})
    k_player = min(candidates, key=lambda c: (c["player_log_loss"], c["k"]))["k"]
    k_team = min(candidates, key=lambda c: (c["team_log_loss"], c["k"]))["k"]
    rows, fitted = replay(observations, k_player, k_team)
    prior = sum(m.y for m in cohorts["train"]) / len(cohorts["train"])
    for row in rows:
        row.update(phase=phase(date.fromisoformat(row["date"][:10])), p_training_prior=prior,
                   p_coin_flip=0.5)
    comparison_name = "team_elo" if comparison_entity_kind == "organization_name" else "lineup_elo"
    columns = {"player_elo": "p_player_elo", comparison_name: "p_team_elo",
               "training_prior": "p_training_prior", "coin_flip": "p_coin_flip"}

    def metrics(group):
        return {model: evaluate_predictions([r["y"] for r in group], [r[col] for r in group])
                for model, col in columns.items()} if group else None

    support = {}
    for name in cohorts:
        group = [r for r in rows if r["phase"] == name]
        support[name] = {"maps": len(group), "has_cold_player": sum(
            r["minimum_prior_player_maps"] == 0 for r in group),
            "cold_player_appearances": sum(json.loads(r["prior_player_maps"]).count(0) for r in group),
            "all_ten_players_at_least_prior_maps": {str(n): sum(
                r["minimum_prior_player_maps"] >= n for r in group) for n in (1, 5, 10, 20, 50)}}
    test = [r for r in rows if r["phase"] == "test"]
    history_counts = {str(n): sum(c >= n for c in fitted.map_counts.values()) for n in (1, 5, 10, 20, 50, 100)}
    series_counts = {str(n): sum(len(s) >= n for s in fitted.series.values()) for n in (1, 5, 10, 20, 50)}
    summary = {"schema_version": 1, "target": "map_winner", "identity_namespace": identity_namespace,
               "comparison_entity_kind": comparison_entity_kind, "time_semantics": time_semantics,
               "selected_k": {"player": k_player, "team": k_team},
               "validation_candidates": candidates, "train_end_exclusive": train_end.isoformat(),
               "validation_end_exclusive": validation_end.isoformat(),
               "cohorts": {name: {"maps": len(c), "series": len({m.series_id for m in c}),
                                  "first": c[0].timestamp.isoformat(), "last": c[-1].timestamp.isoformat()}
                           for name, c in cohorts.items()},
               "metrics": {name: metrics([r for r in rows if r["phase"] == name])
                           for name in ("validation", "test")},
               "test_support_metrics": {str(n): metrics([r for r in test if r["minimum_prior_player_maps"] >= n])
                                        for n in (1, 5, 10, 20)},
               "support": support, "players": len(fitted.players),
               "map_history_thresholds": history_counts, "series_history_thresholds": series_counts,
               "players_observed_with_multiple_comparison_entities": sum(len(s) > 1 for s in fitted.organizations.values()),
               "limitations": (["Nickname identities are source-specific handles, not verified persistent persons/SteamIDs."]
                               if identity_namespace == NAMESPACE else
                               ["Organization attribution is unavailable; the comparator tracks exact lineup fingerprints."]) + [
                               "As-played lineups are historical roster scenarios, not proven pre-match announcements.",
                               "Original result publication clocks are unavailable; chronology is retrospective.",
                               "Shared win/loss updates do not identify individual performance or impact.",
                               "Global map strength only: no map/veto specialization, Glicko uncertainty, neural history, or odds."]}
    model = {"schema_version": 1, "target": "map_winner", "identity_namespace": identity_namespace,
             "comparison_entity_kind": comparison_entity_kind,
             "history_through": fitted.last_day.isoformat(), "selected_k": summary["selected_k"],
             "players": fitted.players, "teams": fitted.teams, "map_counts": dict(fitted.map_counts),
             "series_counts": {p: len(s) for p, s in fitted.series.items()},
             "last_played": fitted.last_played,
             "comparison_entities": {p: sorted(s) for p, s in fitted.organizations.items()}}
    return summary, rows, model


def ranking_rows(model):
    ranked = sorted(model["players"], key=lambda p: (-model["players"][p], p))
    return [{"player_id": p, "player_name": model.get("display_names", {}).get(
                 p, p.removeprefix(model["identity_namespace"])),
             "elo": model["players"][p], "maps": model["map_counts"][p],
             "series": model["series_counts"][p], "last_played": model["last_played"][p],
             "comparison_entities": json.dumps(model["comparison_entities"][p])} for p in ranked]


def backtest(map_path, player_path, output, train_end, validation_end):
    observations, audit, excluded = load_maps(map_path, player_path)
    summary, rows, model = evaluate(observations, train_end, validation_end)
    summary["source_audit"] = audit
    summary["sources"] = {name: {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                          for name, path in (("maps", map_path), ("players", player_path))}
    model["sources"] = summary["sources"]
    save_run(summary, rows, model, excluded, output)
    return summary


def save_run(summary, rows, model, excluded, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", summary), ("model.json", model)):
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    write_csv(output / "predictions.csv", rows, list(rows[0]))
    rankings = ranking_rows(model)
    write_csv(output / "rankings.csv", rankings, list(rankings[0]))
    fields = list(excluded[0]) if excluded else ["match_url", "map", "reasons"]
    write_csv(output / "excluded_maps.csv", excluded, fields)


def predict_rosters(model, roster1, roster2):
    if model.get("schema_version") != 1:
        raise ValueError("Unsupported player model schema")
    players = model["players"]

    def resolve(roster):
        resolved = tuple(p if p in players else model["identity_namespace"] + p for p in roster)
        unknown = [p for p in resolved if p not in players]
        if unknown:
            raise ValueError(f"Unknown player identities: {unknown}; exact source player identifiers are required")
        return resolved

    a, b = resolve(roster1), resolve(roster2)
    validate_rosters(a, b)
    p = win_probability(*(math.fsum(players[player] for player in roster) / 5 for roster in (a, b)))
    return {"target": "map_winner", "forecast_kind": "historical_roster_scenario",
            "history_through": model["history_through"], "roster1": a, "roster2": b,
            "p_roster1": p, "p_roster2": 1 - p,
            "prior_player_maps": {player: model["map_counts"][player] for player in a+b},
            "warning": "Historical player ratings; as-played roster scenario, not a current/live forecast."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("backtest")
    run.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    run.add_argument("--players", type=Path, default=DEFAULT_PLAYERS)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--train-end", type=date.fromisoformat, default=date(2026, 4, 1))
    run.add_argument("--validation-end", type=date.fromisoformat, default=date(2026, 7, 1))
    rank = commands.add_parser("rankings")
    rank.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.json")
    rank.add_argument("--min-maps", type=int, default=20)
    rank.add_argument("--limit", type=int, default=20)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.json")
    predict.add_argument("--roster1", nargs=5, required=True)
    predict.add_argument("--roster2", nargs=5, required=True)
    args = parser.parse_args()
    try:
        if args.command == "backtest":
            report = backtest(args.maps, args.players, args.output, args.train_end, args.validation_end)
            result = {"selected_k": report["selected_k"], "cohorts": report["cohorts"],
                      "players": report["players"], "source_audit": report["source_audit"],
                      "test_metrics": {name: {k: v for k, v in values.items() if k != "calibration_bins"}
                                       for name, values in report["metrics"]["test"].items()}}
        else:
            model = json.loads(args.model.read_text(encoding="utf-8"))
            if args.command == "rankings":
                if args.min_maps < 0 or args.limit < 1:
                    raise ValueError("min-maps must be non-negative and limit positive")
                result = {"history_through": model["history_through"], "identity_namespace": model["identity_namespace"],
                          "warning": "Win/loss roster ratings, not isolated individual-performance rankings.",
                          "players": [r for r in ranking_rows(model) if r["maps"] >= args.min_maps][:args.limit]}
            else:
                result = predict_rosters(model, tuple(args.roster1), tuple(args.roster2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
