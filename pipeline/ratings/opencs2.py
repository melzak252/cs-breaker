"""SteamID player Elo using roster membership and explicit per-round side evidence."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, timezone
from pathlib import Path

from cs_breaker import write_csv
from pipeline.ratings.player_elo import MapObservation, evaluate, save_run

NAMESPACE = "opencs2:steamid:"
DEFAULT_MANIFEST = Path("data/raw/opencs2/manifest.json")
DEFAULT_OUTPUT = Path("artifacts/steam_player_ratings")


def reconstruct_map(row):
    """Do not assign anonymous roster groups to named organizations or fixed CT/T."""
    teams = row.get("teams") or []
    if len(teams) != 2:
        raise ValueError("not_two_rosters")
    ids = [{str(p["steamid"]) for p in team["players"]} for team in teams]
    if any(len(r) != 5 or "0" in r for r in ids) or ids[0] & ids[1]:
        raise ValueError("not_two_disjoint_fives")
    group_by_id = {steam: group for group, roster in enumerate(ids) for steam in roster}
    rounds = row.get("rounds") or []
    if (not rounds or len({r["round_num"] for r in rounds}) != len(rounds)
            or len(rounds) != row.get("rounds_played")):
        raise ValueError("missing_or_duplicate_rounds")
    sides = defaultdict(lambda: defaultdict(set))
    for kill in row.get("kills") or []:
        for prefix in ("attacker", "victim"):
            steam, side = str(kill[f"{prefix}_steamid"]), kill[f"{prefix}_side"]
            if steam == "0":  # World damage is not a third roster member.
                continue
            if steam not in group_by_id or side not in ("ct", "t"):
                raise ValueError("event_player_or_side_not_in_roster")
            sides[kill["round_num"]][group_by_id[steam]].add(side)
    wins = [0, 0]
    for rnd in rounds:
        evidence = sides[rnd["round_num"]]
        if rnd["winner"] not in ("ct", "t") or not evidence:
            raise ValueError("round_winner_or_side_evidence_missing")
        if any(len(s) != 1 for s in evidence.values()):
            raise ValueError("conflicting_sides_within_round")
        assignment = {group: next(iter(s)) for group, s in evidence.items()}
        if len(assignment) == 1:
            group = next(iter(assignment))
            assignment[1 - group] = "t" if assignment[group] == "ct" else "ct"
        if assignment[0] == assignment[1]:
            raise ValueError("rosters_on_same_side")
        winner = next(group for group, side in assignment.items() if side == rnd["winner"])
        wins[winner] += 1
    winner, loser = max(wins), min(wins)
    if not ((winner == 13 and loser <= 11)
            or (winner >= 16 and winner % 3 == 1 and winner - loser in (2, 3, 4))):
        raise ValueError("not_standard_complete_mr12_score")
    timestamp = row.get("match_date")
    if timestamp is None or timestamp.utcoffset() is None:
        raise ValueError("missing_aware_match_date")
    rosters = [tuple(NAMESPACE + steam for steam in sorted(group)) for group in ids]
    entities = ["lineup:" + hashlib.sha256("|".join(roster).encode()).hexdigest() for roster in rosters]
    observation = MapObservation(row["row_id"], row["match_id"], timestamp.astimezone(timezone.utc),
                                 row["map_name"], entities[0], entities[1], rosters[0], rosters[1],
                                 int(wins[0] > wins[1]))
    detail = {"map_id": observation.map_id, "series_id": observation.series_id,
              "date": observation.timestamp.isoformat(), "map": observation.map_name,
              "roster_scores": wins, "roster1": rosters[0], "roster2": rosters[1],
              "aggregate_score_ct": (row.get("match") or {}).get("score_ct"),
              "aggregate_score_t": (row.get("match") or {}).get("score_t")}
    return observation, detail


def load_metadata(manifest_path):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("OpenCS2 inspection requires PyArrow; use the project's .venv Python") from exc
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    observations, labels, excluded, names = [], [], [], {}
    series_days = defaultdict(set)
    total = 0
    for file in manifest["files"]:
        shard = manifest_path.parent / Path(file["path"]).name
        payload = shard.read_bytes()
        if len(payload) != file["bytes"] or hashlib.sha256(payload).hexdigest() != file["sha256"]:
            raise ValueError(f"Pinned source checksum/size mismatch: {shard}")
        for row in pq.read_table(pa.BufferReader(payload)).to_pylist():
            total += 1
            try:
                observation, detail = reconstruct_map(row)
            except ValueError as exc:
                excluded.append({"map_id": row["row_id"], "series_id": row["match_id"],
                                 "source_shard": file["path"], "reasons": str(exc)})
                continue
            observations.append(observation)
            labels.append({**detail, "source_shard": file["path"]})
            series_days[observation.series_id].add(observation.timestamp.date())
            for team in row["teams"]:
                for player in team["players"]:
                    identity = NAMESPACE + str(player["steamid"])
                    candidate = (observation.timestamp, observation.map_id, player["name"])
                    if identity not in names or candidate > names[identity]:
                        names[identity] = candidate
    conflicted = {key for key, days in series_days.items() if len(days) > 1}
    for observation in observations:
        if observation.series_id in conflicted:
            excluded.append({"map_id": observation.map_id, "series_id": observation.series_id,
                             "source_shard": "", "reasons": "series_calendar_date_conflict"})
    observations = [m for m in observations if m.series_id not in conflicted]
    eligible_ids = {m.map_id for m in observations}
    labels = [label for label in labels if label["map_id"] in eligible_ids]
    eligible_players = {p for m in observations for p in m.roster1 + m.roster2}
    observations.sort(key=lambda m: (m.timestamp, m.map_id))
    if len(eligible_ids) != len(observations):
        raise ValueError("Duplicate reconstructed map identity")
    audit = {"raw_maps": total, "eligible_maps": len(observations), "excluded_maps": len(excluded),
             "eligible_series": len({m.series_id for m in observations}),
             "exclusion_reasons": dict(Counter(r["reasons"] for r in excluded)),
             "source_repository": manifest["repository"], "source_revision": manifest["revision"],
             "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
             "verified_shards": len(manifest["files"]), "identity_namespace": NAMESPACE,
             "outcome_method": "For each round, map current CT/T winner to stable roster group using kill participant IDs and their current sides",
             "organization_attribution": "Not inferred; comparison entities are exact roster fingerprints"}
    return observations, audit, excluded, labels, {p: names[p][2] for p in sorted(eligible_players)}


def backtest(manifest_path, output, train_end, validation_end):
    observations, audit, excluded, labels, display_names = load_metadata(manifest_path)
    summary, predictions, model = evaluate(
        observations, train_end, validation_end, identity_namespace=NAMESPACE,
        comparison_entity_kind="exact_lineup_fingerprint",
        time_semantics="UTC match-date batches; original result publication clocks unavailable")
    summary["source_audit"] = audit
    model["source_audit"] = audit
    model["display_names"] = display_names
    save_run(summary, predictions, model, excluded, output)
    fields = list(labels[0])
    csv_labels = [{key: json.dumps(value) if isinstance(value, (list, tuple)) else value
                   for key, value in row.items()} for row in labels]
    write_csv(Path(output) / "reconstructed_labels.csv", csv_labels, fields)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("backtest")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--train-end", type=date.fromisoformat, default=date(2026, 4, 1))
    run.add_argument("--validation-end", type=date.fromisoformat, default=date(2026, 4, 11))
    args = parser.parse_args()
    try:
        report = backtest(args.manifest, args.output, args.train_end, args.validation_end)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    result = {"selected_k": report["selected_k"], "cohorts": report["cohorts"],
              "players": report["players"], "source_audit": report["source_audit"],
              "test_metrics": {name: {k: v for k, v in values.items() if k != "calibration_bins"}
                               for name, values in report["metrics"]["test"].items()}}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
