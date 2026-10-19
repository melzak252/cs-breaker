"""Command-line workflows for the CS2 forecasting pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence



def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", default="data/exports/kaggle/results_only.csv", help="match-results CSV source")
    parser.add_argument("--opencs2-dir", default="data/raw/opencs2", help="OpenCS2 parquet source directory")
    parser.add_argument("--limit-demos", type=int, help="maximum number of demo shards to load")
    parser.add_argument("--csgo-csv", default=None, help="CS:GO historical games CSV (e.g. data/raw/csgo_historical/historic_games_list.csv)")
    parser.add_argument("--limit-csgo", type=int, help="maximum number of CS:GO matches to load")


def _sync(args: argparse.Namespace) -> int:
    from pipeline.ingestion.liquipedia import LiquipediaClient
    client = LiquipediaClient()
    tournaments = client.get_tournaments(year=args.year, tier=args.tier)
    fixtures = []
    for tournament in tournaments:
        fixtures.extend(client.get_tournament_matches(tournament.name))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "year": args.year,
        "tournaments": [tournament.to_dict() for tournament in tournaments],
        "fixtures": [fixture.to_dict() for fixture in fixtures],
    }
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Synced {len(tournaments)} tournaments and {len(fixtures)} fixtures → {output}")
    return 0


def _download_demo(args: argparse.Namespace) -> int:
    from pipeline.ingestion.downloader import DemoDownloader
    records = DemoDownloader().download_and_extract_demo(args.url, args.match_id, args.dest_dir)
    print(f"Downloaded and extracted {len(records)} demo file(s):")
    for record in records:
        print(f"  {record.map_name}: {record.file_path}")
    return 0


def _parse_demo(args: argparse.Namespace) -> int:
    from pipeline.parser.aggregator import EventAggregator
    from pipeline.parser.demoparser import CS2DemoParser
    parser = CS2DemoParser(args.demo_path)
    events = parser.parse_all_events()
    header = events.get("header", {})
    map_name = header.get("map_name") or Path(args.demo_path).stem
    result = EventAggregator.aggregate_from_events(
        match_id=Path(args.demo_path).stem,
        map_name=str(map_name),
        events=events,
        team1_name=args.team1,
        team2_name=args.team2,
    )
    print(f"Map: {result.map_name} | {result.team1_name} {result.team1_score} - {result.team2_score} {result.team2_name}")
    print(f"Rounds: {result.total_rounds} | Winner: {result.winner_team}")
    print("\nPlayer scoreboard")
    print(f"{'Player':<22} {'Team':<18} {'K':>3} {'D':>3} {'A':>3} {'ADR':>7} {'K/D':>6}")
    for player in sorted(result.player_stats, key=lambda item: (-item.kills, item.player_name.casefold())):
        print(f"{player.player_name:<22.22} {player.team_name:<18.18} {player.kills:>3} {player.deaths:>3} {player.assists:>3} {player.adr:>7.1f} {player.kd_ratio:>6.2f}")
    if events.get("parse_error"):
        print(f"Note: parser returned limited data: {events['parse_error']}", file=sys.stderr)
    return 0


def _build_features(args: argparse.Namespace) -> int:
    from pipeline.features.dataset import DatasetPipeline
    pipeline = DatasetPipeline()
    pipeline.load_sources(
        results_csv_path=args.results,
        opencs2_dir=args.opencs2_dir,
        limit_demos=args.limit_demos,
        csgo_csv_path=args.csgo_csv,
        limit_csgo=args.limit_csgo,
    )
    train_rows, test_rows, manifest = pipeline.build_chronological_dataset(args.train_ratio)
    if not DatasetPipeline.verify_zero_leakage(train_rows, test_rows):
        raise ValueError("Chronological feature split failed leakage verification")
    paths = pipeline.save_dataset(train_rows, test_rows, args.output_dir)
    print(f"Built {manifest['match_count']} chronological rows ({len(train_rows)} train, {len(test_rows)} test).")
    print("Zero-leakage check: passed")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from pipeline.models.trainer import ModelTrainer
    trainer = ModelTrainer()
    summary = trainer.train_and_evaluate(args.train, args.test)
    print(f"Best model: {summary['best_model']}")
    print(f"Saved model: {args.model_path}")
    print("Test metrics (lower log loss is better):")
    for name, report in summary["models"].items():
        metrics = report["test"]
        print(f"  {name:<30} log_loss={metrics['log_loss']:.4f}  brier={metrics['brier']:.4f}  accuracy={metrics['accuracy']:.3f}")
    if args.model_path != "artifacts/models/best_model.joblib":
        trainer.save_model(summary["best_model"], args.model_path)
    return 0


def _predict(args: argparse.Namespace) -> int:
    from pipeline.contracts import MatchFixture
    from pipeline.features.builder import ChronologicalFeatureBuilder
    from pipeline.features.dataset import DatasetPipeline
    from pipeline.models.trainer import ModelTrainer
    # Rebuild online state in timestamp order so the forecast uses only completed history.
    dataset = DatasetPipeline()
    matches, demo_lookup = dataset.load_sources(
        results_csv_path=args.results,
        opencs2_dir=args.opencs2_dir,
        limit_demos=args.limit_demos,
        csgo_csv_path=args.csgo_csv,
        limit_csgo=args.limit_csgo,
    )
    builder = ChronologicalFeatureBuilder()
    builder.process_matches(matches, demo_lookup)
    t1_players = [p.strip() for p in args.team1_players.split(",") if p.strip()] if getattr(args, "team1_players", None) else []
    t2_players = [p.strip() for p in args.team2_players.split(",") if p.strip()] if getattr(args, "team2_players", None) else []
    if not t1_players:
        for m in reversed(matches):
            if m.team1.casefold() == args.team1.casefold() and m.team1_players:
                t1_players = m.team1_players
                break
            elif m.team2.casefold() == args.team1.casefold() and m.team2_players:
                t1_players = m.team2_players
                break
    if not t2_players:
        for m in reversed(matches):
            if m.team1.casefold() == args.team2.casefold() and m.team1_players:
                t2_players = m.team1_players
                break
            elif m.team2.casefold() == args.team2.casefold() and m.team2_players:
                t2_players = m.team2_players
                break

    fixture = MatchFixture(
        match_id="upcoming-prediction",
        tournament_name="prediction",
        timestamp=datetime.now(timezone.utc),
        team1=args.team1,
        team2=args.team2,
        team1_players=t1_players,
        team2_players=t2_players,
    )
    features = builder.build_match_features(fixture, args.map_name)
    system = getattr(args, "rating_system", "ml").lower()
    if system == "glicko2":
        probability = builder.glicko2_engine.predict_win(args.team1, args.team2, t1_players, t2_players)
        source_name = "Glicko-2"
    elif system == "trueskill":
        probability = builder.trueskill_engine.predict_win(args.team1, args.team2, t1_players, t2_players)
        source_name = "TrueSkill"
    elif system == "openskill":
        probability = builder.openskill_engine.predict_win(args.team1, args.team2, t1_players, t2_players)
        source_name = "OpenSkill"
    elif system == "elo":
        t1_elo, _, _ = builder.player_engine.compute_lineup_rating(t1_players, team_fallback=args.team1)
        t2_elo, _, _ = builder.player_engine.compute_lineup_rating(t2_players, team_fallback=args.team2)
        z = (t1_elo - t2_elo) * math.log(10) / 400.0
        probability = 1.0 / (1.0 + math.exp(-z))
        source_name = "Elo (Lineup)"
    else:
        bundle = ModelTrainer.load_model(args.model_path)
        trainer = ModelTrainer()
        trainer.models = {bundle["model_name"]: bundle["model"]}
        trainer.best_model_name = bundle["model_name"]
        trainer.feature_columns = bundle["feature_columns"]
        if getattr(args, "symmetric", False):
            probability = trainer.predict_proba_symmetric("best", features, temperature=args.temperature)
            source_name = f"ML Symmetric ({bundle['model_name']})"
        else:
            probability = trainer.predict_proba("best", features)
            if getattr(args, "temperature", None):
                probability = TemperatureCalibrator(args.temperature).transform(probability)
            source_name = f"ML ({bundle['model_name']})"

    confidence = abs(probability - 0.5) * 2.0
    if t1_players:
        print(f"Roster {args.team1}: {', '.join(t1_players)}")
    if t2_players:
        print(f"Roster {args.team2}: {', '.join(t2_players)}")
    print(f"\n[{source_name} Forecast]")
    print(f"  {args.team1}: {probability:.1%}")
    print(f"  {args.team2}: {1.0 - probability:.1%}")
    print(f"  Confidence: {confidence:.1%} ({'high' if confidence >= 0.6 else 'moderate' if confidence >= 0.25 else 'low'})")
    return 0


def _predict_series(args: argparse.Namespace) -> int:
    """Use the persisted broad model and its paired, trusted local history state."""
    import hashlib
    import pickle

    from pipeline.models.series_forecast import predict_series_scenario

    model_bytes = Path(args.model).read_bytes()
    with Path(args.state).open("rb") as stream:
        state = pickle.load(stream)
    if not isinstance(state, dict) or state.get("model_sha256") != hashlib.sha256(model_bytes).hexdigest():
        raise ValueError("Series inference state is not paired with this exact fitted model")
    model = pickle.loads(model_bytes)
    result = predict_series_scenario(model, state, args.team1, args.team2, args.day, args.format)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def _predict_player(args: argparse.Namespace) -> int:
    """Load trusted local Kedro artifacts and score only an explicit roster scenario."""
    import pickle

    from pipeline.pipelines.match_modeling.nodes import predict_player_match
    from pipeline.ratings.player_elo import validate_rosters

    if not args.team1.strip() or not args.team2.strip() or args.team1 == args.team2:
        raise ValueError("Scenario requires two nonempty distinct literal team names")
    try:
        day = datetime.strptime(args.day, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Forecast day must be a valid ISO date (YYYY-MM-DD)") from None
    if day.isoformat() != args.day:
        raise ValueError("Forecast day must be a valid ISO date (YYYY-MM-DD)")
    roster1, roster2 = tuple(args.roster1), tuple(args.roster2)
    if any(not key.strip() for key in roster1 + roster2):
        raise ValueError("Roster player keys must not be empty or whitespace-only")
    validate_rosters(roster1, roster2)

    # Pickle may execute code: these paths must refer to trusted local artifacts.
    def load_artifact(path, name):
        try:
            with Path(path).open("rb") as stream:
                return pickle.load(stream)
        except (pickle.UnpicklingError, EOFError) as exc:
            raise ValueError(f"Invalid trusted {name} pickle artifact {path}: {exc}") from exc

    bundle = load_artifact(args.model, "model")
    state = load_artifact(args.state, "rating state")
    try:
        forecast = predict_player_match(bundle, state, args.team1, args.team2,
                                        roster1, roster2, args.day)
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid saved player artifacts or explicit roster scenario: {exc}") from exc
    print(json.dumps({
        **forecast,
        "forecast_kind": "explicit_roster_scenario",
        "team1": args.team1, "team2": args.team2, "roster1": roster1, "roster2": roster2,
        "day": args.day, "model_artifact": str(args.model), "state_artifact": str(args.state),
        "lineup_assumption": "User-supplied roster scenario; no inferred or certified starters",
        "network_requests": 0,
    }, indent=2, allow_nan=False))
    return 0


def _player(args: argparse.Namespace) -> int:
    from pipeline.features.builder import ChronologicalFeatureBuilder
    from pipeline.features.dataset import DatasetPipeline
    dataset = DatasetPipeline()
    matches, demo_lookup = dataset.load_sources(
        results_csv_path=args.results,
        opencs2_dir=args.opencs2_dir,
        limit_demos=args.limit_demos,
        csgo_csv_path=args.csgo_csv,
        limit_csgo=args.limit_csgo,
    )
    builder = ChronologicalFeatureBuilder()
    builder.process_matches(matches, demo_lookup)
    engine = builder.player_engine

    if args.top:
        top_list = engine.top_players(limit=args.top, min_matches=args.min_matches)
        print(f"\n{'Rank':<5} {'Player':<20} {'Elo':>6} {'Matches':>8} {'Win%':>7}  {'Recent Teams'}")
        print("-" * 75)
        for i, p in enumerate(top_list, 1):
            teams = ", ".join(p['teams']) if p['teams'] else "N/A"
            print(f"{i:<5} {p['player_name']:<20} {p['elo']:>6.1f} {p['matches_played']:>8} {p['win_rate']*100:>6.1f}%  {teams}")
        return 0

    if args.name:
        p = engine.get_player(args.name)
        print(f"\nPlayer Profile: {p.player_name}")
        print(f"  Individual Elo: {p.elo:.1f}")
        print(f"  Matches Played: {p.matches_played}")
        print(f"  Wins: {p.wins} ({p.win_rate*100:.1f}%)")
        print(f"  Teams Played For: {', '.join(p.teams) if p.teams else 'N/A'}")
        if p.last_active:
            print(f"  Last Active: {p.last_active}")
        return 0

    print("Please specify --name <player> or --top <N>", file=sys.stderr)
    return 2


def _compare_ratings(args: argparse.Namespace) -> int:
    from pipeline.features.dataset import DatasetPipeline
    from pipeline.models.rating_benchmark import RatingBenchmark
    dataset = DatasetPipeline()
    matches, _ = dataset.load_sources(
        results_csv_path=args.results,
        opencs2_dir=args.opencs2_dir,
        limit_demos=args.limit_demos,
        csgo_csv_path=args.csgo_csv,
        limit_csgo=args.limit_csgo,
    )
    if args.limit_matches and len(matches) > args.limit_matches:
        matches = matches[-args.limit_matches:]
    bench = RatingBenchmark()
    results = bench.run_benchmark(matches, train_ratio=args.train_ratio)
    bench.print_summary(results)
    if args.output:
        bench.save_results(results, args.output)
        print(f"\nSaved benchmark results to {args.output}")
    return 0


def _compare_bookmakers(args: argparse.Namespace) -> int:
    """Export native HLTV links and score unchanged TEST forecasts offline."""
    import hashlib
    import sqlite3
    from collections import Counter
    from types import SimpleNamespace

    from cs_app.services import latest_quotes
    from pipeline.models.bookmaker_comparison import PAIRED_FIELDS, compare_hltv_bookmakers
    from scripts import map_cs_oddspapi_history as mapper

    paths = {
        "hltv_results": Path(args.hltv_results),
        "predictions": Path(args.predictions),
        "model_metrics": Path(args.model_metrics),
    }
    fingerprints = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            fingerprints[name] = {
                "path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            }
    results = mapper.read_csv(paths["hltv_results"])
    predictions = mapper.read_csv(paths["predictions"])
    metadata = json.loads(paths["model_metrics"].read_text())
    database = Path(args.database).resolve()
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        store = SimpleNamespace(query=lambda sql, parameters=(): [
            dict(row) for row in connection.execute(sql, parameters)
        ])
        fixtures = [
            json.loads(row["metadata_json"])
            for row in store.query("SELECT metadata_json FROM fixtures ORDER BY start_time,fixture_id")
        ]
        quotes = latest_quotes(store)
        anchors = store.query(
            "SELECT l.fixture_id,h.source,h.source_match_id "
            "FROM links l JOIN historical_matches h ON h.history_key=l.history_key "
            "WHERE l.status IN ('automatic','manual') AND h.source IN ('hltv','opencs2','kaggle')"
        )
        native_results = store.query(
            "SELECT source_match_id,score1,score2,winner_1 FROM historical_matches "
            "WHERE source='oddspapi' AND winner_1 IS NOT NULL"
        )
        snapshot = {
            "path": str(database), "read_only": True,
            "fixtures": len(fixtures), "latest_fixture_book_observations": len(quotes),
            "maximum_selected_quote_id": max((row["id"] for row in quotes), default=None),
        }
    finally:
        connection.rollback()
        connection.close()

    links, candidates = mapper.map_hltv_fixtures(fixtures, mapper.hltv_calendar_series(results))
    matched = {row["fixtureId"]: row for row in links if row["mapping_status"] == "matched"}
    anchor_checks = []
    for anchor in anchors:
        row = matched.get(anchor["fixture_id"])
        if row is not None:
            anchor_checks.append({
                **anchor, "hltv_match_id": row["hltv_match_id"],
                "agrees": anchor["source_match_id"] == row["hltv_match_id"],
            })
    result_checks = []
    for native in native_results:
        row = matched.get(native["source_match_id"])
        if row is None or row["label_status"] != "valid_explicit_series_result":
            continue
        scores = (int(row["hltv_score1"]), int(row["hltv_score2"]))
        if not row["participant1_is_hltv_team1"]:
            scores = scores[::-1]
        result_checks.append({
            "fixtureId": row["fixtureId"], "hltv_match_id": row["hltv_match_id"],
            "agrees": scores == (native["score1"], native["score2"])
            and row["outcome_1_won"] == native["winner_1"],
        })
    if any(not row["agrees"] for row in anchor_checks + result_checks):
        raise ValueError("Direct HLTV crosswalk contradicts an existing native-ID or native-result anchor")

    report, pairs = compare_hltv_bookmakers(
        fixtures, quotes, links, predictions, metadata,
        devig_method=args.devig_method, quote_cutoff=args.quote_cutoff,
    )
    report["inputs"] = {**fingerprints, "odds_database_snapshot": snapshot}
    report["protocol"].update(network_requests=0, database_read_only=True)
    audit = {
        "version": 1,
        "inputs": {**fingerprints, "odds_database_snapshot": snapshot},
        "counts": {
            "provider_fixtures": len(fixtures), "hltv_result_rows": len(results),
            "matched_native_hltv_ids": len(matched),
            "matched_played_series": sum(row["label_status"] == "valid_explicit_series_result" for row in matched.values()),
            "candidate_evidence_rows": len(candidates),
            "model_test_predictions": len(predictions),
            "mapped_test_matches": len({row["hltv_match_id"] for row in matched.values()}
                                       & {row["hltv_match_id"] for row in predictions}),
            "test_matches_with_usable_odds": report["matched_test_matches_any_book"],
        },
        "mapping_statuses": dict(Counter(row["mapping_status"] for row in links)),
        "matched_label_statuses": dict(Counter(row["label_status"] for row in matched.values())),
        "matched_calendar_day_offsets": dict(Counter(str(row["calendar_day_offset"]) for row in matched.values())),
        "native_id_anchor_checks": anchor_checks,
        "native_provider_result_checks": result_checks,
        "protocol": {
            "calendar_window": "HLTV source calendar day within ±1 of provider scheduled UTC day",
            "identity": "Mutually unique normalized full/short participant names plus compatible event; no score-based selection",
            "source_time": "HLTV noon is synthetic; source calendar timezone and actual kickoff are not certified",
            "network_requests": 0, "database_read_only": True,
        },
        "limits": [
            "Audited natural-key links are not independently certified cross-provider entity identities.",
            "Ambiguous names/dates, event conflicts, conflicting source metadata and unfinished provider fixtures are excluded.",
            "Forfeit and nonseries identities may map, but do not receive played-series targets.",
            "Native match-ID mapping does not supply missing historical player lineups or create held-out predictions.",
            "Existing application links are cross-checked, not overwritten; the exported native crosswalk is used for this benchmark.",
        ],
    }
    mapping_dir, audit_dir, output_dir = map(Path, (args.mapping_dir, args.audit_dir, args.output_dir))
    for directory in (mapping_dir, audit_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    mapper.write_csv(mapping_dir / "oddspapi_hltv_matches.csv", mapper.HLTV_LINK_FIELDS, links)
    mapper.write_csv(audit_dir / "oddspapi_hltv_candidates.csv", mapper.HLTV_LINK_FIELDS, candidates)
    mapper.write_json(audit_dir / "oddspapi_hltv_mapping.json", audit)
    mapper.write_json(output_dir / "bookmaker_comparison.json", report)
    mapper.write_csv(output_dir / "bookmaker_comparison_matches.csv", PAIRED_FIELDS, pairs)

    def compact(scores):
        return {name: scores[name] for name in ("accuracy", "log_loss", "brier", "roc_auc")}

    common = report["common_all_books"]
    print(json.dumps({
        "mapping": audit["counts"], "mapping_statuses": audit["mapping_statuses"],
        "selected_model": report["model_selected"],
        "devig_method": args.devig_method, "quote_cutoff_policy": args.quote_cutoff,
        "common_all_books": {
            "matches": common["matches"], "model": compact(common["model"]),
            "bookmakers": {book: compact(scores) for book, scores in common["bookmakers"].items()},
            "paired_differences": common["paired_differences"],
        },
        "per_book_paired_cohorts": {
            book: {"matches": cohort["matches"], "model": compact(cohort["model"]),
                   "bookmaker": compact(cohort["bookmaker"]),
                   "paired_log_loss_difference": cohort["paired_log_loss_difference"],
                   "paired_brier_difference": cohort["paired_brier_difference"]}
            for book, cohort in report["per_book_paired_cohorts"].items()
        },
        "outputs": [str(mapping_dir / "oddspapi_hltv_matches.csv"),
                    str(audit_dir / "oddspapi_hltv_mapping.json"),
                    str(audit_dir / "oddspapi_hltv_candidates.csv"),
                    str(output_dir / "bookmaker_comparison.json"),
                    str(output_dir / "bookmaker_comparison_matches.csv")],
        "network_requests": 0,
    }, indent=2, allow_nan=False))
    return 0


def _run(args: argparse.Namespace) -> int:
    from pipeline.features.dataset import DatasetPipeline
    from pipeline.models.trainer import ModelTrainer
    dataset = DatasetPipeline()
    dataset.load_sources(
        results_csv_path=args.results,
        opencs2_dir=args.opencs2_dir,
        limit_demos=args.limit_demos,
        csgo_csv_path=args.csgo_csv,
        limit_csgo=args.limit_csgo,
    )
    train_rows, test_rows, manifest = dataset.build_chronological_dataset(args.train_ratio)
    if not DatasetPipeline.verify_zero_leakage(train_rows, test_rows):
        raise ValueError("Chronological feature split failed leakage verification")
    paths = dataset.save_dataset(train_rows, test_rows, args.output_dir)
    trainer = ModelTrainer()
    summary = trainer.train_and_evaluate(paths["train_features"], paths["test_features"])
    print("CS2 forecasting workflow complete")
    print(f"  Matches: {manifest['match_count']} ({manifest['train_count']} train / {manifest['test_count']} test)")
    print("  Leakage check: passed")
    print(f"  Best model: {summary['best_model']}")
    print("  Test log loss: " + f"{summary['models'][summary['best_model']]['test']['log_loss']:.4f}")
    print("  Model: artifacts/models/best_model.joblib")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m pipeline.cli", description="CS2 match forecasting pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    sync = commands.add_parser("sync", help="sync Liquipedia tournaments and fixtures")
    sync.add_argument("--year", type=int, default=datetime.now().year)
    sync.add_argument("--tier", help="optional tournament tier filter")
    sync.add_argument("--output", default="data/cache/liquipedia/fixtures.json")
    sync.set_defaults(handler=_sync)

    download = commands.add_parser("download-demo", help="download and extract a match demo")
    download.add_argument("--url", required=True)
    download.add_argument("--match-id", required=True)
    download.add_argument("--dest-dir", default="data/raw/demos")
    download.set_defaults(handler=_download_demo)

    parse = commands.add_parser("parse-demo", help="parse a demo and print its scoreboard")
    parse.add_argument("--demo-path", required=True)
    parse.add_argument("--team1", default="Team A")
    parse.add_argument("--team2", default="Team B")
    parse.set_defaults(handler=_parse_demo)

    features = commands.add_parser("build-features", help="build chronological train/test feature splits")
    _add_source_arguments(features)
    features.add_argument("--train-ratio", type=float, default=0.8)
    features.add_argument("--output-dir", default="data/processed")
    features.set_defaults(handler=_build_features)

    train = commands.add_parser("train", help="train and evaluate forecasting models")
    train.add_argument("--train", default="data/processed/train_features.csv")
    train.add_argument("--test", default="data/processed/test_features.csv")
    train.add_argument("--model-path", default="artifacts/models/best_model.joblib")
    train.set_defaults(handler=_train)

    predict = commands.add_parser("predict", help="forecast a team-versus-team match")
    predict.add_argument("--team1", required=True)
    predict.add_argument("--team2", required=True)
    predict.add_argument("--team1-players", help="comma-separated list of 5 players for team 1")
    predict.add_argument("--team2-players", help="comma-separated list of 5 players for team 2")
    predict.add_argument("--map", dest="map_name", help="optional map name")
    predict.add_argument("--rating-system", choices=["ml", "glicko2", "trueskill", "openskill", "elo"], default="ml", help="rating system or ML model to forecast with")
    predict.add_argument("--model-path", default="artifacts/models/best_model.joblib")
    predict.add_argument("--symmetric", action="store_true", help="use dual-pass symmetric order-invariant prediction")
    predict.add_argument("--temperature", type=float, default=None, help="Bayesian temperature logit shrinkage (e.g. 1.15)")
    _add_source_arguments(predict)
    predict.set_defaults(handler=_predict)

    player_predict = commands.add_parser(
        "predict-player", help="forecast an explicit five-player roster scenario from trusted local Kedro pickles",
    )
    player_predict.add_argument("--model", default="artifacts/baseline_model.pkl",
                                help="trusted local Kedro model pickle (unpickling executes code)")
    player_predict.add_argument("--state", default="artifacts/player_rating_state.pkl",
                                help="trusted local Kedro player/archive history pickle")
    player_predict.add_argument("--team1", required=True)
    player_predict.add_argument("--team2", required=True)
    player_predict.add_argument("--roster1", nargs=5, required=True, metavar="PLAYER_KEY",
                                help="five exact source player keys for team 1; no starter inference")
    player_predict.add_argument("--roster2", nargs=5, required=True, metavar="PLAYER_KEY",
                                help="five exact source player keys for team 2; disjoint from roster1")
    player_predict.add_argument("--day", required=True, help="ISO source day strictly after saved history")
    player_predict.set_defaults(handler=_predict_player)

    series_predict = commands.add_parser(
        "predict-series", help="forecast a future organization scenario using trusted broad-history model pickles",
    )
    series_predict.add_argument("--model", default="artifacts/professional_model_20261010/selected_model.pkl",
                                help="trusted local fitted pickle; unpickling executes code")
    series_predict.add_argument("--state", default="artifacts/professional_model_20261010/series_inference_state.pkl",
                                help="trusted local state paired to the exact model hash")
    series_predict.add_argument("--team1", required=True)
    series_predict.add_argument("--team2", required=True)
    series_predict.add_argument("--day", required=True, help="ISO source day strictly after all saved history")
    series_predict.add_argument("--format", choices=("bo3", "bo5"), default="bo3")
    series_predict.set_defaults(handler=_predict_series)

    compare = commands.add_parser("compare-ratings", help="compare standalone Elo, Glicko-2, TrueSkill, and OpenSkill on chronological data")
    compare.add_argument("--limit-matches", type=int, default=5000, help="matches to evaluate (from latest backwards)")
    compare.add_argument("--train-ratio", type=float, default=0.8)
    compare.add_argument("--output", default="artifacts/models/rating_comparison.json")
    _add_source_arguments(compare)
    compare.set_defaults(handler=_compare_ratings)

    bookmakers = commands.add_parser("compare-bookmakers", help="map downloaded OddsPapi fixtures to native HLTV IDs and compare frozen TEST forecasts offline")
    bookmakers.add_argument("--database", default="data/app/app.sqlite3")
    bookmakers.add_argument("--hltv-results", default="data/raw/hltv/matches.csv")
    bookmakers.add_argument("--predictions", default="artifacts/baseline_test_predictions.csv")
    bookmakers.add_argument("--model-metrics", default="artifacts/baseline_model_metrics.json")
    bookmakers.add_argument("--mapping-dir", default="data/processed")
    bookmakers.add_argument("--audit-dir", default="data/audits")
    bookmakers.add_argument("--output-dir", default="artifacts")
    bookmakers.add_argument("--devig-method", choices=("proportional", "power"), default="proportional",
                            help="price-only margin removal; primary proportional or power sensitivity")
    bookmakers.add_argument("--quote-cutoff", choices=("pre_start", "source_day"), default="pre_start",
                            help="primary pre-start gates or nominal source-day UTC proxy sensitivity")
    bookmakers.set_defaults(handler=_compare_bookmakers)

    player_cmd = commands.add_parser("player", help="inspect individual player career ratings and rankings")
    player_cmd.add_argument("--name", help="player handle/name to inspect (e.g. s1mple, donk, NiKo)")
    player_cmd.add_argument("--top", type=int, help="show top N players by Elo rating")
    player_cmd.add_argument("--min-matches", type=int, default=50, help="minimum matches for top rankings")
    _add_source_arguments(player_cmd)
    player_cmd.set_defaults(handler=_player)

    run = commands.add_parser("run", help="build features, train models, and summarize results")
    _add_source_arguments(run)
    run.add_argument("--train-ratio", type=float, default=0.8)
    run.add_argument("--output-dir", default="data/processed")
    run.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return its process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ImportError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

import click

@click.group()
def cli():
    """Command line tools for CS2 pipeline."""
    pass
