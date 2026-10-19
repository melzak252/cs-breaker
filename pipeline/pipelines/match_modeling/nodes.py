"""Retrospective lineup-conditional player forecasts with literal-team archive history."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import json
import logging
import math
import re

import pandas as pd

from pipeline.features.team_history import ARCHIVE_FEATURES, TeamHistoryEngine
from pipeline.features.unified_player_ratings import UnifiedPlayerRatingEngine
from pipeline.models.player_baseline import predict_player_baseline, train_player_baseline
from pipeline.ratings.player_elo import validate_rosters

logger = logging.getLogger(__name__)
RATING_FEATURES = [
    "elo_diff", "elo_prob", "glicko_diff", "glicko_prob", "trueskill_mu_diff",
    "trueskill_prob", "openskill_ordinal_diff", "openskill_prob", "lineup_exp_diff",
    "lineup_win_rate_diff", "glicko_rd_diff", "glicko_joint_rd", "trueskill_sigma_diff",
    "trueskill_joint_sigma", "openskill_sigma_diff", "openskill_joint_sigma",
    "lineup_last10_win_rate_diff", "minimum_prior_matches", "team1_known_players", "team2_known_players",
]
PERFORMANCE_FEATURES = [
    "player_kpr_diff", "player_dpr_diff", "player_assists_per_round_diff", "player_adr_diff",
    "player_kast_diff", "player_last10_kpr_diff", "player_last10_dpr_diff", "player_last10_adr_diff",
    "player_last10_kast_diff", "performance_available", "minimum_prior_measured_maps",
]
FEATURE_COLUMNS = RATING_FEATURES + PERFORMANCE_FEATURES + ARCHIVE_FEATURES
STAT_FIELDS = ("maps", "rounds", "kills", "deaths", "assists", "adr_round_sum", "kast_round_sum")


def _valid_series(row):
    if row.get("format") not in ("bo3", "bo5"):
        return "unsupported_or_forfeit_format"
    try:
        scores = tuple(int(row[field]) for field in ("score1", "score2"))
    except (ValueError, TypeError):
        return "invalid_series_scores"
    needed = 2 if row["format"] == "bo3" else 3
    if min(scores) < 0 or max(scores) != needed or min(scores) >= needed:
        return "incomplete_series_score"
    if row.get("winner") != ("team1" if scores[0] > scores[1] else "team2"):
        return "winner_score_conflict"
    return ""


def _performance_features(history, roster1, roster2):
    keys = roster1 + roster2
    minimum = min((sum(item["maps"] for item in history.get(key, [])) for key in keys), default=0)
    features = {key: 0.0 for key in PERFORMANCE_FEATURES}
    features["minimum_prior_measured_maps"] = float(minimum)
    if minimum == 0:
        # Missing measurements are explicitly marked; zero is not a measured ADR/KAST.
        return features
    features["performance_available"] = 1.0

    def aggregate(roster, window=None):
        means = defaultdict(float)
        for key in roster:
            rows = history[key] if window is None else history[key][-window:]
            totals = {field: math.fsum(item[field] for item in rows) for field in STAT_FIELDS}
            rounds = totals["rounds"]
            means["kpr"] += totals["kills"] / rounds / 5
            means["dpr"] += totals["deaths"] / rounds / 5
            means["assists_per_round"] += totals["assists"] / rounds / 5
            means["adr"] += totals["adr_round_sum"] / rounds / 5
            means["kast"] += totals["kast_round_sum"] / rounds / 5
        return means

    for window, prefix in ((None, "player_"), (10, "player_last10_")):
        left, right = aggregate(roster1, window), aggregate(roster2, window)
        for field in ("kpr", "dpr", "adr", "kast") + (("assists_per_round",) if window is None else ()):
            features[prefix + field + "_diff"] = left[field] - right[field]
    return features


def _source_day(value):
    """Use the literal source calendar, never an invented UTC kickoff order."""
    if not isinstance(value, (str, date)):
        raise ValueError("Source date must contain a valid YYYY-MM-DD calendar day")
    day = str(value)[:10]
    parsed = date.fromisoformat(day)
    if parsed.isoformat() != day:
        raise ValueError("Source date must contain a canonical YYYY-MM-DD calendar day")
    return day


def _archive_games(master_df, identities, duplicate_ids):
    """Read only bounded series fields, not wide snapshot/roster columns."""
    columns = ["hltv_match_id", "date", "format", "score1", "score2", "winner", "team1", "team2"]
    games_by_day = defaultdict(list)
    exclusions = Counter()
    for row, identity, duplicate in zip(
            master_df.loc[:, columns].fillna("").to_dict("records"), identities, duplicate_ids):
        if re.fullmatch(r"[1-9][0-9]*", identity) is None:
            issue = "invalid_native_match_id"
        elif duplicate:
            issue = "duplicate_native_match_id"
        else:
            try:
                issue = _valid_series(row)
            except OverflowError:
                issue = "invalid_series_scores"
        if issue:
            exclusions[issue] += 1
            continue
        try:
            day = _source_day(row["date"])
        except (ValueError, TypeError):
            exclusions["invalid_source_date"] += 1
            continue
        if day < "2024-01-01":
            exclusions["before_modern_era_proxy"] += 1
            continue
        teams = row["team1"], row["team2"]
        if any(not isinstance(team, str) or not team.strip() for team in teams):
            exclusions["invalid_literal_team_name"] += 1
            continue
        if teams[0] == teams[1]:
            exclusions["identical_literal_teams"] += 1
            continue
        games_by_day[day].append({
            "hltv_match_id": identity, "team1": teams[0], "team2": teams[1],
            "y": int(row["winner"] == "team1"),
        })
    return games_by_day, exclusions


def generate_player_rankings_and_features(master_df):
    """Freeze all source-calendar-day features before accepting any day's outcomes.

    Actual participants are retrospective update/conditional-scenario evidence.
    Their capture does not prove historical pre-event lineup announcement time.
    """
    identities = master_df["hltv_match_id"].astype(str)
    duplicate_ids = identities.duplicated(keep=False)
    games_by_day = defaultdict(list)
    lineup_mask = master_df["has_lineup"].astype(str).str.casefold().eq("true")
    if (duplicate_ids & lineup_mask).any():
        raise ValueError("Historical target series identities must be unique")
    archive_by_day, archive_exclusions = _archive_games(master_df, identities, duplicate_ids)
    exclusions = Counter(reason or "no_match_specific_lineup"
                         for reason in master_df.loc[~lineup_mask, "lineup_issue"].fillna(""))
    columns = ["hltv_match_id", "date", "format", "score1", "score2", "winner", "team1", "team2",
               "tournament", "lineup_source", "identity_namespace", "player_match_statistics"]
    columns += [f"team{side}_player_{slot}_{field}" for side in (1, 2)
                for slot in range(1, 6) for field in ("key", "name")]
    for row in master_df.loc[lineup_mask, columns].fillna("").to_dict("records"):
        mid = str(row["hltv_match_id"])
        issue = _valid_series(row)
        if issue:
            exclusions[issue] += 1
            continue
        if re.fullmatch(r"[1-9][0-9]*", mid) is None:
            raise ValueError("Historical target series require positive native HLTV integer IDs")
        day = _source_day(row["date"])
        rosters = tuple(tuple(str(row.get(f"team{side}_player_{slot}_key", "")).strip() for slot in range(1, 6))
                        for side in (1, 2))
        validate_rosters(*rosters)
        names = {row[f"team{side}_player_{slot}_key"]: row[f"team{side}_player_{slot}_name"]
                 for side in (1, 2) for slot in range(1, 6)}
        game = {"hltv_match_id": mid, "participants_a": rosters[0], "participants_b": rosters[1],
                "y": int(row["winner"] == "team1"), "team1": row["team1"], "team2": row["team2"],
                "player_names": names, "row": row}
        games_by_day[day].append(game)
    target_match_ids = {game["hltv_match_id"] for games in games_by_day.values() for game in games}

    engine = UnifiedPlayerRatingEngine()
    team_history = TeamHistoryEngine()
    performance = {}
    feature_rows = []
    replay_days = sorted(set(games_by_day) | set(archive_by_day))
    for day in replay_days:
        games = games_by_day.get(day, [])
        games.sort(key=lambda game: int(game["hltv_match_id"]))
        for game in games:
            row = game["row"]
            rosters = game["participants_a"], game["participants_b"]
            rating = engine.compute_prematch_features(game["team1"], game["team2"], *rosters, day=day)
            statistics = _performance_features(performance, *rosters)
            archive = team_history.compute_prematch_features(game["team1"], game["team2"], day=day)
            feature_rows.append({"hltv_match_id": game["hltv_match_id"], "date": day,
                                 "team1": game["team1"], "team2": game["team2"], "target": game["y"],
                                 "tournament": row["tournament"], "format": row["format"],
                                 "lineup_source": row["lineup_source"], "identity_namespace": row["identity_namespace"],
                                 "history_source_dates_before": day, **rating, **statistics, **archive})
        if games:
            # Archive-only days must not change any player rating/uncertainty state.
            engine.update_day(day, games)
        archive_games = archive_by_day.get(day, [])
        if archive_games:
            team_history.update_day(day, archive_games)
        # Current match box scores enter history only after every same-day forecast.
        for game in games:
            payload = game["row"].get("player_match_statistics", "")
            if not payload:
                continue
            measurements = json.loads(payload)
            for key, values in measurements.items():
                if key not in game["player_names"]:
                    raise ValueError("Box-score identity is not a participant of this series")
                if set(values) != set(STAT_FIELDS) or values["rounds"] <= 0:
                    raise ValueError("Invalid player series measurements")
                if any(not math.isfinite(value) or value < 0 for value in values.values()):
                    raise ValueError("Player series measurements must be finite and non-negative")
                performance.setdefault(key, []).append(values)

    features = pd.DataFrame(feature_rows)
    rankings = pd.DataFrame(engine.get_rankings(min_matches=1))
    if not rankings.empty:
        rankings["player_id"] = rankings["player_id"].astype("Int64")
        for field in ("measured_maps", "measured_rounds", "career_kpr", "career_dpr", "career_adr", "career_kast"):
            rankings[field] = None
        for index, row in rankings.iterrows():
            history = performance.get(row["player_key"])
            if not history:
                continue
            total = {field: math.fsum(item[field] for item in history) for field in STAT_FIELDS}
            rankings.at[index, "measured_maps"] = int(total["maps"])
            rankings.at[index, "measured_rounds"] = int(total["rounds"])
            for name, numerator in (("kpr", "kills"), ("dpr", "deaths"), ("adr", "adr_round_sum"), ("kast", "kast_round_sum")):
                rankings.at[index, "career_" + name] = total[numerator] / total["rounds"]
    snapshot_columns = [column for column in master_df if column.startswith("kaggle_snapshot__")]
    audit = {"master_matches": len(master_df), "eligible_series": len(features), "ranked_observed_players": len(rankings),
             "excluded_matches": len(master_df) - len(features), "exclusion_reasons": dict(sorted(exclusions.items())),
             "eligible_lineup_sources": dict(Counter(features["lineup_source"])) if not features.empty else {},
             "snapshot_fields_preserved_but_not_features": len(snapshot_columns),
             "date_range": [min(games_by_day), max(games_by_day)] if games_by_day else [],
             "forecast_kind": "retrospective_actual_lineup_conditional",
             "archive_history": {
                 "source": "data/processed/master_matches_consolidated.csv",
                 "minimum_source_date": "2024-01-01",
                 "history_rows": sum(map(len, archive_by_day.values())),
                 "history_only_rows": sum(game["hltv_match_id"] not in target_match_ids
                                          for games in archive_by_day.values() for game in games),
                 "excluded_rows": sum(archive_exclusions.values()),
                 "exclusion_reasons": dict(sorted(archive_exclusions.items())),
                 "literal_teams": len(team_history.teams),
                 "source_days": len(archive_by_day),
                 "date_range": [min(archive_by_day), max(archive_by_day)] if archive_by_day else [],
                 "target_rows_with_both_teams_supported": int((features["archive_minimum_prior_matches"] > 0).sum())
                                                            if not features.empty else 0,
                 "target_rows_with_any_cold_team": int((features["archive_minimum_prior_matches"] == 0).sum())
                                                 if not features.empty else 0,
                 "target_literal_teams": len(set(features["team1"]) | set(features["team2"]))
                                         if not features.empty else 0,
                 "limitations": [
                     "Literal source names only; no organization IDs, aliases or identity crosswalk",
                     "Date >= 2024-01-01 is a modern-era proxy, not verified CS2 version evidence",
                     "Source dates and postgame evidence do not certify pre-event availability",
                     "Complete BO3/BO5 results only; history does not require observed rosters",
                 ],
             },
             "assumptions": ["Source-calendar-day chronology; original release times and timezones are unverified",
                             "All forecasts frozen before same-day outcomes or measurements",
                             "No inferred player-ID crosswalk; native IDs and historical nickname scopes stay separate",
                             "Kaggle scrape-time roster/stat snapshots excluded from historical rating updates",
                             "Equal-participation ratings infer outcome contribution, not individual mechanical ability",
                             "Match-page lineup captures do not prove per-map substitution coverage or pre-event availability"]}
    state = {"schema_version": 2, "engine": engine, "team_history_engine": team_history,
             "performance_history": performance,
             "history_through": max(replay_days) if replay_days else None,
             "player_history_through": max(games_by_day) if games_by_day else None,
             "archive_history_through": max(archive_by_day) if archive_by_day else None,
             "feature_columns": list(FEATURE_COLUMNS), "forecast_kind": audit["forecast_kind"]}
    logger.info("Replayed %s eligible series, excluding %s unsupported matches", len(features), audit["excluded_matches"])
    return features, rankings, audit, state


def train_baseline_prediction_models(features_df):
    return train_player_baseline(features_df, FEATURE_COLUMNS)


def predict_player_match(model_bundle, rating_state, team1, team2, roster1, roster2, day):
    """Forecast an explicitly supplied roster scenario, never silently invent a lineup."""
    if not isinstance(rating_state, dict) or rating_state.get("schema_version") != 2:
        raise ValueError("Unsupported rating state schema version; regenerate both rating engines")
    if not isinstance(rating_state.get("engine"), UnifiedPlayerRatingEngine) or not isinstance(
            rating_state.get("team_history_engine"), TeamHistoryEngine):
        raise ValueError("Rating state must contain both player and archive history engines")
    if not isinstance(model_bundle, dict) or model_bundle.get("version") != 4:
        raise ValueError("Unsupported model artifact version; retrain the current feature schema")
    if not isinstance(day, str) or _source_day(day) != day:
        raise ValueError("Forecast day must be a canonical YYYY-MM-DD string")
    cutoff = rating_state.get("history_through")
    if cutoff is not None and day <= cutoff:
        raise ValueError("Forecast day must be after the saved rating history")
    features = rating_state["team_history_engine"].compute_prematch_features(team1, team2, day=day)
    features.update(rating_state["engine"].compute_prematch_features(team1, team2, roster1, roster2, day=day))
    features.update(_performance_features(rating_state["performance_history"], tuple(roster1), tuple(roster2)))
    probability = float(predict_player_baseline(model_bundle, pd.DataFrame([features]))[0])
    return {"prob_team1_win": probability, "prob_team2_win": 1.0 - probability,
            "forecast_kind": "explicit_roster_scenario", "history_through": cutoff,
            "team1": team1, "team2": team2, "day": day,
            "minimum_prior_matches": features["minimum_prior_matches"],
            "team1_known_players": features["team1_known_players"],
            "team2_known_players": features["team2_known_players"],
            "archive_team1_prior_matches": features["archive_team1_prior_matches"],
            "archive_team2_prior_matches": features["archive_team2_prior_matches"],
            "archive_minimum_prior_matches": features["archive_minimum_prior_matches"],
            "performance_available": bool(features["performance_available"]),
            "caveats": [
                "Explicit supplied roster scenario; no inferred current starters or live forecast",
                "Archive identities are literal source names, not organization IDs or inferred aliases",
                "Archive date >= 2024-01-01 is a modern-era proxy, not verified CS2 version evidence",
                "Source-day history and actual played lineups do not certify pre-event availability",
            ]}
