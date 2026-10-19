"""Consolidate match evidence without converting scrape-time snapshots to history."""
from __future__ import annotations

from collections import defaultdict
import json
import logging
import math
import re

import pandas as pd

from pipeline.ratings.player_elo import NAMESPACE, load_map_rows

logger = logging.getLogger(__name__)
MATCH_ID = re.compile(r"/matches/(\d+)(?:/|$)")


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _unique(frame, column):
    frame = frame.copy().fillna("")
    frame[column] = frame[column].map(_text)
    if not frame[column].str.fullmatch(r"[1-9]\d*").all():
        raise ValueError(f"{column} requires positive native HLTV integer IDs")
    frame = frame.drop_duplicates()
    if frame[column].duplicated().any():
        raise ValueError(f"Conflicting rows for {column}")
    return frame


def _orientation(match, team1, team2):
    expected = tuple(_text(match[key]).casefold() for key in ("team1", "team2"))
    actual = (_text(team1).casefold(), _text(team2).casefold())
    return 1 if actual == expected else -1 if actual[::-1] == expected else 0


def _set_lineup(frame, mid, rosters, names, native_ids, source, statistics=None):
    keys = rosters[0] + rosters[1]
    if any(len(r) != 5 or len(set(r)) != 5 for r in rosters) or len(set(keys)) != 10:
        frame.at[mid, "lineup_issue"] = "not_disjoint_five_player_rosters"
        return
    for side, roster in enumerate(rosters, 1):
        for slot, key in enumerate(roster, 1):
            frame.at[mid, f"team{side}_player_{slot}_name"] = names[key]
            frame.at[mid, f"team{side}_p{slot}_id"] = native_ids.get(key, "")
            frame.at[mid, f"team{side}_player_{slot}_key"] = key
    frame.at[mid, "has_lineup"] = True
    frame.at[mid, "lineup_source"] = source
    frame.at[mid, "lineup_status"] = "complete"
    frame.at[mid, "lineup_issue"] = ""
    frame.at[mid, "identity_namespace"] = "hltv:player" if native_ids else "strandedpond:nickname"
    if statistics:
        frame.at[mid, "player_match_statistics"] = json.dumps(statistics, sort_keys=True)


def _historical_series(frame, map_frame, player_frame):
    map_rows = map_frame.fillna("").astype(str).to_dict("records")
    player_rows = player_frame.fillna("").astype(str).to_dict("records")
    observations, _, rejected = load_map_rows(map_rows, player_rows)
    by_series = defaultdict(list)
    for observation in observations:
        by_series[observation.series_id].append(observation)
    rejected_ids = set()
    for row in rejected:
        identity = MATCH_ID.search(row["match_url"])
        if identity:
            mid = identity.group(1)
            rejected_ids.add(mid)
            if mid in frame.index and not frame.at[mid, "has_lineup"]:
                frame.at[mid, "lineup_issue"] = "historical_" + row["reasons"]

    stats_by_join = defaultdict(dict)
    for row in player_rows:
        join = tuple(row[field] for field in ("event", "date", "map", "team"))
        stats_by_join[join][row["player_name"]] = row
    maps_by_id = {}
    for row in map_rows:
        identity = MATCH_ID.search(row["match_url"])
        if identity:
            maps_by_id[(identity.group(1), row["map"])] = row

    for mid, maps in by_series.items():
        if mid not in frame.index or mid in rejected_ids or frame.at[mid, "has_lineup"]:
            continue
        match = frame.loc[mid]
        first = maps[0]
        direction = _orientation(match, first.team1, first.team2)
        issue = ""
        if not direction or any(_orientation(match, m.team1, m.team2) != direction for m in maps):
            issue = "historical_team_orientation_conflict"
        elif any((m.roster1, m.roster2) != (first.roster1, first.roster2) for m in maps):
            issue = "historical_roster_changed_within_series"
        elif first.timestamp.date().isoformat() != _text(match["date"])[:10]:
            issue = "historical_archive_calendar_conflict"
        elif match["format"] not in ("bo3", "bo5"):
            issue = "historical_non_series_result"
        else:
            outcomes = [m.y if direction == 1 else 1 - m.y for m in maps]
            wins = (sum(outcomes), len(outcomes) - sum(outcomes))
            try:
                expected = (int(match["score1"]), int(match["score2"]))
            except (ValueError, TypeError):
                expected = None
            if wins != expected:
                issue = "historical_incomplete_or_conflicting_maps"
        if issue:
            frame.at[mid, "lineup_issue"] = issue
            continue
        rosters = (first.roster1, first.roster2) if direction == 1 else (first.roster2, first.roster1)
        names = {key: key[len(NAMESPACE):] for roster in rosters for key in roster}
        statistics = {key: {"maps": 0, "rounds": 0, "kills": 0.0, "deaths": 0.0,
                            "assists": 0.0, "adr_round_sum": 0.0, "kast_round_sum": 0.0}
                      for key in names}
        valid_statistics = True
        for observation in maps:
            row = maps_by_id[(mid, observation.map_name)]
            rounds = int(row["total_rounds"])
            for side in ("left", "right"):
                join = (row["event"], row["date"], row["map"], row[f"team_{side}"])
                for name, player in stats_by_join[join].items():
                    key = NAMESPACE + name
                    if key not in statistics:
                        valid_statistics = False
                        continue
                    try:
                        values = {field: float(player[field]) for field in ("kills", "deaths", "assists", "adr", "kast")}
                        if any(not math.isfinite(v) or v < 0 for v in values.values()) or values["kast"] > 1:
                            raise ValueError("Invalid historical measurement")
                    except (KeyError, ValueError):
                        valid_statistics = False
                        continue
                    total = statistics[key]
                    total["maps"] += 1
                    total["rounds"] += rounds
                    for field in ("kills", "deaths", "assists"):
                        total[field] += values[field]
                    total["adr_round_sum"] += values["adr"] * rounds
                    total["kast_round_sum"] += values["kast"] * rounds
        _set_lineup(frame, mid, rosters, names, {}, "strandedpond_historical_maps",
                    statistics if valid_statistics else None)


def build_consolidated_dataset(hltv_df, kaggle_df, lineups_df, historical_maps=None, historical_players=None):
    """Preserve snapshot fields separately; only exact match evidence supplies rosters.

    Historical nicknames are source-scoped identities, never inferred HLTV player IDs.
    Match-page captures are retrospective lineup evidence, not pre-event availability.
    """
    base = _unique(hltv_df.rename(columns={"match_id": "hltv_match_id"}), "hltv_match_id")
    for row in base.to_dict("records"):
        identity = MATCH_ID.search(_text(row.get("match_url", "")))
        if identity is None or identity.group(1) != row["hltv_match_id"]:
            raise ValueError("Match URL and native HLTV match ID disagree")
    snapshots = _unique(kaggle_df, "hltv_match_id")
    snapshots = snapshots.rename(columns={c: "kaggle_snapshot__" + c for c in snapshots if c != "hltv_match_id"})
    master = base.merge(snapshots, on="hltv_match_id", how="left", validate="one_to_one").fillna("")
    master = master.set_index("hltv_match_id", drop=False)
    master["has_lineup"] = False
    master["lineup_source"] = ""
    master["lineup_status"] = "not_collected"
    master["lineup_issue"] = "lineup_not_collected"
    master["identity_namespace"] = ""
    master["player_match_statistics"] = ""
    master.loc[master.index.isin(snapshots["hltv_match_id"]), "lineup_issue"] = "unverified_kaggle_snapshot"
    for side in (1, 2):
        for slot in range(1, 6):
            for field in (f"team{side}_player_{slot}_name", f"team{side}_p{slot}_id", f"team{side}_player_{slot}_key"):
                master[field] = ""
    if not lineups_df.empty:
        captures = _unique(lineups_df, "hltv_match_id")
        for capture in captures.to_dict("records"):
            mid = capture["hltv_match_id"]
            if mid not in master.index or capture.get("source") != "hltv_live":
                continue
            if capture.get("lineup_status") != "complete":
                master.at[mid, "lineup_issue"] = "captured_" + _text(capture.get("lineup_status", "invalid"))
                continue
            if any(_text(capture.get(f"team{side}_player_count", "")) not in ("", "5") for side in (1, 2)):
                master.at[mid, "lineup_issue"] = "captured_not_exactly_five_participants"
                continue
            direction = _orientation(master.loc[mid], capture.get("team1"), capture.get("team2"))
            if not direction:
                master.at[mid, "lineup_issue"] = "captured_team_orientation_conflict"
                continue
            rosters, names, native_ids = [], {}, {}
            for side in ((1, 2) if direction == 1 else (2, 1)):
                roster = []
                for slot in range(1, 6):
                    name = _text(capture.get(f"team{side}_p{slot}_name", ""))
                    pid = _text(capture.get(f"team{side}_p{slot}_id", ""))
                    if not name or re.fullmatch(r"[1-9]\d*", pid) is None:
                        continue
                    key = "hltv:player:" + pid
                    roster.append(key)
                    names[key], native_ids[key] = name, pid
                rosters.append(tuple(roster))
            _set_lineup(master, mid, rosters, names, native_ids, "hltv_match_capture")
    if (historical_maps is None) != (historical_players is None):
        raise ValueError("Historical maps and player measurements must be supplied together")
    if historical_maps is not None and not historical_maps.empty:
        _historical_series(master, historical_maps, historical_players)
    master["time_semantics"] = "source_calendar_day_only; noon_is_synthetic"
    result = master.reset_index(drop=True).sort_values(["date", "hltv_match_id"], ascending=[False, True], kind="stable")
    logger.info("Consolidated %s native matches; %s have match-specific 5v5 lineup evidence", len(result), int(result["has_lineup"].sum()))
    return result.reset_index(drop=True)
