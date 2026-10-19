"""Normalize demo events and OpenCS2 metadata into map-result contracts."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Mapping, Optional

from pipeline.contracts import ParsedMapResult, PlayerMapStats


def _value(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first present, non-null value from common parser aliases."""
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return default


def _records(value: Any) -> List[Dict[str, Any]]:
    """Convert event rows from ordinary Python or dataframe-like objects."""
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            rows = value.to_dict(orient="records")
        except TypeError:
            rows = value.to_dicts() if hasattr(value, "to_dicts") else value.to_dict()
        if isinstance(rows, list):
            return [dict(row) for row in rows]
    if isinstance(value, Mapping):
        return [dict(value)]
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _side_number(value: Any) -> int:
    if isinstance(value, str):
        side = value.strip().lower()
        if side in ("t", "terrorist", "terrorists", "2"):
            return 2
        if side in ("ct", "counterterrorist", "counter-terrorist", "3"):
            return 3
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number in (2, 3) else 0


def _identity(record: Mapping[str, Any], prefix: str = "") -> tuple[str, str]:
    name = _value(record, f"{prefix}name", f"{prefix}player_name", f"{prefix}player", default="")
    player_id = _value(
        record,
        f"{prefix}steamid",
        f"{prefix}steam_id",
        f"{prefix}player_steamid",
        f"{prefix}id",
        default=name,
    )
    return str(player_id or name), str(name or player_id or "Unknown player")


class EventAggregator:
    """Build map statistics from demoparser2-style event rows."""

    @staticmethod
    def aggregate_from_events(
        match_id: str,
        map_name: str,
        events: Dict[str, Any],
        team1_name: str = "Team A",
        team2_name: str = "Team B",
    ) -> ParsedMapResult:
        stats: Dict[str, Dict[str, Any]] = {}
        round_scores = {team1_name: 0, team2_name: 0}
        round_data: Dict[Any, Dict[str, Any]] = {}

        def player(record: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
            player_id, name = _identity(record, prefix)
            entry = stats.setdefault(player_id, {
                "player_id": player_id, "player_name": name, "team_name": "",
                "team_num": 0, "kills": 0, "deaths": 0, "assists": 0,
                "headshot_kills": 0, "damage_dealt": 0.0, "rounds": set(),
                "first_kills": 0, "first_deaths": 0,
            })
            if name and entry["player_name"] == player_id:
                entry["player_name"] = name
            team_name = _value(record, f"{prefix}team_name", f"{prefix}team", default="")
            if team_name:
                entry["team_name"] = str(team_name)
            team_num = _side_number(_value(record, f"{prefix}team_num", f"{prefix}side", f"{prefix}team"))
            if team_num:
                entry["team_num"] = team_num
            return entry

        death_rows = _records(_value(events, "player_death", "death", "deaths", default=[]))
        hurt_rows = _records(_value(events, "player_hurt", "hurt", "damages", default=[]))
        round_rows = _records(_value(events, "round_end", "rounds", "round", default=[]))

        first_kill_by_round: Dict[Any, str] = {}
        first_death_by_round: Dict[Any, str] = {}
        for event in death_rows:
            victim = player(event, "user_")
            attacker_id, _ = _identity(event, "attacker_")
            attacker = player(event, "attacker_") if attacker_id not in ("", "None") else None
            assister_id, _ = _identity(event, "assister_")
            assister = player(event, "assister_") if assister_id not in ("", "None") else None
            round_num = _value(event, "round_num", "total_rounds", "round", "round_number")
            if round_num is not None:
                victim["rounds"].add(round_num)
                if attacker is not None:
                    attacker["rounds"].add(round_num)
                if assister is not None:
                    assister["rounds"].add(round_num)
            victim["deaths"] += 1
            if attacker is not None and attacker["player_id"] != victim["player_id"]:
                attacker["kills"] += 1
                if bool(_value(event, "headshot", "is_headshot", default=False)):
                    attacker["headshot_kills"] += 1
                if round_num is not None:
                    first_kill_by_round.setdefault(round_num, attacker["player_id"])
                    first_death_by_round.setdefault(round_num, victim["player_id"])
            if assister is not None and assister["player_id"] not in (victim["player_id"], attacker_id):
                assister["assists"] += 1

        for round_num, player_id in first_kill_by_round.items():
            if round_num in first_death_by_round:
                stats[player_id]["first_kills"] += 1
                stats[first_death_by_round[round_num]]["first_deaths"] += 1

        for event in hurt_rows:
            attacker = player(event, "attacker_")
            round_num = _value(event, "round_num", "total_rounds", "round", "round_number")
            if round_num is not None:
                attacker["rounds"].add(round_num)
            damage = _value(event, "dmg_health", "damage", "damage_health", "health_damage", default=0)
            try:
                attacker["damage_dealt"] += max(float(damage or 0), 0.0)
            except (TypeError, ValueError):
                continue

        # Side/team inference from event team fields, with the caller's team order
        # as a fallback when only T/CT sides are exposed.
        side_names: Dict[int, str] = {}
        for event in death_rows + hurt_rows:
            for prefix in ("attacker_", "user_", "victim_"):
                side = _side_number(_value(event, f"{prefix}team_num", f"{prefix}side"))
                name = _value(event, f"{prefix}team_name", f"{prefix}team")
                if side and isinstance(name, str) and name.lower() not in ("t", "ct", "2", "3"):
                    side_names.setdefault(side, name)
        fallback = {2: team1_name, 3: team2_name}
        for entry in stats.values():
            if not entry["team_name"]:
                entry["team_name"] = side_names.get(entry["team_num"], fallback.get(entry["team_num"], team1_name))

        for index, event in enumerate(round_rows, start=1):
            round_num = _value(event, "round_num", "round_number", "round", default=index)
            winner = _value(event, "winner", "winning_team", "team_name", default="")
            winner_side = _side_number(_value(event, "winner_side", "winner_team_num", "team_num", default=winner))
            explicit_team = ""
            if isinstance(winner, str):
                lower = winner.lower()
                if lower in (team1_name.lower(), "team1", "team 1", "1"):
                    explicit_team = team1_name
                elif lower in (team2_name.lower(), "team2", "team 2"):
                    explicit_team = team2_name
                elif lower in ("t", "terrorist", "terrorists", "ct", "counterterrorist", "counter-terrorist"):
                    winner_side = _side_number(winner)
            if explicit_team:
                round_scores[explicit_team] += 1
            elif winner_side:
                score_team = side_names.get(winner_side)
                if score_team == team1_name:
                    round_scores[team1_name] += 1
                elif score_team == team2_name:
                    round_scores[team2_name] += 1
                else:
                    round_scores[fallback[winner_side]] += 1
            round_data[round_num] = dict(event)

        # Some event exports include round wins on match_end rather than round_end.
        total_rounds = len(round_data) or max(
            (len(entry["rounds"]) for entry in stats.values()), default=0
        )
        player_stats = [
            PlayerMapStats(
                player_id=entry["player_id"], player_name=entry["player_name"],
                team_name=entry["team_name"], team_num=entry["team_num"],
                kills=entry["kills"], deaths=entry["deaths"], assists=entry["assists"],
                headshot_kills=entry["headshot_kills"], damage_dealt=entry["damage_dealt"],
                rounds_played=len(entry["rounds"]) or total_rounds,
                first_kills=entry["first_kills"], first_deaths=entry["first_deaths"],
            )
            for entry in stats.values()
        ]
        team1_score, team2_score = round_scores[team1_name], round_scores[team2_name]
        winner_team = team1_name if team1_score > team2_score else team2_name if team2_score > team1_score else ""
        return ParsedMapResult(
            match_id=str(match_id), map_name=map_name, team1_name=team1_name,
            team2_name=team2_name, team1_score=team1_score, team2_score=team2_score,
            total_rounds=total_rounds, winner_team=winner_team,
            player_stats=player_stats,
            rounds_data=[round_data[key] for key in sorted(round_data, key=lambda key: (str(type(key)), key))],
        )


class OpenCS2Adapter:
    """Read OpenCS2 metadata parquet shards into normalized map contracts."""

    @staticmethod
    def _player_stats(row: Mapping[str, Any], team1: str, team2: str) -> List[PlayerMapStats]:
        players = _value(row, "players_scoped", "players", default=[])
        if isinstance(players, Mapping):
            players = list(players.values())
        if not players:
            players = []
            for field, team_name in (("ct_players", team2), ("t_players", team1)):
                for item in _records(row.get(field)):
                    item = dict(item)
                    item.setdefault("team_name", team_name)
                    players.append(item)
        teams = _records(_value(row, "teams", default=[]))
        starts = [str(_value(team, "side_start", default="")).lower() for team in teams]
        team1_side = str(_value(row, "team1_side", default="")).lower()
        if not team1_side and starts:
            team1_side = "ct" if starts[0] == "ct" and not (len(starts) > 1 and starts[1] == "t") else "t"
        roster: Dict[str, tuple[str, int]] = {}
        for group in teams:
            side = _side_number(_value(group, "side_start"))
            team_name = team1 if (side == 2 and team1_side == "t") or (side == 3 and team1_side == "ct") else team2
            for roster_player in _records(_value(group, "players", default=[])):
                roster_id, _ = _identity(roster_player)
                roster[roster_id] = (team_name, side)
        aggregated: Dict[str, Dict[str, Any]] = {}
        for item in players:
            if not isinstance(item, Mapping):
                continue
            player_id, name = _identity(item)
            roster_team, roster_side = roster.get(player_id, ("", 0))
            side = roster_side or _side_number(_value(item, "team_num", "side", "team"))
            team = roster_team or _value(item, "team_name", "team", default="")
            if str(team).lower() in ("t", "ct", "2", "3") or not team:
                team = team1 if side == 2 else team2 if side == 3 else ""
            data = aggregated.setdefault(player_id, {
                "name": name, "team": str(team or ""), "side": side, "kills": 0,
                "deaths": 0, "assists": 0, "headshots": 0, "damage": 0.0,
                "rounds": 0, "first_kills": 0, "first_deaths": 0,
                "utility_damage": 0.0, "flash_assists": 0, "clutches_won": 0,
            })
            data["name"] = name or data["name"]
            if team:
                data["team"] = str(team)
            if side:
                data["side"] = side
            for target, aliases in (
                ("kills", ("kills",)), ("deaths", ("deaths",)),
                ("assists", ("assists",)), ("headshots", ("headshots", "headshot_kills", "hs")),
                ("damage", ("dmg", "damage_dealt", "damage")),
                ("rounds", ("n_rounds", "rounds_played", "rounds")),
                ("first_kills", ("first_kills", "opening_kills")),
                ("first_deaths", ("first_deaths", "opening_deaths")),
                ("utility_damage", ("utility_damage", "damage_utility")),
                ("flash_assists", ("flash_assists",)), ("clutches_won", ("clutches_won", "clutches")),
            ):
                raw = _value(item, *aliases, default=0)
                try:
                    data[target] += float(raw or 0) if target in ("damage", "utility_damage") else int(raw or 0)
                except (TypeError, ValueError):
                    pass
        return [
            PlayerMapStats(
                player_id=player_id, player_name=value["name"], team_name=value["team"],
                team_num=value["side"], kills=value["kills"], deaths=value["deaths"],
                assists=value["assists"], headshot_kills=value["headshots"],
                damage_dealt=value["damage"], rounds_played=value["rounds"],
                first_kills=value["first_kills"], first_deaths=value["first_deaths"],
                utility_damage=value["utility_damage"], flash_assists=value["flash_assists"],
                clutches_won=value["clutches_won"],
            ) for player_id, value in aggregated.items()
        ]

    @staticmethod
    def _parse_row(row: Mapping[str, Any]) -> ParsedMapResult:
        match = _value(row, "match", default={}) or {}
        if not isinstance(match, Mapping):
            match = {}
        team1 = str(_value(row, "team1", "team1_name", default="Team A") or "Team A")
        team2 = str(_value(row, "team2", "team2_name", default="Team B") or "Team B")
        rounds = _records(_value(row, "rounds", default=[]))
        score1 = _value(row, "team1_score", "score_team1")
        score2 = _value(row, "team2_score", "score_team2")

        if score1 is None or score2 is None:
            ct_score = _value(match, "score_ct")
            t_score = _value(match, "score_t")
            if ct_score is None or t_score is None:
                ct_score = sum(str(_value(r, "winner", default="")).lower() == "ct" for r in rounds)
                t_score = sum(str(_value(r, "winner", default="")).lower() == "t" for r in rounds)
            # OpenCS2 roster order is CT then T when side-start metadata is present.
            teams = _records(_value(row, "teams", default=[]))
            starts = [str(_value(team, "side_start", default="")).lower() for team in teams]
            team1_side = str(_value(row, "team1_side", default="")).lower()
            if not team1_side and starts:
                team1_side = "ct" if starts[0] == "ct" and not (len(starts) > 1 and starts[1] == "t") else "t"
            if team1_side == "ct":
                score1, score2 = ct_score, t_score
            elif team1_side == "t":
                score1, score2 = t_score, ct_score
            else:
                score1, score2 = ct_score, t_score
        score1, score2 = int(score1 or 0), int(score2 or 0)
        winner_value = str(_value(row, "winner_team", default="") or "")
        if winner_value.lower() in ("team1", "team 1"):
            winner = team1
        elif winner_value.lower() in ("team2", "team 2"):
            winner = team2
        elif winner_value in (team1, team2):
            winner = winner_value
        else:
            winner = team1 if score1 > score2 else team2 if score2 > score1 else ""
        player_stats = OpenCS2Adapter._player_stats(row, team1, team2)
        return ParsedMapResult(
            match_id=str(_value(row, "match_id", default=_value(_value(row, "meta", default={}), "match_id", default=""))),
            map_name=str(_value(row, "map_name", default=_value(match, "map_name", default="Unknown map")) or "Unknown map"),
            team1_name=team1, team2_name=team2, team1_score=score1, team2_score=score2,
            total_rounds=int(_value(match, "rounds_played", default=_value(row, "rounds_played", default=len(rounds))) or 0),
            winner_team=winner, player_stats=player_stats, rounds_data=rounds,
        )

    @staticmethod
    def parse_opencs2_parquet(file_path: str) -> List[ParsedMapResult]:
        """Parse every map row from one OpenCS2 parquet shard."""
        try:
            import pyarrow.parquet as parquet
            rows = parquet.read_table(file_path).to_pylist()
        except ImportError:
            try:
                import pandas as pd
            except ImportError as error:
                raise ImportError("Reading OpenCS2 parquet requires pyarrow or pandas") from error
            rows = pd.read_parquet(file_path).to_dict(orient="records")
        return [OpenCS2Adapter._parse_row(row) for row in rows]

    @staticmethod
    def batch_load_opencs2(
        directory: str = "data/raw/opencs2", limit: Optional[int] = None
    ) -> List[ParsedMapResult]:
        """Load parquet map rows from shards in deterministic filename order."""
        paths = sorted(glob.glob(os.path.join(directory, "*.parquet")))
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative or None")
            paths = paths[:limit]
        return [result for path in paths for result in OpenCS2Adapter.parse_opencs2_parquet(path)]
