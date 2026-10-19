"""Sports-only, literal-team history with read-only forecasts and atomic days.

The source calendar is a retrospective chronology proxy, not a publication or
kickoff clock. A 2024 start is a modern-game proxy, not certified game-version
metadata; names are never merged into guessed organization aliases. Map-coded
rows update history but are never relabeled as BO3/BO5 targets.

Map Elo forecasts use the conditional-independent, exchangeable-map majority
approximation. Glicko has daily periods: opening RD diffuses for *preceding*
inactive days, capped at 350; the public period primitive adds current-period
diffusion when that day's series are committed. Its forecast uses joint RD so
that reversing the sides complements the probability exactly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import json
import math
from numbers import Integral, Real
import re
from urllib.parse import urlsplit

import pandas as pd

from pipeline.features.unified_player_ratings import glicko2_rating_period
from pipeline.ratings.player_elo import validate_rosters


STAT_FIELDS = ("maps", "rounds", "kills", "deaths", "assists", "adr_round_sum", "kast_round_sum")
MAP_FORMATS = frozenset(("mrg", "inf", "nuke", "d2", "anc", "anb", "ovp", "vtg", "trn", "cch", "cbl"))
HISTORY_COLUMNS = [
    "hltv_match_id", "date", "team1", "team2", "score1", "score2", "winner", "format",
    "match_url", "tournament", "has_lineup", "lineup_source", "lineup_status",
    "identity_namespace", "player_match_statistics",
    *[f"team{side}_player_{slot}_key" for side in (1, 2) for slot in range(1, 6)],
]
_CORE_COLUMNS = ("hltv_match_id", "date", "team1", "team2", "score1", "score2", "winner", "format", "match_url")
_SERIES_CHANNELS = tuple((f"series_elo_{k}_{half if half is not None else 'none'}", k, half)
                         for k in (16, 32, 64) for half in (90, 180, 365, None))
_MAP_CHANNELS = tuple((f"map_elo_{k}_{half}", k, half) for k in (16, 32, 64) for half in (180, 365))
_ELO_CHANNELS = _SERIES_CHANNELS + _MAP_CHANNELS
_GLICKO_PRIORS = (200, 350)
_GLICKO_SCALE = 173.7178
_FORM_REFERENCE = tuple(name for name, _, _ in _SERIES_CHANNELS).index("series_elo_32_180")

RATING_COLUMNS = [name + suffix for name, _, _ in _ELO_CHANNELS for suffix in ("_prob", "_diff")]
RATING_COLUMNS += [f"glicko_{rd}_{suffix}" for rd in _GLICKO_PRIORS
                   for suffix in ("prob", "diff", "rd_diff", "joint_rd")]
PROBABILITY_COLUMNS = [name + "_prob" for name, _, _ in _ELO_CHANNELS] + [f"glicko_{rd}_prob" for rd in _GLICKO_PRIORS]
FORM_COLUMNS = [
    "series_win_rate_5_diff", "series_win_rate_10_diff", "series_win_rate_30_diff",
    "series_map_margin_10_diff", "series_map_margin_30_diff",
    "series_residual_form_10_diff", "series_residual_form_30_diff",
    "series_count_diff", "series_log_count_diff", "series_rest_days_diff", "series_log_rest_diff",
    "series_activity_30d_diff", "minimum_prior_series", "minimum_activity_30d",
    "team1_prior_series", "team2_prior_series", "team1_cold", "team2_cold", "either_team_cold",
]
_PERFORMANCE_RATES = (("kpr", "kills"), ("dpr", "deaths"),
                      ("assists_per_player_round", "assists"), ("adr", "adr_round_sum"),
                      ("kast", "kast_round_sum"))
PERFORMANCE_COLUMNS = [f"team_{prefix}{name}_diff" for prefix in ("", "last10_") for name, _ in _PERFORMANCE_RATES]
PERFORMANCE_COLUMNS += ["performance_available", "last10_performance_available",
                        "minimum_prior_measured_maps", "minimum_prior_measured_series", "performance_age_days"]
FEATURE_COLUMNS = RATING_COLUMNS + FORM_COLUMNS + PERFORMANCE_COLUMNS
FEATURE_GROUPS = {"ratings": list(RATING_COLUMNS), "form": list(FORM_COLUMNS),
                  "performance": list(PERFORMANCE_COLUMNS), "probabilities": list(PROBABILITY_COLUMNS),
                  "all": list(FEATURE_COLUMNS)}
# No suffix-based fallback: every predictor's side transformation is registered.
_NEGATED = [name + "_diff" for name, _, _ in _ELO_CHANNELS]
_NEGATED += [f"glicko_{rd}_{suffix}" for rd in _GLICKO_PRIORS for suffix in ("diff", "rd_diff")]
_NEGATED += FORM_COLUMNS[:12] + PERFORMANCE_COLUMNS[:10]
_INVARIANT = [f"glicko_{rd}_joint_rd" for rd in _GLICKO_PRIORS]
_INVARIANT += ["minimum_prior_series", "minimum_activity_30d", "either_team_cold"] + PERFORMANCE_COLUMNS[10:]
MIRROR_RULES = {
    **{name: {"operation": "negate", "source": name} for name in _NEGATED},
    **{name: {"operation": "complement", "source": name} for name in PROBABILITY_COLUMNS},
    **{name: {"operation": "identity", "source": name} for name in _INVARIANT},
    "team1_prior_series": {"operation": "swap", "source": "team2_prior_series"},
    "team2_prior_series": {"operation": "swap", "source": "team1_prior_series"},
    "team1_cold": {"operation": "swap", "source": "team2_cold"},
    "team2_cold": {"operation": "swap", "source": "team1_cold"},
}
_METADATA_COLUMNS = ["hltv_match_id", "source_date", "team1", "team2", "target", "format", "kind",
                     "score1", "score2", "history_through", "history_cutoff_date", "history_lag_days"]


def _day(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("day must be a canonical YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("day must be a canonical YYYY-MM-DD string") from exc
    if parsed.isoformat() != value:
        raise ValueError("day must be a canonical YYYY-MM-DD string")
    return parsed


def _source_day(value) -> str:
    if not isinstance(value, (str, date)):
        raise ValueError("source date requires a source-calendar date")
    return _day(str(value)[:10]).isoformat()


def _teams(team1, team2) -> None:
    if any(not isinstance(team, str) or not team.strip() for team in (team1, team2)) or team1 == team2:
        raise ValueError("opposing literal team names must be nonempty distinct strings")


def _native_id(value) -> str:
    if isinstance(value, Integral) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError("hltv_match_id requires a canonical positive native HLTV integer ID")
    return value


def _url_identity(value) -> str:
    if not isinstance(value, str):
        raise ValueError("native match URL must be an HLTV URL")
    parsed = urlsplit(value)
    match = re.fullmatch(r"/matches/([1-9][0-9]*)(?:/[^?#]*)?", parsed.path)
    if (parsed.scheme not in ("http", "https") or parsed.netloc not in ("hltv.org", "www.hltv.org")
            or parsed.query or parsed.fragment or match is None):
        raise ValueError("native match URL must be a canonical HLTV match URL")
    return match.group(1)


def _integer(value) -> int:
    if isinstance(value, bool):
        raise ValueError("scores must be nonnegative integers")
    if isinstance(value, str):
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
            raise ValueError("scores must be nonnegative integers")
        return int(value)
    if not isinstance(value, Real) or not math.isfinite(value) or value < 0 or value != int(value):
        raise ValueError("scores must be nonnegative integers")
    return int(value)


def _score_issue(format, score1: int, score2: int) -> str:
    winner, loser = max(score1, score2), min(score1, score2)
    if format in ("bo3", "bo5"):
        needed = 2 if format == "bo3" else 3
        return "" if winner == needed and loser < needed else "incomplete_series_score"
    if format in MAP_FORMATS:
        regulation = winner == 13 and loser <= 11
        overtime = winner >= 16 and (winner - 16) % 3 == 0 and winner - 4 <= loser <= winner - 2
        return "" if regulation or overtime else "incomplete_or_invalid_map_score"
    return "unsupported_or_forfeit_format"


def _stat_values(values: dict) -> dict:
    if not isinstance(values, dict) or set(values) != set(STAT_FIELDS):
        raise ValueError("measurements require exactly STAT_FIELDS")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0
           for value in values.values()):
        raise ValueError("measurements must be finite and nonnegative")
    if values["maps"] <= 0 or values["maps"] != int(values["maps"]) or values["rounds"] <= 0 or values["rounds"] != int(values["rounds"]):
        raise ValueError("maps and player-round exposure must be positive integers")
    if values["kast_round_sum"] > values["rounds"]:
        raise ValueError("KAST sum exceeds player-round exposure")
    return {field: float(values[field]) for field in STAT_FIELDS}


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate verified measurement key")
        result[key] = value
    return result


def _verified_statistics(row: dict, score1: int, score2: int, kind: str) -> dict | None:
    payload = row.get("player_match_statistics", "")
    if str(row.get("has_lineup", "")).casefold() != "true" or payload in ("", None):
        return None
    rosters = tuple(tuple(row.get(f"team{side}_player_{slot}_key", "") for slot in range(1, 6)) for side in (1, 2))
    if any(not isinstance(key, str) or not key.strip() for roster in rosters for key in roster):
        raise ValueError("verified measurement participants must be explicit five-key sides")
    validate_rosters(*rosters)
    measurements = json.loads(payload, object_pairs_hook=_json_object) if isinstance(payload, str) else payload
    if not isinstance(measurements, dict) or set(measurements) != set(rosters[0] + rosters[1]):
        raise ValueError("verified measurements must cover exactly all ten participants")
    expected_maps = score1 + score2 if kind == "series" else 1
    validated = {key: _stat_values(value) for key, value in measurements.items()}
    if any(value["maps"] != expected_maps for value in validated.values()):
        raise ValueError("verified measured maps must match completed observed maps")
    exposures = {value["rounds"] for value in validated.values()}
    if len(exposures) != 1:
        raise ValueError("five-player complete measurements require matching round exposure")
    if kind == "map" and next(iter(exposures)) != score1 + score2:
        raise ValueError("measured rounds must match completed map rounds")
    return {f"team{side}": {
        field: float(expected_maps) if field == "maps" else math.fsum(validated[key][field] for key in roster)
        for field in STAT_FIELDS
    } for side, roster in enumerate(rosters, 1)}


def load_history(master_df: pd.DataFrame, minimum_day: str = "2024-01-01") -> tuple[list[dict], dict]:
    """Select bounded native fields and audit every rejected source row.

    Unverified/missing box scores are absent measurements, never snapshot
    replacements. A malformed nonempty claimed-verified payload fails the load:
    measurement quality must not silently alter the frozen series target cohort.
    """
    minimum = _day(minimum_day).isoformat()
    if not isinstance(master_df, pd.DataFrame) or not master_df.columns.is_unique:
        raise ValueError("master_df requires a DataFrame with unique column names")
    missing = set(_CORE_COLUMNS) - set(master_df.columns)
    if missing:
        raise ValueError(f"missing native history columns: {sorted(missing)}")
    columns = [column for column in HISTORY_COLUMNS if column in master_df.columns]
    rows = master_df.loc[:, columns].fillna("").to_dict("records")
    identities = []
    for row in rows:
        try:
            identities.append(_native_id(row["hltv_match_id"]))
        except ValueError:
            identities.append(None)
    frequencies = Counter(identity for identity in identities if identity is not None)
    exclusions = Counter()
    games = []
    measured = 0
    for row, identity in zip(rows, identities):
        issue = ""
        if identity is None:
            issue = "invalid_native_match_id"
        elif frequencies[identity] > 1:
            issue = "duplicate_native_match_id"
        else:
            try:
                if _url_identity(row["match_url"]) != identity:
                    issue = "native_match_url_id_conflict"
            except ValueError:
                issue = "invalid_native_match_url"
        if not issue:
            try:
                day = _source_day(row["date"])
            except ValueError:
                issue = "invalid_source_date"
            else:
                if day < minimum:
                    issue = "before_minimum_day"
        if not issue:
            try:
                _teams(row["team1"], row["team2"])
            except ValueError:
                issue = "invalid_literal_team_name"
        if not issue:
            if not isinstance(row["format"], str) or (row["format"] not in ("bo3", "bo5") and row["format"] not in MAP_FORMATS):
                issue = "unsupported_or_forfeit_format"
            else:
                try:
                    score1, score2 = _integer(row["score1"]), _integer(row["score2"])
                except ValueError:
                    issue = "invalid_scores"
                else:
                    issue = _score_issue(row["format"], score1, score2)
                    if not issue and row["winner"] != ("team1" if score1 > score2 else "team2"):
                        issue = "winner_score_conflict"
        if not issue:
            kind = "series" if row["format"] in ("bo3", "bo5") else "map"
            try:
                statistics = _verified_statistics(row, score1, score2, kind)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"invalid_verified_measurements for native match {identity}: {exc}") from exc
        if issue:
            exclusions[issue] += 1
            continue
        game = {"hltv_match_id": identity, "source_date": day,
                "team1": row["team1"], "team2": row["team2"], "target": int(score1 > score2),
                "format": row["format"], "score1": score1, "score2": score2, "kind": kind}
        if statistics is not None:
            game["statistics"] = statistics
            game["statistics_source"] = row.get("lineup_source", "")
            measured += 1
        games.append(game)
    games.sort(key=lambda game: (game["source_date"], int(game["hltv_match_id"])))
    return games, {
        "input_rows": len(rows), "unique_native_ids": len(frequencies), "minimum_day": minimum,
        "accepted_games": len(games), "accepted_series": sum(game["kind"] == "series" for game in games),
        "accepted_maps": sum(game["kind"] == "map" for game in games),
        "accepted_measured_games": measured, "accepted_unmeasured_games": len(games) - measured,
        "exclusions": dict(sorted(exclusions.items())), "loaded_columns": columns,
        "limitations": [
            "Source calendar dates are retrospective proxies; result publication and exact kickoff clocks are unknown.",
            "The minimum source day is a modern-game proxy, not a certified game-version field.",
            "Literal source team names are not organization IDs; aliases and roster changes are not merged.",
            "Native map-coded history is history only, never an invented BO3/BO5 target.",
            "Performance uses historical verified participants only, not current target rosters or Kaggle snapshots.",
        ],
    }


def _validated_game(game: dict, current: date | None = None) -> dict:
    if not isinstance(game, dict):
        raise ValueError("history games must be dictionaries")
    identity = _native_id(game.get("hltv_match_id"))
    day = _day(game.get("source_date"))
    if current is not None and day != current:
        raise ValueError("source_date must equal the complete batch update day")
    team1, team2 = game.get("team1"), game.get("team2")
    _teams(team1, team2)
    format = game.get("format")
    if not isinstance(format, str):
        raise ValueError("format must be a native format string")
    score1, score2 = _integer(game.get("score1")), _integer(game.get("score2"))
    issue = _score_issue(format, score1, score2)
    if issue:
        raise ValueError(issue)
    kind = "series" if format in ("bo3", "bo5") else "map"
    if game.get("kind") != kind:
        raise ValueError("map-coded history and ordinary series must retain their native kinds")
    target = game.get("target")
    if isinstance(target, bool) or not isinstance(target, Real) or not math.isfinite(target) or target not in (0, 1):
        raise ValueError("target must be finite binary zero or one")
    if int(target) != int(score1 > score2):
        raise ValueError("target must agree with completed scores")
    statistics = game.get("statistics")
    if statistics is not None:
        if not isinstance(statistics, dict) or set(statistics) != {"team1", "team2"}:
            raise ValueError("statistics require exactly both measured sides")
        statistics = {side: _stat_values(values) for side, values in statistics.items()}
        observed_maps = score1 + score2 if kind == "series" else 1
        for values in statistics.values():
            if values["maps"] != observed_maps or values["rounds"] % 5 != 0:
                raise ValueError("team measurements require observed maps and five-player round exposure")
            if kind == "map" and values["rounds"] != 5 * (score1 + score2):
                raise ValueError("team exposure must match completed map rounds")
        if statistics["team1"]["rounds"] != statistics["team2"]["rounds"]:
            raise ValueError("opposing complete five-player exposures must match")
    return {"hltv_match_id": identity, "source_date": day.isoformat(), "team1": team1, "team2": team2,
            "target": int(target), "format": format, "score1": score1, "score2": score2,
            "kind": kind, "statistics": statistics}


def _probability(difference: float, scale: float = 400.0) -> float:
    exponent = math.log(10.0) * difference / scale
    tail = math.exp(-abs(exponent))
    return 1 / (1 + tail) if exponent >= 0 else tail / (1 + tail)


def _majority(p: float, format: str) -> float:
    # Evaluate the small tail directly for numerical stability and complement symmetry.
    q = min(p, 1 - p)
    small = q * q * (3 - 2 * q) if format == "bo3" else q ** 3 * (10 - 15 * q + 6 * q * q)
    return small if p <= .5 else 1 - small


@dataclass(frozen=True)
class TeamSeriesProfile:
    elo: tuple[float, ...] = (1500.0,) * len(_ELO_CHANNELS)
    glicko: tuple[tuple[float, float, float], ...] = ((1500.0, 200.0, .06), (1500.0, 350.0, .06))
    series_count: int = 0
    last_series_day: date | None = None
    last_map_day: date | None = None
    recent_series: tuple[tuple[int, float, float], ...] = ()
    activity: tuple[tuple[date, int], ...] = ()
    measured_totals: tuple[float, ...] = (0.0,) * len(STAT_FIELDS)
    measured_series: int = 0
    last_measurement_day: date | None = None
    recent_measured_series: tuple[tuple[float, ...], ...] = ()


_EMPTY_PROFILE = TeamSeriesProfile()


def _opening(profile: TeamSeriesProfile, current: date) -> tuple[tuple, tuple]:
    elo = []
    for index, (_, _, half) in enumerate(_ELO_CHANNELS):
        last = profile.last_series_day if index < len(_SERIES_CHANNELS) else profile.last_map_day
        elapsed = (current - last).days if last is not None else 0
        factor = 2 ** (-elapsed / half) if half is not None else 1.0
        elo.append(1500 + (profile.elo[index] - 1500) * factor)
    inactive = max(0, (current - profile.last_series_day).days - 1) if profile.last_series_day is not None else 0
    glicko = tuple((rating, min(350.0, math.sqrt(rd * rd + inactive * (_GLICKO_SCALE * sigma) ** 2)), sigma)
                   for rating, rd, sigma in profile.glicko)
    return tuple(elo), glicko


def _recent_mean(profile: TeamSeriesProfile, window: int, field: int) -> float:
    values = profile.recent_series[-window:]
    return math.fsum(value[field] for value in values) / len(values) if values else (.5 if field == 0 else 0.0)


def _sum_measurements(rows: tuple) -> tuple:
    return tuple(math.fsum(row[index] for row in rows) for index in range(len(STAT_FIELDS)))


class SeriesHistoryEngine:
    """Independent fixed rating bank and prior-only form/measurement aggregates."""

    def __init__(self) -> None:
        self.teams: dict[str, TeamSeriesProfile] = {}
        self._last_day: date | None = None
        self._match_ids: set[str] = set()

    @property
    def history_through(self) -> str | None:
        return self._last_day.isoformat() if self._last_day is not None else None

    def _future_day(self, day: str) -> date:
        current = _day(day)
        if self._last_day is not None and current <= self._last_day:
            raise ValueError("day must be later than all completed engine history")
        return current

    def features(self, team1: str, team2: str, day: str, format: str = "bo3") -> dict:
        """Read-only source-day-opening predictors; no target roster is required."""
        current = self._future_day(day)
        _teams(team1, team2)
        if format not in ("bo3", "bo5"):
            raise ValueError("forecast format must be ordinary bo3 or bo5")
        left, right = self.teams.get(team1, _EMPTY_PROFILE), self.teams.get(team2, _EMPTY_PROFILE)
        elo1, glicko1 = _opening(left, current)
        elo2, glicko2 = _opening(right, current)
        result = {}
        for index, (name, _, _) in enumerate(_ELO_CHANNELS):
            difference = elo1[index] - elo2[index]
            probability = _probability(difference)
            result[name + "_prob"] = probability if index < len(_SERIES_CHANNELS) else _majority(probability, format)
            result[name + "_diff"] = difference
        for prior, (r1, rd1, _), (r2, rd2, _) in zip(_GLICKO_PRIORS, glicko1, glicko2):
            joint = math.hypot(rd1, rd2)
            g = 1 / math.sqrt(1 + 3 * (joint / _GLICKO_SCALE) ** 2 / math.pi ** 2)
            result.update({f"glicko_{prior}_prob": _probability(g * (r1 - r2)),
                           f"glicko_{prior}_diff": r1 - r2,
                           f"glicko_{prior}_rd_diff": rd1 - rd2, f"glicko_{prior}_joint_rd": joint})
        for window in (5, 10, 30):
            result[f"series_win_rate_{window}_diff"] = _recent_mean(left, window, 0) - _recent_mean(right, window, 0)
        for window in (10, 30):
            result[f"series_map_margin_{window}_diff"] = _recent_mean(left, window, 1) - _recent_mean(right, window, 1)
            result[f"series_residual_form_{window}_diff"] = _recent_mean(left, window, 2) - _recent_mean(right, window, 2)
        rest1 = (current - left.last_series_day).days if left.last_series_day is not None else 0
        rest2 = (current - right.last_series_day).days if right.last_series_day is not None else 0
        activity1 = sum(count for active, count in left.activity if 0 < (current - active).days <= 30)
        activity2 = sum(count for active, count in right.activity if 0 < (current - active).days <= 30)
        result.update({
            "series_count_diff": left.series_count - right.series_count,
            "series_log_count_diff": math.log1p(left.series_count) - math.log1p(right.series_count),
            "series_rest_days_diff": rest1 - rest2, "series_log_rest_diff": math.log1p(rest1) - math.log1p(rest2),
            "series_activity_30d_diff": activity1 - activity2,
            "minimum_prior_series": min(left.series_count, right.series_count),
            "minimum_activity_30d": min(activity1, activity2),
            "team1_prior_series": left.series_count, "team2_prior_series": right.series_count,
            "team1_cold": int(left.series_count == 0), "team2_cold": int(right.series_count == 0),
            "either_team_cold": int(min(left.series_count, right.series_count) == 0),
        })
        available = left.measured_totals[1] > 0 and right.measured_totals[1] > 0
        recent_available = bool(left.recent_measured_series and right.recent_measured_series)
        for prefix, totals1, totals2, supported in (
            ("", left.measured_totals, right.measured_totals, available),
            ("last10_", _sum_measurements(left.recent_measured_series),
             _sum_measurements(right.recent_measured_series), recent_available),
        ):
            for rate, numerator in _PERFORMANCE_RATES:
                index = STAT_FIELDS.index(numerator)
                result[f"team_{prefix}{rate}_diff"] = (totals1[index] / totals1[1] - totals2[index] / totals2[1]) if supported else 0.0
        age = max((current - profile.last_measurement_day).days for profile in (left, right)) if available else 0
        result.update({"performance_available": int(available), "last10_performance_available": int(recent_available),
                       "minimum_prior_measured_maps": min(left.measured_totals[0], right.measured_totals[0]),
                       "minimum_prior_measured_series": min(left.measured_series, right.measured_series),
                       "performance_age_days": age})
        return {name: result[name] for name in FEATURE_COLUMNS}

    def update_day(self, day: str, games: list[dict]) -> None:
        """Validate/compute the entire batch, then commit all channels together.

        Native-ID sorting, fsum and common opening opponents make input row order
        irrelevant. Map-only observations do not refresh series clocks or RD.
        Empty/invalid batches never reserve IDs or advance history_through.
        """
        current = self._future_day(day)
        if not isinstance(games, list) or not games:
            raise ValueError("games must be a nonempty complete history batch")
        ordered = [_validated_game(game, current) for game in games]
        ids = [game["hltv_match_id"] for game in ordered]
        if len(set(ids)) != len(ids) or self._match_ids.intersection(ids):
            raise ValueError("duplicate native history match ID")
        ordered.sort(key=lambda game: int(game["hltv_match_id"]))
        names = sorted({game[side] for game in ordered for side in ("team1", "team2")})
        profiles = {team: self.teams.get(team, _EMPTY_PROFILE) for team in names}
        openings = {team: _opening(profile, current) for team, profile in profiles.items()}
        deltas = {team: [[] for _ in _ELO_CHANNELS] for team in names}
        opponents = {team: [[] for _ in _GLICKO_PRIORS] for team in names}
        series_rows = defaultdict(list)
        measurements = defaultdict(list)
        measured_series = defaultdict(list)
        for game in ordered:
            team1, team2, outcome = game["team1"], game["team2"], game["target"]
            elo1, glicko1 = openings[team1]
            elo2, glicko2 = openings[team2]
            series = game["kind"] == "series"
            map_score = game["score1"] / (game["score1"] + game["score2"]) if series else outcome
            for index, (_, k, _) in enumerate(_ELO_CHANNELS):
                if index < len(_SERIES_CHANNELS) and not series:
                    continue
                observed = outcome if index < len(_SERIES_CHANNELS) else map_score
                delta = k * (observed - _probability(elo1[index] - elo2[index]))
                deltas[team1][index].append(delta)
                deltas[team2][index].append(-delta)
            if series:
                expectation = _probability(elo1[_FORM_REFERENCE] - elo2[_FORM_REFERENCE])
                margin = (game["score1"] - game["score2"]) / (game["score1"] + game["score2"])
                residual = outcome - expectation
                series_rows[team1].append((outcome, margin, residual))
                series_rows[team2].append((1 - outcome, -margin, -residual))
                for index, ((r1, rd1, _), (r2, rd2, _)) in enumerate(zip(glicko1, glicko2)):
                    opponents[team1][index].append((r2, rd2, outcome))
                    opponents[team2][index].append((r1, rd1, 1 - outcome))
            if game["statistics"] is not None:
                for side, team in (("team1", team1), ("team2", team2)):
                    values = tuple(game["statistics"][side][field] for field in STAT_FIELDS)
                    measurements[team].append(values)
                    if series:
                        measured_series[team].append(values)
        updated = {}
        for team in names:
            profile = profiles[team]
            elo, glicko = openings[team]
            added_series = series_rows[team]
            # Do not store a decayed series rating at an unchanged series clock.
            new_elo = tuple(elo[index] + math.fsum(deltas[team][index])
                            if deltas[team][index] else profile.elo[index] for index in range(len(_ELO_CHANNELS)))
            new_glicko = tuple(glicko2_rating_period(*glicko[index], opponents[team][index], tau=.5)
                               if added_series else profile.glicko[index] for index in range(len(_GLICKO_PRIORS)))
            new_glicko = tuple((rating, min(350.0, rd), sigma) for rating, rd, sigma in new_glicko)
            activity = profile.activity
            if added_series:
                activity = tuple((active, count) for active, count in activity if (current - active).days < 30)
                activity += ((current, len(added_series)),)
            totals = _sum_measurements((profile.measured_totals, *measurements[team]))
            updated[team] = TeamSeriesProfile(
                elo=new_elo, glicko=new_glicko, series_count=profile.series_count + len(added_series),
                last_series_day=current if added_series else profile.last_series_day, last_map_day=current,
                recent_series=(profile.recent_series + tuple(added_series))[-30:], activity=activity,
                measured_totals=totals, measured_series=profile.measured_series + len(measured_series[team]),
                last_measurement_day=current if measurements[team] else profile.last_measurement_day,
                recent_measured_series=(profile.recent_measured_series + tuple(measured_series[team]))[-10:],
            )
        self.teams.update(updated)
        self._match_ids.update(ids)
        self._last_day = current


def build_series_features(games: list[dict], history_lag_days: int = 1) -> tuple[pd.DataFrame, SeriesHistoryEngine]:
    """Independent chronological replay with a full source-calendar-day lag.

    Only series targets are emitted. Every target day queries the same opening
    state through day minus lag, including maps only when old enough. The
    returned engine subsequently replays all completed input history so saved
    inference state has no missing trailing lag days. Partition selection and
    reserved-TEST forecast embargoes remain the caller's responsibility.
    """
    if isinstance(history_lag_days, bool) or not isinstance(history_lag_days, Integral) or history_lag_days < 1:
        raise ValueError("history_lag_days must be a positive integer")
    if not isinstance(games, list):
        raise ValueError("games must be a list of native history records")
    checked = [_validated_game(game) for game in games]
    identities = [game["hltv_match_id"] for game in checked]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate native history match ID")
    by_day = defaultdict(list)
    for game in checked:
        by_day[game["source_date"]].append(game)
    days = sorted(by_day)
    for batch in by_day.values():
        batch.sort(key=lambda game: int(game["hltv_match_id"]))
    engine = SeriesHistoryEngine()
    cursor = 0
    rows = []
    for day in days:
        targets = [game for game in by_day[day] if game["kind"] == "series"]
        if not targets:
            continue
        cutoff = (_day(day) - timedelta(days=int(history_lag_days))).isoformat()
        while cursor < len(days) and days[cursor] <= cutoff:
            prior_day = days[cursor]
            engine.update_day(prior_day, by_day[prior_day])
            cursor += 1
        for game in targets:
            features = engine.features(game["team1"], game["team2"], day, format=game["format"])
            rows.append({**{name: game[name] for name in _METADATA_COLUMNS[:9]},
                         "history_through": engine.history_through, "history_cutoff_date": cutoff,
                         "history_lag_days": int(history_lag_days), **features})
    while cursor < len(days):
        prior_day = days[cursor]
        engine.update_day(prior_day, by_day[prior_day])
        cursor += 1
    return pd.DataFrame(rows, columns=_METADATA_COLUMNS + FEATURE_COLUMNS), engine
