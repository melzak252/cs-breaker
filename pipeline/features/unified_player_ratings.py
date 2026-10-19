"""Four independent, outcome-only player rating channels for complete 5v5 series.

The caller must freeze every source-calendar day's forecasts before update_day.
Elo and marginal Glicko-2 batch against frozen day openings; native Gaussian
models update in deterministic match-ID order, not an invented intraday order.
Glicko's independent-player effective-opponent approximation is not a canonical
team model. Display strength differences are roster means; its probability uses
team strength sums and the joint uncertainty of all ten players.
"""
from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Sequence

import trueskill
from openskill.models import PlackettLuce


_GLICKO_SCALE = 173.7178
_Q = math.log(10.0) / 400.0
_PERIOD_DAYS = 7.0
_GLICKO_TAU = 0.5


def _sigmoid(exponent: float) -> float:
    tail = math.exp(-abs(exponent))
    return 1.0 / (1.0 + tail) if exponent >= 0 else tail / (1.0 + tail)


def _rating_day(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("day must be a canonical YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("day must be a canonical YYYY-MM-DD string") from exc
    if parsed.isoformat() != value:
        raise ValueError("day must be a canonical YYYY-MM-DD string")
    return parsed


def _identity(key: str) -> tuple[str, int | None]:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("player keys must be explicit nonempty identity strings")
    if key.startswith("hltv:player:"):
        suffix = key[len("hltv:player:"):]
        if re.fullmatch(r"[1-9][0-9]*", suffix):
            return "hltv:player", int(suffix)
    elif key.startswith("strandedpond:nickname:"):
        if key[len("strandedpond:nickname:"):].strip():
            return "strandedpond:nickname", None
    raise ValueError(f"unsupported or invalid player identity: {key!r}")


def _lineups(a: Sequence[str], b: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rosters = []
    for roster in (a, b):
        if not isinstance(roster, (list, tuple)) or len(roster) != 5:
            raise ValueError("each lineup must contain exactly five player keys")
        for key in roster:
            _identity(key)
        if len(set(roster)) != 5:
            raise ValueError("each lineup must contain five distinct player keys")
        rosters.append(tuple(sorted(roster)))
    if set(rosters[0]) & set(rosters[1]):
        raise ValueError("opposing lineups must be disjoint")
    return rosters[0], rosters[1]


def _team_name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("team names must be nonempty strings")


def glicko2_rating_period(
    rating: float,
    rd: float,
    volatility: float,
    opponents: Sequence[tuple[float, float, float]],
    *,
    tau: float,
) -> tuple[float, float, float]:
    """Official Glicko-2 one-period equations (Glickman, example updated 2022).

    Pure numerical primitive, including draws for the published worked example;
    the player-series API accepts binary results only. No dependency on Elo or
    on an external repository. An empty period diffuses RD once.
    """
    for name, value in (("rating", rating), ("rd", rd), ("volatility", volatility), ("tau", tau)):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name != "rating" and value <= 0:
            raise ValueError(f"{name} must be positive")
    phi = rd / _GLICKO_SCALE
    information = 0.0
    residual = 0.0
    for opponent_rating, opponent_rd, score in opponents:
        if not all(math.isfinite(v) for v in (opponent_rating, opponent_rd, score)):
            raise ValueError("Glicko observations must be finite")
        if opponent_rd <= 0 or not 0 <= score <= 1:
            raise ValueError("Glicko observations require positive RD and score in [0, 1]")
        opponent_phi = opponent_rd / _GLICKO_SCALE
        g = 1.0 / math.sqrt(1.0 + 3.0 * opponent_phi**2 / math.pi**2)
        exponent = g * (rating - opponent_rating) / _GLICKO_SCALE
        tail = math.exp(-abs(exponent))
        information += g * g * tail / (1.0 + tail)**2
        residual += g * (score - _sigmoid(exponent))
    if not opponents:
        return rating, math.hypot(rd, _GLICKO_SCALE * volatility), volatility
    if not math.isfinite(information) or information <= 0:
        raise ValueError("Glicko rating period exceeds the supported numerical range")
    variance = 1.0 / information
    delta_squared = (variance * residual)**2
    phi_squared = phi * phi
    a = 2.0 * math.log(volatility)
    tau_squared = tau * tau
    if not all(math.isfinite(v) and v > 0 for v in (variance, phi_squared, tau_squared)) or not math.isfinite(delta_squared):
        raise ValueError("Glicko rating period exceeds the supported numerical range")

    def f(x: float) -> float:
        exp_x = math.exp(x)
        total = phi_squared + variance + exp_x
        return exp_x * (delta_squared - phi_squared - variance - exp_x) / (2 * total * total) - (x - a) / tau_squared

    lower = a
    if delta_squared > phi_squared + variance:
        upper = math.log(delta_squared - phi_squared - variance)
    else:
        k = 1
        upper = a - k * tau
        while f(upper) < 0:
            k += 1
            upper = a - k * tau
    f_lower, f_upper = f(lower), f(upper)
    while abs(upper - lower) > 0.000001:
        candidate = lower + (lower - upper) * f_lower / (f_upper - f_lower)
        f_candidate = f(candidate)
        if f_candidate * f_upper <= 0:
            lower, f_lower = upper, f_upper
        else:
            f_lower /= 2
        upper, f_upper = candidate, f_candidate
    new_volatility = math.exp(lower / 2)
    phi_star = math.hypot(phi, new_volatility)
    new_phi = 1.0 / math.sqrt(1.0 / phi_star**2 + information)
    return rating + _GLICKO_SCALE * new_phi**2 * residual, _GLICKO_SCALE * new_phi, new_volatility


@dataclass
class PlayerUnifiedProfile:
    """Persistent identity, independent ratings, and observed series outcomes."""

    player_key: str
    player_name: str
    matches_played: int = 0
    wins: int = 0
    elo: float = 1500.0
    glicko_rating: float = 1500.0
    glicko_rd: float = 350.0
    glicko_volatility: float = 0.06
    trueskill_mu: float = 25.0
    trueskill_sigma: float = 25.0 / 3.0
    openskill_mu: float = 25.0
    openskill_sigma: float = 25.0 / 3.0
    teams: list[str] = field(default_factory=list)
    last_active: str | None = None
    recent_outcomes: deque[int] = field(default_factory=lambda: deque(maxlen=10))

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches_played if self.matches_played else 0.0

    @property
    def last10_win_rate(self) -> float:
        return sum(self.recent_outcomes) / len(self.recent_outcomes) if self.recent_outcomes else 0.0

    @property
    def trueskill_conservative(self) -> float:
        return self.trueskill_mu - 3 * self.trueskill_sigma

    @property
    def openskill_ordinal(self) -> float:
        return self.openskill_mu - 3 * self.openskill_sigma

    def to_dict(self, *, glicko_rd: float | None = None) -> dict[str, Any]:
        namespace, native_id = _identity(self.player_key)
        rd = self.glicko_rd if glicko_rd is None else glicko_rd
        return {
            "player_key": self.player_key,
            "identity_namespace": namespace,
            "player_id": native_id,
            "player_name": self.player_name,
            "matches_played": self.matches_played,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "last10_win_rate": self.last10_win_rate,
            "elo": self.elo,
            "glicko_rating": self.glicko_rating,
            "glicko_rd": rd,
            "glicko_volatility": self.glicko_volatility,
            "glicko_conservative": self.glicko_rating - 2 * rd,
            "trueskill_mu": self.trueskill_mu,
            "trueskill_sigma": self.trueskill_sigma,
            "trueskill_conservative": self.trueskill_conservative,
            "openskill_mu": self.openskill_mu,
            "openskill_sigma": self.openskill_sigma,
            "openskill_ordinal": self.openskill_ordinal,
            "teams": list(self.teams),
            "last_active": self.last_active,
        }


class UnifiedPlayerRatingEngine:
    """Fixed-prior, player-keyed ratings; forecasts never insert or mutate state."""

    def __init__(self) -> None:
        self.players: dict[str, PlayerUnifiedProfile] = {}
        self._last_day: date | None = None
        self._match_ids: set[str] = set()
        self.ts_env = trueskill.TrueSkill(
            mu=25.0, sigma=25.0 / 3.0, beta=25.0 / 6.0,
            tau=25.0 / 300.0, draw_probability=0.0,
        )
        self.os_model = PlackettLuce(
            mu=25.0, sigma=25.0 / 3.0, beta=25.0 / 6.0, tau=25.0 / 300.0,
        )

    def __getstate__(self) -> dict:
        # Persist independent numeric state, not native environments' closures.
        return {"players": self.players, "last_day": self._last_day, "match_ids": self._match_ids}

    def __setstate__(self, state: dict) -> None:
        self.__init__()
        self.players = state["players"]
        self._last_day = state["last_day"]
        self._match_ids = state["match_ids"]

    def _future_day(self, day: str) -> date:
        current = _rating_day(day)
        if self._last_day is not None and current <= self._last_day:
            raise ValueError("day must be later than the last completed update day")
        return current

    def _profile(self, key: str) -> PlayerUnifiedProfile:
        profile = self.players.get(key)
        return profile if profile is not None else PlayerUnifiedProfile(key, key)

    @staticmethod
    def _opening(profile: PlayerUnifiedProfile, day: date) -> tuple[float, float, float]:
        rd = profile.glicko_rd
        if profile.last_active is not None:
            # The result update already diffused its own period. Calendar queries
            # add only intervening empty days, at the fixed seven-day time scale.
            empty_days = max(0, (day - date.fromisoformat(profile.last_active)).days - 1)
            rd = math.hypot(rd, _GLICKO_SCALE * profile.glicko_volatility * math.sqrt(empty_days / _PERIOD_DAYS))
        return profile.glicko_rating, rd, profile.glicko_volatility

    def compute_prematch_features(
        self,
        team1: str,
        team2: str,
        t1_players: Sequence[str],
        t2_players: Sequence[str],
        *,
        day: str,
    ) -> dict[str, float | int]:
        """Read full-precision features using only preceding-day outcomes.

        Strength, experience and win-rate differences are roster means. Joint
        uncertainty fields combine ten independent variances; uncertainty
        differences compare the two roster-sum standard deviations. An unseen
        player's observed career/recent win rate is zero (no invented outcomes);
        support counts explicitly identify cold priors.
        """
        current = self._future_day(day)
        _team_name(team1)
        _team_name(team2)
        a, b = _lineups(t1_players, t2_players)
        p1, p2 = [self._profile(k) for k in a], [self._profile(k) for k in b]
        g1, g2 = [self._opening(p, current) for p in p1], [self._opening(p, current) for p in p2]
        elo_diff = math.fsum(p.elo for p in p1) / 5 - math.fsum(p.elo for p in p2) / 5
        glicko_sum_diff = math.fsum(s[0] for s in g1) - math.fsum(s[0] for s in g2)
        gr1, gr2 = math.hypot(*(s[1] for s in g1)), math.hypot(*(s[1] for s in g2))
        g_joint = math.hypot(gr1, gr2)
        g = 1 / math.sqrt(1 + 3 * _Q**2 * g_joint**2 / math.pi**2)
        ts_sum_diff = math.fsum(p.trueskill_mu for p in p1) - math.fsum(p.trueskill_mu for p in p2)
        ts1, ts2 = math.hypot(*(p.trueskill_sigma for p in p1)), math.hypot(*(p.trueskill_sigma for p in p2))
        ts_joint = math.hypot(ts1, ts2)
        ts_denom = math.hypot(ts_joint, math.sqrt(10) * self.ts_env.beta)
        os1, os2 = math.hypot(*(p.openskill_sigma for p in p1)), math.hypot(*(p.openskill_sigma for p in p2))
        # Native OpenSkill CDF tails may differ by rounding on side reversal.
        # Predict in one identity order and complement for the other side.
        canonical = (p1, p2) if a < b else (p2, p1)
        os_teams = [[self.os_model.rating(mu=p.openskill_mu, sigma=p.openskill_sigma) for p in roster] for roster in canonical]
        os_prob = float(self.os_model.predict_win(os_teams)[0])
        if a > b:
            os_prob = 1 - os_prob
        return {
            "elo_diff": elo_diff,
            "elo_prob": _sigmoid(_Q * elo_diff),
            "glicko_diff": glicko_sum_diff / 5,
            "glicko_prob": _sigmoid(_Q * g * glicko_sum_diff),
            "trueskill_mu_diff": ts_sum_diff / 5,
            "trueskill_prob": 0.5 * (1 + math.erf(ts_sum_diff / ts_denom / math.sqrt(2))),
            "openskill_ordinal_diff": math.fsum(p.openskill_ordinal for p in p1) / 5 - math.fsum(p.openskill_ordinal for p in p2) / 5,
            "openskill_prob": os_prob,
            "lineup_exp_diff": math.fsum(p.matches_played for p in p1) / 5 - math.fsum(p.matches_played for p in p2) / 5,
            "lineup_win_rate_diff": math.fsum(p.win_rate for p in p1) / 5 - math.fsum(p.win_rate for p in p2) / 5,
            "glicko_rd_diff": gr1 - gr2,
            "glicko_joint_rd": g_joint,
            "trueskill_sigma_diff": ts1 - ts2,
            "trueskill_joint_sigma": ts_joint,
            "openskill_sigma_diff": os1 - os2,
            "openskill_joint_sigma": math.hypot(os1, os2),
            "lineup_last10_win_rate_diff": math.fsum(p.last10_win_rate for p in p1) / 5 - math.fsum(p.last10_win_rate for p in p2) / 5,
            "minimum_prior_matches": min(p.matches_played for p in (*p1, *p2)),
            "team1_known_players": sum(k in self.players for k in a),
            "team2_known_players": sum(k in self.players for k in b),
        }

    def update_day(self, day: str, games: list[dict]) -> None:
        """Validate a whole day, then atomically commit one result per series.

        Match IDs are unique across the replay. If a record carries a `day`
        field it must equal the batch day. Optional display-name entries update
        names without affecting identity. Empty batches are rejected.
        """
        current = self._future_day(day)
        if not isinstance(games, list) or not games:
            raise ValueError("games must be a nonempty list of series records")
        ordered = []
        match_ids = set()
        for record in games:
            if not isinstance(record, dict):
                raise ValueError("each series must be a dictionary")
            match_id = record.get("hltv_match_id")
            if not isinstance(match_id, str) or not match_id.strip():
                raise ValueError("hltv_match_id must be a nonempty string")
            if match_id in match_ids or match_id in self._match_ids:
                raise ValueError("duplicate series match ID")
            match_ids.add(match_id)
            if "day" in record and record["day"] != day:
                raise ValueError("series day must match its update day")
            a, b = _lineups(record.get("participants_a"), record.get("participants_b"))
            y = record.get("y")
            if isinstance(y, bool) or not isinstance(y, (int, float)) or y not in (0, 1):
                raise ValueError("series outcome y must be zero or one")
            team1, team2 = record.get("team1"), record.get("team2")
            _team_name(team1)
            _team_name(team2)
            names = record.get("player_names")
            if not isinstance(names, dict):
                raise ValueError("player_names must be a dictionary keyed by player identity")
            for key, name in names.items():
                _identity(key)
                if key not in (*a, *b) or not isinstance(name, str) or not name.strip():
                    raise ValueError("display names must be nonempty and belong to the series participants")
            if a > b:
                a, b, y, team1, team2 = b, a, 1 - y, team2, team1
            ordered.append((match_id, a, b, int(y), team1, team2, names))
        ordered.sort(key=lambda g: g[0])
        participant_keys = sorted({k for _, a, b, *_ in ordered for k in (*a, *b)})
        opening = {key: self._profile(key) for key in participant_keys}
        glicko_opening = {key: self._opening(p, current) for key, p in opening.items()}
        # Work on only affected profiles; even a native update failure cannot
        # leave a partially applied day or altered career display metadata.
        updated = {key: replace(p, teams=list(p.teams), recent_outcomes=deque(p.recent_outcomes, maxlen=10)) for key, p in opening.items()}
        elo_deltas = dict.fromkeys(participant_keys, 0.0)
        glicko_results = {key: [] for key in participant_keys}
        for _, a, b, y, team1, team2, names in ordered:
            elo_a = math.fsum(opening[k].elo for k in a) / 5
            elo_b = math.fsum(opening[k].elo for k in b) / 5
            delta = 32 * (y - _sigmoid(_Q * (elo_a - elo_b)))
            for own, other, score, elo_delta in ((a, b, y, delta), (b, a, 1 - y, -delta)):
                other_strength = math.fsum(glicko_opening[k][0] for k in other)
                for key in own:
                    elo_deltas[key] += elo_delta
                    teammates = [k for k in own if k != key]
                    effective_rating = other_strength - math.fsum(glicko_opening[k][0] for k in teammates)
                    effective_rd = math.hypot(*(glicko_opening[k][1] for k in (*other, *teammates)))
                    glicko_results[key].append((effective_rating, effective_rd, score))
            ranks = [0, 1] if y == 1 else [1, 0]
            ts_teams = [[self.ts_env.create_rating(mu=updated[k].trueskill_mu, sigma=updated[k].trueskill_sigma) for k in roster] for roster in (a, b)]
            os_teams = [[self.os_model.rating(mu=updated[k].openskill_mu, sigma=updated[k].openskill_sigma) for k in roster] for roster in (a, b)]
            ts_rated = self.ts_env.rate(ts_teams, ranks=ranks)
            os_rated = self.os_model.rate(os_teams, ranks=ranks)
            for roster, ts_team, os_team, score, team in zip((a, b), ts_rated, os_rated, (y, 1 - y), (team1, team2)):
                for key, ts_rating, os_rating in zip(roster, ts_team, os_team):
                    p = updated[key]
                    p.trueskill_mu, p.trueskill_sigma = ts_rating.mu, ts_rating.sigma
                    p.openskill_mu, p.openskill_sigma = os_rating.mu, os_rating.sigma
                    p.matches_played += 1
                    p.wins += score
                    p.recent_outcomes.append(score)
                    p.last_active = day
                    if team not in p.teams:
                        p.teams.append(team)
                    if key in names:
                        p.player_name = names[key]
        for key, p in updated.items():
            p.elo = opening[key].elo + elo_deltas[key]
            p.glicko_rating, p.glicko_rd, p.glicko_volatility = glicko2_rating_period(
                *glicko_opening[key], glicko_results[key], tau=_GLICKO_TAU,
            )
        self.players.update(updated)
        self._match_ids.update(match_ids)
        self._last_day = current

    def get_rankings(self, min_matches: int = 1) -> list[dict[str, Any]]:
        """All observed players by default, with independent competition ranks.

        Elo and Glicko ranks use means; Gaussian ranks use their conservative
        scores. Glicko conservative rank is also supplied. Exact ties share a
        rank, leaving gaps, and player keys break only display-order ties.
        Unplayed-calendar-day Glicko RD is projected at the latest replay day.
        """
        if isinstance(min_matches, bool) or not isinstance(min_matches, int) or min_matches < 0:
            raise ValueError("min_matches must be a nonnegative integer")
        rows = [p.to_dict(glicko_rd=self._opening(p, self._last_day)[1]) for p in self.players.values() if p.matches_played >= min_matches]
        for rank_name, score_name in (
            ("elo_rank", "elo"),
            ("glicko_rank", "glicko_rating"),
            ("glicko_conservative_rank", "glicko_conservative"),
            ("trueskill_rank", "trueskill_conservative"),
            ("openskill_rank", "openskill_ordinal"),
        ):
            ordered = sorted(rows, key=lambda r: (-r[score_name], r["player_key"]))
            last_score = None
            rank = 0
            for index, row in enumerate(ordered, 1):
                if row[score_name] != last_score:
                    rank = index
                    last_score = row[score_name]
                row[rank_name] = rank
        return sorted(rows, key=lambda r: (-r["elo"], r["player_key"]))
