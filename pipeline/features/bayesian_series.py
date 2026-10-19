"""Forward-only approximate Bayesian strength under exchangeable map outcomes.

Conditional on a Gaussian team-skill difference, maps are independent Bernoulli
trials with logistic probability. Ordinary zero-start BO3/BO5 likelihoods account
for stopping at the required wins; this is not a map/veto-specific model. A
completed map-coded record supplies one Bernoulli observation, never one per
round. Skill units are natural log odds.

Inference uses fixed 64-point Gauss-Hermite integration and Gaussian marginal
moment matching. Each source day's likelihood sites use its common opening
state; their natural-parameter increments are added before an atomic commit.
Cross-team covariance is discarded, so this is an approximation, not exact
joint Bayesian filtering. Fixed quadrature is not an adaptive accuracy guarantee
for very broad variances or extremely surprising outcomes. Team identities are
literal native names and chronology uses validated source-calendar days; there
is no roster identity resolution, backward smoothing, or same-day ordering.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pipeline.features.series_history import _day, _teams, _validated_game


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class SkillSpec:
    """A Gaussian prior with random-walk or stationary OU time dynamics."""

    name: str
    update_mode: str
    prior_variance: float
    variance_per_day: float
    mean_half_life_days: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("skill channel name must be a nonempty string")
        if self.update_mode not in ("winner", "score"):
            raise ValueError("update_mode must be winner or score")
        prior = _finite_real(self.prior_variance, "prior_variance")
        diffusion = _finite_real(self.variance_per_day, "variance_per_day")
        if prior <= 0:
            raise ValueError("prior_variance must be positive")
        if self.mean_half_life_days is None:
            if diffusion <= 0:
                raise ValueError("random-walk variance_per_day must be positive")
        else:
            half_life = _finite_real(self.mean_half_life_days, "mean_half_life_days")
            if half_life <= 0 or diffusion != 0:
                raise ValueError("OU requires a positive mean half-life and zero variance_per_day")


DEFAULT_SPECS = tuple(
    SkillSpec(f"bayes_{mode}_s{sd}_{dynamic}", mode, float(sd * sd), diffusion, half_life)
    for mode in ("winner", "score")
    for sd in (1, 2)
    for dynamic, diffusion, half_life in (
        ("rw0025", .0025, None), ("rw01", .01, None),
        ("ou90", 0., 90.), ("ou365", 0., 365.),
    )
)
BAYES_FEATURE_COLUMNS = [spec.name + "_prob" for spec in DEFAULT_SPECS]
BAYES_MIRROR_RULES = {
    name: {"operation": "complement", "source": name}
    for name in BAYES_FEATURE_COLUMNS
}
_METADATA_COLUMNS = ["hltv_match_id", "source_date", "team1", "team2", "target", "format",
                     "history_through", "history_cutoff_date", "history_lag_days"]

# Cache the mandated rule once. Transforming these nodes gives a standard normal.
_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(64)
_NORMAL_NODES = math.sqrt(2.) * _HERMITE_NODES
_LOG_WEIGHTS = np.log(_HERMITE_WEIGHTS) - .5 * math.log(math.pi)
_NORMAL_NODES.setflags(write=False)
_LOG_WEIGHTS.setflags(write=False)
_LOG_COMBINATIONS = {
    needed: tuple(math.log(math.comb(needed + loser - 1, loser)) for loser in range(needed))
    for needed in (2, 3)
}


def _required_wins(format: str) -> int:
    if format not in ("bo3", "bo5"):
        raise ValueError("forecast format must be ordinary bo3 or bo5")
    return 2 if format == "bo3" else 3


def _log_map_probabilities(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -np.logaddexp(0., -z), -np.logaddexp(0., z)


def _validate_gaussians(means: np.ndarray, variances: np.ndarray, *, positive: bool) -> None:
    if (not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances))
            or np.any(variances <= 0 if positive else variances < 0)):
        raise ValueError("Gaussian moments must be finite with valid variance")


def _terminal_probabilities(means: np.ndarray, variances: np.ndarray, needed: int) -> np.ndarray:
    """Integrate all terminal scores for a vector of nonnegative differences."""
    offsets = np.sqrt(variances)[:, None] * _NORMAL_NODES
    z = means[:, None] + offsets
    if not np.all(np.isfinite(z)):
        raise ValueError("nonfinite Gaussian quadrature nodes")
    log_p, log_q = _log_map_probabilities(z)
    probabilities = np.empty((len(means), 2 * needed))
    for loser, log_combination in enumerate(_LOG_COMBINATIONS[needed]):
        probabilities[:, loser] = np.exp(logsumexp(
            _LOG_WEIGHTS + log_combination + needed * log_p + loser * log_q, axis=1))
        probabilities[:, needed + loser] = np.exp(logsumexp(
            _LOG_WEIGHTS + log_combination + needed * log_q + loser * log_p, axis=1))
    # At zero difference the two sides are identical, including floating-point
    # representation. Canonical positive orientation also enforces side reversal.
    zero = means == 0
    paired = .5 * (probabilities[zero, :needed] + probabilities[zero, needed:])
    probabilities[zero, :needed] = paired
    probabilities[zero, needed:] = paired
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or np.any(np.abs(probabilities.sum(axis=1) - 1.) > 1e-12)):
        raise ValueError("invalid integrated terminal score probabilities")
    return probabilities


def _win_probabilities(means: np.ndarray, probabilities: np.ndarray, needed: int) -> np.ndarray:
    # Evaluate the small tail directly, and complement in the positive orientation.
    tail = probabilities[:, needed:].sum(axis=1)
    return np.where(means == 0, .5, np.where(means > 0, 1. - tail, tail))


def score_distribution(mean_difference: float, variance_sum: float, format: str) -> dict[str, float]:
    """Stopped-series probabilities, integrating uncertainty before aggregation.

    The distribution uses a fixed 64-point Gaussian rule, including at zero
    variance, and log-stable likelihoods. Only ordinary zero-start series apply.
    """
    mean = _finite_real(mean_difference, "mean_difference")
    variance = _finite_real(variance_sum, "variance_sum")
    if variance < 0:
        raise ValueError("variance_sum must be nonnegative")
    needed = _required_wins(format)
    probabilities = _terminal_probabilities(np.array([abs(mean)]), np.array([variance]), needed)[0]
    if mean < 0:
        probabilities = np.concatenate((probabilities[needed:], probabilities[:needed]))
    return {**{f"{needed}-{loser}": float(probabilities[loser]) for loser in range(needed)},
            **{f"{loser}-{needed}": float(probabilities[needed + loser]) for loser in range(needed)}}


@dataclass(frozen=True)
class _TeamSkill:
    means: tuple[float, ...]
    variances: tuple[float, ...]
    last_day: date | None = None
    series_count: int = 0
    map_count: int = 0


class BayesianSeriesEngine:
    """Independent Gaussian marginals with transactional day-parallel updates."""

    def __init__(self, specs=DEFAULT_SPECS) -> None:
        try:
            self.specs = tuple(specs)
        except TypeError as exc:
            raise ValueError("specs must be a nonempty collection of SkillSpec") from exc
        if not self.specs or any(not isinstance(spec, SkillSpec) for spec in self.specs):
            raise ValueError("specs must be a nonempty collection of SkillSpec")
        if len({spec.name for spec in self.specs}) != len(self.specs):
            raise ValueError("skill channel names must be unique")
        self.feature_columns = [spec.name + "_prob" for spec in self.specs]
        self._prior_variances = np.array([spec.prior_variance for spec in self.specs], dtype=float)
        self._diffusion = np.array([spec.variance_per_day for spec in self.specs], dtype=float)
        self._half_lives = np.array([
            np.inf if spec.mean_half_life_days is None else spec.mean_half_life_days for spec in self.specs
        ], dtype=float)
        self._winner_mode = np.array([spec.update_mode == "winner" for spec in self.specs])
        self._empty = _TeamSkill((0.,) * len(self.specs), tuple(self._prior_variances))
        self.teams: dict[str, _TeamSkill] = {}
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

    def _opening(self, profile: _TeamSkill, current: date) -> tuple[np.ndarray, np.ndarray]:
        means = np.array(profile.means)
        variances = np.array(profile.variances)
        if profile.last_day is not None:
            elapsed = (current - profile.last_day).days
            rho = np.exp2(-elapsed / self._half_lives)
            means *= rho
            variances = (rho * rho * variances + (1. - rho * rho) * self._prior_variances
                         + self._diffusion * elapsed)
        _validate_gaussians(means, variances, positive=True)
        return means, variances

    def _query(self, team1: str, team2: str, day: str, format: str):
        current = self._future_day(day)
        _teams(team1, team2)
        needed = _required_wins(format)
        left = self.teams.get(team1, self._empty)
        right = self.teams.get(team2, self._empty)
        means1, variances1 = self._opening(left, current)
        means2, variances2 = self._opening(right, current)
        means, variances = means1 - means2, variances1 + variances2
        _validate_gaussians(means, variances, positive=True)
        probabilities = _terminal_probabilities(np.abs(means), variances, needed)
        return left, right, means, variances, probabilities, needed

    def features(self, team1: str, team2: str, day: str, format: str = "bo3") -> dict[str, float]:
        """Pure future-day probabilities; unseen teams remain unstored priors."""
        _, _, means, _, probabilities, needed = self._query(team1, team2, day, format)
        wins = _win_probabilities(means, probabilities, needed)
        return {name: float(value) for name, value in zip(self.feature_columns, wins)}

    def diagnostics(self, team1: str, team2: str, day: str, format: str = "bo3", channel=None) -> dict:
        if channel is None:
            channel = self.specs[0].name
        indices = [index for index, spec in enumerate(self.specs) if spec.name == channel]
        if not indices:
            raise ValueError("unknown Bayesian skill channel")
        index = indices[0]
        left, right, means, variances, probabilities, needed = self._query(team1, team2, day, format)
        scores = probabilities[index]
        if means[index] < 0:
            scores = np.concatenate((scores[needed:], scores[:needed]))
        return {
            "channel": channel,
            "mean_difference": float(means[index]), "variance_sum": float(variances[index]),
            "prob_team1_win": float(_win_probabilities(means, probabilities, needed)[index]),
            "score_probabilities": {
                **{f"{needed}-{loser}": float(scores[loser]) for loser in range(needed)},
                **{f"{loser}-{needed}": float(scores[needed + loser]) for loser in range(needed)},
            },
            "team1_prior_series": left.series_count, "team2_prior_series": right.series_count,
            "team1_prior_maps": left.map_count, "team2_prior_maps": right.map_count,
        }

    def _tilted_moments(self, means: np.ndarray, variances: np.ndarray, game: dict):
        """Likelihood-weighted difference moments, oriented winner minus loser."""
        offsets = np.sqrt(variances)[:, None] * _NORMAL_NODES
        z = means[:, None] + offsets
        if not np.all(np.isfinite(z)):
            raise ValueError("nonfinite Gaussian quadrature nodes")
        log_p, log_q = _log_map_probabilities(z)
        if game["kind"] == "map":
            log_likelihood = log_p
        else:
            needed = _required_wins(game["format"])
            loser = min(game["score1"], game["score2"])
            log_likelihood = (_LOG_COMBINATIONS[needed][loser]
                              + needed * log_p + loser * log_q)
            if np.any(self._winner_mode):
                winner_logs = np.stack([
                    combination + needed * log_p + lost * log_q
                    for lost, combination in enumerate(_LOG_COMBINATIONS[needed])
                ])
                log_likelihood[self._winner_mode] = logsumexp(winner_logs, axis=0)[self._winner_mode]
        log_mass = _LOG_WEIGHTS + log_likelihood
        log_normalizer = logsumexp(log_mass, axis=1)
        if not np.all(np.isfinite(log_normalizer)):
            raise ValueError("nonfinite tilted likelihood normalizer")
        weights = np.exp(log_mass - log_normalizer[:, None])
        shifts = np.sum(weights * offsets, axis=1)
        tilted_variances = np.sum(weights * (offsets - shifts[:, None]) ** 2, axis=1)
        _validate_gaussians(shifts, tilted_variances, positive=True)
        return shifts, tilted_variances

    def update_day(self, day: str, games: list[dict]) -> None:
        """Validate every record and compute every site before changing state.

        Marginal Gaussian sites are posterior-minus-opening natural parameters.
        All matches use the same time-projected opening marginals. Native-ID
        ordering and fsum make input order irrelevant, including repeated teams.
        Any invalid record or numerical failure leaves all history unchanged.
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
        profiles = {team: self.teams.get(team, self._empty) for team in names}
        openings = {team: self._opening(profile, current) for team, profile in profiles.items()}
        sites = {team: [] for team in names}
        counts = {team: [0, 0] for team in names}
        for game in ordered:
            winner = game["team1"] if game["target"] else game["team2"]
            loser = game["team2"] if game["target"] else game["team1"]
            mean1, variance1 = openings[winner]
            mean2, variance2 = openings[loser]
            means, variances = mean1 - mean2, variance1 + variance2
            _validate_gaussians(means, variances, positive=True)
            shifts, tilted_variances = self._tilted_moments(means, variances, game)
            for team, own_mean, own_variance, opponent_variance, sign in (
                (winner, mean1, variance1, variance2, 1.),
                (loser, mean2, variance2, variance1, -1.),
            ):
                fraction = own_variance / variances
                posterior_mean = own_mean + sign * fraction * shifts
                # Conditional residual plus tilted uncertainty avoids subtracting
                # nearly equal variances, especially for unequal team uncertainty.
                posterior_variance = (own_variance * (opponent_variance / variances)
                                      + fraction * fraction * tilted_variances)
                _validate_gaussians(posterior_mean, posterior_variance, positive=True)
                precision_increment = 1. / posterior_variance - 1. / own_variance
                location_increment = posterior_mean / posterior_variance - own_mean / own_variance
                if (not np.all(np.isfinite(precision_increment))
                        or not np.all(np.isfinite(location_increment))):
                    raise ValueError("nonfinite Gaussian site natural parameters")
                sites[team].append((precision_increment, location_increment))
                counts[team][0] += int(game["kind"] == "series")
                counts[team][1] += game["score1"] + game["score2"] if game["kind"] == "series" else 1
        updated = {}
        for team in names:
            means, variances = openings[team]
            precision = 1. / variances + np.array([
                math.fsum(site[0][index] for site in sites[team]) for index in range(len(self.specs))
            ])
            location = means / variances + np.array([
                math.fsum(site[1][index] for site in sites[team]) for index in range(len(self.specs))
            ])
            if (not np.all(np.isfinite(precision)) or np.any(precision <= 0)
                    or not np.all(np.isfinite(location))):
                raise ValueError("invalid aggregated Gaussian site precision or location")
            posterior_variances = 1. / precision
            posterior_means = location * posterior_variances
            _validate_gaussians(posterior_means, posterior_variances, positive=True)
            profile = profiles[team]
            updated[team] = _TeamSkill(
                tuple(float(value) for value in posterior_means),
                tuple(float(value) for value in posterior_variances), current,
                profile.series_count + counts[team][0], profile.map_count + counts[team][1],
            )
        self.teams.update(updated)
        self._match_ids.update(ids)
        self._last_day = current


def build_bayesian_features(games: list[dict], history_lag_days: int = 1) -> tuple[pd.DataFrame, BayesianSeriesEngine]:
    """Replay native history with a source-calendar lag and full trailing state.

    Only completed series targets produce rows; map-only history still updates
    skills once eligible. All targets on a day see an identical opening state.
    Prefix forecasts are independent of later input outcomes. The caller owns
    partition filtering and any reserved-TEST forecast embargo.
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
    engine = BayesianSeriesEngine()
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
            rows.append({**{name: game[name] for name in _METADATA_COLUMNS[:6]},
                         "history_through": engine.history_through, "history_cutoff_date": cutoff,
                         "history_lag_days": int(history_lag_days), **features})
    while cursor < len(days):
        prior_day = days[cursor]
        engine.update_day(prior_day, by_day[prior_day])
        cursor += 1
    return pd.DataFrame(rows, columns=_METADATA_COLUMNS + BAYES_FEATURE_COLUMNS), engine
