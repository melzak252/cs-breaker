"""Causal player-series rating contracts; no external repository dependency."""
from copy import deepcopy
import math
import pickle

import pytest
import trueskill
from openskill.models import PlackettLuce

from pipeline.features.unified_player_ratings import (
    UnifiedPlayerRatingEngine,
    glicko2_rating_period,
)


A = tuple(f"hltv:player:{i}" for i in range(1, 6))
B = tuple(f"hltv:player:{i}" for i in range(6, 11))


def game(match_id="1", a=A, b=B, y=1, **extra):
    return {
        "hltv_match_id": match_id,
        "participants_a": a,
        "participants_b": b,
        "y": y,
        "team1": "Alpha",
        "team2": "Beta",
        "player_names": {key: f"Name {key}" for key in (*a, *b)},
        **extra,
    }


def forecast(engine, day="2024-01-02", a=A, b=B):
    return engine.compute_prematch_features("Alpha", "Beta", a, b, day=day)


def state(engine):
    return deepcopy((engine.players, engine._last_day, engine._match_ids))


def test_prediction_reads_cold_prior_without_inserting_or_aging_players():
    engine = UnifiedPlayerRatingEngine()
    before = state(engine)
    cold = forecast(engine, day="2024-01-01")
    assert state(engine) == before
    assert cold["elo_prob"] == cold["glicko_prob"] == 0.5
    assert cold["trueskill_prob"] == pytest.approx(0.5)
    assert cold["openskill_prob"] == pytest.approx(0.5)
    assert cold["minimum_prior_matches"] == 0
    assert cold["team1_known_players"] == cold["team2_known_players"] == 0
    engine.update_day("2024-01-01", [game()])
    before = state(engine)
    forecast(engine, day="2025-01-01")
    assert state(engine) == before


def test_saved_engine_restores_native_environments_and_preserves_future_updates():
    engine = UnifiedPlayerRatingEngine()
    engine.update_day("2024-01-01", [game()])
    restored = pickle.loads(pickle.dumps(engine))
    assert forecast(restored) == forecast(engine)
    assert restored.get_rankings() == engine.get_rankings()
    assert forecast(restored, day="2025-01-01") == forecast(engine, day="2025-01-01")
    engine.update_day("2024-01-02", [game("2", y=0)])
    restored.update_day("2024-01-02", [game("2", y=0)])
    assert restored.get_rankings() == engine.get_rankings()


def test_glicko_is_independent_of_elo_in_updates_and_forecasts():
    original = UnifiedPlayerRatingEngine()
    original.update_day("2024-01-01", [game()])
    changed = UnifiedPlayerRatingEngine()
    changed.update_day("2024-01-01", [game()])
    for key in A:
        changed.players[key].elo += 500
    assert forecast(changed)["elo_prob"] != forecast(original)["elo_prob"]
    for name in ("glicko_diff", "glicko_prob", "glicko_joint_rd"):
        assert forecast(changed)[name] == forecast(original)[name]
    original.update_day("2024-01-02", [game("2", y=0)])
    changed.update_day("2024-01-02", [game("2", y=0)])
    for key in (*A, *B):
        for attr in ("glicko_rating", "glicko_rd", "glicko_volatility"):
            assert getattr(original.players[key], attr) == getattr(changed.players[key], attr)


def test_first_update_matches_native_equal_five_player_gaussian_models():
    engine = UnifiedPlayerRatingEngine()
    ts = trueskill.TrueSkill(mu=25, sigma=25 / 3, beta=25 / 6, tau=25 / 300, draw_probability=0)
    os = PlackettLuce(mu=25, sigma=25 / 3, beta=25 / 6, tau=25 / 300)
    expected_ts = ts.rate([[ts.create_rating() for _ in range(5)] for _ in range(2)], ranks=[0, 1])
    expected_os = os.rate([[os.rating() for _ in range(5)] for _ in range(2)], ranks=[0, 1])
    engine.update_day("2024-01-01", [game()])
    for side, roster in enumerate((A, B)):
        for i, key in enumerate(roster):
            player = engine.players[key]
            assert player.trueskill_mu == pytest.approx(expected_ts[side][i].mu)
            assert player.trueskill_sigma == pytest.approx(expected_ts[side][i].sigma)
            assert player.openskill_mu == pytest.approx(expected_os[side][i].mu)
            assert player.openskill_sigma == pytest.approx(expected_os[side][i].sigma)
            assert player.elo == (1516 if side == 0 else 1484)


def test_glicko_update_uses_all_other_nine_variances_and_sum_strengths():
    engine = UnifiedPlayerRatingEngine()
    expected = glicko2_rating_period(1500, 350, 0.06, [(1500, 1050, 1)], tau=0.5)
    engine.update_day("2024-01-01", [game()])
    p = engine.players[A[0]]
    assert (p.glicko_rating, p.glicko_rd, p.glicko_volatility) == pytest.approx(expected)
    f = forecast(engine)
    strength_diff = sum(engine.players[k].glicko_rating for k in A) - sum(engine.players[k].glicko_rating for k in B)
    joint_rd = math.sqrt(sum(engine.players[k].glicko_rd**2 for k in (*A, *B)))
    q = math.log(10) / 400
    g = 1 / math.sqrt(1 + 3 * q*q * joint_rd*joint_rd / math.pi**2)
    assert f["glicko_prob"] == pytest.approx(1 / (1 + math.exp(-q * g * strength_diff)))


def test_identity_namespaces_case_and_renames_do_not_merge_skill():
    nickname_a = "strandedpond:nickname:SameName"
    nickname_b = "strandedpond:nickname:samename"
    a = (nickname_a, nickname_b, *A[2:])
    engine = UnifiedPlayerRatingEngine()
    first = game(a=a)
    first["player_names"][nickname_a] = "SameName"
    first["player_names"][B[0]] = "SameName"
    engine.update_day("2024-01-01", [first])
    second = game("2", a=a)
    second["player_names"][nickname_a] = "Renamed"
    engine.update_day("2024-01-02", [second])
    rows = {r["player_key"]: r for r in engine.get_rankings()}
    assert len(rows) == 10
    assert rows[nickname_a]["matches_played"] == 2
    assert rows[nickname_a]["player_name"] == "Renamed"
    assert rows[nickname_a]["player_id"] is None
    assert rows[nickname_a]["identity_namespace"] == "strandedpond:nickname"
    assert rows[B[0]]["player_id"] == 6
    assert rows[B[0]]["identity_namespace"] == "hltv:player"


@pytest.mark.parametrize("a,b", [
    ((), B), (A[:4], B), ((A[0],) * 5, B), (A, A),
    (("", *A[1:]), B), (("nickname", *A[1:]), B),
    (("hltv:player:0", *A[1:]), B),
    (("strandedpond:nickname:", *A[1:]), B),
])
def test_invalid_prediction_and_update_lineups_raise_without_mutation(a, b):
    engine = UnifiedPlayerRatingEngine()
    before = state(engine)
    with pytest.raises(ValueError):
        forecast(engine, a=a, b=b)
    assert state(engine) == before
    with pytest.raises(ValueError):
        engine.update_day("2024-01-01", [game(a=a, b=b)])
    assert state(engine) == before


@pytest.mark.parametrize("bad_games", [
    [], [game(), game()], [game("2"), game("3", y=2)],
    [game("2"), game("3", day="2024-01-03")],
    [game("2"), game("3", player_names={A[0]: ""})],
    [game("2"), game("", y=0)],
    [game("2"), game("3", team1="")],
])
def test_entire_invalid_day_is_atomic(bad_games):
    engine = UnifiedPlayerRatingEngine()
    engine.update_day("2024-01-01", [game("old")])
    before = state(engine)
    with pytest.raises(ValueError):
        engine.update_day("2024-01-02", bad_games)
    assert state(engine) == before


@pytest.mark.parametrize("day", ["2024-01-01", "2023-12-31", "2024-02-30", "2024-1-2", "2024-01-02T12:00:00Z", ""])
def test_days_are_canonical_and_strictly_increasing(day):
    engine = UnifiedPlayerRatingEngine()
    engine.update_day("2024-01-01", [game()])
    before = state(engine)
    with pytest.raises(ValueError):
        engine.update_day(day, [game("2")])
    with pytest.raises(ValueError):
        forecast(engine, day=day)
    assert state(engine) == before


def test_replayed_match_id_rejected_even_on_later_day():
    engine = UnifiedPlayerRatingEngine()
    engine.update_day("2024-01-01", [game()])
    before = state(engine)
    with pytest.raises(ValueError):
        engine.update_day("2024-01-02", [game()])
    assert state(engine) == before


def test_day_batch_uses_frozen_elo_and_stable_match_id_gaussian_order():
    games = [game("2", y=0), game("1", y=1), game("3", y=1)]
    first = UnifiedPlayerRatingEngine()
    second = UnifiedPlayerRatingEngine()
    first.update_day("2024-01-01", games)
    reversed_games = [game(g["hltv_match_id"], tuple(reversed(A)), tuple(reversed(B)), g["y"]) for g in reversed(games)]
    second.update_day("2024-01-01", reversed_games)
    assert first.get_rankings() == second.get_rankings()
    assert first.players[A[0]].elo == 1516
    expected = glicko2_rating_period(1500, 350, 0.06, [(1500, 1050, 1), (1500, 1050, 0), (1500, 1050, 1)], tau=0.5)
    p = first.players[A[0]]
    assert (p.glicko_rating, p.glicko_rd, p.glicko_volatility) == pytest.approx(expected)
    ts = first.ts_env
    os = first.os_model
    ts_teams = [[ts.create_rating() for _ in range(5)] for _ in range(2)]
    os_teams = [[os.rating() for _ in range(5)] for _ in range(2)]
    for g in sorted(games, key=lambda record: record["hltv_match_id"]):
        ranks = [0, 1] if g["y"] else [1, 0]
        ts_teams = ts.rate(ts_teams, ranks=ranks)
        os_teams = os.rate(os_teams, ranks=ranks)
    for side, roster in enumerate((tuple(sorted(A)), tuple(sorted(B)))):
        for i, key in enumerate(roster):
            assert first.players[key].trueskill_mu == pytest.approx(ts_teams[side][i].mu)
            assert first.players[key].trueskill_sigma == pytest.approx(ts_teams[side][i].sigma)
            assert first.players[key].openskill_mu == pytest.approx(os_teams[side][i].mu)
            assert first.players[key].openskill_sigma == pytest.approx(os_teams[side][i].sigma)


def test_side_representation_and_game_permutations_preserve_multiday_replay():
    first = UnifiedPlayerRatingEngine()
    second = UnifiedPlayerRatingEngine()
    for day, games in (
        ("2024-01-01", [game("2", y=0), game("1", y=1)]),
        ("2024-01-08", [game("4", y=1), game("3", y=0)]),
    ):
        first.update_day(day, games)
        swapped = []
        for original in reversed(games):
            reverse = game(original["hltv_match_id"], tuple(reversed(B)), tuple(reversed(A)), 1 - original["y"])
            reverse["team1"], reverse["team2"] = "Beta", "Alpha"
            swapped.append(reverse)
        second.update_day(day, swapped)
    assert first.get_rankings() == second.get_rankings()
    assert forecast(first, day="2024-01-09") == forecast(second, day="2024-01-09")


def test_family_rankings_are_independent_with_competition_ties_and_all_observed_players():
    engine = UnifiedPlayerRatingEngine()
    engine.update_day("2024-01-01", [game()])
    assert len(engine.get_rankings()) == 10
    for key in (*A, *B):
        p = engine.players[key]
        p.elo = 1500
        p.glicko_rating = 1500
        p.trueskill_mu, p.trueskill_sigma = 25, 1
        p.openskill_mu, p.openskill_sigma = 25, 1
    engine.players[A[0]].elo = 1600
    engine.players[A[1]].glicko_rating = 1600
    engine.players[A[2]].trueskill_mu = 30
    engine.players[A[3]].openskill_mu = 30
    rows = {r["player_key"]: r for r in engine.get_rankings()}
    for i, family in enumerate(("elo", "glicko", "trueskill", "openskill")):
        assert rows[A[i]][f"{family}_rank"] == 1
        assert rows[B[0]][f"{family}_rank"] == 2
    assert engine.get_rankings(min_matches=2) == []


def test_inactivity_projects_only_intervening_calendar_days_without_mutation():
    engine = UnifiedPlayerRatingEngine()
    engine.update_day("2024-01-01", [game()])
    p = engine.players[A[0]]
    before = state(engine)
    immediate = forecast(engine, day="2024-01-02")
    later = forecast(engine, day="2024-01-16")
    assert later["glicko_joint_rd"] > immediate["glicko_joint_rd"]
    expected_variance = sum(
        engine.players[k].glicko_rd**2 + (173.7178 * engine.players[k].glicko_volatility)**2 * 14 / 7
        for k in (*A, *B)
    )
    assert later["glicko_joint_rd"] == pytest.approx(math.sqrt(expected_variance))
    assert state(engine) == before
    assert p.glicko_rd < 350


def test_published_glicko2_numerical_example():
    # Glickman's worked example: r=1500, RD=200, sigma=.06, tau=.5.
    r, rd, volatility = glicko2_rating_period(
        1500, 200, 0.06, [(1400, 30, 1), (1550, 100, 0), (1700, 300, 0)], tau=0.5,
    )
    assert r == pytest.approx(1464.06, abs=0.01)
    assert rd == pytest.approx(151.52, abs=0.01)
    assert volatility == pytest.approx(0.059996, abs=0.000001)


def test_recent_outcomes_and_full_precision_support_are_career_not_future_statistics():
    engine = UnifiedPlayerRatingEngine()
    for i in range(12):
        engine.update_day(f"2024-01-{i + 1:02d}", [game(str(i), y=int(i >= 2))])
    rows = {r["player_key"]: r for r in engine.get_rankings()}
    assert rows[A[0]]["matches_played"] == 12
    assert rows[A[0]]["wins"] == 10
    assert rows[A[0]]["win_rate"] == 10 / 12
    assert rows[A[0]]["last10_win_rate"] == 1
    assert rows[A[0]]["last_active"] == "2024-01-12"
    f = forecast(engine, day="2024-01-13")
    assert f["minimum_prior_matches"] == 12
    assert f["team1_known_players"] == f["team2_known_players"] == 5
    assert f["lineup_last10_win_rate_diff"] == 1
    assert f["lineup_win_rate_diff"] == pytest.approx(2 / 3)
    assert f["elo_prob"] != round(f["elo_prob"], 4)


def test_roster_swap_complements_probabilities_and_negates_differences():
    engine = UnifiedPlayerRatingEngine()
    engine.update_day("2024-01-01", [game()])
    forward = forecast(engine)
    reverse = forecast(engine, a=B, b=A)
    for name in ("elo_prob", "glicko_prob", "trueskill_prob", "openskill_prob"):
        assert reverse[name] == pytest.approx(1 - forward[name], abs=1e-14)
    for name in ("elo_diff", "glicko_diff", "trueskill_mu_diff", "openskill_ordinal_diff", "lineup_exp_diff", "lineup_win_rate_diff"):
        assert reverse[name] == -forward[name]
