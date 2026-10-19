"""Literal-team archive history is causal, day-frozen, and read-only at query time."""
import math
import pickle

import pytest


@pytest.fixture
def engine():
    from pipeline.features.team_history import TeamHistoryEngine

    return TeamHistoryEngine()


def game(identity="1", team1="Alpha", team2="Beta", y=1):
    return {"hltv_match_id": identity, "team1": team1, "team2": team2, "y": y}


def forecast(engine, day="2024-01-02", team1="Alpha", team2="Beta"):
    return engine.compute_prematch_features(team1, team2, day=day)


def test_cold_start_has_neutral_probabilities_and_explicit_zero_support(engine):
    before = pickle.dumps(engine)
    features = forecast(engine, day="2024-01-01")
    assert features == {
        "archive_elo_diff": 0.0,
        "archive_elo_prob": 0.5,
        "archive_decay_diff": 0.0,
        "archive_decay_prob": 0.5,
        "archive_team1_prior_matches": 0,
        "archive_team2_prior_matches": 0,
        "archive_minimum_prior_matches": 0,
    }
    assert pickle.dumps(engine) == before


def test_reversal_complements_probabilities_negates_strength_and_swaps_support(engine):
    engine.update_day("2024-01-01", [game()])
    engine.update_day("2024-01-02", [game("2", team2="Gamma")])
    forward = forecast(engine, day="2024-01-03")
    reverse = forecast(engine, day="2024-01-03", team1="Beta", team2="Alpha")
    for channel in ("archive_elo", "archive_decay"):
        assert forward[channel + "_prob"] > 0.5
        assert forward[channel + "_prob"] + reverse[channel + "_prob"] == pytest.approx(1)
        assert forward[channel + "_diff"] == pytest.approx(-reverse[channel + "_diff"])
    assert forward["archive_team1_prior_matches"] == reverse["archive_team2_prior_matches"] == 2
    assert forward["archive_team2_prior_matches"] == reverse["archive_team1_prior_matches"] == 1
    assert forward["archive_minimum_prior_matches"] == reverse["archive_minimum_prior_matches"] == 1


def test_inactivity_halves_decay_logit_in_180_days_but_not_nondecay_strength(engine):
    engine.update_day("2024-01-01", [game()])
    soon = forecast(engine, day="2024-01-02")
    later = forecast(engine, day="2024-06-30")
    logit = lambda p: math.log(p / (1 - p))
    assert soon["archive_elo_diff"] == pytest.approx(32)
    assert later["archive_elo_diff"] == soon["archive_elo_diff"]
    assert later["archive_elo_prob"] == soon["archive_elo_prob"]
    assert 0 < logit(later["archive_decay_prob"]) < logit(soon["archive_decay_prob"])
    assert logit(later["archive_decay_prob"]) == pytest.approx(logit(soon["archive_decay_prob"]) / 2)
    assert later["archive_team1_prior_matches"] == soon["archive_team1_prior_matches"] == 1


def test_unknown_and_far_future_queries_do_not_register_teams_or_age_saved_state(engine):
    engine.update_day("2024-01-01", [game()])
    before = pickle.dumps(engine)
    original = forecast(engine)
    cold = forecast(engine, day="2030-01-01", team1="Unknown", team2="Other")
    forecast(engine, day="2030-01-01")
    assert cold["archive_elo_prob"] == cold["archive_decay_prob"] == 0.5
    assert cold["archive_minimum_prior_matches"] == 0
    assert pickle.dumps(engine) == before
    assert forecast(engine) == original


@pytest.mark.parametrize("literal", ["alpha", "ALPHA", "Alpha Academy", " Alpha"])
def test_literal_team_names_never_inherit_another_names_evidence(engine, literal):
    engine.update_day("2024-01-01", [game()])
    features = forecast(engine, team1=literal, team2="Unseen")
    assert features["archive_elo_prob"] == features["archive_decay_prob"] == 0.5
    assert features["archive_team1_prior_matches"] == features["archive_team2_prior_matches"] == 0


def test_same_day_batch_uses_start_of_day_ratings_and_is_order_invariant(engine):
    from pipeline.features.team_history import TeamHistoryEngine

    other = TeamHistoryEngine()
    games = [game("2"), game("1"), game("3", team2="Gamma", y=0)]
    engine.update_day("2024-01-01", games)
    other.update_day("2024-01-01", list(reversed(games)))
    result = forecast(engine)
    assert result == forecast(other)
    # Alpha gains 16+16-16, Beta loses 16+16: every expectation was 0.5.
    assert result["archive_elo_diff"] == pytest.approx(48)
    assert result["archive_team1_prior_matches"] == 3
    assert result["archive_team2_prior_matches"] == 2


@pytest.mark.parametrize("invalid", [
    game("3", team1=""), game("3", team1="  "), game("3", team2="Alpha"),
    game("3", team1=None), game("3", team2=float("nan")),
    game("3", y=2), game("3", y=float("nan")), game(""), game("not-native"),
    game("1"), game("2"),
])
def test_invalid_whole_batch_has_no_partial_mutation(engine, invalid):
    engine.update_day("2024-01-01", [game()])
    before = pickle.dumps(engine)
    with pytest.raises(ValueError):
        engine.update_day("2024-01-02", [game("2"), invalid])
    assert pickle.dumps(engine) == before
    assert forecast(engine)["archive_team1_prior_matches"] == 1


def test_pickle_roundtrip_preserves_queries_and_subsequent_updates(engine):
    engine.update_day("2024-01-01", [game()])
    restored = pickle.loads(pickle.dumps(engine))
    assert forecast(restored) == forecast(engine)
    engine.update_day("2024-01-02", [game("2", y=0)])
    restored.update_day("2024-01-02", [game("2", y=0)])
    assert forecast(restored, day="2024-01-03") == forecast(engine, day="2024-01-03")
