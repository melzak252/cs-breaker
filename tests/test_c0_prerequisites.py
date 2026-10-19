"""Comprehensive unit tests for CS2 C0 prerequisites.

Tests:
1. CS2 tokenization, vocabulary maps, and token embeddings.
2. Chronological sequence builder, rolling history, and temporal causality.
3. PyTorch C0 architecture:
   - Masked player history attention (no NaNs on empty history).
   - Permutation-invariant 5-player Team Set Transformer.
   - Strict pairwise anti-symmetric series winner probability head.
4. Serialized player sequence bank loading and verification.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.features.player_sequence_builder import (
    PlayerHistoryTracker,
    build_match_input_tensors,
)
from src.models.c0_architecture import (
    CS2C0Model,
    CS2PlayerHistoryAttention,
    CS2TeamSetTransformer,
)
from src.models.c0_tokens import (
    CS2TokenConfig,
    CS2TokenEmbedding,
    encode_player_map_token,
    encode_player_stats,
    get_map_id,
    get_side_id,
)


class TestCS2Tokenization:
    def test_map_and_side_vocab(self):
        assert get_map_id("de_mirage") == 1
        assert get_map_id("MIRAGE") == 1
        assert get_map_id("inferno") == 2
        assert get_map_id("de_inferno") == 2
        assert get_map_id("dust 2") == 4
        assert get_map_id("dust2") == 4
        assert get_map_id("de_dust2") == 4
        assert get_map_id("ancient") == 5
        assert get_map_id("non_existent_map") == 0

        assert get_side_id("CT") == 1
        assert get_side_id("counter-terrorist") == 1
        assert get_side_id("t") == 2
        assert get_side_id("terrorist") == 2
        assert get_side_id("neutral") == 0

    def test_encode_player_stats(self):
        stats = {
            "kills": 22,
            "deaths": 14,
            "assists": 6,
            "adr": 85.4,
            "kast": "75.0%",
            "rating": 1.25,
            "hs_pct": 0.45,
            "won": 1,
        }
        vec = encode_player_stats(stats)
        assert vec.shape == (16,)
        assert vec.dtype == np.float32
        assert vec[0] == 22.0  # kills
        assert vec[1] == 14.0  # deaths
        assert vec[5] == 1.25  # rating
        assert np.isfinite(vec).all()

    def test_token_embedding_forward(self):
        config = CS2TokenConfig(hidden_dim=32)
        embedder = CS2TokenEmbedding(config)

        # Batch 2, 16 maps, 18-dim token [map_id, side_id, 16 stats]
        tokens = torch.zeros(2, 16, 18, dtype=torch.float32)
        tokens[:, :, 0] = 1.0  # mirage
        tokens[:, :, 1] = 1.0  # CT
        tokens[:, :, 2:] = torch.randn(2, 16, 16)

        out = embedder(tokens)
        assert out.shape == (2, 16, 32)
        assert torch.isfinite(out).all()


class TestPlayerSequenceBuilder:
    def test_history_tracker_temporal_causality(self):
        tracker = PlayerHistoryTracker(max_history=16)

        # Add 3 records on different dates
        tracker.add_player_map_record(
            player_id="zywoo",
            date="2026-01-01",
            map_name="mirage",
            side="CT",
            stats_dict={"kills": 25, "rating": 1.50},
            won=1,
        )
        tracker.add_player_map_record(
            player_id="zywoo",
            date="2026-01-05",
            map_name="inferno",
            side="T",
            stats_dict={"kills": 20, "rating": 1.20},
            won=1,
        )
        tracker.add_player_map_record(
            player_id="zywoo",
            date="2026-01-10",
            map_name="nuke",
            side="CT",
            stats_dict={"kills": 15, "rating": 0.95},
            won=0,
        )

        # Query with cutoff 2026-01-08: must ONLY include Jan 01 and Jan 05 records
        seq, mask = tracker.get_player_sequence("zywoo", cutoff_date="2026-01-08")
        assert mask.sum() == 2
        assert mask[0] is True or mask[0] == 1
        assert mask[1] is True or mask[1] == 1
        assert not mask[2]

        # Query with cutoff 2026-01-01: must be empty (strictly earlier than cutoff)
        seq_empty, mask_empty = tracker.get_player_sequence("zywoo", cutoff_date="2026-01-01")
        assert mask_empty.sum() == 0

        # Query unknown player: must be empty
        _, mask_unk = tracker.get_player_sequence("unknown_player", cutoff_date="2026-01-15")
        assert mask_unk.sum() == 0

    def test_build_match_input_tensors(self):
        tracker = PlayerHistoryTracker(max_history=16)
        tracker.add_player_map_record("p1", "2026-02-01", "mirage", "CT", {"kills": 20}, 1)
        tracker.add_player_map_record("p2", "2026-02-02", "nuke", "T", {"kills": 18}, 0)

        t1 = ["p1", "p2", "p3", "p4", "p5"]
        t2 = ["q1", "q2", "q3", "q4", "q5"]

        tensors = build_match_input_tensors(tracker, t1, t2, cutoff_date="2026-02-10")
        assert tensors["history"].shape == (1, 2, 5, 16, 18)
        assert tensors["mask"].shape == (1, 2, 5, 16)
        assert tensors["player_counts"].shape == (1, 2, 5)

        # p1 has 1 map, p2 has 1 map, others have 0
        assert tensors["player_counts"][0, 0, 0] == 1
        assert tensors["player_counts"][0, 0, 1] == 1
        assert tensors["player_counts"][0, 0, 2] == 0
        assert (tensors["player_counts"][0, 1] == 0).all()

    def test_tracker_serialization(self, tmp_path):
        tracker = PlayerHistoryTracker(max_history=16)
        tracker.add_player_map_record("m0NESY", "2026-03-01", "mirage", "CT", {"kills": 30}, 1)

        save_path = tmp_path / "tracker.pt"
        tracker.save(save_path)
        assert save_path.exists()

        loaded = PlayerHistoryTracker.load(save_path)
        seq, mask = loaded.get_player_sequence("m0NESY", cutoff_date="2026-03-05")
        assert mask.sum() == 1


class TestCS2C0Architecture:
    def test_masked_player_history_attention(self):
        config = CS2TokenConfig(hidden_dim=32)
        attn = CS2PlayerHistoryAttention(config)

        # 2 batches, 2 teams, 5 players, 16 maps, 18 features
        history = torch.randn(2, 2, 5, 16, 18)
        mask = torch.ones(2, 2, 5, 16, dtype=torch.bool)

        # Player 0 of team 0 has empty history (all-False mask)
        mask[0, 0, 0, :] = False

        attended = attn(history, mask)
        assert attended.shape == (2, 2, 5, 32)
        # Empty history must not produce NaN!
        assert not torch.isnan(attended).any()
        assert torch.isfinite(attended).all()

    def test_team_set_transformer_permutation_invariance(self):
        set_transformer = CS2TeamSetTransformer(d_model=32, nhead=4)

        # Batch 2, 5 players, 32 dims
        players = torch.randn(2, 5, 32)

        # Original representation
        team_vec_orig = set_transformer(players)
        assert team_vec_orig.shape == (2, 64)

        # Shuffled player permutation: e.g. [3, 0, 4, 1, 2]
        perm = [3, 0, 4, 1, 2]
        players_shuffled = players[:, perm, :]

        team_vec_shuffled = set_transformer(players_shuffled)

        # Must be identical to machine precision due to permutation-invariant pooling
        assert torch.allclose(team_vec_orig, team_vec_shuffled, atol=1e-5)

    def test_c0_model_pairwise_antisymmetry(self):
        config = CS2TokenConfig(hidden_dim=32)
        model = CS2C0Model(token_config=config)
        model.eval()

        batch_size = 4
        history = torch.randn(batch_size, 2, 5, 16, 18)
        mask = torch.randint(0, 2, (batch_size, 2, 5, 16), dtype=torch.bool)

        with torch.no_grad():
            # A vs B
            p_ab = model(history, mask)
            assert p_ab.shape == (batch_size, 1)
            assert (p_ab >= 0.0).all() and (p_ab <= 1.0).all()

            # B vs A (swap teams)
            history_swapped = history[:, [1, 0], :, :, :]
            mask_swapped = mask[:, [1, 0], :, :]
            p_ba = model(history_swapped, mask_swapped)

            # Probabilities MUST complement: P(A > B) + P(B > A) == 1.0
            sum_prob = p_ab + p_ba
            assert torch.allclose(sum_prob, torch.ones_like(sum_prob), atol=1e-6)

            # Logits MUST invert: logit(A, B) == -logit(B, A)
            logit_ab = model.compute_logits(history, mask)
            logit_ba = model.compute_logits(history_swapped, mask_swapped)
            assert torch.allclose(logit_ab, -logit_ba, atol=1e-5)

    def test_c0_model_symmetric_self_matchup(self):
        config = CS2TokenConfig(hidden_dim=32)
        model = CS2C0Model(token_config=config)
        model.eval()

        # Team A plays Team A (identical histories)
        team_history = torch.randn(2, 1, 5, 16, 18)
        history = torch.cat([team_history, team_history], dim=1)  # shape (2, 2, 5, 16, 18)
        mask = torch.ones(2, 2, 5, 16, dtype=torch.bool)

        with torch.no_grad():
            prob = model(history, mask)
            # Self-matchup win probability MUST be exactly 0.50
            assert torch.allclose(prob, torch.full_like(prob, 0.50), atol=1e-6)


class TestSequenceBankGeneration:
    def test_sequence_bank_file_exists_and_loads(self):
        bank_path = Path("data/04_feature/player_sequence_bank.pt")
        assert bank_path.exists(), "player_sequence_bank.pt was not generated"

        tracker = PlayerHistoryTracker.load(bank_path)
        assert len(tracker) > 500, f"Expected >500 tracked players, got {len(tracker)}"
