"""Unit tests for CS2 C0 dataset, collator, and chronological splits."""

from pathlib import Path
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.features.c0_dataset import (
    CS2MatchDataset,
    collate_c0_batch,
    create_chronological_splits,
    extract_5v5_matches,
)
from src.features.player_sequence_builder import PlayerHistoryTracker


class TestCS2MatchDataset:
    def test_dataset_indexing_and_shapes(self):
        tracker = PlayerHistoryTracker(max_history=16)
        tracker.add_player_map_record("p1", "2026-01-01", "mirage", "CT", {"kills": 20}, 1)
        tracker.add_player_map_record("q1", "2026-01-02", "nuke", "T", {"kills": 15}, 0)

        matches = [
            {
                "team1_players": ["p1", "p2", "p3", "p4", "p5"],
                "team2_players": ["q1", "q2", "q3", "q4", "q5"],
                "date": "2026-01-05",
                "y": 1.0,
            },
            {
                "team1_players": ["p1", "p2", "p3", "p4", "p5"],
                "team2_players": ["q1", "q2", "q3", "q4", "q5"],
                "date": "2026-01-06",
                "y": 0.0,
            },
        ]

        ds = CS2MatchDataset(matches, tracker)
        assert len(ds) == 2

        item = ds[0]
        assert "history" in item
        assert "mask" in item
        assert "label" in item

        assert item["history"].shape == (2, 5, 16, 18)
        assert item["mask"].shape == (2, 5, 16)
        assert item["label"].shape == (1,)

        assert item["history"].dtype == torch.float32
        assert item["mask"].dtype == torch.bool
        assert item["label"].dtype == torch.float32

        assert not torch.isnan(item["history"]).any()
        assert not torch.isnan(item["label"]).any()
        assert item["label"].item() == 1.0

        # p1 has 1 recorded map, q1 has 1 recorded map
        assert item["mask"][0, 0].sum().item() == 1
        assert item["mask"][1, 0].sum().item() == 1
        # other players have 0
        assert item["mask"][0, 1:].sum().item() == 0

    def test_dataset_temporal_causality(self):
        tracker = PlayerHistoryTracker(max_history=16)
        # Event at Jan 05
        tracker.add_player_map_record("p1", "2026-01-05", "mirage", "CT", {"kills": 25}, 1)

        # Match at Jan 03: player has 0 prior maps strictly before cutoff
        match_early = {
            "team1_players": ["p1", "p2", "p3", "p4", "p5"],
            "team2_players": ["q1", "q2", "q3", "q4", "q5"],
            "date": "2026-01-03",
            "y": 1.0,
        }
        # Match at Jan 06: player has 1 prior map
        match_late = {
            "team1_players": ["p1", "p2", "p3", "p4", "p5"],
            "team2_players": ["q1", "q2", "q3", "q4", "q5"],
            "date": "2026-01-06",
            "y": 1.0,
        }

        ds = CS2MatchDataset([match_early, match_late], tracker)
        early_item = ds[0]
        late_item = ds[1]

        assert early_item["mask"][0, 0].sum().item() == 0
        assert late_item["mask"][0, 0].sum().item() == 1


class TestCollateBatch:
    def test_collate_c0_batch(self):
        tracker = PlayerHistoryTracker(max_history=16)
        matches = [
            {
                "team1_players": [f"p{i}" for i in range(5)],
                "team2_players": [f"q{i}" for i in range(5)],
                "date": "2026-01-05",
                "y": 1.0,
            },
            {
                "team1_players": [f"a{i}" for i in range(5)],
                "team2_players": [f"b{i}" for i in range(5)],
                "date": "2026-01-06",
                "y": 0.0,
            },
            {
                "team1_players": [f"c{i}" for i in range(5)],
                "team2_players": [f"d{i}" for i in range(5)],
                "date": "2026-01-07",
                "y": 1.0,
            },
        ]
        ds = CS2MatchDataset(matches, tracker)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_c0_batch)

        batches = list(loader)
        assert len(batches) == 2

        b0 = batches[0]
        assert b0["history"].shape == (2, 2, 5, 16, 18)
        assert b0["mask"].shape == (2, 2, 5, 16)
        assert b0["label"].shape == (2, 1)
        assert not torch.isnan(b0["history"]).any()
        assert not torch.isnan(b0["label"]).any()

        b1 = batches[1]
        assert b1["history"].shape == (1, 2, 5, 16, 18)
        assert b1["mask"].shape == (1, 2, 5, 16)
        assert b1["label"].shape == (1, 1)


class TestChronologicalSplits:
    def test_mock_splits_strict_ordering(self):
        tracker = PlayerHistoryTracker(max_history=16)

        # 10 mock matches
        matches_data = []
        for i in range(10):
            matches_data.append(
                {
                    "date": f"2026-01-{i+1:02d} 12:00:00",
                    "team1_p1_name": f"p{i}_0",
                    "team1_p2_name": f"p{i}_1",
                    "team1_p3_name": f"p{i}_2",
                    "team1_p4_name": f"p{i}_3",
                    "team1_p5_name": f"p{i}_4",
                    "team2_p1_name": f"q{i}_0",
                    "team2_p2_name": f"q{i}_1",
                    "team2_p3_name": f"q{i}_2",
                    "team2_p4_name": f"q{i}_3",
                    "team2_p5_name": f"q{i}_4",
                    "winner": "team1" if i % 2 == 0 else "team2",
                }
            )

        matches_df = pd.DataFrame(matches_data)
        train_ds, val_ds, test_ds = create_chronological_splits(
            tracker,
            matches_df,
            train_ratio=0.70,
            val_ratio=0.15,
        )

        assert len(train_ds) == 7
        assert len(val_ds) == 2
        assert len(test_ds) == 1

        train_dates = [m["date"] for m in train_ds.matches]
        val_dates = [m["date"] for m in val_ds.matches]
        test_dates = [m["date"] for m in test_ds.matches]

        assert max(train_dates) <= min(val_dates)
        assert max(val_dates) <= min(test_dates)

    def test_strandedpond_splits_if_available(self):
        m_path = Path("data/raw/strandedpond/matches.csv")
        p_path = Path("data/raw/strandedpond/player_stats.csv")
        if not m_path.exists() or not p_path.exists():
            pytest.skip("StrandedPond CSVs not found")

        tracker = PlayerHistoryTracker(max_history=16)
        train_ds, val_ds, test_ds = create_chronological_splits(
            tracker,
            m_path,
            p_path,
            train_ratio=0.70,
            val_ratio=0.15,
        )

        assert len(train_ds) > 1000
        assert len(val_ds) > 200
        assert len(test_ds) > 200

        train_dates = [m["date"] for m in train_ds.matches]
        val_dates = [m["date"] for m in val_ds.matches]
        test_dates = [m["date"] for m in test_ds.matches]

        assert max(train_dates) <= min(val_dates)
        assert max(val_dates) <= min(test_dates)
