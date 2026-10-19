import copy
import unittest
from datetime import datetime, timezone

from pipeline.ratings.opencs2 import reconstruct_map


def demo_row():
    teams = [{"side_start":side, "players":[{"name":f"p{i}","steamid":i} for i in ids]}
             for side,ids in (("ct",range(1,6)),("t",range(6,11)))]
    rounds, kills = [], []
    for number in range(1,18):
        group0_side = "ct" if number<=12 else "t"
        group1_side = "t" if number<=12 else "ct"
        group0_wins = number in (1,2,13,14)
        winner_side = group0_side if group0_wins else group1_side
        rounds.append({"round_num":number,"winner":winner_side})
        kills.append({"round_num":number,"attacker_steamid":1,"attacker_side":group0_side,
                      "victim_steamid":6,"victim_side":group1_side})
    return {"row_id":"m1", "match_id":"series1", "map_name":"de_nuke",
            "match_date":datetime(2026,4,1,12,tzinfo=timezone.utc), "teams":teams,
            "rounds":rounds,"kills":kills,"rounds_played":17,
            "match":{"score_ct":5,"score_t":12}, "team1":"unmapped A","team2":"unmapped B"}


class StableIdOutcomeTests(unittest.TestCase):
    def test_round_side_changes_recover_roster_winner_not_aggregate_ct_winner(self):
        observation, detail = reconstruct_map(demo_row())
        self.assertEqual(detail["roster_scores"], [4,13])
        self.assertEqual(observation.y, 0)
        self.assertEqual(observation.roster1, tuple(f"opencs2:steamid:{i}" for i in range(1,6)))
        self.assertTrue(observation.team1.startswith("lineup:"))
        self.assertNotIn("unmapped", observation.team1)

    def test_side_assignment_is_taken_from_events_not_assumed_from_half_number(self):
        row = demo_row()
        for rnd in row["rounds"]:
            rnd["winner"] = "t" if rnd["winner"]=="ct" else "ct"
        for kill in row["kills"]:
            for key in ("attacker_side","victim_side"):
                kill[key] = "t" if kill[key]=="ct" else "ct"
        _, detail = reconstruct_map(row)
        self.assertEqual(detail["roster_scores"], [4,13])

    def test_unlisted_participant_is_not_silently_ignored(self):
        row = demo_row()
        row["kills"][0]["attacker_steamid"] = 999
        with self.assertRaisesRegex(ValueError, "event_player_or_side_not_in_roster"):
            reconstruct_map(row)

    def test_round_without_identity_side_evidence_is_not_imputed(self):
        row = demo_row()
        row["kills"] = row["kills"][1:]
        with self.assertRaisesRegex(ValueError, "round_winner_or_side_evidence_missing"):
            reconstruct_map(row)

    def test_duplicate_rounds_overlapping_rosters_and_incomplete_scores_are_rejected(self):
        duplicate = demo_row()
        duplicate["rounds"].append(copy.deepcopy(duplicate["rounds"][0]))
        overlap = demo_row()
        overlap["teams"][1]["players"][0]["steamid"] = 1
        incomplete = demo_row()
        incomplete["rounds"] = incomplete["rounds"][:10]
        incomplete["rounds_played"] = 10
        for row in (duplicate, overlap, incomplete):
            with self.subTest(row=row), self.assertRaises(ValueError):
                reconstruct_map(row)


if __name__ == "__main__":
    unittest.main()
