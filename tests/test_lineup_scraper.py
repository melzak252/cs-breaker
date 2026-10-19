import unittest

from scripts.scrape_lineups import create_lineup_row


class LineupScraperTests(unittest.TestCase):
    def players(self, first):
        return [{"id": str(first + i), "name": f"Player{first + i}"} for i in range(5)]

    def test_six_participants_are_not_silently_presented_as_a_complete_five(self):
        left, right = self.players(1), self.players(10)
        left.append({"id": "6", "name": "Substitute"})
        row = create_lineup_row({"match_id": "123"}, left, right, "complete", "hltv_live")
        self.assertNotEqual(row["lineup_status"], "complete")

    def test_name_only_roster_cannot_be_marked_as_a_native_identity_capture(self):
        left, right = self.players(1), self.players(10)
        left[0]["id"] = ""
        row = create_lineup_row({"match_id": "123"}, left, right, "complete", "hltv_live")
        self.assertNotEqual(row["lineup_status"], "complete")

    def test_overlapping_native_player_cannot_play_for_both_teams(self):
        left, right = self.players(1), self.players(10)
        right[0]["id"] = left[0]["id"]
        row = create_lineup_row({"match_id": "123"}, left, right, "complete", "hltv_live")
        self.assertNotEqual(row["lineup_status"], "complete")


if __name__ == "__main__":
    unittest.main()
