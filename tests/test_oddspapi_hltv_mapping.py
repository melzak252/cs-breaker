"""Native HLTV calendar linkage must preserve ambiguity and side orientation."""
import unittest

from scripts.map_cs_oddspapi_history import hltv_calendar_series, map_hltv_fixtures


def fixture(identity="id1", one="Team Alpha", two="Beta Esports", start="2026-04-14T18:00:00Z", event="PGL Bucharest"):
    return {"fixtureId": identity, "participant1Name": one, "participant2Name": two,
            "startTime": start, "sportId": 17, "statusId": 2, "tournamentName": event}


def result(identity="100", one="Alpha", two="Beta", day="2026-04-14", event="PGL Bucharest 2026", score1="2", score2="1", winner="team1", kind="bo3"):
    return {"match_id": identity, "date": day + " 12:00:00", "team1": one, "team2": two,
            "tournament": event, "score1": score1, "score2": score2, "winner": winner,
            "format": kind, "match_url": f"https://www.hltv.org/matches/{identity}/alpha-vs-beta"}


class HLTVCalendarMappingTests(unittest.TestCase):
    def links(self, fixtures, rows):
        links, _ = map_hltv_fixtures(fixtures, hltv_calendar_series(rows))
        return {row["fixtureId"]: row for row in links}

    def test_synthetic_noon_is_not_compared_to_real_kickoff(self):
        link = self.links([fixture()], [result()])["id1"]
        self.assertEqual(link["mapping_status"], "matched")
        self.assertEqual(link["hltv_match_id"], "100")
        self.assertEqual(link["hltv_calendar_day"], "2026-04-14")
        self.assertEqual(link["outcome_1_won"], 1)

    def test_reversed_sides_attach_the_native_id_and_complement_result(self):
        link = self.links([fixture(one="Beta", two="Alpha")], [result()])["id1"]
        self.assertEqual(link["mapping_status"], "matched")
        self.assertFalse(link["participant1_is_hltv_team1"])
        self.assertEqual(link["outcome_1_won"], 0)

    def test_adjacent_calendar_day_is_supported_but_two_days_is_not(self):
        for day, accepted in (("2026-04-13", True), ("2026-04-15", True), ("2026-04-16", False)):
            with self.subTest(day=day):
                link = self.links([fixture()], [result(day=day)])["id1"]
                self.assertEqual(link["mapping_status"] == "matched", accepted)

    def test_nearest_day_and_matching_score_do_not_resolve_multiple_candidates(self):
        rows = [result(), result("101", day="2026-04-15", score1="1", score2="2", winner="team2")]
        link = self.links([fixture()], rows)["id1"]
        self.assertEqual(link["mapping_status"], "ambiguous_fixture")
        self.assertEqual(link["outcome_1_won"], "")

    def test_two_provider_fixtures_cannot_claim_one_hltv_game(self):
        links = self.links([fixture("id1"), fixture("id2")], [result()])
        self.assertEqual({r["mapping_status"] for r in links.values()}, {"ambiguous_hltv_match"})
        self.assertTrue(all(r["outcome_1_won"] == "" for r in links.values()))

    def test_incompatible_event_and_academy_names_are_not_accepted(self):
        for f in (fixture(event="IEM Rio"), fixture(one="Alpha Academy")):
            with self.subTest(fixture=f):
                self.assertNotEqual(self.links([f], [result()])["id1"]["mapping_status"], "matched")

    def test_conflicting_native_id_rows_block_link_but_exact_duplicates_do_not(self):
        self.assertEqual(self.links([fixture()], [result(), result()])["id1"]["mapping_status"], "matched")
        rows = [result(), result(score1="1", score2="2", winner="team2")]
        link = self.links([fixture()], rows)["id1"]
        self.assertEqual(link["mapping_status"], "hltv_metadata_conflict")
        self.assertEqual(link["outcome_1_won"], "")

    def test_foreign_or_disagreeing_match_url_is_rejected(self):
        row = result()
        row["match_url"] = "https://www.hltv.org/matches/101/other"
        with self.assertRaises(ValueError):
            hltv_calendar_series([row])

    def test_forfeit_identity_can_be_linked_without_becoming_a_played_result(self):
        link = self.links([fixture()], [result(kind="def")])["id1"]
        self.assertEqual(link["mapping_status"], "matched")
        self.assertEqual(link["outcome_1_won"], "")

    def test_unfinished_provider_fixture_is_preserved_without_result_link(self):
        upcoming = fixture()
        upcoming["statusId"] = 0
        link = self.links([upcoming], [result()])["id1"]
        self.assertEqual(link["mapping_status"], "not_finished")
        self.assertEqual(link["outcome_1_won"], "")

    def test_event_preference_cannot_remove_a_second_candidate(self):
        rows = [result(), result("101", event="IEM Rio")]
        links, candidates = map_hltv_fixtures([fixture()], hltv_calendar_series(rows))
        self.assertEqual(links[0]["mapping_status"], "ambiguous_fixture")
        self.assertEqual({row["hltv_match_id"] for row in candidates}, {"100", "101"})
        self.assertEqual(links[0]["candidate_count"], 2)

    def test_event_conflicting_fixture_still_counts_for_native_uniqueness(self):
        links = self.links([fixture(), fixture("id2", event="IEM Rio")], [result()])
        self.assertEqual({row["mapping_status"] for row in links.values()}, {"ambiguous_hltv_match"})

    def test_duplicate_rows_are_preserved_in_candidate_evidence(self):
        series = hltv_calendar_series([result(), result()])
        links, candidates = map_hltv_fixtures([fixture()], series)
        self.assertEqual(links[0]["mapping_status"], "matched")
        self.assertEqual(links[0]["source_row_count"], 2)
        self.assertEqual(len(candidates), 2)
        self.assertEqual([row["source_row_index"] for row in candidates], [1, 2])

    def test_conflict_evidence_includes_rows_that_do_not_match_this_fixture(self):
        rows = [result(), result(one="Other Alpha", day="2026-04-20")]
        links, candidates = map_hltv_fixtures([fixture()], hltv_calendar_series(rows))
        self.assertEqual(links[0]["mapping_status"], "hltv_metadata_conflict")
        self.assertEqual(len(candidates), 2)
        self.assertEqual([row["candidate_edge"] for row in candidates], [True, False])
        self.assertTrue(all(row["outcome_1_won"] == "" for row in candidates))

    def test_native_ids_must_be_canonical_positive_ascii(self):
        for identity in ("0", "0100", "-100", "１００", "١٠٠", "OpenCS2:100"):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                hltv_calendar_series([result(identity)])

    def test_foreign_url_cannot_certify_native_identity(self):
        for url in ("https://opencs2.com/matches/100", "https://www.hltv.org.evil/matches/100",
                    "https://www.hltv.org/matches/100?match=101"):
            row = result()
            row["match_url"] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                hltv_calendar_series([row])

    def test_valid_bo5_and_invalid_explicit_results_remain_distinct(self):
        valid = result(kind="bo5", score1="1", score2="3", winner="team2")
        link = self.links([fixture()], [valid])["id1"]
        self.assertEqual(link["label_status"], "valid_explicit_series_result")
        self.assertEqual(link["outcome_1_won"], 0)
        for row in (result(winner="team2"), result(score1="2", score2="2"),
                    result(score1="2", score2="-1"), result(kind="bo1", score1="1", score2="0")):
            with self.subTest(row=row):
                link = self.links([fixture()], [row])["id1"]
                self.assertEqual(link["mapping_status"], "matched")
                self.assertEqual(link["outcome_1_won"], "")

    def test_utc_scheduled_day_and_short_names_define_edges(self):
        provider = fixture(one="Unrelated Full Name", start="2026-04-15T00:30:00+02:00")
        provider["participant1ShortName"] = "Alpha"
        provider["trueStartTime"] = "2026-04-15T01:00:00Z"
        link = self.links([provider], [result(day="2026-04-13")])["id1"]
        self.assertEqual(link["mapping_status"], "matched")
        self.assertEqual(link["calendar_day_offset"], -1)
        self.assertEqual(link["true_start"], provider["trueStartTime"])

    def test_overlapping_name_aliases_do_not_invent_orientation(self):
        provider = fixture(one="Alpha", two="Beta")
        provider.update(participant1ShortName="Beta", participant2ShortName="Alpha")
        link = self.links([provider], [result()])["id1"]
        self.assertEqual(link["mapping_status"], "unmatched")
        self.assertEqual(link["participant1_is_hltv_team1"], "")

    def test_display_team_suffix_matches_without_erasing_academy_qualifier(self):
        row = result(one="BETBOOM", two="9z")
        provider = fixture(one="Betboom Team", two="9Z Team")
        self.assertEqual(self.links([provider], [row])["id1"]["mapping_status"], "matched")
        provider["participant1Name"] = "Betboom Academy Team"
        self.assertEqual(self.links([provider], [row])["id1"]["mapping_status"], "unmatched")


if __name__ == "__main__":
    unittest.main()
