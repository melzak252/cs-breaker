"""Conservative cross-source series identity and outcome regressions."""
import unittest

from scripts import map_cs_oddspapi_history as mapper


def fixture(identity='id1', one='Team Alpha', two='Beta Esports', start='2026-04-14T10:00:00Z', event='PGL Bucharest'):
    return {'fixtureId': identity, 'participant1Name': one, 'participant2Name': two,
            'startTime': start, 'sportId': 17, 'statusId': 2, 'tournamentName': event}


def historical(identity='100', one='Alpha', two='Beta', date='2026-04-14T12:00:00Z', event='PGL Bucharest 2026', score1='2', score2='1', winner='team1'):
    return {'match_id': identity, 'team1': one, 'team2': two, 'match_date': date,
            'event': event, 'score1': score1, 'score2': score2, 'winner': winner,
            'format': 'bo3', 'match_url': f'https://www.hltv.org/matches/{identity}/example'}


class HistoricalMappingTests(unittest.TestCase):
    def links(self, fixtures, rows):
        links, _ = mapper.map_fixtures(fixtures, mapper.historical_series(rows))
        return {r['fixtureId']: r for r in links}

    def test_reversed_orientation_flips_winner_to_provider_participant_one(self):
        result = self.links([fixture()], [historical(one='Beta', two='Alpha')])['id1']
        self.assertEqual(result['mapping_status'], 'matched')
        self.assertEqual(result['participant1_is_historical_team1'], False)
        self.assertEqual(result['outcome_1_won'], 0)

    def test_multiple_same_pair_matches_are_not_resolved_by_nearest_time(self):
        result = self.links([fixture()], [historical('100'), historical('101', date='2026-04-14T13:00:00Z')])['id1']
        self.assertEqual(result['mapping_status'], 'ambiguous_fixture')
        self.assertEqual(result['outcome_1_won'], '')

    def test_two_provider_fixtures_cannot_claim_one_historical_identity(self):
        results = self.links([fixture('id1'), fixture('id2', start='2026-04-14T11:00:00Z')], [historical()])
        self.assertEqual({r['mapping_status'] for r in results.values()}, {'ambiguous_historical_match'})
        self.assertTrue(all(r['outcome_1_won'] == '' for r in results.values()))

    def test_academy_and_former_team_qualifiers_are_not_removed(self):
        for name in ('Team Alpha Academy', 'Ex-Alpha'):
            with self.subTest(name=name):
                result = self.links([fixture(one=name)], [historical()])['id1']
                self.assertNotEqual(result['mapping_status'], 'matched')
                self.assertEqual(result['outcome_1_won'], '')

    def test_region_conflict_blocks_otherwise_unique_name_and_time_pair(self):
        result = self.links([fixture(event='ESL Challenger League North America')],
                            [historical(event='ESL Challenger League Season 51 Oceania Cup 3')])['id1']
        self.assertEqual(result['mapping_status'], 'event_conflict')
        self.assertEqual(result['outcome_1_won'], '')

    def test_unrelated_events_are_not_accepted_from_names_alone(self):
        result = self.links([fixture(event='PGL Bucharest')], [historical(event='IEM Rio 2026')])['id1']
        self.assertEqual(result['mapping_status'], 'event_unconfirmed')

    def test_utc_midnight_crossing_retains_same_series(self):
        result = self.links([fixture(start='2026-04-14T23:00:00Z')],
                            [historical(date='2026-04-15T01:00:00+00:00')])['id1']
        self.assertEqual(result['mapping_status'], 'matched')
        self.assertEqual(result['outcome_1_won'], 1)

    def test_outside_time_window_does_not_attach_result(self):
        for date in ('2026-04-14T09:59:59Z', '2026-04-14T16:00:01Z'):
            with self.subTest(date=date):
                result = self.links([fixture()], [historical(date=date)])['id1']
                self.assertNotEqual(result['mapping_status'], 'matched')
                self.assertEqual(result['outcome_1_won'], '')

    def test_naive_historical_clock_is_rejected_instead_of_assumed_utc(self):
        with self.assertRaises(ValueError):
            mapper.historical_series([historical(date='2026-04-14T12:00:00')])

    def test_conflicting_top_level_results_across_maps_block_label(self):
        result = self.links([fixture()], [historical(), historical(score1='1', score2='2', winner='team2')])['id1']
        self.assertEqual(result['mapping_status'], 'historical_metadata_conflict')
        self.assertEqual(result['outcome_1_won'], '')

    def test_round_scores_and_inconsistent_winner_are_not_series_labels(self):
        for row in (historical(score1='13', score2='7'), historical(winner='team2')):
            with self.subTest(row=row):
                result = self.links([fixture()], [row])['id1']
                self.assertEqual(result['mapping_status'], 'matched')
                self.assertEqual(result['label_status'], 'invalid_series_result')
                self.assertEqual(result['outcome_1_won'], '')

    def test_missing_bookmaker_quotes_are_not_zero_loss(self):
        links = self.links([fixture()], [historical()])
        quote = {'fixtureId': 'id1', 'bookmaker': 'pinnacle', 'price_1': '2', 'price_2': '3',
                 'no_vig_probability_1': '0.6'}
        scored = mapper.score_quotes([quote], list(links.values()))
        metrics = mapper.bookmaker_metrics(scored, ['pinnacle', 'bet365'])
        self.assertAlmostEqual(metrics['available_cohorts']['pinnacle']['log_loss'], 0.5108256237659907)
        self.assertIsNone(metrics['available_cohorts']['bet365']['log_loss'])
        self.assertEqual(metrics['common_all_requested_books']['matches'], 0)


if __name__ == '__main__':
    unittest.main()
