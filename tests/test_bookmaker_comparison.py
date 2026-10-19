"""Bookmakers and the frozen player forecaster must share labels and cohorts."""
import json
import math
import unittest

from pipeline.models.bookmaker_comparison import compare_hltv_bookmakers


def fixture(identity="id1"):
    return {"fixtureId": identity, "participant1Name": "Beta Esports", "participant2Name": "Team Alpha",
            "startTime": "2026-04-14T18:00:00Z", "trueStartTime": "2026-04-14T18:05:00Z",
            "sportId": 17, "statusId": 2, "tournamentName": "PGL Bucharest"}


def link(identity="id1", hltv_id="100"):
    return {"fixtureId": identity, "mapping_status": "matched", "hltv_match_id": hltv_id,
            "hltv_match_url": f"https://www.hltv.org/matches/{hltv_id}/alpha-vs-beta",
            "hltv_team1": "Alpha", "hltv_team2": "Beta", "participant1_is_hltv_team1": False,
            "hltv_calendar_day": "2026-04-14", "hltv_format": "bo3", "hltv_score1": "2", "hltv_score2": "1",
            "hltv_winner_1": 1, "outcome_1_won": 0, "label_status": "valid_explicit_series_result",
            "matching_method": "audited_calendar_natural_keys"}


def prediction(hltv_id="100"):
    return {"hltv_match_id": hltv_id, "source_date": "2026-04-14", "date": "2026-04-14",
            "team1": "Alpha", "team2": "Beta", "target": "1",
            "prob_team1_win": "0.7", "prob_team2_win": "0.3"}


def quote(identity="id1", book="pinnacle"):
    return {"id": f"{identity}-{book}", "fixture_id": identity, "bookmaker": book,
            "kind": "historical", "status": "admitted", "price1": 3.0, "price2": 2.0,
            "source_time_1": "2026-04-14T17:00:00Z", "source_time_2": "2026-04-14T17:00:00Z",
            "observed_at": "2026-10-01T00:00:00Z", "raw_id": 1,
            "details_json": json.dumps({"quote": {"fixtureId": identity, "bookmaker": book,
                 "market_id": "171", "outcome_1": "171", "outcome_2": "172",
                 "binding_status": "provider_catalog_convention_only", "execution_eligibility": "not_certified"}})}


METADATA = {"selected_model": "pure_elo", "splits": {
    "train": {"end_date": "2026-01-01"}, "validation": {"end_date": "2026-04-01"},
    "test": {"start_date": "2026-04-14", "end_date": "2026-04-14", "dates": ["2026-04-14"]}}}


class BookmakerComparisonTests(unittest.TestCase):
    def compare(self, quotes, predictions=None, fixtures=None, links=None,
                books=("pinnacle", "bet365", "unibet"), **options):
        return compare_hltv_bookmakers(fixtures or [fixture()], quotes, links or [link()],
                                      predictions or [prediction()], METADATA, books=books, **options)

    def test_reversed_provider_orientation_normalizes_prices_not_just_probability(self):
        report, rows = self.compare([quote()])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["hltv_match_id"], "100")
        self.assertEqual(row["target"], 1)
        self.assertAlmostEqual(row["p_model"], 0.7)
        self.assertAlmostEqual(row["p_bookmaker"], 0.6)
        self.assertEqual((row["price_team1"], row["price_team2"]), (2.0, 3.0))
        paired = report["per_book_paired_cohorts"]["pinnacle"]
        self.assertAlmostEqual(paired["model"]["log_loss"], -math.log(0.7))
        self.assertAlmostEqual(paired["bookmaker"]["log_loss"], -math.log(0.6))
        self.assertAlmostEqual(paired["bookmaker"]["brier"], 0.16)

    def test_common_all_book_cohort_does_not_mix_different_fixture_populations(self):
        quotes = [quote(book=b) for b in ("pinnacle", "bet365", "unibet")]
        quotes += [quote("id2", "pinnacle")]
        report, _ = self.compare(quotes, [prediction(), prediction("101")],
                                 [fixture(), fixture("id2")], [link(), link("id2", "101")])
        self.assertEqual(report["common_all_books"]["hltv_match_ids"], ["100"])
        self.assertEqual(report["common_all_books"]["matches"], 1)
        self.assertEqual(report["per_book_paired_cohorts"]["pinnacle"]["matches"], 2)
        self.assertEqual(report["per_book_paired_cohorts"]["bet365"]["matches"], 1)

    def test_source_clock_at_or_after_start_and_current_receipt_after_start_are_excluded(self):
        for change in ({"source_time_1": "2026-04-14T18:00:00Z"},
                       {"source_time_2": "2026-04-14T18:00:01Z"},
                       {"kind": "current", "observed_at": "2026-04-14T18:00:00Z"},
                       {"kind": "current"}):
            q = quote()
            q.update(change)
            with self.subTest(change=change):
                report, rows = self.compare([q])
                self.assertEqual(rows, [])
                self.assertEqual(report["common_all_books"]["matches"], 0)
                self.assertIsNone(report["per_book_paired_cohorts"]["pinnacle"]["model"]["log_loss"])

    def test_wrong_market_scope_and_invalid_decimal_prices_are_excluded(self):
        for change in ({"price1": 1.0}, {"price2": float("nan")},
                       {"details_json": json.dumps({"quote": {"market_id": "999"}})}):
            q = quote()
            q.update(change)
            with self.subTest(change=change):
                self.assertEqual(self.compare([q])[1], [])

    def test_conflicting_target_is_not_silently_overwritten(self):
        p = prediction()
        p["target"] = "0"
        with self.assertRaises(ValueError):
            self.compare([quote()], [p])

    def test_duplicate_prediction_and_duplicate_book_quote_are_not_double_counted(self):
        with self.assertRaises(ValueError):
            self.compare([quote()], [prediction(), prediction()])
        with self.assertRaises(ValueError):
            self.compare([quote(), quote()])

    def test_train_or_validation_predictions_cannot_enter_held_out_benchmark(self):
        p = prediction()
        p["source_date"] = p["date"] = "2026-03-01"
        with self.assertRaises(ValueError):
            self.compare([quote()], [p])

    def test_single_class_auc_and_empty_cohort_scores_are_null(self):
        report, _ = self.compare([quote()])
        self.assertIsNone(report["per_book_paired_cohorts"]["pinnacle"]["model"]["roc_auc"])
        self.assertIsNone(report["per_book_paired_cohorts"]["bet365"]["bookmaker"]["accuracy"])
        self.assertIsNone(report["common_all_books"]["model"]["ece"])

    def test_reversed_prediction_orientation_preserves_labels_and_offered_prices(self):
        p = prediction()
        p.update(team1="Beta", team2="Alpha", target="0",
                 prob_team1_win="0.3", prob_team2_win="0.7")
        _, rows = self.compare([quote()], [p])
        self.assertEqual(rows[0]["target"], 0)
        self.assertAlmostEqual(rows[0]["p_bookmaker"], 0.4)
        self.assertEqual((rows[0]["price_team1"], rows[0]["price_team2"]), (3.0, 2.0))

    def test_corrupt_names_probabilities_labels_and_native_ids_raise(self):
        for changes in ({"team1": "Unrelated"}, {"prob_team2_win": "0.4"},
                        {"prob_team1_win": "nan"}, {"target": "2"},
                        {"hltv_match_id": "0100"}):
            p = prediction()
            p.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.compare([quote()], [p])
        for changes in ({"participant1_is_hltv_team1": True},
                        {"hltv_score1": "1"}, {"hltv_match_url": "https://www.hltv.org/matches/101/x"}):
            mapped = link()
            mapped.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.compare([quote()], links=[mapped])

    def test_naive_source_clock_map_scope_and_nonzero_player_are_excluded(self):
        for changes in ({"source_time_1": "2026-04-14T17:00:00"},
                        {"scope": "map"}, {"playerId": 9}, {"isMap": True}):
            q = quote()
            if "source_time_1" in changes:
                q.update(changes)
            else:
                details = json.loads(q["details_json"])
                details["quote"].update(changes)
                q["details_json"] = json.dumps(details)
            with self.subTest(changes=changes):
                self.assertEqual(self.compare([q])[1], [])

    def test_nominal_source_day_proxy_must_precede_provider_start(self):
        f = fixture()
        f["startTime"] = "2026-04-13T23:30:00Z"
        q = quote()
        q["source_time_1"] = q["source_time_2"] = "2026-04-13T23:00:00Z"
        self.assertEqual(self.compare([q], fixtures=[f])[1], [])

    def test_unmatched_and_broader_pool_rows_are_explicit_not_model_forecasts(self):
        mappings = [link(), link("id2", "101"), {
            "fixtureId": "id3", "mapping_status": "unmatched", "mapping_reason": "no_candidate"}]
        report, rows = self.compare([quote(), quote("id2")], fixtures=[
            fixture(), fixture("id2"), fixture("id3")], links=mappings)
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["coverage"]["mapped_without_test_prediction_ids"], ["101"])
        self.assertEqual(report["coverage"]["mapping_status_counts"]["unmatched"], 1)
        self.assertIn("historical", rows[0]["quote_kind"])
        self.assertEqual(rows[0]["execution_eligibility"], "not_certified")

    def test_day_block_interval_is_deterministic_and_empty_interval_is_null(self):
        report, _ = self.compare([quote()])
        repeated, _ = self.compare([quote()])
        interval = report["per_book_paired_cohorts"]["pinnacle"]["paired_log_loss_difference"]
        self.assertEqual(interval, repeated["per_book_paired_cohorts"]["pinnacle"]["paired_log_loss_difference"])
        self.assertEqual(interval["source_day_count"], 1)
        self.assertAlmostEqual(interval["estimate"], math.log(0.6 / 0.7))
        self.assertEqual(interval["confidence_interval_95"], [interval["estimate"], interval["estimate"]])
        self.assertIsNone(report["per_book_paired_cohorts"]["bet365"]["paired_log_loss_difference"]["estimate"])

    def test_test_selected_model_metadata_is_rejected(self):
        metadata = {**METADATA, "selection": {"partition": "test", "fit_partition": "train"}}
        with self.assertRaises(ValueError):
            compare_hltv_bookmakers([fixture()], [quote()], [link()], [prediction()], metadata)

    def test_empty_inputs_do_not_become_zero_scores(self):
        report, rows = compare_hltv_bookmakers([], [], [], [], METADATA)
        self.assertEqual(rows, [])
        self.assertEqual(report["model_test_matches"], 0)
        self.assertIsNone(report["common_all_books"]["model"]["log_loss"])
        self.assertEqual(report["common_all_books"]["model"]["n_samples"], 0)

    def test_current_prestart_receipt_is_admitted_but_historical_late_receipt_is_not_execution_evidence(self):
        q = quote()
        q.update(kind="current", observed_at="2026-04-14T17:30:00Z")
        _, current_rows = self.compare([q])
        _, historical_rows = self.compare([quote()])
        self.assertEqual(len(current_rows), 1)
        self.assertEqual(len(historical_rows), 1)
        self.assertEqual(historical_rows[0]["observed_at"], "2026-10-01T00:00:00Z")
        self.assertEqual(historical_rows[0]["source_time_1"], "2026-04-14T17:00:00Z")
        self.assertEqual(historical_rows[0]["execution_eligibility"], "not_certified")

    def test_two_source_day_blocks_preserve_within_day_cluster_and_match_weighting(self):
        fixtures = [fixture(), fixture("id2"), fixture("id3")]
        mappings = [link(), link("id2", "101"), link("id3", "102")]
        predictions = [prediction(), prediction("101"), prediction("102")]
        quotes = [quote(), quote("id2"), quote("id3")]
        predictions[1].update(prob_team1_win="0.9", prob_team2_win="0.1")
        predictions[2].update(source_date="2026-04-15", date="2026-04-15",
                              prob_team1_win="0.4", prob_team2_win="0.6")
        fixtures[2].update(startTime="2026-04-15T18:00:00Z", trueStartTime=None)
        mappings[2]["hltv_calendar_day"] = "2026-04-15"
        quotes[2].update(source_time_1="2026-04-15T17:00:00Z",
                         source_time_2="2026-04-15T17:00:00Z")
        metadata = {**METADATA, "splits": {**METADATA["splits"], "test": {
            "start_date": "2026-04-14", "end_date": "2026-04-15",
            "dates": ["2026-04-14", "2026-04-15"]}}}
        report, _ = compare_hltv_bookmakers(fixtures, quotes, mappings, predictions, metadata)
        interval = report["per_book_paired_cohorts"]["pinnacle"]["paired_log_loss_difference"]
        first_day_mean = (math.log(0.6 / 0.7) + math.log(0.6 / 0.9)) / 2
        second_day_mean = math.log(0.6 / 0.4)
        self.assertEqual(interval["source_day_count"], 2)
        self.assertAlmostEqual(interval["estimate"], (2 * first_day_mean + second_day_mean) / 3)
        self.assertAlmostEqual(interval["confidence_interval_95"][0], first_day_mean)
        self.assertAlmostEqual(interval["confidence_interval_95"][1], second_day_mean)
        self.assertTrue(interval["interval_crosses_zero"])
        self.assertFalse(interval["statistical_significance_claim"])

    def test_invalid_label_status_is_excluded_instead_of_inventing_a_series_result(self):
        mapped = link()
        mapped.update(label_status="invalid_series_result", outcome_1_won="",
                      hltv_winner_1="", hltv_format="bo1")
        report, rows = self.compare([quote()], links=[mapped])
        self.assertEqual(rows, [])
        self.assertEqual(report["exclusions"]["quote_reason_counts"]["no_valid_explicit_series_label"], 1)

    def test_power_devig_uses_prices_alone_and_complements_reversed_orientation(self):
        q = quote()
        # In provider orientation q1=.6, q2=.8, so the unique exponent is 2.
        q.update(price1=1 / 0.6, price2=1 / 0.8)
        _, rows = self.compare([q], devig_method="power")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["p_bookmaker"], 0.64)
        self.assertAlmostEqual(rows[0]["devig_power_exponent"], 2.0)
        self.assertEqual(rows[0]["devig_method"], "power")
        self.assertEqual((rows[0]["price_team1"], rows[0]["price_team2"]),
                         (1 / 0.8, 1 / 0.6))
        reverse = prediction()
        reverse.update(team1="Beta", team2="Alpha", target="0",
                       prob_team1_win="0.3", prob_team2_win="0.7")
        _, reversed_rows = self.compare([q], predictions=[reverse], devig_method="power")
        self.assertAlmostEqual(reversed_rows[0]["p_bookmaker"], 1 - rows[0]["p_bookmaker"])
        self.assertAlmostEqual(reversed_rows[0]["devig_power_exponent"], 2.0)
        opposite_result = link()
        opposite_result.update(hltv_score1="1", hltv_score2="2", hltv_winner_1=0, outcome_1_won=1)
        opposite_prediction = prediction()
        opposite_prediction.update(target="0", prob_team1_win="0.2", prob_team2_win="0.8")
        _, opposite_rows = self.compare([q], predictions=[opposite_prediction],
                                        links=[opposite_result], devig_method="power")
        self.assertAlmostEqual(opposite_rows[0]["p_bookmaker"], 0.64)
        self.assertAlmostEqual(opposite_rows[0]["devig_power_exponent"], 2.0)
        _, proportional_rows = self.compare([q], devig_method="proportional")
        self.assertAlmostEqual(proportional_rows[0]["p_bookmaker"], 0.8 / 1.4)
        self.assertEqual(proportional_rows[0]["devig_power_exponent"], "")

    def test_power_devig_is_finite_for_fair_near_fair_and_underround_prices(self):
        for inverse_one, inverse_two in ((0.25, 0.75), (0.50000001, 0.5), (1 / 3, 0.5)):
            q = quote()
            q.update(price1=1 / inverse_one, price2=1 / inverse_two)
            with self.subTest(inverse_prices=(inverse_one, inverse_two)):
                _, rows = self.compare([q], devig_method="power")
                self.assertEqual(len(rows), 1)
                exponent = rows[0]["devig_power_exponent"]
                probability = rows[0]["p_bookmaker"]
                self.assertTrue(math.isfinite(exponent))
                self.assertGreater(exponent, 0)
                self.assertTrue(math.isfinite(probability))
                self.assertGreater(probability, 0)
                self.assertLess(probability, 1)
                self.assertAlmostEqual(inverse_one ** exponent + inverse_two ** exponent, 1.0)
                self.assertAlmostEqual(probability, inverse_two ** exponent)
                if inverse_one + inverse_two < 1:
                    self.assertLess(exponent, 1)
                elif inverse_one + inverse_two == 1:
                    self.assertAlmostEqual(exponent, 1)
                    self.assertAlmostEqual(probability, 0.75)

    def test_unsupported_devig_methods_and_cutoff_policies_raise(self):
        for options in ({"devig_method": "outcome_fitted"}, {"devig_method": ""},
                        {"quote_cutoff": "closing"}, {"quote_cutoff": ""}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.compare([quote()], **options)

    def test_source_day_requires_each_side_and_current_receipt_strictly_before_midnight(self):
        early = "2026-04-13T23:59:59Z"
        midnight = "2026-04-14T00:00:00Z"
        q = quote()
        q.update(kind="current", source_time_1=early, source_time_2=early, observed_at=early)
        _, rows = self.compare([q], quote_cutoff="source_day")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["quote_cutoff_policy"], "source_day")
        for clock in ("source_time_1", "source_time_2", "observed_at"):
            for value in (midnight, "2026-04-14T00:00:01Z", "2026-04-13T23:59:59-01:00"):
                invalid = {**q, clock: value}
                with self.subTest(clock=clock, value=value):
                    self.assertEqual(len(self.compare([invalid], quote_cutoff="pre_start")[1]), 1)
                    self.assertEqual(self.compare([invalid], quote_cutoff="source_day")[1], [])
        # A historical receipt records retrieval, not when either offered price existed.
        historical = {**q, "kind": "historical", "observed_at": "2026-10-01T00:00:00Z"}
        _, historical_rows = self.compare([historical], quote_cutoff="source_day")
        self.assertEqual(len(historical_rows), 1)
        self.assertEqual(historical_rows[0]["observed_at"], historical["observed_at"])
        self.assertEqual(historical_rows[0]["execution_eligibility"], "not_certified")
        for clock in ("source_time_1", "source_time_2"):
            with self.subTest(historical_clock=clock):
                self.assertEqual(self.compare([{**historical, clock: midnight}],
                                              quote_cutoff="source_day")[1], [])

    def test_invalid_latest_observation_is_excluded_not_reconstructed_from_earlier_prices(self):
        earlier = quote()
        earlier.update(id="earlier", source_time_1="2026-04-13T23:00:00Z",
                       source_time_2="2026-04-13T23:00:00Z")
        self.assertEqual(len(self.compare([earlier], quote_cutoff="source_day")[1]), 1)
        # The API consumes the caller's latest observation, not a quote-history archive.
        for changes in ({"source_time_2": "2026-04-14T00:00:00Z"}, {"price1": 1.0},
                        {"status": "rejected"}):
            latest = {**earlier, "id": "latest", **changes}
            with self.subTest(changes=changes):
                report, rows = self.compare([latest], quote_cutoff="source_day")
                self.assertEqual(rows, [])
                excluded = report["exclusions"]["quote_rows"]
                self.assertEqual(len(excluded), 1)
                self.assertEqual(excluded[0]["quote_id"], "latest")
                self.assertEqual(report["per_book_paired_cohorts"]["pinnacle"]["matches"], 0)

    def test_paired_brier_single_day_and_empty_intervals_use_the_paired_cohort(self):
        report, _ = self.compare([quote()])
        interval = report["per_book_paired_cohorts"]["pinnacle"]["paired_brier_difference"]
        expected = (0.7 - 1) ** 2 - (0.6 - 1) ** 2
        self.assertAlmostEqual(interval["estimate"], expected)
        self.assertEqual(interval["source_day_count"], 1)
        self.assertEqual(interval["confidence_interval_95"], [interval["estimate"], interval["estimate"]])
        self.assertFalse(interval["statistical_significance_claim"])
        repeated, _ = self.compare([quote()])
        self.assertEqual(interval, repeated["per_book_paired_cohorts"]["pinnacle"]["paired_brier_difference"])
        empty = report["per_book_paired_cohorts"]["bet365"]["paired_brier_difference"]
        self.assertIsNone(empty["estimate"])
        self.assertIsNone(empty["confidence_interval_95"])
        self.assertEqual(empty["source_day_count"], 0)
        for book in ("pinnacle", "bet365", "unibet"):
            differences = report["common_all_books"]["paired_differences"][book]
            for metric in ("paired_log_loss_difference", "paired_brier_difference"):
                self.assertIsNone(differences[metric]["estimate"])
                self.assertIsNone(differences[metric]["confidence_interval_95"])
                self.assertEqual(differences[metric]["source_day_count"], 0)
        common_report, _ = self.compare([quote(book=book) for book in ("pinnacle", "bet365", "unibet")])
        for book in ("pinnacle", "bet365", "unibet"):
            differences = common_report["common_all_books"]["paired_differences"][book]
            for metric, expected_loss in (("paired_log_loss_difference", math.log(0.6 / 0.7)),
                                          ("paired_brier_difference", expected)):
                interval = differences[metric]
                self.assertAlmostEqual(interval["estimate"], expected_loss)
                self.assertEqual(interval["source_day_count"], 1)
                self.assertEqual(interval["confidence_interval_95"],
                                 [interval["estimate"], interval["estimate"]])
                self.assertFalse(interval["statistical_significance_claim"])

    def test_common_and_per_book_paired_losses_use_identical_ids_and_day_weighted_intervals(self):
        books = ("pinnacle", "bet365", "unibet")
        fixtures = [fixture(f"id{i}") for i in range(1, 5)]
        mappings = [link(f"id{i}", str(99 + i)) for i in range(1, 5)]
        predictions = [prediction(str(99 + i)) for i in range(1, 5)]
        # Row two is reversed, hence its target is zero and probabilities complement.
        predictions[1].update(team1="Beta", team2="Alpha", target="0",
                              prob_team1_win="0.1", prob_team2_win="0.9")
        predictions[2].update(prob_team1_win="0.4", prob_team2_win="0.6")
        predictions[3].update(prob_team1_win="0.99", prob_team2_win="0.01")
        for index in (2, 3):
            fixtures[index].update(startTime="2026-04-15T18:00:00Z", trueStartTime=None)
            mappings[index]["hltv_calendar_day"] = "2026-04-15"
            predictions[index].update(date="2026-04-15", source_date="2026-04-15")
        metadata = {**METADATA, "splits": {**METADATA["splits"], "test": {
            "start_date": "2026-04-14", "end_date": "2026-04-15",
            "dates": ["2026-04-14", "2026-04-15"]}}}
        book_probabilities = {"pinnacle": 0.6, "bet365": 0.55, "unibet": 0.65}
        quotes = []
        for index in range(4):
            for book in (books if index < 3 else ("pinnacle",)):
                probability = book_probabilities[book]
                q = quote(f"id{index + 1}", book)
                q.update(price1=1 / (1 - probability), price2=1 / probability)
                if index >= 2:
                    q.update(source_time_1="2026-04-15T17:00:00Z",
                             source_time_2="2026-04-15T17:00:00Z")
                quotes.append(q)
        report, rows = compare_hltv_bookmakers(fixtures, quotes, mappings, predictions, metadata)
        common = report["common_all_books"]
        self.assertEqual(common["hltv_match_ids"], ["100", "101", "102"])
        self.assertEqual(common["model"]["n_samples"], 3)
        for book in books:
            probability = book_probabilities[book]
            expected_ids = ["100", "101", "102", "103"] if book == "pinnacle" else ["100", "101", "102"]
            paired = report["per_book_paired_cohorts"][book]
            self.assertEqual(paired["hltv_match_ids"], expected_ids)
            self.assertEqual([row["hltv_match_id"] for row in rows if row["bookmaker"] == book], expected_ids)
            self.assertEqual(paired["model"]["n_samples"], len(expected_ids))
            self.assertEqual(paired["bookmaker"]["n_samples"], len(expected_ids))
            self.assertEqual(common["bookmakers"][book]["n_samples"], 3)
            # These losses are computed from fixture inputs, never implementation scores.
            inputs = [(1, 0.7, probability), (0, 0.1, 1 - probability),
                      (1, 0.4, probability), (1, 0.99, probability)]
            for metric in ("paired_log_loss_difference", "paired_brier_difference"):
                if metric == "paired_brier_difference":
                    losses = [(p - y) ** 2 - (b - y) ** 2 for y, p, b in inputs]
                else:
                    losses = [-math.log(p if y else 1 - p) + math.log(b if y else 1 - b)
                              for y, p, b in inputs]
                for size, interval in (
                        (len(expected_ids), paired[metric]),
                        (3, common["paired_differences"][book][metric])):
                    with self.subTest(book=book, metric=metric, cohort_size=size):
                        day_means = [sum(losses[:2]) / 2, sum(losses[2:size]) / (size - 2)]
                        self.assertAlmostEqual(interval["estimate"], sum(losses[:size]) / size)
                        self.assertEqual(interval["source_day_count"], 2)
                        self.assertAlmostEqual(interval["confidence_interval_95"][0], min(day_means))
                        self.assertAlmostEqual(interval["confidence_interval_95"][1], max(day_means))
                        self.assertEqual(interval["interval_crosses_zero"],
                                         min(day_means) <= 0 <= max(day_means))
                        self.assertFalse(interval["statistical_significance_claim"])

    def test_forecast_origin_is_explicit_aware_bounded_and_strict_for_both_sides(self):
        origin = "2026-04-14T17:00:00Z"
        p = {**prediction(), "forecast_origin_utc": origin}
        early = {**quote(), "source_time_1": "2026-04-14T16:59:59Z",
                 "source_time_2": "2026-04-14T16:59:59Z"}
        _, rows = self.compare([early], [p], quote_cutoff="forecast_origin")
        self.assertEqual(rows[0]["forecast_origin_utc"], origin)
        for key in ("source_time_1", "source_time_2"):
            self.assertEqual(self.compare([{**early, key: origin}], [p],
                                         quote_cutoff="forecast_origin")[1], [])
        for value in (None, "2026-04-14T17:00:00", "2026-04-13T23:59:59Z",
                      "2026-04-14T18:00:00Z"):
            invalid = {**p, "forecast_origin_utc": value}
            with self.subTest(origin=value), self.assertRaises(ValueError):
                self.compare([early], [invalid], quote_cutoff="forecast_origin")
        current = {**early, "kind": "current", "observed_at": origin}
        self.assertEqual(self.compare([current], [p], quote_cutoff="forecast_origin")[1], [])
        wrong_origin = {**early, "forecast_origin_utc": "2026-04-14T16:00:00Z"}
        self.assertEqual(self.compare([wrong_origin], [p], quote_cutoff="forecast_origin")[1], [])

    def test_available_consensus_counts_each_native_match_once_with_equal_book_weight(self):
        qs = [quote(book=book) for book in ("pinnacle", "bet365", "unibet")]
        qs[1].update(price1=4.0, price2=1.0 / 0.75)
        qs[2].update(status="excluded")
        qs.append(quote("id2"))
        report, _ = self.compare(qs, [prediction(), prediction("101")],
                                 [fixture(), fixture("id2")], [link(), link("id2", "101")])
        consensus = report["available_book_consensus"]
        self.assertEqual(consensus["matches"], 2)
        self.assertEqual(consensus["model"]["n_samples"], 2)
        self.assertEqual(consensus["bookmaker"]["n_samples"], 2)
        self.assertEqual(consensus["coverage_pattern_counts"],
                         {"pinnacle": 1, "pinnacle+bet365": 1})
        self.assertAlmostEqual(consensus["rows"][0]["p_bookmaker"], (0.6 + 0.75) / 2)
        self.assertEqual(consensus["rows"][0]["bookmakers"], ["pinnacle", "bet365"])
        self.assertEqual(consensus["rows"][1]["hltv_match_id"], "101")

    def test_derived_selection_and_hash_provenance_reaches_pair_and_report(self):
        origin = "2026-04-14T17:30:00Z"
        p = {**prediction(), "forecast_origin_utc": origin}
        q = {**quote(), "forecast_origin_utc": origin, "raw_path": "raw/evidence.gz",
             "file_sha256": "1" * 64, "decoded_sha256": "2" * 64}
        details = json.loads(q["details_json"])
        details["derived_selection"] = {"method": "latest_requested_receipt_then_asof",
                                         "horizon": "one_hour", "raw_id": 1}
        q["details_json"] = json.dumps(details)
        report, rows = self.compare([q], [p], quote_cutoff="forecast_origin")
        self.assertEqual(rows[0]["derived_selection_method"], details["derived_selection"]["method"])
        self.assertEqual(rows[0]["derived_horizon"], "one_hour")
        self.assertEqual(rows[0]["raw_path"], q["raw_path"])
        self.assertEqual(rows[0]["decoded_sha256"], q["decoded_sha256"])
        self.assertEqual(report["protocol"]["forecast_origins_utc"], {"100": origin})
        self.assertEqual(report["protocol"]["derived_selection_method_counts"],
                         {"latest_requested_receipt_then_asof": 1})

    def test_declared_numerical_ties_do_not_allow_material_selection_gaps(self):
        selection = {"partition": "validation", "fit_partition": "train", "metric": "log_loss",
                     "tie_tolerance": 1e-12,
                     "candidate_log_loss": {"pure_elo": 0.6 + 5e-13, "alternative": 0.6}}
        metadata = {**METADATA, "selection": selection}
        report, _ = compare_hltv_bookmakers([fixture()], [quote()], [link()], [prediction()], metadata)
        self.assertEqual(report["matched_test_matches_any_book"], 1)
        selection["candidate_log_loss"]["pure_elo"] = 0.601
        with self.assertRaises(ValueError):
            compare_hltv_bookmakers([fixture()], [quote()], [link()], [prediction()], metadata)
        selection["tie_tolerance"] = 0.1
        with self.assertRaises(ValueError):
            compare_hltv_bookmakers([fixture()], [quote()], [link()], [prediction()], metadata)



if __name__ == "__main__":
    unittest.main()
