"""Behavioral regressions for the isolated provider-native acquisition."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

from scripts import collect_cs_oddspapi as collector


class BoundaryTests(unittest.TestCase):
    def arguments(self, directory, *extra):
        return ['--from', '2026-04-14T00:00:00Z', '--to', '2026-04-20T23:59:59Z',
                '--output-dir', directory, *extra]

    def test_missing_key_precedes_output_creation_and_http(self):
        with tempfile.TemporaryDirectory() as root:
            directory = str(Path(root) / 'absent')
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(collector, 'urlopen') as http:
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(collector.main(self.arguments(directory)), 2)
                self.assertIn('ODDSPAPI_API_KEY', stderr.getvalue())
                http.assert_not_called()
                self.assertFalse(Path(directory).exists())

    def test_interval_and_book_validation(self):
        for extra in (['--from', '2026-04-14'], ['--to', '2026-04-24T00:00:00Z'],
                      ['--bookmakers', 'pinnacle,pinnacle'], ['--bookmakers', '../unsafe']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    collector.parse_args(self.arguments('unused', *extra))

    def test_http_errors_are_charged_and_secret_echo_is_not_retained(self):
        secret = 'private/key+value'
        config = collector.parse_args(self.arguments('unused', '--max-requests', '1'))
        error = HTTPError('https://example/?apiKey=' + secret, 429, secret, {},
                          io.BytesIO(json.dumps({'error': {'code': 'REQUEST_LIMIT_EXCEEDED',
                                                         'message': secret}}).encode()))
        client = collector.Client(config, secret)
        with mock.patch.object(collector, 'urlopen', side_effect=error):
            record, body, payload = client.request('/historical-odds', {'fixtureId': 'id1'})
            self.assertEqual(record['error_code'], 'SECRET_ECHO')
            self.assertTrue(record['stop'])
            self.assertIsNone(body)
            self.assertIsNone(payload)
            with self.assertRaises(collector.Stop):
                client.request('/account', {})
        self.assertEqual(client.attempts['/historical-odds'], 1)
        self.assertNotIn(secret, json.dumps(record))

    def test_caps_charge_failed_billable_attempts(self):
        config = collector.parse_args(self.arguments('unused', '--max-billable-requests', '1'))
        client = collector.Client(config, 'secret-test')
        with mock.patch.object(collector, 'urlopen', side_effect=OSError('secret-test')):
            record, _, _ = client.request('/markets', {})
            self.assertEqual(record['error_code'], 'NETWORK_ERROR')
            with self.assertRaises(collector.Stop):
                client.request('/fixtures', {})
        self.assertEqual(client.billable, 1)
        self.assertEqual(sum(client.attempts.values()), 1)

    def test_application_endpoint_policy_charges_failed_odds_before_cap(self):
        class ApplicationClient(collector.Client):
            cooldowns = {'/odds': 0}
            billable_endpoints = {'/odds'}
        config = collector.parse_args(self.arguments('unused', '--max-billable-requests', '1'))
        client = ApplicationClient(config, 'secret-test')
        with mock.patch.object(collector, 'urlopen', side_effect=OSError('private URL')):
            record, _, _ = client.request('/odds', {'fixtureId': 'id1'})
            self.assertEqual(record['error_code'], 'NETWORK_ERROR')
            with self.assertRaises(collector.Stop):
                client.request('/odds', {'fixtureId': 'id2'})
        self.assertEqual(client.billable, 1)


class ContractTests(unittest.TestCase):
    def account(self, count=7):
        return {'api_key': 'secret', 'current_subscription_id': 's',
                'subscriptions': [{'subscription_id': 's', 'is_active': True,
                                   'sport_ids': [17], 'bookmakers': {'pinnacle': {}},
                                   'request_limit': 10, 'request_count': count}]}

    def market(self):
        return {'sportId': 17, 'marketId': 171, 'marketName': 'Winner',
                'marketType': 'moneyline', 'period': 'result', 'playerProp': False,
                'marketLength': 2,
                'outcomes': [{'outcomeId': 999, 'outcomeName': '2'},
                             {'outcomeId': 888, 'outcomeName': '1'}]}

    def fixture(self, fixture_id='id1'):
        return {'fixtureId': fixture_id, 'sportId': 17, 'statusId': 2,
                'startTime': '2026-04-15T10:00:00Z', 'trueStartTime': None,
                'participant1Id': 1, 'participant2Id': 2,
                'participant1Name': 'A', 'participant2Name': 'B'}

    def test_quota_requires_strict_headroom_and_active_unique_subscription(self):
        snapshot, books = collector.account_gate(self.account(), ['pinnacle', 'unibet'], 2)
        self.assertEqual(books, ['pinnacle'])
        self.assertNotIn('api_key', snapshot)
        with self.assertRaises(collector.Stop):
            collector.account_gate(self.account(8), ['pinnacle'], 2)
        for field, value in [('request_limit', None), ('request_count', True), ('is_active', False),
                             ('sport_ids', [18]), ('bookmakers', {'unibet': {}})]:
            payload = self.account()
            payload['subscriptions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(collector.Stop):
                collector.account_gate(payload, ['pinnacle'], 2)
        payload = self.account()
        payload['subscriptions'] *= 2
        with self.assertRaises(collector.Stop):
            collector.account_gate(payload, ['pinnacle'], 2)

    def test_domain_book_slugs_do_not_block_requested_entitled_books(self):
        payload = self.account()
        payload['subscriptions'][0]['bookmakers'].update({'bookmaker.eu': {}, 'betpawa.cm': {}})
        snapshot, books = collector.account_gate(payload, ['pinnacle'], 2)
        self.assertEqual(books, ['pinnacle'])
        self.assertIn('bookmaker.eu', snapshot['bookmaker_slugs'])
        args = collector.parse_args(BoundaryTests().arguments('unused', '--bookmakers', 'bookmaker.eu'))
        self.assertEqual(args.bookmakers, ['bookmaker.eu'])
        for unsafe in ('../unsafe', 'book..eu', 'book/eu', '.book', 'book.'):
            with self.subTest(unsafe=unsafe), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    collector.parse_args(BoundaryTests().arguments('unused', '--bookmakers', unsafe))

    def test_catalog_labels_not_array_order_and_map_player_rejection(self):
        mapping = collector.resolve_market([self.market()])
        self.assertEqual(mapping, {'market_id': '171', 'outcome_1': '888', 'outcome_2': '999'})
        for field, value in [('period', 'map1'), ('playerProp', True), ('sportId', 18),
                             ('marketName', 'Map Winner')]:
            item = self.market()
            item[field] = value
            with self.subTest(field=field), self.assertRaises(collector.Stop):
                collector.resolve_market([item])
        with self.assertRaises(collector.Stop):
            collector.resolve_market([self.market(), self.market()])
        item = self.market()
        item['outcomes'][0]['outcomeName'] = '1'
        with self.assertRaises(collector.Stop):
            collector.resolve_market([item])

    def test_fixture_freeze_deduplicates_before_bounded_ordered_selection(self):
        args = collector.parse_args(BoundaryTests().arguments('unused', '--max-history', '1'))
        ordered = collector.discover_fixtures([self.fixture('id2'), self.fixture('id1'), self.fixture('id1')], args)
        self.assertEqual([f['fixtureId'] for f in ordered], ['id1', 'id2'])
        for field, value in [('startTime', '2026-04-15'), ('startTime', '2026-04-21T00:00:00Z'),
                             ('sportId', 18), ('fixtureId', '../unsafe'), ('statusId', 1)]:
            item = self.fixture()
            item[field] = value
            with self.subTest(field=field), self.assertRaises(collector.Stop):
                collector.discover_fixtures([item], args)
        conflict = self.fixture()
        conflict['participant1Name'] = 'Different'
        with self.assertRaises(collector.Stop):
            collector.discover_fixtures([self.fixture(), conflict], args)

    def test_numeric_quota_and_malformed_catalog_are_handled_without_assumptions(self):
        payload = self.account()
        payload['subscriptions'][0]['request_limit'] = 10.0
        payload['subscriptions'][0]['request_count'] = 7.0
        snapshot, _ = collector.account_gate(payload, ['pinnacle'], 2)
        self.assertEqual(snapshot['request_count'], 7.0)
        market = self.market()
        market['outcomes'][0]['outcomeName'] = ['2']
        with self.assertRaises(collector.Stop):
            collector.resolve_market([market])


class SelectionTests(unittest.TestCase):
    cutoff = collector.utc('2026-04-15T10:00:00Z')

    def state(self, when='2026-04-15T09:00:00Z', price=2.0, active=True, **extra):
        return {'createdAt': when, 'price': price, 'active': active, **extra}

    def test_unordered_and_identical_ties_choose_latest_but_conflicts_block(self):
        old = self.state('2026-04-15T08:00:00Z', 3.0)
        latest = self.state(price=2.5)
        result = collector.select_state([latest, old, latest.copy()], self.cutoff)
        self.assertEqual(result['price'], 2.5)
        conflicting = self.state(price=2.6)
        self.assertEqual(collector.select_state([old, latest, conflicting], self.cutoff)['status'],
                         'conflicting_latest_states')

    def test_latest_invalid_state_never_revives_older_active(self):
        old = self.state('2026-04-15T08:00:00Z', 3)
        for price, active, status in [(2, False, 'latest_inactive'), (2, None, 'unknown_active'),
                                      (True, True, 'invalid_price'), (1, True, 'invalid_price'),
                                      ('NaN', True, 'invalid_price'), ('Infinity', True, 'invalid_price')]:
            with self.subTest(price=price, active=active):
                self.assertEqual(collector.select_state([self.state(price=price, active=active), old],
                                                       self.cutoff)['status'], status)

    def test_any_unorderable_clock_blocks_even_with_usable_prices(self):
        for when in [None, '2026-04-15T09:01:00', 'bad']:
            with self.subTest(when=when):
                self.assertEqual(collector.select_state([self.state(), self.state(when)], self.cutoff)['status'],
                                 'unorderable_clock')

    def test_at_and_post_start_states_do_not_supply_prices(self):
        result = collector.select_state([self.state('2026-04-15T10:00:00Z', 8),
                                         self.state('2026-04-15T11:00:00Z', 9),
                                         self.state(price=2)], self.cutoff)
        self.assertEqual(result['price'], 2)
        self.assertEqual(collector.select_state([self.state('2026-04-15T10:00:00Z')], self.cutoff)['status'],
                         'no_pre_start_state')
        fixture = ContractTests().fixture()
        fixture['trueStartTime'] = '2026-04-15T08:30:00Z'
        self.assertEqual(collector.fixture_cutoff(fixture), collector.utc(fixture['trueStartTime']))
        fixture['trueStartTime'] = 'invalid'
        with self.assertRaises(collector.Stop):
            collector.fixture_cutoff(fixture)

    def test_explicit_live_or_map_scope_is_rejected(self):
        for scope in [{'live': True}, {'period': 'map1'}, {'playerProp': True}, {'playerId': 42}]:
            with self.subTest(scope=scope):
                self.assertEqual(collector.select_state([self.state(**scope)], self.cutoff)['status'],
                                 'explicit_live_or_map_scope')

    def test_two_sides_compute_exact_source_probability_without_binding_claim(self):
        fixture = ContractTests().fixture()
        mapping = collector.resolve_market([ContractTests().market()])
        history = {'bookmakers': {'pinnacle': {'markets': {'171': {'outcomes': {
            '888': {'players': {'0': [self.state(price=2.0)]}},
            '999': {'players': {'0': [self.state(price=4.0)]}}}}}}}}
        coverage, row = collector.select_pair(fixture, 'pinnacle', history, mapping)
        self.assertEqual(coverage['status'], 'admitted')
        self.assertEqual((row['price_1'], row['price_2']), (2.0, 4.0))
        self.assertAlmostEqual(row['no_vig_probability_1'], 2 / 3)
        self.assertAlmostEqual(row['margin'], -0.25)
        self.assertEqual(row['binding_status'], 'provider_catalog_convention_only')
        self.assertEqual(row['execution_eligibility'], 'not_certified')
        history['bookmakers']['pinnacle']['markets']['171']['outcomes']['999']['players']['0'].append(
            self.state('2026-04-15T09:30:00Z', 4.0, False))
        coverage, row = collector.select_pair(fixture, 'pinnacle', history, mapping)
        self.assertEqual(coverage['side_2_status'], 'latest_inactive')
        self.assertIsNone(row)

    def test_only_documented_404_is_terminal_absence(self):
        absent = {'error': {'code': 'NOT_FOUND', 'message': 'No historical odds found.'}}
        self.assertTrue(collector.no_history(404, absent))
        for status, payload in [(403, absent), (500, absent), (404, {}),
                                (404, {'error': {'code': 'FIXTURE_NOT_FOUND'}})]:
            self.assertFalse(collector.no_history(status, payload))


class AcquisitionTests(unittest.TestCase):
    def simulate(self, run, replies):
        def response(request, timeout):
            status, payload = replies.pop(0)
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            if status != 200:
                raise HTTPError(request.full_url, status, 'provider failure', {}, io.BytesIO(body))
            item = mock.MagicMock()
            item.__enter__.return_value = item
            item.status = status
            item.read.return_value = body
            return item
        return mock.patch.object(collector, 'urlopen', side_effect=response)

    def history(self, fixture_id='id1'):
        state = SelectionTests().state
        return {'fixtureId': fixture_id, 'bookmakers': {'pinnacle': {'markets': {'171': {'outcomes': {
            '888': {'players': {'0': [state(price=2)]}},
            '999': {'players': {'0': [state(price=4)]}}}}}}}}

    def test_explicit_absence_and_failures_are_distinct_and_only_terminal_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            args = collector.parse_args(BoundaryTests().arguments(directory))
            run = collector.Run(args, 'private-test')
            run.protocol['effective_books'] = ['pinnacle']
            absent = {'error': {'code': 'NOT_FOUND', 'message': 'No historical odds found.'}}
            replies = [(404, absent), (404, {'error': {'code': 'NOT_FOUND'}}), (500, {}), (403, {})]
            with self.simulate(run, replies), mock.patch.object(collector.time, 'sleep'):
                for fixture_id in ['id1', 'id2', 'id3']:
                    record, _ = run.fetch('/historical-odds', run.parameters('/historical-odds', fixture_id),
                                          lambda p: collector.validate_history(p, fixture_id))
                    self.assertEqual(record['terminal'], fixture_id == 'id1')
                with self.assertRaises(collector.Stop):
                    run.fetch('/historical-odds', run.parameters('/historical-odds', 'id4'))
            self.assertEqual(len(run.cache), 1)
            self.assertEqual(run.client.attempts['/historical-odds'], 4)
            self.assertEqual(run.client.billable, 0)

    def test_acquisition_saves_exact_raw_and_reports_each_original_book(self):
        contract = ContractTests()
        raw = json.dumps(self.history(), indent=3).encode()
        with tempfile.TemporaryDirectory() as directory:
            args = collector.parse_args(BoundaryTests().arguments(directory))
            run = collector.Run(args, 'private-test')
            replies = [(200, contract.account()), (200, [contract.market()]),
                       (200, [contract.fixture()]), (200, raw), (200, contract.account())]
            with self.simulate(run, replies), mock.patch.object(collector.time, 'sleep'):
                summary = run.execute()
            self.assertTrue(summary['acquisition_complete'])
            self.assertEqual(summary['selected_pairs'], 1)
            self.assertEqual(summary['coverage_by_book']['unibet'], {'not_entitled': 1})
            self.assertEqual(summary['actual_attempts_this_invocation'],
                             {'/account': 2, '/markets': 1, '/fixtures': 1, '/historical-odds': 1})
            record = next(r for r in run.receipts if r['endpoint'] == '/historical-odds')
            import gzip
            saved = (Path(directory) / record['raw_path']).read_bytes()
            self.assertEqual(gzip.decompress(saved), raw)
            self.assertEqual(collector.sha(raw), record['decoded_sha256'])
            self.assertEqual(collector.sha(saved), record['file_sha256'])
            self.assertEqual(len(run.protocol['accounts']), 2)


class ResumeTests(unittest.TestCase):
    def acquire(self, directory, history_status=200, history=None):
        args = collector.parse_args(BoundaryTests().arguments(directory))
        run = collector.Run(args, 'private-test')
        contract, acquisition = ContractTests(), AcquisitionTests()
        payload = history if history is not None else acquisition.history()
        replies = [(200, contract.account()), (200, [contract.market()]),
                   (200, [contract.fixture()]), (history_status, payload)]
        if history_status == 200 or collector.no_history(history_status, payload):
            replies.append((200, contract.account()))
        with acquisition.simulate(run, replies), mock.patch.object(collector.time, 'sleep'):
            run.execute()
        return args, run

    def test_terminal_resume_without_key_reuses_receipts_and_identical_rows(self):
        for status, payload in [(200, None), (404, {'error': {'code': 'NOT_FOUND',
                                                           'message': 'No historical odds found.'}})]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                args, original = self.acquire(directory, status, payload)
                stable = {name: (Path(directory) / name).read_bytes() for name in
                          ['coverage.csv', 'selected_pre_match_quotes.csv', 'receipts.jsonl', 'protocol.json']}
                with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(collector, 'urlopen') as http:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = collector.main(BoundaryTests().arguments(directory, '--resume'))
                self.assertEqual(result, 0)
                http.assert_not_called()
                for name, body in stable.items():
                    self.assertEqual((Path(directory) / name).read_bytes(), body, name)
                summary = json.loads((Path(directory) / 'summary.json').read_text())
                self.assertEqual(summary['actual_attempts_this_invocation'], {})
                self.assertEqual(summary['cached_responses_this_invocation']['/historical-odds'], 1)

    def test_failed_cache_requires_key_then_current_gate_without_rebilling_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            args, original = self.acquire(directory, 500, {})
            args.resume = True
            resume = collector.Run(args, 'private-test')
            resume.load_resume()
            account = ContractTests().account(count=9)  # One remaining credit, no new billable calls.
            with AcquisitionTests().simulate(resume, [(200, account), (200, AcquisitionTests().history()),
                                                      (200, account)]), mock.patch.object(collector.time, 'sleep'):
                summary = resume.execute()
            self.assertTrue(summary['acquisition_complete'])
            self.assertEqual(summary['billable_attempts_this_invocation'], 0)
            self.assertEqual(summary['actual_attempts_this_invocation'], {'/account': 2, '/historical-odds': 1})

    def test_corruption_and_query_mismatch_fail_before_http_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args, original = self.acquire(directory)
            args.resume = True
            raw = Path(directory) / next(r['raw_path'] for r in original.receipts if r['terminal'])
            raw.write_bytes(b'corrupt')
            with mock.patch.object(collector, 'urlopen') as http, self.assertRaises(collector.Stop):
                collector.Run(args, '').load_resume()
            http.assert_not_called()
            self.assertEqual(raw.read_bytes(), b'corrupt')
        with tempfile.TemporaryDirectory() as directory:
            args, original = self.acquire(directory)
            args.resume = True
            args.bookmakers = ['pinnacle']
            with self.assertRaises(collector.Stop):
                collector.Run(args, '').load_resume()


class SecurityAndStopTests(unittest.TestCase):
    def test_secret_echo_and_access_rate_quota_stops_leave_frozen_remainder_incomplete(self):
        secret = 'private/key+value'
        failures = [(200, {'fixtureId': 'id1', 'bookmakers': {}, 'echo': secret}, 'SECRET_ECHO'),
                    (403, {'error': {'message': secret}}, 'SECRET_ECHO'),
                    (429, {'error': {'code': 'REQUEST_LIMIT_EXCEEDED'}}, 'REQUEST_LIMIT_EXCEEDED'),
                    (429, {}, 'RATE_LIMIT_EXCEEDED'),
                    (302, {}, 'UNSUPPORTED_REDIRECT')]
        for status, failure, code in failures:
            with self.subTest(status=status, code=code), tempfile.TemporaryDirectory() as directory:
                args = collector.parse_args(BoundaryTests().arguments(directory))
                run = collector.Run(args, secret)
                account = ContractTests().account()
                account['api_key'] = secret
                account['private_unlisted_field'] = secret
                replies = [(200, account), (200, [ContractTests().market()]),
                           (200, [ContractTests().fixture('id1'), ContractTests().fixture('id2')]),
                           (status, failure)]
                with AcquisitionTests().simulate(run, replies), mock.patch.object(collector.time, 'sleep'):
                    summary = run.execute()
                self.assertFalse(summary['acquisition_complete'])
                self.assertEqual(summary['actual_attempts_this_invocation']['/historical-odds'], 1)
                self.assertEqual(summary['actual_attempts_this_invocation']['/account'], 1)
                self.assertIn(code, summary['stop_reason'])
                self.assertEqual(summary['coverage_by_book']['pinnacle'],
                                 {'failed_request': 1, 'not_attempted': 1})
                import gzip
                for path in Path(directory).rglob('*'):
                    if path.is_file():
                        body = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
                        self.assertFalse(run.client.has_secret(body))
                if code == 'SECRET_ECHO':
                    self.assertIsNone(run.receipts[-1]['raw_path'])
                    self.assertIn('decoded_sha256', run.receipts[-1])

    def test_encoded_secret_echo_is_detected_and_wrong_fixture_success_is_not_cached(self):
        from urllib.parse import quote_plus
        with tempfile.TemporaryDirectory() as directory:
            args = collector.parse_args(BoundaryTests().arguments(directory))
            run = collector.Run(args, 'private/key+value')
            run.protocol['effective_books'] = ['pinnacle']
            with AcquisitionTests().simulate(run, [(500, {'message': quote_plus(run.client.key)})]):
                with self.assertRaises(collector.Stop):
                    run.fetch('/historical-odds', run.parameters('/historical-odds', 'id1'))
            self.assertFalse(run.cache)
        with tempfile.TemporaryDirectory() as directory:
            args = collector.parse_args(BoundaryTests().arguments(directory))
            run = collector.Run(args, 'private-test')
            run.protocol['effective_books'] = ['pinnacle']
            with AcquisitionTests().simulate(run, [(200, AcquisitionTests().history('id2'))]):
                with self.assertRaises(collector.Stop):
                    run.fetch('/historical-odds', run.parameters('/historical-odds', 'id1'),
                              lambda p: collector.validate_history(p, 'id1'))
            self.assertFalse(run.cache)


class ResponseContractTests(unittest.TestCase):
    def test_success_with_null_or_non_json_body_is_not_terminal_or_retried(self):
        for raw in (b'null', b'not json'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                args = collector.parse_args(BoundaryTests().arguments(directory))
                run = collector.Run(args, 'private-test')
                run.protocol['effective_books'] = ['pinnacle']
                with AcquisitionTests().simulate(run, [(200, raw)]):
                    with self.assertRaises(collector.Stop):
                        run.fetch('/historical-odds', run.parameters('/historical-odds', 'id1'),
                                  lambda p: collector.validate_history(p, 'id1'))
                self.assertFalse(run.cache)
                self.assertFalse(run.receipts[-1]['terminal'])


class TransportFailureTests(unittest.TestCase):
    def test_protocol_exceptions_cannot_expose_secret_bearing_error_text(self):
        from http.client import BadStatusLine, IncompleteRead
        args = collector.parse_args(BoundaryTests().arguments('unused'))
        for error in (BadStatusLine('private-test'), IncompleteRead(b'private-test', 30)):
            client = collector.Client(args, 'private-test')
            with self.subTest(error=type(error).__name__), mock.patch.object(collector, 'urlopen', side_effect=error):
                record, body, payload = client.request('/historical-odds', {'fixtureId': 'id1'})
            self.assertEqual(record['error_code'], 'NETWORK_ERROR')
            self.assertEqual(sum(client.attempts.values()), 1)
            self.assertIsNone(body)
            self.assertIsNone(payload)
            self.assertNotIn('private-test', json.dumps(record))

    def test_invalid_deflate_cache_fails_explicitly_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            args, original = ResumeTests().acquire(directory)
            args.resume = True
            raw = Path(directory) / next(r['raw_path'] for r in original.receipts if r['terminal'])
            raw.write_bytes(b'\x1f\x8b\x08\x00' + b'\x00' * 6 + b'\x06' + b'\x00' * 8)
            with self.assertRaises(collector.Stop):
                collector.Run(args, '').load_resume()


if __name__ == '__main__':
    unittest.main()
