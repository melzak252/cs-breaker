"""Worker ownership must not invalidate another process's authorized job."""
import tempfile
import unittest
from pathlib import Path

from cs_app.config import Config
from cs_app.store import Store
from cs_app.worker import Worker


class WorkerOwnershipTests(unittest.TestCase):
    def test_non_owner_shutdown_does_not_cancel_running_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = Store(root)
            config = Config(data_dir=root)
            owner, standby = Worker(store, config), Worker(store, config)
            self.assertTrue(owner.acquire())
            try:
                identity = store.create_job('remap', {})
                store.claim_job(owner.worker_id)
                self.assertFalse(standby.acquire())
                standby.stop()
                row = store.get_job(identity)
                self.assertEqual(row['status'], 'running')
                self.assertFalse(row['cancel_requested'])
            finally:
                owner.close()

    def test_cancellation_wins_all_error_terminal_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            for outcome in ('partial', 'failed', 'budget_paused'):
                with self.subTest(outcome=outcome):
                    identity = store.create_job('remap', {})
                    store.claim_job('authorized-worker')
                    store.cancel_job(identity)
                    store.update_job(identity, status=outcome, error='Sanitized failure retained')
                    row = store.get_job(identity)
                    self.assertEqual(row['status'], 'cancelled')
                    self.assertEqual(row['error'], 'Sanitized failure retained')


if __name__ == '__main__':
    unittest.main()
