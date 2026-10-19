"""Single durable worker with explicitly enabled scheduling and resumable jobs."""
from __future__ import annotations

import fcntl
import gzip
import json
import os
import threading
from datetime import datetime, timedelta, timezone

from scripts.collect_cs_oddspapi import Stop
from .config import Config
from .services import import_existing, import_history, rebuild_mappings, now
from .store import Store


class Worker:
    def __init__(self, store: Store, config: Config):
        self.store, self.config = store, config
        self.stopping = threading.Event()
        self.thread = None
        self.lock = None
        self.worker_id = f'pid-{os.getpid()}'
        self.active_id = None

    def acquire(self):
        handle = (self.store.root / 'worker.lock').open('a')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self.lock = handle
        self.store.recover_jobs()
        return True

    def close(self):
        if self.lock:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def schedule(self):
        settings = self.store.get_settings()
        if not settings['scheduler_enabled'] or not self.config.key:
            return
        row = self.store.one("SELECT value_json FROM settings WHERE key='scheduler_state'")
        state = json.loads(row['value_json']) if row else {}
        current = datetime.now(timezone.utc)
        if state.get('last_at') and current < datetime.fromisoformat(state['last_at']) + timedelta(minutes=settings['scheduler_interval_minutes']):
            return
        if self.store.one("SELECT id FROM jobs WHERE status IN ('queued','running') LIMIT 1"):
            return
        # Cycle source discovery, completed histories and native results fairly.
        # Every job retains its normal local and provider allowance checks.
        sequence = ('upcoming', 'finished', 'results')
        index = int(state.get('index', 0)) % len(sequence)
        self.store.create_job(sequence[index], {'scheduled': True})
        self.store.execute("INSERT INTO settings(key,value_json) VALUES('scheduler_state',?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                           (json.dumps({'last_at': current.isoformat(), 'index': (index + 1) % len(sequence)}),))

    def run_next(self):
        job = self.store.claim_job(self.worker_id)
        if job is None:
            return None
        identity = job['id']
        self.active_id = identity
        try:
            if job['kind'] == 'import_existing':
                result = import_existing(self.store, self.config.repo, identity)
            elif job['kind'] == 'import_history':
                payload = job['payload']
                evidence = self.store.one('SELECT * FROM raw_responses WHERE id=?', (payload['raw_id'],))
                if not evidence or not evidence['raw_path']:
                    raise ValueError('Historical import raw evidence is missing')
                path = self.store.root / evidence['raw_path']
                if not path.resolve().is_relative_to(self.store.raw_root.resolve()):
                    raise ValueError('Unsafe historical import evidence path')
                report = import_history(self.store, payload['source'], gzip.decompress(path.read_bytes()), identity)
                result = {'status': 'completed', 'import': report, 'mapping': rebuild_mappings(self.store), 'network_requests': 0}
            elif job['kind'] == 'remap':
                result = {'status': 'completed', 'mapping': rebuild_mappings(self.store), 'network_requests': 0}
            else:
                from .collection import Collector
                result = Collector(self.store, self.config.key).run(job)
                result['mapping'] = rebuild_mappings(self.store)
            state = result.get('status', 'completed')
            if state not in ('completed', 'partial', 'budget_paused', 'cancelled'):
                raise ValueError('Unsupported collector completion state')
            if self.store.cancelled(identity):
                state = 'cancelled'
            self.store.update_job(identity, status=state, result=result, error=None)
        except Stop as error:
            # Stop is deliberately sanitized by the transport; guard against a
            # misbehaving adapter before retaining or returning diagnostics.
            message = str(error)
            if self.config.key and self.config.key in message:
                message = 'Provider stopped acquisition; private diagnostic omitted'
            self.store.update_job(identity, status='partial', error=message)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.store.update_job(identity, status='failed', error='Input or provider contract validation failed; inspect source schema and retained evidence')
        except Exception:
            self.store.update_job(identity, status='failed', error='Application job failed; retained progress and evidence are available')
        self.active_id = None
        return self.store.get_job(identity)

    def loop(self):
        while not self.stopping.is_set() and not self.acquire():
            self.stopping.wait(1)
        if self.stopping.is_set():
            self.close()
            return
        try:
            while not self.stopping.is_set():
                self.schedule()
                completed = self.run_next()
                if completed is None:
                    self.stopping.wait(1)
        finally:
            self.close()

    def start(self):
        self.thread = threading.Thread(target=self.loop, name='cs-app-worker', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        # Cancellation is observed between bounded requests/import windows;
        # the provider's finite socket timeout bounds shutdown waiting.
        if self.lock is not None and self.active_id is not None:
            self.store.cancel_job(self.active_id)
        if self.thread:
            self.thread.join(timeout=50)
