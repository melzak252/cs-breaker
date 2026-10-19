"""Run the web workspace, explicit jobs, historical imports and research exports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .services import status, iter_export, now
from .store import Store
from .worker import Worker


def run_job(store, config, identity):
    worker = Worker(store, config)
    if not worker.acquire():
        return {'id': identity, 'status': 'queued', 'reason': 'Another worker owns the job queue'}
    try:
        while True:
            result = worker.run_next()
            if result is None:
                return store.get_job(identity)
            if result['id'] == identity:
                return result
    finally:
        worker.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data-dir', type=Path)
    common.add_argument('--env-file', type=Path, help='Explicit private .env file; process environment takes precedence')
    commands = parser.add_subparsers(dest='command', required=True)
    initialize = commands.add_parser('init', parents=[common])
    initialize.add_argument('--seed-existing', action='store_true')
    serve = commands.add_parser('serve', parents=[common])
    serve.add_argument('--host', choices=['127.0.0.1', '::1', '0.0.0.0'], default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8017)
    commands.add_parser('status', parents=[common])
    work = commands.add_parser('worker', parents=[common])
    work.add_argument('--once', action='store_true')
    collect = commands.add_parser('collect', parents=[common])
    collect.add_argument('kind', choices=['backfill', 'upcoming', 'finished', 'results'])
    collect.add_argument('--from', dest='start')
    collect.add_argument('--to', dest='end')
    collect.add_argument('--bookmakers')
    collect.add_argument('--max-history', type=int)
    collect.add_argument('--max-requests', type=int)
    collect.add_argument('--max-fixtures', type=int)
    collect.add_argument('--billable-per-run', type=int)
    collect.add_argument('--refresh', action='store_true', help='Results only: reacquire existing native scores for explicit corrections')
    import_csv = commands.add_parser('import-history', parents=[common])
    import_csv.add_argument('--source', choices=['kaggle', 'opencs2', 'strandedpond', 'ektarr', 'hltv'], required=True)
    import_csv.add_argument('--file', type=Path, required=True)
    commands.add_parser('remap', parents=[common])
    export = commands.add_parser('export', parents=[common])
    export.add_argument('dataset', choices=['fixtures', 'history', 'quotes', 'mapped_quotes', 'states', 'mappings', 'jobs'])
    export.add_argument('--output', type=Path, required=True)
    settings = commands.add_parser('settings', parents=[common])
    settings.add_argument('--json', help='Validated JSON settings patch; omitted displays current settings')
    job = commands.add_parser('job', parents=[common])
    job.add_argument('id')
    action = job.add_mutually_exclusive_group()
    action.add_argument('--retry', action='store_true')
    action.add_argument('--cancel', action='store_true')
    args = parser.parse_args(argv)
    try:
        config = Config.from_args(args.data_dir, args.env_file)
        store = Store(config.data_dir)
        if args.command == 'serve':
            if not 1 <= args.port <= 65535:
                parser.error('Port must be between 1 and 65535')
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(config), host=args.host, port=args.port, log_level='warning', access_log=False)
            return 0
        if args.command == 'init':
            result = run_job(store, config, store.create_job('import_existing', {})) if args.seed_existing else status(store, bool(config.key))
        elif args.command == 'status':
            result = status(store, bool(config.key))
        elif args.command == 'worker':
            worker = Worker(store, config)
            if args.once:
                if not worker.acquire():
                    parser.error('Another worker already owns the queue')
                try:
                    result = worker.run_next() or {'status': 'idle'}
                finally:
                    worker.close()
            else:
                try:
                    worker.loop()
                except KeyboardInterrupt:
                    worker.stopping.set()
                return 0
        elif args.command == 'collect':
            from .api import validate_payload
            payload = {k: getattr(args, k) for k in ('max_history', 'max_requests', 'max_fixtures', 'billable_per_run') if getattr(args, k) is not None}
            if args.refresh:
                payload['refresh'] = True
            if args.start:
                payload['from'] = args.start
            if args.end:
                payload['to'] = args.end
            if args.bookmakers:
                payload['bookmakers'] = args.bookmakers.split(',')
            validate_payload(args.kind, payload, store.get_settings())
            result = run_job(store, config, store.create_job(args.kind, payload))
        elif args.command == 'import-history':
            body = args.file.read_bytes()
            if len(body) > 16 * 1024 * 1024:
                parser.error('Historical CSV exceeds 16 MiB import limit')
            if config.key and config.key.encode() in body:
                parser.error('Private credential-bearing input cannot be retained')
            stamp = now()
            raw = store.put_raw({'endpoint': 'local://history/upload', 'parameters': {'source': args.source},
                'request_started_at': stamp, 'received_at': stamp, 'http_status': 0, 'terminal': False}, body, origin='import', billable=False)
            result = run_job(store, config, store.create_job('import_history', {'source': args.source, 'raw_id': raw}))
        elif args.command == 'remap':
            result = run_job(store, config, store.create_job('remap', {}))
        elif args.command == 'export':
            if args.output.exists():
                parser.error('Research export target exists; choose a new output path')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8', newline='') as target:
                for chunk in iter_export(store, args.dataset):
                    target.write(chunk)
            result = {'export': args.dataset, 'path': str(args.output), 'bytes': args.output.stat().st_size}
        elif args.command == 'settings':
            patch = json.loads(args.json) if args.json else None
            if patch and patch.get('scheduler_enabled') and not config.key:
                parser.error('Configure provider credentials before explicitly enabling scheduling')
            result = store.save_settings(patch) if patch is not None else store.get_settings()
        else:
            if args.retry:
                store.retry_job(args.id)
                result = run_job(store, config, args.id)
            elif args.cancel:
                result = {'ok': store.cancel_job(args.id)}
            else:
                result = store.get_job(args.id)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result and result.get('status') in ('failed', 'partial', 'budget_paused', 'cancelled') else 0
    except (ValueError, KeyError, TypeError, OSError):
        print('Application input/configuration error; no private diagnostic printed', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
