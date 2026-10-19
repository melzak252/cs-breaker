"""Same-origin local research API; server startup never initiates provider requests."""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from scripts.collect_cs_oddspapi import utc, SAFE_BOOK
from .config import Config
from .services import (status, list_fixtures, fixture_detail, historical_list, states_list,
                       decide_mapping, compute_metrics, iter_export, now)
from .store import Store
from .worker import Worker


class JobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['backfill', 'upcoming', 'finished', 'results', 'import_existing', 'remap']
    payload: dict = Field(default_factory=dict)


class MappingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decision: Literal['accept', 'reject', 'clear']
    history_key: str | None = None
    orientation: Literal[0, 1] | None = None
    reason: str = Field(min_length=1, max_length=2000)


class HistoryImport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: Literal['opencs2', 'kaggle', 'strandedpond', 'ektarr', 'hltv']
    csv: str = Field(min_length=1, max_length=16 * 1024 * 1024)


def validate_payload(kind, payload, settings):
    allowed = {'from', 'to', 'bookmakers', 'max_history', 'max_requests', 'billable_per_run', 'max_fixtures', 'refresh'}
    if set(payload) - allowed:
        raise ValueError('Unsupported collection parameter')
    if 'refresh' in payload and (kind != 'results' or type(payload['refresh']) is not bool):
        raise ValueError('refresh is a results-only boolean')
    if kind != 'backfill' and ('from' in payload or 'to' in payload):
        raise ValueError('Explicit date bounds apply to backfill jobs only')
    if kind == 'backfill':
        if 'from' not in payload or 'to' not in payload or utc(payload['from']) >= utc(payload['to']):
            raise ValueError('Backfill requires ordered timezone-aware from/to dates')
    for key in ('max_history', 'max_requests', 'billable_per_run', 'max_fixtures'):
        if key in payload and (type(payload[key]) is not int or not 0 <= payload[key] <= settings[key]):
            raise ValueError('Requested job caps must be nonnegative integers within configured limits')
    if 'bookmakers' in payload:
        books = payload['bookmakers']
        if (not isinstance(books, list) or not 1 <= len(books) <= 3
                or not all(isinstance(b, str) and SAFE_BOOK.fullmatch(b) for b in books)
                or len(set(books)) != len(books)):
            raise ValueError('Bookmakers must contain one to three distinct safe slugs')
    return payload


def create_app(config: Config | None = None, *, run_worker=True):
    config = config or Config.from_args()
    store = Store(config.data_dir)
    worker = Worker(store, config)

    @asynccontextmanager
    async def lifespan(app):
        if run_worker:
            worker.start()
        yield
        if run_worker:
            worker.stop()

    app = FastAPI(title='CS2 Research Workspace', version='1.0.0', lifespan=lifespan)
    app.state.store = store
    app.state.config = config
    app.state.worker = worker
    static = Path(__file__).with_name('static')
    app.mount('/static', StaticFiles(directory=static), name='static')

    @app.middleware('http')
    async def local_boundary(request: Request, call_next):
        host = urlsplit('http://' + request.headers.get('host', '')).hostname
        if host not in ('localhost', '127.0.0.1', '::1'):
            return JSONResponse({'detail': 'Application is restricted to local hosts'}, status_code=403)
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            if request.headers.get('x-cs-app') != '1' or (origin and urlsplit(origin).netloc.lower() != request.headers.get('host', '').lower()):
                return JSONResponse({'detail': 'Same-origin application header required'}, status_code=403)
        length = request.headers.get('content-length')
        if length and (not length.isdigit() or int(length) > 24 * 1024 * 1024):
            return JSONResponse({'detail': 'Request body exceeds local import limit'}, status_code=413)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.exception_handler(RequestValidationError)
    async def field_error(request, error):
        # Never echo submitted private input in Pydantic's default error payload.
        return JSONResponse({'detail': [{'field': '.'.join(str(p) for p in e['loc']), 'message': e['msg']} for e in error.errors()]}, status_code=422)

    @app.exception_handler(ValueError)
    async def input_error(request, error):
        message = str(error)
        if config.key and config.key in message:
            message = 'Private input rejected'
        return JSONResponse({'detail': message}, status_code=400)

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        return JSONResponse({'detail': 'Application request failed; no private diagnostic returned'}, status_code=500)

    @app.get('/')
    def index():
        return FileResponse(static / 'index.html')

    @app.get('/api/status')
    def application_status():
        return status(store, bool(config.key))

    @app.get('/api/fixtures')
    def fixtures(status: Literal['all', 'upcoming', 'finished'] = 'all', q: str = '', bookmaker: str = '',
                 mapping: Literal['all', 'mapped', 'unmapped'] = 'all', offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        return list_fixtures(store, status, q, bookmaker, mapping, offset, limit)

    @app.get('/api/fixtures/{identity}')
    def detail(identity: str):
        result = fixture_detail(store, identity)
        if result is None:
            raise HTTPException(404, 'Fixture not found')
        return result

    @app.get('/api/fixtures/{identity}/odds')
    def odds(identity: str, bookmaker: str = '', offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        return states_list(store, identity, bookmaker, offset, limit)

    @app.get('/api/history')
    def history(q: str = '', source: str = '', offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        return historical_list(store, q, source, offset, limit)

    @app.post('/api/history/import', status_code=202)
    def historical_import(data: HistoryImport):
        body = data.csv.encode()
        if len(body) > 16 * 1024 * 1024:
            raise HTTPException(413, 'CSV exceeds 16 MiB')
        if config.key and config.key.encode() in body:
            raise ValueError('Credential-bearing input cannot be retained')
        stamp = now()
        record = {'endpoint': 'local://history/upload', 'parameters': {'source': data.source},
                  'request_started_at': stamp, 'received_at': stamp, 'http_status': 0, 'terminal': False, 'error_code': None}
        raw_id = store.put_raw(record, body, origin='import', billable=False)
        identity = store.create_job('import_history', {'source': data.source, 'raw_id': raw_id})
        return {'id': identity, 'status': 'queued'}

    @app.get('/api/jobs')
    def jobs():
        return {'items': store.list_jobs()}

    @app.get('/api/jobs/{identity}')
    def job(identity: str):
        result = store.get_job(identity)
        if result is None:
            raise HTTPException(404, 'Job not found')
        return result

    @app.post('/api/jobs', status_code=202)
    def enqueue(data: JobRequest):
        if data.kind in ('import_existing', 'remap'):
            if data.payload:
                raise ValueError('Local maintenance jobs do not accept filesystem or command arguments')
        else:
            validate_payload(data.kind, data.payload, store.get_settings())
            if not config.key:
                raise ValueError('ODDSPAPI_API_KEY is not configured; imports and research are available offline')
        return {'id': store.create_job(data.kind, data.payload), 'status': 'queued'}

    @app.post('/api/jobs/{identity}/cancel')
    def cancel(identity: str):
        return {'ok': store.cancel_job(identity)}

    @app.post('/api/jobs/{identity}/retry')
    def retry(identity: str):
        current = store.get_job(identity)
        if current and current['kind'] in ('backfill', 'upcoming', 'finished', 'results') and not config.key:
            raise ValueError('Provider credential required before resuming unfinished collection')
        return {'ok': store.retry_job(identity)}

    @app.get('/api/settings')
    def settings():
        return status(store, bool(config.key))['settings']

    @app.put('/api/settings')
    def update_settings(values: dict):
        if values.get('scheduler_enabled') is True and not config.key:
            raise ValueError('Configure a provider credential before enabling scheduled acquisition')
        return store.save_settings(values)

    @app.post('/api/mappings/{identity}')
    def mapping(identity: str, data: MappingRequest):
        return decide_mapping(store, identity, data.decision, data.history_key, data.orientation, data.reason)

    @app.get('/api/metrics')
    def metrics():
        return compute_metrics(store)

    @app.get('/api/export/{dataset}.csv')
    def export(dataset: Literal['fixtures', 'history', 'quotes', 'states', 'mappings', 'jobs', 'mapped_quotes']):
        return StreamingResponse(iter_export(store, dataset), media_type='text/csv', headers={'Content-Disposition': f'attachment; filename="cs2-{dataset}.csv"'})

    @app.get('/api/evidence/{identity}')
    def evidence(identity: int):
        row = store.one('SELECT raw_path,file_sha256 FROM raw_responses WHERE id=?', (identity,))
        if not row or not row['raw_path']:
            raise HTTPException(404, 'Raw body not retained for this receipt')
        path = store.root / row['raw_path']
        if not path.resolve().is_relative_to(store.raw_root.resolve()) or not path.is_file():
            raise HTTPException(404, 'Evidence file unavailable')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['file_sha256']:
            raise HTTPException(409, 'Evidence integrity mismatch')
        return FileResponse(path, media_type='application/gzip', filename=path.name)

    return app
