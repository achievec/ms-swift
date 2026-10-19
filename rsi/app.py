# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from rsi import __version__
from rsi.backends import build_backend, detect_gpu
from rsi.pipeline import Pipeline, PipelineError
from rsi.schemas import public_schema

STATIC_DIR = Path(__file__).resolve().parent / 'static'
# run.sh appends here: rsi/.rsi.log next to this package, never a request path.
SERVER_LOG_PATH = Path(__file__).resolve().parent / '.rsi.log'
SERVER_LOG_TAIL = 400 * 1024
logger = logging.getLogger(__name__)


def server_log_path() -> Path:
    return SERVER_LOG_PATH


def read_server_log(path: Path, limit: int | None = None) -> str:
    """Return the launch log, keeping the tail when the file is large."""
    size_limit = SERVER_LOG_TAIL if limit is None else limit
    data = path.read_bytes()
    if len(data) <= size_limit:
        return data.decode('utf-8', errors='replace')
    tail = data[-size_limit:]
    text = tail.decode('utf-8', errors='replace')
    return f'[截断] 文件共 {len(data)} 字节，以下为最后 {size_limit} 字节。\n{text}'


async def read_json(request: Request) -> dict:
    """Parse a JSON object body. An empty body is an empty object, never a 500."""
    raw = await request.body()
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PipelineError('请求体不是合法的 JSON') from exc
    if not isinstance(data, dict):
        raise PipelineError('请求体必须是 JSON 对象')
    return data


def create_app(home: Path | None = None) -> FastAPI:
    if home is None:
        home = Path(os.environ.get('RSI_HOME', Path(__file__).resolve().parent / 'workspace'))
    home = Path(home)
    pipeline = Pipeline(home, build_backend())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.pipeline.serve.stop_all()

    app = FastAPI(title='大模型训练平台', version=__version__, lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.home = home

    @app.exception_handler(PipelineError)
    async def _pipeline_error(_request: Request, exc: PipelineError):
        return JSONResponse({'detail': str(exc)}, status_code=exc.status)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        logger.exception('未处理的请求错误')
        detail = str(exc).strip() or '服务器内部错误'
        return JSONResponse({'detail': detail}, status_code=500)

    @app.get('/api/health')
    def health():
        schema = public_schema()
        return {
            'ok': True,
            'version': __version__,
            'backend': pipeline.backend.name,
            'serve_backend': pipeline._serve_backend_name(),
            'gpu': detect_gpu(),
            'hardware': schema.get('hardware') or {},
        }

    @app.get('/api/schema')
    def schema():
        return public_schema()

    @app.get('/api/server-log')
    def server_log():
        path = server_log_path()
        if not path.is_file():
            return PlainTextResponse('rsi/.rsi.log 不存在\n', status_code=404)
        try:
            text = read_server_log(path)
        except OSError:
            return PlainTextResponse('rsi/.rsi.log 无法读取\n', status_code=404)
        return PlainTextResponse(text)

    @app.get('/api/runs')
    def list_runs():
        return {'runs': pipeline.list_runs()}

    @app.post('/api/runs')
    async def create_run(request: Request):
        body = await read_json(request)
        return pipeline.create_run(
            body.get('name', ''), body.get('scenario', ''), body.get('base_model', ''),
            body.get('model_type', '') or '', body.get('template', '') or '')

    @app.get('/api/runs/{run_id}')
    def get_run(run_id: str, round: int | None = None):
        return pipeline.view(run_id, round)

    def _delete_run(run_id: str):
        return pipeline.delete_run(run_id)

    @app.delete('/api/runs/{run_id}')
    def delete_run(run_id: str):
        return _delete_run(run_id)

    @app.delete('/api/experiments/{experiment_id}')
    def delete_experiment(experiment_id: str):
        return _delete_run(experiment_id)

    @app.get('/api/runs/{run_id}/audit')
    def audit(run_id: str):
        view = pipeline.view(run_id)
        return {'steps': view['steps'], 'services': view['services'], 'checkpoints': view['checkpoints']}

    @app.get('/api/runs/{run_id}/stages/{stage}/log')
    def stage_log(run_id: str, stage: str, round: int | None = None, offset: int | None = None,
                  limit: int | None = None):
        return pipeline.stage_log(run_id, stage, round, offset, limit)

    @app.post('/api/datasets/validate')
    async def validate_dataset(request: Request):
        body = await read_json(request)
        params = body.get('params') if isinstance(body, dict) and 'params' in body else body
        return pipeline.validate_request(params or {})

    @app.post('/api/datasets/upload')
    async def upload_dataset(file: UploadFile = File(...)):
        data = await file.read()
        return pipeline.save_upload(file.filename or 'data.jsonl', data)

    @app.post('/api/runs/{run_id}/fetch')
    async def refresh_fetch(run_id: str, request: Request):
        body = await read_json(request)
        return pipeline.refresh_fetch(run_id, body or {})

    @app.post('/api/runs/{run_id}/stages/{stage}/preview')
    async def preview_stage(run_id: str, stage: str, request: Request):
        body = await read_json(request)
        params = body.get('params') if isinstance(body, dict) else {}
        return pipeline.preview(run_id, stage, params or {})

    @app.post('/api/runs/{run_id}/decide')
    async def decide(run_id: str, request: Request):
        body = await read_json(request)
        return pipeline.decide(run_id, (body or {}).get('action') or '')

    @app.post('/api/runs/{run_id}/stages/{stage}/run')
    async def run_stage(run_id: str, stage: str, request: Request):
        body = await read_json(request)
        params = body.get('params') if isinstance(body, dict) else {}
        return pipeline.run_stage(run_id, stage, params or {})

    @app.post('/api/runs/{run_id}/serve')
    async def start_serve(run_id: str, request: Request):
        body = await read_json(request)
        params = body.get('params') if isinstance(body, dict) else {}
        return pipeline.start_serve(run_id, params or {})

    @app.post('/api/runs/{run_id}/serve/stop')
    async def stop_serve(run_id: str, request: Request):
        await read_json(request)
        return pipeline.stop_serve(run_id)

    @app.post('/api/runs/{run_id}/chat')
    async def chat(run_id: str, request: Request):
        body = await read_json(request)
        return pipeline.chat(run_id, body.get('messages') or [], body.get('temperature'))

    @app.get('/')
    def index():
        return FileResponse(STATIC_DIR / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

    @app.exception_handler(HTTPException)
    async def _http_error(_request: Request, exc: HTTPException):
        return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)

    return app
