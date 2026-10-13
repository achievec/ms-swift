# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rsi import __version__
from rsi.backends import build_backend
from rsi.pipeline import Pipeline, PipelineError
from rsi.schemas import public_schema

STATIC_DIR = Path(__file__).resolve().parent / 'static'


def create_app(home: Path | None = None) -> FastAPI:
    if home is None:
        home = Path(os.environ.get('RSI_HOME', Path(__file__).resolve().parent / 'workspace'))
    home = Path(home)
    pipeline = Pipeline(home, build_backend())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.pipeline.serve.stop_all()

    app = FastAPI(title='ms-swift RSI', version=__version__, lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.swift_ui = False
    app.state.home = home

    @app.exception_handler(PipelineError)
    async def _pipeline_error(_request: Request, exc: PipelineError):
        return JSONResponse({'detail': str(exc)}, status_code=exc.status)

    @app.get('/api/health')
    def health():
        schema = public_schema()
        return {
            'ok': True,
            'version': __version__,
            'backend': pipeline.backend.name,
            'serve_backend': pipeline._serve_backend_name(),
            'hardware': schema.get('hardware') or {},
            'swift_ui': bool(getattr(app.state, 'swift_ui', False)),
            'port': int(os.environ.get('WEBUI_PORT') or os.environ.get('RSI_PORT') or '7860'),
        }

    @app.get('/api/schema')
    def schema():
        return public_schema()

    @app.get('/api/runs')
    def list_runs():
        return {'runs': pipeline.list_runs()}

    @app.post('/api/runs')
    async def create_run(request: Request):
        body = await request.json()
        meta = body.get('model_meta') if isinstance(body.get('model_meta'), dict) else {}
        return pipeline.create_run(body.get('name', ''), body.get('scenario', ''), body.get('base_model', ''), meta)

    @app.get('/api/runs/{run_id}')
    def get_run(run_id: str, round: int | None = None):
        return pipeline.view(run_id, round)

    @app.get('/api/runs/{run_id}/audit')
    def audit(run_id: str):
        view = pipeline.view(run_id)
        return {'steps': view['steps'], 'services': view['services'], 'checkpoints': view['checkpoints']}

    @app.post('/api/datasets/validate')
    async def validate_dataset(request: Request):
        body = await request.json()
        params = body.get('params') if isinstance(body, dict) and 'params' in body else body
        return pipeline.validate_request(params or {})

    @app.post('/api/datasets/upload')
    async def upload_dataset(file: UploadFile = File(...)):
        data = await file.read()
        return pipeline.save_upload(file.filename or 'data.jsonl', data)

    @app.post('/api/runs/{run_id}/stages/{stage}/preview')
    async def preview_stage(run_id: str, stage: str, request: Request):
        body = await request.json()
        params = body.get('params') if isinstance(body, dict) else {}
        return pipeline.preview(run_id, stage, params or {})

    @app.post('/api/runs/{run_id}/stages/{stage}/run')
    async def run_stage(run_id: str, stage: str, request: Request):
        body = await request.json()
        params = body.get('params') if isinstance(body, dict) else {}
        return pipeline.run_stage(run_id, stage, params or {})

    @app.post('/api/runs/{run_id}/serve')
    async def start_serve(run_id: str, request: Request):
        body = await request.json()
        params = body.get('params') if isinstance(body, dict) else {}
        return pipeline.start_serve(run_id, params or {})

    @app.post('/api/runs/{run_id}/serve/stop')
    def stop_serve(run_id: str):
        return pipeline.stop_serve(run_id)

    @app.post('/api/runs/{run_id}/chat')
    async def chat(run_id: str, request: Request):
        body = await request.json()
        return pipeline.chat(run_id, body.get('messages') or [], body.get('temperature'))

    @app.get('/')
    def index():
        return FileResponse(STATIC_DIR / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

    @app.exception_handler(HTTPException)
    async def _http_error(_request: Request, exc: HTTPException):
        return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)

    return app
