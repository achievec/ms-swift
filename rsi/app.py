# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rsi import __version__
from rsi.backends import build_backend, detect_gpu
from rsi.model_fetch import resolve_rsi_home
from rsi.pipeline import Pipeline, PipelineError
from rsi.schemas import public_schema
from rsi.store import StoreError
from rsi import workbench

STATIC_DIR = Path(__file__).resolve().parent / 'static'
logger = logging.getLogger(__name__)


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
        home = resolve_rsi_home(Path(__file__).resolve().parent / 'workspace')
        logger.info('RSI_HOME=%s', home)
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

    @app.exception_handler(StoreError)
    async def _store_error(_request: Request, exc: StoreError):
        logger.error('实验记录库写入失败：%s', exc)
        return JSONResponse({'detail': str(exc)}, status_code=500)

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

    @app.get('/api/runs/{run_id}/stages/{stage}/report')
    def stage_report(run_id: str, stage: str, round: int | None = None):
        return FileResponse(pipeline.eval_report_file(run_id, stage, round), media_type='text/html')

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

    @app.get('/api/runs/{run_id}/serve/log')
    def serve_log(run_id: str):
        return pipeline.serve_log(run_id)

    @app.post('/api/runs/{run_id}/chat')
    async def chat(run_id: str, request: Request):
        body = await read_json(request) or {}
        messages = body.get('messages') or []
        options = {
            'temperature': body.get('temperature'),
            'top_k': body.get('top_k'),
            'top_p': body.get('top_p'),
            'repetition_penalty': body.get('repetition_penalty'),
            'max_new_tokens': body.get('max_new_tokens'),
            'model': body.get('infer_model_type') or '',
            'system': body.get('system') or '',
        }
        if body.get('stream'):
            chunks = pipeline.iter_chat_sse(run_id, messages, **options)
            return StreamingResponse(chunks, media_type='text/event-stream')
        return pipeline.chat(run_id, messages, **options)

    def _workbench_text(body: dict) -> tuple:
        text = body.get('text') or ''
        if not str(text).strip():
            raise PipelineError('请贴上 jsonl，或选择一个文件')
        return str(text), body.get('filename') or 'data.jsonl'

    @app.post('/api/workbench/inspect')
    async def workbench_inspect(request: Request):
        text, filename = _workbench_text(await read_json(request))
        return workbench.inspect_text(text, filename)

    @app.post('/api/workbench/split')
    async def workbench_split(request: Request):
        body = await read_json(request)
        text, filename = _workbench_text(body)
        ratio = body.get('eval_ratio')
        seed = body.get('seed')
        return workbench.write_split(
            home, text, filename, 0.2 if ratio is None or ratio == '' else ratio,
            0 if seed is None or seed == '' else seed)

    @app.post('/api/workbench/next-pack')
    async def workbench_next_pack(request: Request):
        body = await read_json(request)
        text, filename = _workbench_text(body)
        minimum = body.get('min_score')
        return workbench.pack_next_round(
            home, text, filename, 1 if minimum is None or minimum == '' else minimum)

    @app.get('/api/runs/{run_id}/report')
    def run_report(run_id: str):
        return PlainTextResponse(workbench.render_report(pipeline.view(run_id)))

    @app.get('/api/runs/{run_id}/serve/ready')
    def serve_ready(run_id: str):
        pipeline._require_run(run_id)
        service = pipeline.store.latest_service(run_id)
        return workbench.probe_service(service)

    @app.post('/api/runs/{run_id}/export')
    def start_export(run_id: str):
        view = pipeline.view(run_id)
        return workbench.start_export(
            home, run_id, workbench.promoted_checkpoint(view), (view.get('run') or {}).get('base_model') or '')

    @app.post('/api/runs/{run_id}/workbench/complete')
    async def workbench_complete(run_id: str, request: Request):
        pipeline._require_run(run_id)
        body = await read_json(request)
        text, filename = _workbench_text(body)
        rows, _report = workbench.load_rows(text, filename)

        def answer(messages):
            payload = pipeline.chat(run_id, messages, temperature=0)
            return workbench.completion_text(payload)

        return workbench.write_completed(home, rows, answer)

    @app.get('/api/runs/{run_id}/export')
    def export_status(run_id: str):
        pipeline._require_run(run_id)
        return workbench.export_status(run_id)

    @app.get('/tools')
    def tools_page():
        return FileResponse(STATIC_DIR / 'tools.html')

    @app.get('/')
    def index():
        return FileResponse(STATIC_DIR / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

    @app.exception_handler(HTTPException)
    async def _http_error(_request: Request, exc: HTTPException):
        return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)

    return app
