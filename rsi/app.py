# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from rsi import __version__
from rsi.backends import build_backend, detect_gpu
from rsi.logtail import read_log_window
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


def wants_html(request: Request) -> bool:
    """Browser navigation asks for text/html. curl's ``*/*`` does not."""
    return 'text/html' in (request.headers.get('accept') or '').lower()


def server_log_window(path: Path, offset: int | None) -> dict:
    """Tail on the first read, then bytes after ``offset``. The file is fixed."""
    tail = SERVER_LOG_TAIL
    empty = {'offset': 0, 'next_offset': 0, 'size': 0, 'text': '', 'missing': True, 'reset': True}
    if not path.is_file():
        return empty
    try:
        size = path.stat().st_size
    except OSError:
        return empty
    reset = offset is None or offset > size
    if offset is not None and offset > size:
        offset = None
    window = read_log_window(path, offset, tail, tail=tail)
    text = window['text']
    if reset and window['offset'] > 0:
        kept = window['size'] - window['offset']
        text = f'[截断] 文件共 {window["size"]} 字节，以下为最后 {kept} 字节。\n{text}'
    window['text'] = text
    window['missing'] = False
    window['reset'] = reset
    return window


_SERVER_LOG_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>进程日志</title>
  <style>
    html, body { height: 100%; margin: 0; background: #1c1915; color: #f4efe6; }
    body { display: flex; flex-direction: column; font-family: "Noto Sans SC", "PingFang SC", "Source Han Sans SC", sans-serif; }
    header { display: flex; align-items: baseline; gap: 16px; padding: 12px 16px; border-bottom: 1px solid #3a332c; }
    h1 { margin: 0; font-size: 16px; font-weight: 600; }
    a { color: #e7c27a; text-decoration: none; }
    p { margin: 0; color: #b5aa9e; font-size: 13px; }
    pre { flex: 1; min-height: 0; margin: 0; padding: 12px 16px 28px; overflow: auto;
      white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.45;
      font-family: ui-monospace, "Sarasa Mono SC", "Noto Sans Mono CJK SC", "Source Han Mono SC", monospace; }
  </style>
</head>
<body>
  <header>
    <h1>进程日志</h1>
    <p>自动跟上新日志</p>
    <a href="/">返回</a>
  </header>
  <pre id="log"></pre>
  <script>
    const view = document.getElementById('log');
    let offset = null;
    let pinned = true;
    view.addEventListener('scroll', () => {
      const gap = view.scrollHeight - view.scrollTop - view.clientHeight;
      pinned = gap < 40;
    });
    function follow() {
      if (pinned) view.scrollTop = view.scrollHeight;
    }
    async function pull(depth) {
      const url = offset == null ? '/api/server-log/stream' : '/api/server-log/stream?offset=' + offset;
      const response = await fetch(url, {headers: {accept: 'application/json'}});
      if (!response.ok) return;
      const body = await response.json();
      if (body.missing) {
        view.textContent = 'rsi/.rsi.log 不存在\n';
        offset = null;
        follow();
        return;
      }
      if (body.reset || offset == null) view.textContent = body.text || '';
      else if (body.text) view.textContent += body.text;
      offset = body.next_offset;
      follow();
      if (depth < 8 && body.size > body.next_offset) await pull(depth + 1);
    }
    async function tick() {
      try { await pull(0); } catch (_err) {}
    }
    tick();
    setInterval(tick, 1000);
  </script>
</body>
</html>
"""


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
    def server_log(request: Request):
        if wants_html(request):
            return HTMLResponse(_SERVER_LOG_PAGE)
        path = server_log_path()
        if not path.is_file():
            return PlainTextResponse('rsi/.rsi.log 不存在\n', status_code=404)
        try:
            text = read_server_log(path)
        except OSError:
            return PlainTextResponse('rsi/.rsi.log 无法读取\n', status_code=404)
        return PlainTextResponse(text)

    @app.get('/server-log')
    def server_log_page():
        return HTMLResponse(_SERVER_LOG_PAGE)

    @app.get('/api/server-log/stream')
    def server_log_stream(offset: int | None = None):
        return server_log_window(server_log_path(), offset)

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
