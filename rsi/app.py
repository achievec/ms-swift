# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rsi import __version__
from rsi.backends import build_backend, detect_gpu
from rsi.logtail import read_log_window
from rsi.model_fetch import resolve_rsi_home
from rsi.pipeline import Pipeline, PipelineError
from rsi.schemas import public_schema
from rsi.store import StoreError
from rsi import workbench

STATIC_DIR = Path(__file__).resolve().parent / 'static'
# run.sh appends here: rsi/.rsi.log next to this package, never a request path.
SERVER_LOG_PATH = Path(__file__).resolve().parent / '.rsi.log'
SERVER_LOG_TAIL = 400 * 1024
SERVER_LOG_POLL_SECONDS = 0.5
SERVER_LOG_HEARTBEAT_SECONDS = 15
# Proxies often buffer the first few KB of a chunked response, and some stall on one huge line.
SSE_FLUSH_BYTES = 4096
SSE_CHUNK_CHARS = 8192
_SSE_CONTROLS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
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


def sse_safe_text(text: str) -> str:
    """Drop bytes that terminate an SSE line. ``\\r`` becomes a normal newline."""
    if not text:
        return ''
    cleaned = text.replace('\r\n', '\n').replace('\r', '\n')
    return _SSE_CONTROLS.sub('', cleaned)


def sse_comment_block(label: str, min_bytes: int = SSE_FLUSH_BYTES) -> str:
    """Comment lines only. Enough bytes that a buffering proxy will forward them."""
    line = f': {label}\n'
    count = max(1, (min_bytes + len(line) - 1) // len(line))
    return line * count


def format_server_log_event(event: str, data: str, event_id: int | None = None) -> str:
    """One SSE event. A trailing newline in ``data`` survives the browser's framing."""
    lines = []
    if event_id is not None:
        lines.append(f'id: {int(event_id)}')
    lines.append(f'event: {event}')
    body = sse_safe_text(data)
    if body == '':
        lines.append('data:')
    else:
        parts = body.split('\n')
        trailing = body.endswith('\n')
        if trailing:
            parts = parts[:-1]
        for part in parts:
            lines.append(f'data: {part}')
        if trailing:
            lines.append('data:')
    framed = '\n'.join(lines) + '\n\n'
    return framed


def _event_frames(kind: str, text: str, event_id: int) -> list:
    """Split a payload so no single SSE write is a huge line."""
    body = sse_safe_text(text)
    if body == '':
        return [format_server_log_event(kind, '', event_id)]
    frames = []
    start = 0
    first = True
    while start < len(body):
        piece = body[start:start + SSE_CHUNK_CHARS]
        start += SSE_CHUNK_CHARS
        frames.append(format_server_log_event(kind if first else 'append', piece, event_id))
        first = False
    return frames


async def iter_server_log_events():
    """Tail rsi/.rsi.log as SSE: flush padding, one snapshot, then appends."""
    path = server_log_path()
    yield sse_comment_block('flush')
    window = server_log_window(path, None)
    if window['missing']:
        yield format_server_log_event('missing', 'rsi/.rsi.log 不存在\n')
        return
    for frame in _event_frames('snapshot', window['text'], window['next_offset']):
        yield frame
    offset = window['next_offset']
    last_beat = time.monotonic()
    while True:
        window = server_log_window(path, offset)
        if window['missing']:
            yield format_server_log_event('missing', 'rsi/.rsi.log 不存在\n')
            return
        if window['text'] or window['reset']:
            kind = 'snapshot' if window['reset'] else 'append'
            for frame in _event_frames(kind, window['text'], window['next_offset']):
                yield frame
            offset = window['next_offset']
            last_beat = time.monotonic()
            if window['next_offset'] < window['size']:
                continue
        elif time.monotonic() - last_beat >= SERVER_LOG_HEARTBEAT_SECONDS:
            yield ': ping\n' + sse_comment_block('flush')
            last_beat = time.monotonic()
        await asyncio.sleep(SERVER_LOG_POLL_SECONDS)


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
    <p id="mode">正在连接…</p>
    <a href="/">返回</a>
  </header>
  <pre id="log">正在加载…
</pre>
  <script>
    const view = document.getElementById('log');
    const mode = document.getElementById('mode');
    const MISSING = 'rsi/.rsi.log 不存在\n';
    const EMPTY = '日志是空的\n';
    const FAILED = '加载失败\n';
    let haveText = false;
    let saw = false;
    let polling = false;
    let offset = null;
    let pinned = true;
    let source = null;
    let pollTimer = null;
    view.addEventListener('scroll', () => {
      const gap = view.scrollHeight - view.scrollTop - view.clientHeight;
      pinned = gap < 40;
    });
    function follow() {
      if (pinned) view.scrollTop = view.scrollHeight;
    }
    function remember(ev) {
      const id = Number(ev.lastEventId);
      if (!Number.isNaN(id)) offset = id;
    }
    function paint(text, replace) {
      if (replace || !haveText) {
        view.textContent = text || EMPTY;
        haveText = Boolean(text);
      } else if (text) {
        view.textContent += text;
      }
      follow();
    }
    async function poll() {
      const url = offset == null ? '/api/server-log/stream' : '/api/server-log/stream?offset=' + offset;
      let response;
      try {
        response = await fetch(url, {headers: {accept: 'application/json'}, cache: 'no-store'});
      } catch (_err) {
        if (!saw) view.textContent = FAILED;
        return;
      }
      if (!response.ok) {
        if (!saw) view.textContent = FAILED;
        return;
      }
      const body = await response.json();
      saw = true;
      if (body.missing) {
        view.textContent = MISSING;
        offset = null;
        haveText = false;
        follow();
        return;
      }
      paint(body.text || '', Boolean(body.reset) || !haveText);
      offset = body.next_offset;
    }
    function startPoll() {
      if (polling) return;
      polling = true;
      if (source) {
        source.close();
        source = null;
      }
      mode.textContent = '定时拉取';
      poll();
      pollTimer = setInterval(poll, 1500);
    }
    const arm = setTimeout(startPoll, 3000);
    source = new EventSource('/api/server-log/events');
    function note() {
      if (polling) return;
      saw = true;
      clearTimeout(arm);
      mode.textContent = '实时推送';
    }
    source.addEventListener('snapshot', (ev) => {
      if (polling) return;
      note();
      remember(ev);
      paint(ev.data, true);
    });
    source.addEventListener('append', (ev) => {
      if (polling) return;
      note();
      remember(ev);
      paint(ev.data, false);
    });
    source.addEventListener('missing', (ev) => {
      if (polling) return;
      note();
      view.textContent = ev.data || MISSING;
      haveText = false;
      source.close();
    });
    source.onerror = () => {
      if (!polling) startPoll();
    };
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
        logger.error('审计库写入失败：%s', exc)
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

    @app.get('/server-log')
    def server_log_page():
        return HTMLResponse(_SERVER_LOG_PAGE)

    @app.get('/api/server-log/stream')
    def server_log_stream(offset: int | None = None):
        return server_log_window(server_log_path(), offset)

    @app.get('/api/server-log/events')
    async def server_log_events():
        return StreamingResponse(
            iter_server_log_events(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache, no-transform',
                'X-Accel-Buffering': 'no',
            },
        )

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

    @app.post('/api/runs/{run_id}/chat')
    async def chat(run_id: str, request: Request):
        body = await read_json(request)
        return pipeline.chat(run_id, body.get('messages') or [], body.get('temperature'))

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
        return workbench.start_export(home, run_id, workbench.promoted_checkpoint(view))

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
