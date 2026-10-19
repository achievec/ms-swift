# Copyright (c) ModelScope Contributors. All rights reserved.
"""OpenAI-compatible server used when serving is mocked."""
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import uvicorn
from fastapi import FastAPI, HTTPException, Request


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def port_is_free(host: str, port: int) -> bool:
    probe = '127.0.0.1' if host in {'0.0.0.0', '::'} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((probe, port))
        except OSError:
            return False
    return True


def loopback(host: str) -> str:
    return '127.0.0.1' if host in {'0.0.0.0', '::'} else host


def build_openai_app(model_name: str, api_key: str, scenario: str, checkpoint: str) -> FastAPI:
    app = FastAPI(title='RSI mock OpenAI')

    def authorize(request: Request):
        if not api_key:
            return
        header = request.headers.get('authorization', '')
        if header != f'Bearer {api_key}':
            raise HTTPException(status_code=401, detail='invalid api key')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': model_name, 'mock': True}

    @app.get('/v1/models')
    def models(request: Request):
        authorize(request)
        return {
            'object': 'list',
            'data': [{
                'id': model_name,
                'object': 'model',
                'owned_by': 'swift',
                'root': checkpoint,
            }],
        }

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        authorize(request)
        body = await request.json()
        messages = body.get('messages') or []
        user_text = ''
        for message in reversed(messages):
            if isinstance(message, dict) and message.get('role') == 'user':
                content = message.get('content')
                user_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                break
        reply = (f'【mock 推理】已挂载模型 {model_name}。'
                 f'检查点：{checkpoint}。场景：{scenario}。'
                 f'针对「{user_text}」，这里返回的是本地 Mock 文本，权重没有加载。'
                 '在 A100 上把 RSI_SERVE_BACKEND 设为 swift 后，同一条路径会换成 swift deploy。')
        now = int(time.time())
        return {
            'id': f'chatcmpl-mock-{now}',
            'object': 'chat.completion',
            'created': now,
            'model': body.get('model') or model_name,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': reply
                },
                'finish_reason': 'stop',
            }],
            'usage': {
                'prompt_tokens': max(1, len(user_text)),
                'completion_tokens': len(reply),
                'total_tokens': max(1, len(user_text)) + len(reply),
            },
        }

    @app.post('/v1/completions')
    async def completions(request: Request):
        authorize(request)
        body = await request.json()
        prompt = body.get('prompt') or ''
        text = f'【mock 推理】{model_name} / {checkpoint} / {prompt}'
        return {
            'id': 'cmpl-mock',
            'object': 'text_completion',
            'model': body.get('model') or model_name,
            'choices': [{
                'index': 0,
                'text': text,
                'finish_reason': 'stop'
            }],
        }

    return app


class _ThreadServer:

    def __init__(self, app: FastAPI, host: str, port: int):
        self.config = uvicorn.Config(app, host=host, port=port, log_level='warning', access_log=False)
        self.server = uvicorn.Server(self.config)
        self.server.install_signal_handlers = lambda: None
        self.thread = threading.Thread(target=self.server.run, name=f'rsi-serve-{port}', daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)


def wait_http(url: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f'服务未就绪：{url} ({last_error})')


class _ProcessServer:

    def __init__(self, process: subprocess.Popen):
        self.process = process

    def stop(self):
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


class ServeManager:

    def __init__(self):
        self._lock = threading.Lock()
        self._handles = {}

    def start_mock(self, run_id: str, host: str, port: int, model_name: str, api_key: str, scenario: str,
                   checkpoint: str):
        self.stop(run_id)
        app = build_openai_app(model_name, api_key or '', scenario, checkpoint)
        server = _ThreadServer(app, host, port)
        server.start()
        try:
            wait_http(f'http://{loopback(host)}:{port}/health')
        except TimeoutError:
            server.stop()
            raise
        with self._lock:
            self._handles[run_id] = server

    def track(self, run_id: str, process: subprocess.Popen):
        """Remember a real `swift deploy` process so stop can terminate it."""
        self.stop(run_id)
        with self._lock:
            self._handles[run_id] = _ProcessServer(process)

    def stop(self, run_id: str):
        with self._lock:
            server = self._handles.pop(run_id, None)
        if server is not None:
            server.stop()

    def stop_all(self):
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for server in handles:
            server.stop()
