# Copyright (c) ModelScope Contributors. All rights reserved.
import html
import logging
import os
import sys
import traceback

import uvicorn
from fastapi.responses import HTMLResponse

from rsi.app import create_app
from rsi.model_fetch import describe_startup, kick_hope_login


def _trim_error(text: str) -> str:
    lines = (text or '').strip().splitlines() or ['未知错误']
    if len(lines) > 40:
        lines = lines[:8] + ['...'] + lines[-30:]
    trimmed = '\n'.join(lines)
    if len(trimmed) > 4000:
        trimmed = trimmed[:4000] + '\n...'
    return trimmed


def swift_ui_python() -> str:
    """Interpreter that should own Gradio. Linux prefers fetch.yaml train_python."""
    if sys.platform != 'linux':
        return sys.executable
    explicit = os.environ.get('RSI_TRAIN_PYTHON', '').strip()
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit
    try:
        from rsi.model_fetch import configured_train_python
        found = configured_train_python()
    except Exception:
        found = ''
    return found or sys.executable


def swift_ui_fallback_page(error: str) -> str:
    """Explain why /swift-ui/ did not mount. The traceback is escaped."""
    detail = html.escape(_trim_error(error))
    executable = html.escape(sys.executable)
    pip_python = html.escape(swift_ui_python())
    lowered = (error or '').lower()
    hint = ''
    if 'gradio' in lowered or 'no module named' in lowered:
        hint = (
            '<p>在 train_python 这个解释器里装上 Web-UI 依赖，然后重新执行 <code>./run.sh</code>：</p>'
            f'<pre>{pip_python} -m pip install &quot;gradio&lt;5.33&quot;</pre>'
        )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>SWIFT Web-UI</title>
  <style>
    body {{ margin: 0; font-family: "Noto Sans SC", "PingFang SC", sans-serif; background: #efe7da; color: #241c15; }}
    main {{ max-width: 720px; margin: 48px auto; padding: 0 20px; }}
    a {{ color: #0e6b60; }}
    pre {{ white-space: pre-wrap; background: #fffdf8; padding: 12px 14px; border-radius: 8px; overflow: auto; }}
  </style>
</head>
<body>
  <main>
    <h1>SWIFT Web-UI</h1>
    <p>这个页面暂时打不开。</p>
    <p>当前解释器：<code>{executable}</code></p>
    {hint}
    <p>启动时的错误（也写在 rsi/.rsi.log）：</p>
    <pre>{detail}</pre>
    <p><a href="/">返回大模型训练平台</a></p>
  </main>
</body>
</html>
"""


def _mount_gradio(gr, app, blocks):
    """Mount at /swift-ui. root_path keeps Gradio's assets on that prefix when the argument exists."""
    import inspect
    kwargs = {'path': '/swift-ui'}
    parameters = inspect.signature(gr.mount_gradio_app).parameters
    if 'root_path' in parameters:
        kwargs['root_path'] = '/swift-ui'
    mounted = gr.mount_gradio_app(app, blocks, **kwargs)
    return mounted or app


def mount_swift_ui(app) -> bool:
    """Mount the original Gradio pages at /swift-ui when swift can be imported.

    The RSI page stays at /. If the import fails, /swift-ui/ shows the
    exception and the interpreter ./run.sh selected.
    """
    try:
        import gradio as gr
        from swift.ui.app import build_blocks
        lang = os.environ.get('SWIFT_UI_LANG') or 'zh'
        blocks = build_blocks(lang)
        _mount_gradio(gr, app, blocks)
    except Exception as exc:
        app.state.swift_ui = False
        detail = _trim_error(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        app.state.swift_ui_error = detail

        @app.get('/swift-ui', response_class=HTMLResponse)
        @app.get('/swift-ui/', response_class=HTMLResponse)
        def swift_ui_unavailable():
            return swift_ui_fallback_page(app.state.swift_ui_error)

        print(f'swift web-ui 未挂载。解释器 {sys.executable}\n{detail}')
        return False
    app.state.swift_ui = True
    app.state.swift_ui_error = ''
    print('SWIFT Web-UI 已挂在 /swift-ui ，RSI 闭环仍是默认页面。')
    return True


def main():
    host = os.environ.get('WEBUI_SERVER') or os.environ.get('RSI_HOST', '0.0.0.0')
    port = int(os.environ.get('WEBUI_PORT') or os.environ.get('RSI_PORT', '8418'))
    app = create_app()
    mount_swift_ui(app)
    shown = '127.0.0.1' if host in {'0.0.0.0', '::', '[::]'} else host
    startup = describe_startup()
    print(f'RSI 启动 {startup}')
    logging.getLogger('rsi.model_fetch').info('RSI 启动 %s', startup)
    kick_hope_login()
    print(f'RSI UI: http://{shown}:{port}/  (listen {host}:{port}, backend={app.state.pipeline.backend.name})')
    uvicorn.run(app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
