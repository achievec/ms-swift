# Copyright (c) ModelScope Contributors. All rights reserved.
import logging
import os

import uvicorn
from fastapi.responses import HTMLResponse

from rsi.app import create_app
from rsi.model_fetch import describe_startup

_SWIFT_UI_FALLBACK = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>SWIFT Web-UI</title>
  <style>
    body { margin: 0; font-family: "Noto Sans SC", "PingFang SC", sans-serif; background: #efe7da; color: #241c15; }
    main { max-width: 720px; margin: 48px auto; padding: 0 20px; }
    a { color: #0e6b60; }
    code { background: #fffdf8; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <main>
    <h1>SWIFT Web-UI</h1>
    <p>这个页面暂时打不开。</p>
    <p><a href="/">返回大模型训练平台</a></p>
  </main>
</body>
</html>
"""


def mount_swift_ui(app) -> bool:
    """Mount the original Gradio pages at /swift-ui when swift can be imported.

    The RSI page stays at /. If the import fails, /swift-ui/ explains how to
    open `swift web-ui` on this same port after stopping RSI.
    """
    try:
        import gradio as gr
        from swift.ui.app import build_blocks
        lang = os.environ.get('SWIFT_UI_LANG') or 'zh'
        blocks = build_blocks(lang)
        gr.mount_gradio_app(app, blocks, path='/swift-ui')
    except Exception as exc:
        app.state.swift_ui = False
        app.state.swift_ui_error = str(exc)

        @app.get('/swift-ui', response_class=HTMLResponse)
        @app.get('/swift-ui/', response_class=HTMLResponse)
        def swift_ui_unavailable():
            return _SWIFT_UI_FALLBACK

        print(f'swift web-ui 未挂载（{exc}）。入口仍是 /swift-ui/ ；装好 swift 依赖后这里会换成原生表单。')
        return False
    app.state.swift_ui = True
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
    print(f'RSI UI: http://{shown}:{port}/  (listen {host}:{port}, backend={app.state.pipeline.backend.name})')
    uvicorn.run(app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
