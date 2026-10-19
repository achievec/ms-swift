# Copyright (c) ModelScope Contributors. All rights reserved.
import os

import uvicorn
from fastapi.responses import HTMLResponse

from rsi.app import create_app

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
    <p>原生页面的地址是 <a href="/swift-ui/"><code>/swift-ui/</code></a>，和 RSI 闭环同一个进程、同一个端口（默认 7860）。</p>
    <p>当前进程没有挂上 Gradio 训练页（通常还没装 torch / transformers / modelscope）。依赖装好后，这个地址会打开原来的 LLM 预训练/微调、人类对齐、GRPO、推理、导出、评测、采样。</p>
    <p>也可以先 <code>./run.sh stop</code>，再单独运行 <code>swift web-ui</code>。它默认同样占用 <code>http://127.0.0.1:7860/</code>，不要和 RSI 同时开。</p>
    <p><a href="/">返回 RSI 闭环</a></p>
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
    host = os.environ.get('WEBUI_SERVER') or os.environ.get('RSI_HOST', '127.0.0.1')
    port = int(os.environ.get('WEBUI_PORT') or os.environ.get('RSI_PORT', '7860'))
    app = create_app()
    mount_swift_ui(app)
    print(f'RSI UI: http://{host}:{port}  (backend={app.state.pipeline.backend.name})')
    uvicorn.run(app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
