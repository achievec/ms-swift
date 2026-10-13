# Copyright (c) ModelScope Contributors. All rights reserved.
"""Launch the V1 RSI page on the Web-UI port (7860)."""
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
    body { margin: 0; font-family: "Source Sans Pro", ui-sans-serif, system-ui, sans-serif; background: #fff; color: #27272a; }
    main { max-width: 720px; margin: 48px auto; padding: 0 20px; }
    a.back { color: #2563eb; }
    code { background: #f4f4f5; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <main>
    <h1>SWIFT Web-UI</h1>
    <p>原生页面的地址是 <a href="/swift-ui/"><code>http://127.0.0.1:7860/swift-ui/</code></a>，和 RSI 闭环同一个进程、同一个端口。</p>
    <p>当前环境没有加载到 swift（通常还没装 torch / transformers），所以这里还不是训练表单。装好依赖后，这个地址会打开原来的 LLM预训练/微调、人类对齐、GRPO、推理、导出、评测、采样。</p>
    <p>也可以先停止 RSI（<code>./run.sh stop</code>），再运行 <code>swift web-ui</code>。它默认也是 <code>http://127.0.0.1:7860/</code>。两边不要同时占用这个端口。</p>
    <p><a class="back" href="/">返回 RSI 闭环</a></p>
  </main>
</body>
</html>
"""


def resolve_server():
    host = os.environ.get('WEBUI_SERVER') or os.environ.get('RSI_HOST') or '127.0.0.1'
    port = int(os.environ.get('WEBUI_PORT') or os.environ.get('RSI_PORT') or '7860')
    return host, port


def mount_swift_ui(app) -> bool:
    """Mount the original Gradio pages at /swift-ui when torch/swift import.

    The RSI page at / stays the V1 theme. If this import fails, RSI still serves
    on 7860 and the header explains how to open `swift web-ui` on that same port.
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
    host, port = resolve_server()
    os.environ['WEBUI_SERVER'] = host
    os.environ['WEBUI_PORT'] = str(port)
    app = create_app()
    mount_swift_ui(app)
    print(f'RSI UI: http://{host}:{port}/  (backend={app.state.pipeline.backend.name})')
    uvicorn.run(app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
