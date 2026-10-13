# Copyright (c) ModelScope Contributors. All rights reserved.
"""Launch the V1 RSI page on the Web-UI port (7860)."""
import os

import uvicorn

from rsi.app import create_app


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
        print(f'swift web-ui 未挂载（{exc}）。RSI 闭环仍使用本页面；原训练页请先停止本服务再运行 swift web-ui（同一默认端口）。')
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
