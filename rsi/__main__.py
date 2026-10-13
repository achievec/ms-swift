# Copyright (c) ModelScope Contributors. All rights reserved.
"""Launch RSI and the SWIFT Web-UI as one Gradio app on the Web-UI port."""
import os

from rsi.gradio_app import launch_standalone, resolve_server


def main():
    host, port = resolve_server()
    os.environ['WEBUI_SERVER'] = host
    os.environ['WEBUI_PORT'] = str(port)
    try:
        from swift.ui.app import webui_main
    except Exception as exc:
        print(f'swift web-ui 未加载（{exc}）。RSI 闭环仍在 {host}:{port}，其余标签会在装好 swift 依赖后出现。')
        launch_standalone(host, port)
        return
    print(f'SWIFT WebUI + RSI: http://{host}:{port}  （默认标签：RSI闭环）')
    webui_main()


if __name__ == '__main__':
    main()
