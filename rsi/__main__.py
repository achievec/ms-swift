# Copyright (c) ModelScope Contributors. All rights reserved.
import os

import uvicorn

from rsi.app import create_app


def main():
    host = os.environ.get('WEBUI_SERVER') or os.environ.get('RSI_HOST', '127.0.0.1')
    port = int(os.environ.get('WEBUI_PORT') or os.environ.get('RSI_PORT', '7860'))
    app = create_app()
    print(f'RSI UI: http://{host}:{port}  (backend={app.state.pipeline.backend.name})')
    uvicorn.run(app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
