# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_swift_logger_imports_without_modelscope():
    script = textwrap.dedent("""\
        import sys
        sys.modules['modelscope'] = None
        import swift
        from swift.utils import get_logger
        log = get_logger()
        log.info('soft-import-ok')
        from swift.utils.logger import get_ms_logger, ms_logger
        assert get_ms_logger() is ms_logger
        ms_logger.info('fallback')
        print('SOFT_OK')
    """)
    env = os.environ.copy()
    previous = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(ROOT) if not previous else f'{ROOT}:{previous}'
    completed = subprocess.run(
        [sys.executable, '-c', script], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert 'SOFT_OK' in completed.stdout


def test_installed_modelscope_logger_is_still_used():
    probe = subprocess.run(
        [sys.executable, '-c', 'import modelscope.utils.logger'],
        capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        return
    script = textwrap.dedent("""\
        import swift.utils.logger as mod
        assert mod.get_ms_logger.__module__.startswith('modelscope')
        print('MS_OK')
    """)
    env = os.environ.copy()
    previous = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(ROOT) if not previous else f'{ROOT}:{previous}'
    completed = subprocess.run(
        [sys.executable, '-c', script], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert 'MS_OK' in completed.stdout
