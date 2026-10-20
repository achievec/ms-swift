# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from rsi.backends import BackendError, resolve_training_python, swift_launch


def _python(path: Path, numpy_ok: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "-c" ] && [ "$2" = "import numpy" ]; then\n'
        f'  exit {"0" if numpy_ok else "1"}\n'
        'fi\n'
        'exit 0\n',
        encoding='utf-8',
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_root_venv_beats_the_lean_ui_interpreter(tmp_path, monkeypatch):
    import rsi.backends as backends
    root = tmp_path / 'repo'
    full = root / '.venv' / 'bin' / 'python'
    lean = tmp_path / 'rsi-venv' / 'bin' / 'python'
    _python(full, True)
    _python(lean, False)
    monkeypatch.setattr(backends, 'REPO_ROOT', root)
    monkeypatch.setattr(backends.sys, 'executable', str(lean))
    monkeypatch.delenv('RSI_TRAIN_PYTHON', raising=False)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    argv, _env = swift_launch(['swift', 'sft', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[0] == str(full)
    assert argv[1:4] == ['-m', 'swift.cli.main', 'sft']
    assert resolve_training_python() == str(full)


def test_current_interpreter_is_used_when_it_has_numpy(tmp_path, monkeypatch):
    import rsi.backends as backends
    root = tmp_path / 'empty-repo'
    root.mkdir()
    good = tmp_path / 'current' / 'bin' / 'python'
    _python(good, True)
    monkeypatch.setattr(backends, 'REPO_ROOT', root)
    monkeypatch.setattr(backends.sys, 'executable', str(good))
    monkeypatch.delenv('RSI_TRAIN_PYTHON', raising=False)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    assert resolve_training_python() == str(good)


def test_train_python_override_and_swift_bin(tmp_path, monkeypatch):
    import rsi.backends as backends
    root = tmp_path / 'repo'
    full = root / '.venv' / 'bin' / 'python'
    chosen = tmp_path / 'train' / 'bin' / 'python'
    broken = tmp_path / 'bad' / 'bin' / 'python'
    _python(full, True)
    _python(chosen, True)
    _python(broken, False)
    monkeypatch.setattr(backends, 'REPO_ROOT', root)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    monkeypatch.setenv('RSI_TRAIN_PYTHON', str(chosen))
    argv, _env = swift_launch(['swift', 'eval', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[0] == str(chosen)
    assert argv[1:4] == ['-m', 'swift.cli.main', 'eval']
    monkeypatch.setenv('RSI_TRAIN_PYTHON', str(broken))
    with pytest.raises(BackendError, match='无法 import numpy') as caught:
        resolve_training_python()
    assert 'RSI_TRAIN_PYTHON' in str(caught.value)
    assert 'rsi/.venv' in str(caught.value)
    script = tmp_path / 'fake-swift'
    script.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    script.chmod(0o755)
    monkeypatch.setenv('RSI_SWIFT_BIN', str(script))
    argv, _env = swift_launch(['swift', 'sft', '--model', 'x'])
    assert argv[0] == str(script)
    assert argv[1] == 'sft'


def test_missing_training_python_fails_before_launch(tmp_path, monkeypatch):
    import rsi.backends as backends
    root = tmp_path / 'repo'
    lean = tmp_path / 'ui' / 'bin' / 'python'
    _python(root / '.venv' / 'bin' / 'python', False)
    _python(lean, False)
    monkeypatch.setattr(backends, 'REPO_ROOT', root)
    monkeypatch.setattr(backends.sys, 'executable', str(lean))
    monkeypatch.delenv('RSI_TRAIN_PYTHON', raising=False)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    with pytest.raises(BackendError, match='找不到可训练的 Python') as caught:
        swift_launch(['swift', 'sft', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    text = str(caught.value)
    assert 'import numpy' in text
    assert 'pip install -e .' in text
    assert 'RSI_TRAIN_PYTHON' in text
    assert 'rsi/.venv' in text


def test_run_sh_prefers_root_venv_with_numpy(tmp_path):
    root = tmp_path / 'repo'
    full = root / '.venv' / 'bin' / 'python'
    lean = root / 'rsi' / '.venv' / 'bin' / 'python'
    _python(full, True)
    lean.parent.mkdir(parents=True, exist_ok=True)
    lean.write_text('#!/bin/sh\nexit 1\n', encoding='utf-8')
    lean.chmod(0o755)
    script = Path(__file__).resolve().parents[2] / 'run.sh'
    env = os.environ.copy()
    env.pop('VIRTUAL_ENV', None)
    chosen = subprocess.run(
        ['bash', '-c', 'source "$1"; ROOT="$2"; RSI_DIR="$2/rsi"; unset VIRTUAL_ENV; resolve_python >&2; printf %s "$PY"',
         'bash', str(script), str(root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert chosen.returncode == 0, chosen.stderr
    assert chosen.stdout == str(full)

    full.write_text('#!/bin/sh\nexit 1\n', encoding='utf-8')
    full.chmod(0o755)
    fallback = subprocess.run(
        ['bash', '-c', 'source "$1"; ROOT="$2"; RSI_DIR="$2/rsi"; unset VIRTUAL_ENV; resolve_python >&2; printf %s "$PY"',
         'bash', str(script), str(root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert fallback.returncode == 0, fallback.stderr
    assert fallback.stdout == str(lean)
