# Copyright (c) ModelScope Contributors. All rights reserved.
import stat
import sys
from pathlib import Path

from rsi.backends import swift_launch, training_python


def _executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_fetch_yaml_train_python_is_used_when_present(tmp_path, monkeypatch):
    configured = tmp_path / 'conda' / 'bin' / 'python'
    _executable(configured)
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text(f'train_python: {configured}\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.delenv('RSI_TRAIN_PYTHON', raising=False)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    argv, _env = swift_launch(['swift', 'sft', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[0] == str(configured)
    assert argv[1:4] == ['-m', 'swift.cli.main', 'sft']
    assert training_python() == str(configured)


def test_missing_config_python_uses_the_current_interpreter(tmp_path, monkeypatch):
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('train_python: /tmp/rsi-missing-train-python/bin/python\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.delenv('RSI_TRAIN_PYTHON', raising=False)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    argv, _env = swift_launch(['swift', 'eval', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[0] == sys.executable
    assert argv[1:4] == ['-m', 'swift.cli.main', 'eval']


def test_train_python_and_swift_bin_override_fetch_yaml(tmp_path, monkeypatch):
    configured = tmp_path / 'conda' / 'bin' / 'python'
    chosen = tmp_path / 'override' / 'bin' / 'python'
    _executable(configured)
    _executable(chosen)
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text(f'train_python: {configured}\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    monkeypatch.setenv('RSI_TRAIN_PYTHON', str(chosen))
    argv, _env = swift_launch(['swift', 'rlhf', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[0] == str(chosen)
    assert argv[1:4] == ['-m', 'swift.cli.main', 'rlhf']
    script = tmp_path / 'fake-swift'
    script.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    script.chmod(0o755)
    monkeypatch.setenv('RSI_SWIFT_BIN', str(script))
    argv, _env = swift_launch(['swift', 'sft', '--model', 'x'])
    assert argv[0] == str(script)
    assert argv[1] == 'sft'


def test_bundled_train_python_is_the_gpu_conda_env():
    text = Path(__file__).resolve().parents[1].joinpath('configs', 'fetch.yaml').read_text(encoding='utf-8')
    assert 'train_python: /usr/local/conda/envs/model-train/bin/python' in text
