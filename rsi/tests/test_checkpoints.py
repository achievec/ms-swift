# Copyright (c) ModelScope Contributors. All rights reserved.
from pathlib import Path

from rsi.backends import find_checkpoint
from rsi.model_fetch import resolve_rsi_home


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_find_checkpoint_reads_versioned_swift_output(tmp_path):
    output = tmp_path / 'sft'
    version = _mkdir(output / 'v1-20261020-122840' / 'checkpoint-1')
    (version / 'args.json').write_text('{}', encoding='utf-8')
    assert find_checkpoint(output) == version


def test_find_checkpoint_prefers_newer_version_and_higher_step(tmp_path):
    output = tmp_path / 'sft'
    _mkdir(output / 'v1-20261020-122840' / 'checkpoint-3')
    newer = _mkdir(output / 'v2-20261020-130000' / 'checkpoint-1')
    _mkdir(output / 'checkpoint-9')
    assert find_checkpoint(output) == newer
    later_step = _mkdir(output / 'v2-20261020-130000' / 'checkpoint-4')
    assert find_checkpoint(output) == later_step


def test_find_checkpoint_uses_last_model_checkpoint_from_the_log(tmp_path):
    output = tmp_path / 'sft'
    first = _mkdir(output / 'v1-20261020-122840' / 'checkpoint-1')
    _mkdir(output / 'v1-20261020-122840' / 'checkpoint-5')
    logs = f'last_model_checkpoint: {first}\ntrain finished\n'
    assert find_checkpoint(output, logs) == first
    outside = _mkdir(tmp_path / 'elsewhere' / 'checkpoint-9')
    ignored = f'last_model_checkpoint: {outside}\n'
    assert find_checkpoint(output, ignored).name == 'checkpoint-5'


def test_rsi_home_env_wins_and_missing_dolphinfs_falls_back(tmp_path, monkeypatch):
    fallback = tmp_path / 'repo-workspace'
    monkeypatch.setenv('RSI_HOME', str(tmp_path / 'from-env'))
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('rsi_home: /opt/meituan/dolphinfs_liubo108/ms-swift/workspace\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    assert resolve_rsi_home(fallback) == (tmp_path / 'from-env')

    monkeypatch.delenv('RSI_HOME', raising=False)
    assert resolve_rsi_home(fallback) == fallback

    parent = tmp_path / 'dolphin'
    parent.mkdir()
    configured = parent / 'workspace'
    cfg.write_text(f'rsi_home: {configured}\n', encoding='utf-8')
    chosen = resolve_rsi_home(fallback)
    assert chosen == configured
    assert configured.is_dir()
    assert not (configured / '.rsi-write-probe').exists()
