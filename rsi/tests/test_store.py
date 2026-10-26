# Copyright (c) ModelScope Contributors. All rights reserved.
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi.store import Store, StoreError, _CONNECT_TIMEOUT_SECONDS, db_error_message


def test_store_creates_parent_and_uses_shared_fs_pragmas(tmp_path, monkeypatch):
    seen = {}
    real_connect = sqlite3.connect

    def wrapped(path, *args, **kwargs):
        seen['path'] = path
        seen['timeout'] = kwargs.get('timeout')
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', wrapped)
    db = tmp_path / 'nested' / 'audit' / 'rsi.sqlite'
    store = Store(db)
    assert db.parent.is_dir()
    assert store.db_path == db.resolve()
    assert Path(seen['path']) == db.resolve()
    assert seen['timeout'] == _CONNECT_TIMEOUT_SECONDS
    assert seen['timeout'] >= 5
    with store.transaction() as conn:
        journal = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        temp_store = conn.execute('PRAGMA temp_store').fetchone()[0]
    assert journal == 'truncate'
    assert journal != 'wal'
    assert int(temp_store) == 2


def test_operational_error_names_the_database(tmp_path, monkeypatch):
    store = Store(tmp_path / 'workspace' / 'rsi.sqlite')
    real_connect = store._connect

    class _CommitFails:
        def __init__(self, conn):
            self._conn = conn

        def commit(self):
            raise sqlite3.OperationalError('unable to open database file')

        def rollback(self):
            self._conn.rollback()

        def close(self):
            self._conn.close()

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(store, '_connect', lambda: _CommitFails(real_connect()))
    with pytest.raises(StoreError) as caught:
        store.list_runs()
    text = str(caught.value)
    assert str(store.db_path) in text
    assert '实验记录库无法写入' in text
    assert '目录不可写' in text
    assert '磁盘满' in text
    assert '可设置 RSI_HOME' in text
    assert 'dolphinfs' in text
    assert 'unable to open database file' in text


def test_api_returns_store_error_as_json(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    home = tmp_path / 'home'
    app = create_app(home)
    message = db_error_message(home / 'rsi.sqlite', sqlite3.OperationalError('unable to open database file'))

    def boom():
        raise StoreError(message)

    app.state.pipeline.store.list_runs = boom
    with TestClient(app, raise_server_exceptions=False) as api:
        response = api.get('/api/runs')
    assert response.status_code == 500, response.text
    assert response.headers['content-type'].startswith('application/json')
    detail = response.json()['detail']
    assert '实验记录库无法写入' in detail
    assert str((home / 'rsi.sqlite').resolve()) in detail
    assert 'dolphinfs' in detail
    assert 'Traceback' not in response.text
