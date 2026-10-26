# Copyright (c) ModelScope Contributors. All rights reserved.
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from rsi.backends import MockBackend
from rsi.dataset_archive import (
    extract_archive,
    find_dataset_file,
    process_extracted_dataset,
    rewrite_media_paths,
)
from rsi.pipeline import Pipeline, PipelineError


def _pipeline(home: Path) -> Pipeline:
    return Pipeline(home, MockBackend())


def _zip_bytes(mapping: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, raw in mapping.items():
            zf.writestr(name, raw)
    return buf.getvalue()


def test_rewrite_relative_media_to_absolute(tmp_path):
    root = tmp_path / 'extract'
    (root / 'images').mkdir(parents=True)
    img = root / 'images' / 'a.jpg'
    img.write_bytes(b'fake')
    rows = [{
        'messages': [
            {'role': 'user', 'content': '<image>描述'},
            {'role': 'assistant', 'content': 'ok'},
        ],
        'images': ['images/a.jpg'],
        'videos': [['frames/1.jpg', 'frames/2.jpg']],
    }]
    rewritten, changed, warnings = rewrite_media_paths(rows, root)
    assert changed
    assert rewritten[0]['images'] == [str(img.resolve())]
    assert rewritten[0]['messages'][0]['content'] == '<image>描述'
    assert rewritten[0]['videos'][0][0].startswith(str(root.resolve()))
    assert any('不存在' in w for w in warnings)


def test_leave_absolute_and_uri(tmp_path):
    rows = [{
        'messages': [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'}],
        'images': ['/abs/x.jpg', 'https://example.com/a.png', 'data:image/png;base64,xx'],
        'image': 'http://cdn/b.jpg',
    }]
    rewritten, changed, warnings = rewrite_media_paths(rows, tmp_path)
    assert not changed
    assert rewritten[0]['images'] == rows[0]['images']
    assert rewritten[0]['image'] == rows[0]['image']
    assert warnings == []


def test_path_traversal_rejected(tmp_path):
    data = _zip_bytes({'../../etc/passwd': b'nope', 'data.jsonl': b'{}\n'})
    dest = tmp_path / 'out'
    with pytest.raises(ValueError, match='不安全路径'):
        extract_archive('evil.zip', data, dest)


def test_missing_jsonl_chinese_error(tmp_path):
    data = _zip_bytes({'readme.txt': b'hello\n', 'images/a.jpg': b'x'})
    dest = tmp_path / 'out'
    extract_archive('empty.zip', data, dest)
    with pytest.raises(ValueError, match='未找到可用的数据集'):
        find_dataset_file(dest)


def test_process_extracted_writes_resolved(tmp_path):
    root = tmp_path / 'pack'
    (root / 'images').mkdir(parents=True)
    (root / 'images' / 'a.jpg').write_bytes(b'img')
    row = {
        'messages': [
            {'role': 'user', 'content': '<image>看图'},
            {'role': 'assistant', 'content': '猫'},
        ],
        'images': ['images/a.jpg'],
    }
    (root / 'data.jsonl').write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    data = _zip_bytes({
        'data.jsonl': (root / 'data.jsonl').read_bytes(),
        'images/a.jpg': b'img',
    })
    extract_dir = tmp_path / 'extract'
    extract_archive('demo.zip', data, extract_dir)
    resolved, warnings = process_extracted_dataset(extract_dir)
    assert resolved.name == 'resolved.jsonl'
    assert warnings == []
    loaded = json.loads(resolved.read_text(encoding='utf-8').splitlines()[0])
    assert Path(loaded['images'][0]).is_file()
    assert loaded['images'][0].startswith(str(extract_dir.resolve()))


def test_save_upload_zip_and_plain_jsonl(tmp_path):
    pipeline = _pipeline(tmp_path / 'home')
    row = {
        'messages': [
            {'role': 'user', 'content': '<image>看'},
            {'role': 'assistant', 'content': '答'},
        ],
        'images': ['images/a.jpg'],
    }
    zdata = _zip_bytes({
        'data.jsonl': (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'),
        'images/a.jpg': b'img',
    })
    result = pipeline.save_upload('demo.zip', zdata)
    path = Path(result['path'])
    assert path.is_file()
    assert result['validation']['ok'] is True
    loaded = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
    assert Path(loaded['images'][0]).is_absolute()

    plain = (json.dumps({
        'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'ok'},
        ],
    }, ensure_ascii=False) + '\n').encode('utf-8')
    plain_result = pipeline.save_upload('plain.jsonl', plain)
    assert Path(plain_result['path']).is_file()
    assert plain_result['validation']['ok'] is True


def test_save_upload_zip_without_dataset_errors(tmp_path):
    pipeline = _pipeline(tmp_path / 'home')
    with pytest.raises(PipelineError, match='未找到可用的数据集'):
        pipeline.save_upload('empty.zip', _zip_bytes({'note.txt': b'x'}))


def test_tar_gz_extract(tmp_path):
    row = {
        'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'ok'},
        ],
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        raw = (json.dumps(row) + '\n').encode('utf-8')
        info = tarfile.TarInfo(name='train.jsonl')
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))
    pipeline = _pipeline(tmp_path / 'home')
    result = pipeline.save_upload('pack.tgz', buf.getvalue())
    assert result['validation']['ok'] is True
