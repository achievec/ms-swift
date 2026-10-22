# Copyright (c) ModelScope Contributors. All rights reserved.
"""Safe extract of uploaded dataset archives and media-path rewriting."""
from __future__ import annotations

import io
import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path

from rsi.dataset_validate import classify_row, parse_rows

logger = logging.getLogger(__name__)

MEDIA_KEYS = ('images', 'videos', 'audios', 'image', 'video', 'audio')
PREFERRED_JSONL = ('data.jsonl', 'train.jsonl')
ARCHIVE_SUFFIXES = ('.zip', '.tar.gz', '.tgz', '.tar')


def is_archive_filename(filename: str) -> bool:
    name = (filename or '').lower()
    return name.endswith(ARCHIVE_SUFFIXES)


def archive_max_bytes() -> int:
    raw = os.environ.get('RSI_UPLOAD_ARCHIVE_MAX_BYTES', '').strip()
    if raw.isdigit():
        return int(raw)
    return 500 * 1024 * 1024


def text_upload_max_bytes() -> int:
    raw = os.environ.get('RSI_UPLOAD_TEXT_MAX_BYTES', '').strip()
    if raw.isdigit():
        return int(raw)
    from rsi.dataset_validate import MAX_BYTES
    return MAX_BYTES


def _is_within(root: Path, candidate: Path) -> bool:
    root = root.resolve()
    candidate = candidate.resolve()
    return candidate == root or root in candidate.parents


def _reject_member_name(name: str) -> str | None:
    if not name or name.endswith('/'):
        return None
    normalized = name.replace('\\', '/')
    if normalized.startswith('/') or normalized.startswith('../') or '/../' in f'/{normalized}/':
        return f'压缩包包含不安全路径：{name}'
    parts = Path(normalized).parts
    if any(part in {'.', '..'} for part in parts):
        return f'压缩包包含不安全路径：{name}'
    if Path(normalized).is_absolute():
        return f'压缩包包含不安全路径：{name}'
    return None


def _safe_join(dest: Path, member_name: str) -> Path:
    err = _reject_member_name(member_name)
    if err:
        raise ValueError(err)
    target = (dest / member_name.replace('\\', '/')).resolve()
    if not _is_within(dest, target):
        raise ValueError(f'压缩包包含不安全路径：{member_name}')
    return target


def _extract_zip(data: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith('/'):
                continue
            err = _reject_member_name(name)
            if err:
                raise ValueError(err)
            # Reject symlink / absolute link escapes (ZipInfo external_attr / create_system).
            if info.is_dir():
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:  # symlink
                raise ValueError(f'压缩包不允许符号链接：{name}')
            target = _safe_join(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, 'r') as src, open(target, 'wb') as out:
                out.write(src.read())


def _extract_tar(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as tf:
        for member in tf.getmembers():
            name = member.name
            if member.isdir():
                continue
            err = _reject_member_name(name)
            if err:
                raise ValueError(err)
            if member.issym() or member.islnk():
                raise ValueError(f'压缩包不允许符号链接：{name}')
            if not member.isfile():
                raise ValueError(f'压缩包包含不支持的成员类型：{name}')
            target = _safe_join(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f'无法读取压缩包成员：{name}')
            with extracted, open(target, 'wb') as out:
                out.write(extracted.read())


def extract_archive(filename: str, data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = (filename or '').lower()
    try:
        if name.endswith('.zip'):
            _extract_zip(data, dest)
        elif name.endswith(('.tar', '.tar.gz', '.tgz')):
            _extract_tar(data, dest)
        else:
            raise ValueError(f'不支持的压缩格式：{filename}')
    except (zipfile.BadZipFile, tarfile.TarError) as exc:
        raise ValueError(f'压缩包损坏或无法解析：{exc}') from exc


def _looks_like_chat_rows(path: Path) -> bool:
    try:
        text = path.read_text(encoding='utf-8')
        rows = parse_rows(text, path.name)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    if not rows:
        return False
    ok = 0
    for row in rows[:20]:
        kind, error = classify_row(row)
        if kind and not error:
            ok += 1
    return ok > 0


def find_dataset_file(extract_root: Path) -> Path:
    root = extract_root.resolve()
    jsonl_files = sorted(p for p in root.rglob('*.jsonl') if p.is_file())
    if jsonl_files:
        preferred = []
        for path in jsonl_files:
            if path.name.lower() in PREFERRED_JSONL:
                preferred.append(path)
        if preferred:
            # Prefer shallower preferred names; then larger size.
            preferred.sort(key=lambda p: (len(p.relative_to(root).parts), -p.stat().st_size, str(p)))
            chosen = preferred[0]
        else:
            jsonl_files.sort(key=lambda p: (-p.stat().st_size, str(p)))
            chosen = jsonl_files[0]
        logger.info('RSI archive dataset file chosen: %s', chosen)
        return chosen

    candidates = []
    for pattern in ('*.json', '*.txt'):
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() == '.json' and path.name.lower().endswith('.jsonl'):
                continue
            if _looks_like_chat_rows(path):
                candidates.append(path)
    if not candidates:
        raise ValueError('压缩包中未找到可用的数据集文件（需要 .jsonl，或符合对话格式的 .json / .txt）')
    candidates.sort(key=lambda p: (-p.stat().st_size, str(p)))
    chosen = candidates[0]
    logger.info('RSI archive dataset file chosen: %s', chosen)
    return chosen


def _leave_media_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    lower = text.lower()
    if lower.startswith(('http://', 'https://', 'data:')):
        return True
    return Path(text).is_absolute()


def _rewrite_media_path(value: str, extract_root: Path) -> tuple[str, bool, str | None]:
    if _leave_media_path(value):
        return value, False, None
    absolute = (extract_root / value).resolve()
    if not _is_within(extract_root, absolute):
        # Relative path that escapes extract root — keep original, warn via missing check.
        return value, False, f'媒体路径越界未改写：{value}'
    return str(absolute), True, None if absolute.is_file() else f'媒体文件不存在：{absolute}'


def _rewrite_media_node(node, extract_root: Path) -> tuple[object, bool, list[str]]:
    warnings: list[str] = []
    changed = False
    if isinstance(node, str):
        new_value, did, warning = _rewrite_media_path(node, extract_root)
        if warning:
            warnings.append(warning)
        return new_value, did, warnings
    if isinstance(node, list):
        out = []
        for item in node:
            rewritten, did, item_warnings = _rewrite_media_node(item, extract_root)
            out.append(rewritten)
            changed = changed or did
            warnings.extend(item_warnings)
        return out, changed, warnings
    return node, False, warnings


def rewrite_media_paths(rows: list, extract_root: Path) -> tuple[list, bool, list[str]]:
    extract_root = extract_root.resolve()
    changed = False
    warnings: list[str] = []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            out.append(row)
            continue
        new_row = dict(row)
        for key in MEDIA_KEYS:
            if key not in new_row:
                continue
            rewritten, did, item_warnings = _rewrite_media_node(new_row[key], extract_root)
            new_row[key] = rewritten
            changed = changed or did
            warnings.extend(item_warnings)
        out.append(new_row)
    # Deduplicate warnings while preserving order.
    seen = set()
    unique = []
    for item in warnings:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return out, changed, unique


def write_jsonl(path: Path, rows: list) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def process_extracted_dataset(extract_root: Path) -> tuple[Path, list[str]]:
    """Find dataset file, rewrite relative media paths, return resolved jsonl path."""
    extract_root = extract_root.resolve()
    source = find_dataset_file(extract_root)
    text = source.read_text(encoding='utf-8')
    rows = parse_rows(text, source.name)
    rewritten, changed, warnings = rewrite_media_paths(rows, extract_root)
    if changed or source.suffix.lower() != '.jsonl':
        resolved = extract_root / 'resolved.jsonl'
        write_jsonl(resolved, rewritten)
        return resolved, warnings
    return source, warnings
