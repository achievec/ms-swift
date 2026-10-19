# Copyright (c) ModelScope Contributors. All rights reserved.
"""Live stage logs: tqdm carriage-return collapse, ANSI stripping, and a console tee."""
import codecs
import re
import subprocess
import sys
import time
from pathlib import Path

# First paint shows the tail of the file. Later polls can ask for a byte offset.
TAIL_BYTES = 64 * 1024
MAX_READ = 256 * 1024

# CSI / OSC, plus leftover tqdm sequences whose ESC byte was already consumed (`[A`, `[2K`, `[32m`).
_ANSI_RE = re.compile(
    r'\x1b\[[0-?]*[ -/]*[@-~]'
    r'|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?'
    r'|\x1b[@-_]'
    r'|\[[0-9;?]*m'
    r'|\[[0-9;?]*[ABCDEFGHJKSTf]'
)


def strip_ansi(text: str) -> str:
    if not text:
        return ''
    return _ANSI_RE.sub('', text)


def collapse_cr(text: str) -> str:
    """Keep the latest snapshot of each carriage-return progress line.

    ``\\r\\n`` stays a normal newline. A lone ``\\r`` overwrites the current line,
    which is how tqdm redraws a bar. A trailing ``\\r`` with nothing after it keeps
    the last snapshot so a poll mid-update is not blank.
    """
    if not text:
        return ''
    ended_nl = text.endswith('\n')
    normalized = text.replace('\r\n', '\n')
    lines = []
    for part in normalized.split('\n'):
        if '\r' in part:
            segments = part.split('\r')
            if segments[-1] == '':
                segments = segments[:-1]
            part = segments[-1] if segments else ''
        lines.append(part)
    if ended_nl and lines and lines[-1] == '':
        lines.pop()
        body = '\n'.join(lines)
        return (body + '\n') if body else ''
    return '\n'.join(lines)


def normalize_log(text: str) -> str:
    return collapse_cr(strip_ansi(text or ''))


def window_bounds(size: int, offset: int | None, limit: int | None) -> tuple[int, int]:
    if limit is None or int(limit) <= 0:
        limit = TAIL_BYTES
    limit = min(int(limit), MAX_READ)
    if offset is None:
        window = min(TAIL_BYTES, limit)
        start = max(0, size - window)
        return start, size - start
    start = min(max(0, int(offset)), size)
    return start, min(limit, size - start)


def slice_bytes(raw: bytes, offset: int | None, limit: int | None) -> dict:
    size = len(raw)
    start, count = window_bounds(size, offset, limit)
    blob = raw[start:start + count]
    return {
        'offset': start,
        'next_offset': start + len(blob),
        'size': size,
        'text': normalize_log(blob.decode('utf-8', errors='replace')),
    }


def read_log_window(path: Path | None, offset: int | None, limit: int | None) -> dict:
    if path is None or not path.is_file():
        return slice_bytes(b'', offset, limit)
    try:
        size = path.stat().st_size
        start, count = window_bounds(size, offset, limit)
        with path.open('rb') as handle:
            handle.seek(start)
            blob = handle.read(count)
    except OSError:
        return slice_bytes(b'', offset, limit)
    text = normalize_log(blob.decode('utf-8', errors='replace'))
    return {'offset': start, 'next_offset': start + len(blob), 'size': size, 'text': text}


class ConsoleTee:
    """Copy subprocess output to the server console with a stage prefix.

    Newline-terminated lines are always printed. Carriage-return progress
    snapshots are printed at most once per ``interval`` seconds, and the latest
    snapshot is flushed when the stream ends.
    """

    def __init__(self, prefix: str, interval: float = 1.0, now=None, write=None):
        self.prefix = prefix
        self.interval = interval
        self.now = now or time.monotonic
        self.write = write or _write_stdout
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.line = ''
        self.pending_cr = False
        self.held = None
        self.last_progress = None

    def feed(self, data: bytes):
        if data:
            self._push(self.decoder.decode(data))

    def finish(self):
        self._push(self.decoder.decode(b'', final=True))
        if self.pending_cr:
            self._progress(self.line)
            self.line = ''
            self.pending_cr = False
        if self.line:
            self._emit_line(self.line)
            self.line = ''
        elif self.held:
            self.write(f'{self.prefix} {self.held}')
            self.held = None

    def _push(self, text: str):
        for char in text:
            if char == '\r':
                self.pending_cr = True
                continue
            if char == '\n':
                if self.pending_cr and not self.line:
                    self.pending_cr = False
                    if self.held:
                        self.write(f'{self.prefix} {self.held}')
                        self.held = None
                    continue
                self.pending_cr = False
                self._emit_line(self.line)
                self.line = ''
                continue
            if self.pending_cr:
                self._progress(self.line)
                self.line = ''
                self.pending_cr = False
            self.line += char

    def _progress(self, text: str):
        text = strip_ansi(text).strip()
        if not text:
            return
        self.held = text
        now = self.now()
        if self.last_progress is not None and now - self.last_progress < self.interval:
            return
        self.last_progress = now
        self.held = None
        self.write(f'{self.prefix} {text}')

    def _emit_line(self, text: str):
        self.held = None
        text = strip_ansi(text).strip()
        if text:
            self.write(f'{self.prefix} {text}')


def _write_stdout(line: str):
    try:
        sys.stdout.write(line + '\n')
        sys.stdout.flush()
    except (OSError, ValueError):
        return


def run_and_tee(argv: list, env: dict | None, log_file, prefix: str, interval: float = 1.0) -> int:
    """Run ``argv``, write combined stdout/stderr to ``log_file``, and echo it.

    ``log_file`` must be a binary file. The file keeps raw bytes (including
    ``\\r`` and ANSI). The console copy is normalized and progress is throttled.
    """
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )
    tee = ConsoleTee(prefix, interval=interval)
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            log_file.write(chunk)
            log_file.flush()
            tee.feed(chunk)
        tee.finish()
    except Exception:
        if proc.poll() is None:
            proc.kill()
        raise
    finally:
        proc.wait()
    return int(proc.returncode if proc.returncode is not None else 0)
