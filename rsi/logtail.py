# Copyright (c) ModelScope Contributors. All rights reserved.
"""Live stage logs: tqdm carriage-return collapse, ANSI stripping, and a console tee."""
import codecs
import re
import subprocess
import sys
import time
from pathlib import Path

# View/DB payloads keep a short tail so /api/runs/{id} stays small.
TAIL_BYTES = 64 * 1024
# Live stage-log panel first paint; later polls append from next_offset.
# Still capped — use offset=0 (or「加载更早」) to page through the full file.
LIVE_TAIL_BYTES = 256 * 1024
# One request window for follow /「加载更早」chunks. Huge files stay on disk.
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


def align_start_to_newline(raw: bytes, start: int) -> tuple[int, bytes]:
    """Drop a partial first line when ``start`` is not the beginning of the file.

    Raw byte tails (e.g. last 64KB of a multi‑MB ``swift.log``) often land mid‑tqdm
    line. Skipping forward to the next ``\\n`` keeps the panel readable.
    """
    if start <= 0 or not raw:
        return start, raw
    nl = raw.find(b'\n')
    if nl < 0:
        return start, raw
    return start + nl + 1, raw[nl + 1:]


def truncate_log_text(text: str, max_bytes: int | None = None) -> str:
    """Keep only the tail of a log for API/DB payloads.

    Full training output lives in ``swift.log`` on disk and is served by the
    stage log endpoint. Embedding multi‑MB text in every ``/api/runs/{id}``
    poll (forms.logs / steps[].logs) makes the view request explode.
    """
    if not text:
        return ''
    limit = TAIL_BYTES if max_bytes is None else max(1, int(max_bytes))
    raw = text.encode('utf-8')
    if len(raw) <= limit:
        return text
    cut = len(raw) - limit
    # Avoid splitting a multi-byte UTF-8 sequence at the start of the window.
    while cut < len(raw) and raw[cut] & 0xC0 == 0x80:
        cut += 1
    cut, clipped = align_start_to_newline(raw[cut:], cut)
    return clipped.decode('utf-8', errors='replace')


def window_bounds(size: int, offset: int | None, limit: int | None, tail: int | None = None) -> tuple[int, int]:
    """Byte window. Omit ``offset`` to keep the tail. ``tail`` overrides the default tail size."""
    # Default stays TAIL_BYTES (view/DB). Live panel passes LIVE_TAIL_BYTES.
    tail_bytes = TAIL_BYTES if tail is None else max(1, int(tail))
    if limit is None or int(limit) <= 0:
        limit = tail_bytes
    limit = min(int(limit), max(MAX_READ, tail_bytes))
    if offset is None:
        window = min(tail_bytes, limit)
        start = max(0, size - window)
        return start, size - start
    start = min(max(0, int(offset)), size)
    return start, min(limit, size - start)


def _window_payload(raw: bytes, start: int, size: int, *, align: bool) -> dict:
    # Only auto-tails align: follow reads from next_offset must not drop live bytes.
    if align and start > 0:
        start, raw = align_start_to_newline(raw, start)
    return {
        'offset': start,
        'next_offset': start + len(raw),
        'size': size,
        'truncated': start > 0,
        'text': normalize_log(raw.decode('utf-8', errors='replace')),
    }


def slice_bytes(raw: bytes, offset: int | None, limit: int | None, tail: int | None = None) -> dict:
    size = len(raw)
    start, count = window_bounds(size, offset, limit, tail=tail)
    return _window_payload(raw[start:start + count], start, size, align=offset is None)


def read_log_window(path: Path | None, offset: int | None, limit: int | None, tail: int | None = None) -> dict:
    if path is None or not path.is_file():
        return slice_bytes(b'', offset, limit, tail=tail)
    try:
        size = path.stat().st_size
        start, count = window_bounds(size, offset, limit, tail=tail)
        with path.open('rb') as handle:
            handle.seek(start)
            blob = handle.read(count)
    except OSError:
        return slice_bytes(b'', offset, limit, tail=tail)
    return _window_payload(blob, start, size, align=offset is None)


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


def announce_log(prefix: str, log_path: Path):
    """One process-log line pointing at an experiment log. The stream stays in the file."""
    _write_stdout(f'{prefix} 实验日志：{log_path}')


def run_and_tee(argv: list, env: dict | None, log_file, prefix: str, interval: float = 1.0,
                echo: bool = True, on_start=None, new_session: bool = False) -> int:
    """Run ``argv`` and write combined stdout/stderr to ``log_file``.

    ``log_file`` must be a binary file. The file keeps raw bytes (including
    ``\\r`` and ANSI). When ``echo`` is true, a normalized copy is also written
    to the process stdout, with carriage-return progress throttled. Training
    stages pass ``echo=False`` so the process log is not the experiment log.

    ``on_start`` receives the ``Popen`` handle once (for stop/cancel tracking).
    ``new_session`` starts a process group so stop can SIGTERM/SIGKILL the tree.
    """
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
        start_new_session=new_session,
    )
    if on_start is not None:
        on_start(proc)
    tee = ConsoleTee(prefix, interval=interval) if echo else None
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            log_file.write(chunk)
            log_file.flush()
            if tee is not None:
                tee.feed(chunk)
        if tee is not None:
            tee.finish()
    except Exception:
        if proc.poll() is None:
            proc.kill()
        raise
    finally:
        proc.wait()
    return int(proc.returncode if proc.returncode is not None else 0)
