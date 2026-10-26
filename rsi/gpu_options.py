# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPU multiselect choices for RSI forms.

Values stay CUDA device ids (``0``, ``1``, …) or ``cpu`` so
``CUDA_VISIBLE_DEVICES`` mapping is unchanged. Labels show the index plus
the nvidia-smi product name when available, so the chip is not a bare ``0``.
"""
from __future__ import annotations

import re

# ``GPU 0: NVIDIA A100-SXM4-40GB (UUID: …)`` from ``nvidia-smi -L``.
_DEVICE_LINE = re.compile(
    r'^GPU\s+(\d+)\s*:\s*(.+?)(?:\s*\(UUID:.*\))?\s*$',
    re.IGNORECASE,
)

GPU_INFO = '选择要用的 GPU（编号与型号）；CUDA 不可用时只能选 CPU'


def parse_device_line(line: str) -> dict | None:
    """Pull index + product name from one ``nvidia-smi -L`` line."""
    match = _DEVICE_LINE.match((line or '').strip())
    if not match:
        return None
    return {'index': match.group(1), 'name': match.group(2).strip()}


def gpu_choices(
    devices=None,
    *,
    device_count: int | None = None,
    ensure_gpu_zero: bool = True,
) -> list[dict]:
    """Build ``{value, label}`` entries for the GPU chip field.

    ``ensure_gpu_zero`` keeps a synthetic ``GPU 0`` when the probe is empty —
    stage / export forms historically always offered ``0`` as the default.
    Inference with no CUDA devices uses ``ensure_gpu_zero=False`` so only CPU
    appears.
    """
    entries: list[dict] = []
    seen: set[str] = set()
    lines = [str(item) for item in (devices or []) if str(item).strip()]
    if lines:
        for line in lines:
            parsed = parse_device_line(line)
            if not parsed:
                continue
            value = parsed['index']
            if value in seen:
                continue
            name = parsed['name']
            label = f'GPU {value} · {name}' if name else f'GPU {value}'
            entries.append({'value': value, 'label': label})
            seen.add(value)
    else:
        count = max(0, int(device_count or 0))
        for index in range(count):
            value = str(index)
            entries.append({'value': value, 'label': f'GPU {value}'})
            seen.add(value)
    if ensure_gpu_zero and '0' not in seen:
        entries.insert(0, {'value': '0', 'label': 'GPU 0'})
        seen.add('0')
    entries.append({'value': 'cpu', 'label': 'CPU'})
    return entries


def gpu_default(choices: list[dict]) -> str:
    """Prefer the first CUDA id; fall back to CPU when that is all there is."""
    for item in choices:
        value = str(item.get('value') or '')
        if value and value != 'cpu':
            return value
    return 'cpu'
