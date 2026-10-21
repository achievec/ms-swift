# Copyright (c) ModelScope Contributors. All rights reserved.
"""Live TensorBoard training curves for RSI stage panels.

Mirrors ``swift.ui.llm_train.runtime.Runtime`` plot specs and the
``read_tensorboard_file`` / ``tensorboard_smoothing`` path Web-UI uses.
RSI exposes a small JSON summary instead of matplotlib figures.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Same keys / smooth factors as Web-UI Runtime.sft_plot / grpo_plot.
SFT_PLOT = [
    {'name': 'train/loss', 'smooth': 0.9},
    {'name': 'train/acc', 'smooth': None},
    {'name': 'train/learning_rate', 'smooth': None},
    {'name': 'eval/loss', 'smooth': 0.9},
    {'name': 'eval/acc', 'smooth': None},
]

GRPO_PLOT = [
    {'name': 'train/loss', 'smooth': 0.9},
    {'name': 'train/reward', 'smooth': 0.9},
    {'name': 'train/learning_rate', 'smooth': None},
    {'name': 'train/completions/mean_length', 'smooth': 0.9},
    {'name': 'train/kl', 'smooth': 0.9},
]

STAGE_PLOTS = {
    'sft': SFT_PLOT,
    'rl': GRPO_PLOT,
}

# Cap points per series so the metrics endpoint stays light.
MAX_POINTS = 400


def plot_specs(stage: str) -> list[dict[str, Any]]:
    return list(STAGE_PLOTS.get(stage) or [])


def find_event_file(roots: list[Path | str | None]) -> Path | None:
    """First ``events.out*`` under known logging locations (Web-UI style).

    Search order per root: the root itself, then ``runs/``, then a shallow walk
    for version folders such as ``v1-*/runs``.
    """
    seen: set[Path] = set()
    for raw in roots:
        if not raw:
            continue
        root = Path(raw)
        if not root.is_dir():
            continue
        for candidate in _event_candidates(root):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
    return None


def _event_candidates(root: Path):
    yield from _events_in(root)
    runs = root / 'runs'
    if runs.is_dir():
        yield from _events_in(runs)
        for child in sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if child.is_dir():
                yield from _events_in(child)
    # Versioned swift output: output_dir/v1-…/runs/events.out*
    try:
        children = sorted(
            (p for p in root.iterdir() if p.is_dir() and p.name.startswith('v')),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
    except OSError:
        children = []
    for child in children[:8]:
        yield from _events_in(child)
        nested = child / 'runs'
        if nested.is_dir():
            yield from _events_in(nested)


def _events_in(folder: Path):
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for name in sorted(names):
        if name.startswith('events.out'):
            path = folder / name
            if path.is_file():
                yield path


def _resolve_tag(name: str, data: dict) -> str | None:
    """Match Web-UI acc aliases (token_acc / seq_acc)."""
    if name in data:
        return name
    if name == 'train/acc':
        for alt in ('train/token_acc', 'train/seq_acc'):
            if alt in data:
                return alt
    if name == 'eval/acc':
        for alt in ('eval/token_acc', 'eval/seq_acc'):
            if alt in data:
                return alt
    return None


def _downsample(steps: list, values: list, limit: int = MAX_POINTS) -> tuple[list, list]:
    n = len(steps)
    if n <= limit or limit < 2:
        return steps, values
    # Keep first, last, and evenly spaced middle points.
    idxs = [0]
    span = n - 1
    for i in range(1, limit - 1):
        idxs.append(round(i * span / (limit - 1)))
    idxs.append(n - 1)
    uniq = sorted(set(idxs))
    return [steps[i] for i in uniq], [values[i] for i in uniq]


def read_tb_scalars(event_path: Path | None) -> dict[str, list[dict]]:
    if event_path is None or not event_path.is_file():
        return {}
    try:
        from swift.utils import read_tensorboard_file
    except Exception:
        try:
            from swift.utils.tb_utils import read_tensorboard_file
        except Exception:
            return {}
    try:
        return read_tensorboard_file(str(event_path))
    except Exception:
        return {}


def series_for_stage(stage: str, event_path: Path | None = None, data: dict | None = None) -> list[dict]:
    """Build the chart payload for one stage. Empty series keep placeholder cards."""
    specs = plot_specs(stage)
    if not specs:
        return []
    raw = data if data is not None else read_tb_scalars(event_path)
    try:
        from swift.utils import tensorboard_smoothing
    except Exception:
        try:
            from swift.utils.tb_utils import tensorboard_smoothing
        except Exception:
            tensorboard_smoothing = None

    out = []
    for spec in specs:
        name = spec['name']
        smooth = spec.get('smooth')
        tag = _resolve_tag(name, raw) if raw else None
        steps: list = []
        values: list = []
        smoothed: list | None = None
        if tag and tag in raw:
            points = raw[tag] or []
            steps = [int(p['step']) for p in points]
            values = [float(p['value']) for p in points]
            steps, values = _downsample(steps, values)
            if smooth is not None and values and tensorboard_smoothing is not None:
                smoothed = [float(v) for v in tensorboard_smoothing(values, float(smooth))]
        item = {
            'name': name,
            'tag': tag or name,
            'smooth': smooth,
            'steps': steps,
            'values': values,
        }
        if smoothed is not None:
            item['smoothed'] = smoothed
        out.append(item)
    return out


def collect_stage_metrics(stage: str, search_roots: list[Path | str | None]) -> dict:
    """Return a ready-to-JSON metrics summary for the stage panel."""
    specs = plot_specs(stage)
    if not specs:
        return {'stage': stage, 'event_file': None, 'series': []}
    event = find_event_file(search_roots)
    return {
        'stage': stage,
        'event_file': str(event) if event else None,
        'series': series_for_stage(stage, event),
    }
