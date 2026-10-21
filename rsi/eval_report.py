# Copyright (c) ModelScope Contributors. All rights reserved.
"""Read the evalscope report swift already wrote under a stage's eval_output_dir."""
import json
import re
from pathlib import Path

_BLEU = re.compile(r'^(?:mean_)?[Bb]leu[-_](\d+)$')
_ROUGE = re.compile(r'^(?:mean_)?[Rr]ouge-([12Ll])-([RPFrpf])$')
_STAT = {'r': 'r', 'p': 'p', 'f': 'f', 'recall': 'r', 'precision': 'p', 'f1': 'f'}
_SCORE_ORDER = (
    'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4',
    'rouge-1-r', 'rouge-1-p', 'rouge-1-f',
    'rouge-2-r', 'rouge-2-p', 'rouge-2-f',
    'rouge-l-r', 'rouge-l-p', 'rouge-l-f',
)
_MAX_REPORT_BYTES = 2_000_000


def canonical_score_name(raw: str) -> str:
    text = (raw or '').strip()
    bleu = _BLEU.fullmatch(text)
    if bleu:
        return f'bleu-{int(bleu.group(1))}'
    rouge = _ROUGE.fullmatch(text)
    if rouge:
        return f'rouge-{rouge.group(1).lower()}-{rouge.group(2).lower()}'
    return ''


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric_name(metric: dict) -> str:
    for key in ('legacy_name', 'name'):
        name = canonical_score_name(str(metric.get(key) or ''))
        if name:
            return name
    identity = metric.get('identity') if isinstance(metric.get('identity'), dict) else {}
    kind = str(identity.get('name') or '')
    dimensions = identity.get('dimensions') if isinstance(identity.get('dimensions'), dict) else {}
    if kind == 'bleu' and dimensions.get('ngram') is not None:
        return f'bleu-{int(dimensions["ngram"])}'
    if kind == 'rouge':
        variant = dimensions.get('variant', dimensions.get('ngram'))
        statistic = _STAT.get(str(dimensions.get('statistic') or 'f1').lower())
        if variant is not None and statistic:
            variant_text = 'l' if str(variant).lower() == 'l' else str(variant)
            return f'rouge-{variant_text}-{statistic}'
    return ''


def _metric_score(metric: dict):
    score = _number(metric.get('score'))
    if score is not None:
        return score
    found = []
    for category in metric.get('categories') or []:
        if not isinstance(category, dict):
            continue
        for subset in category.get('subsets') or []:
            if isinstance(subset, dict):
                number = _number(subset.get('score'))
                if number is not None:
                    found.append(number)
    if len(found) == 1:
        return found[0]
    return None


def scores_from_report(data: dict) -> dict:
    """Map one evalscope report object to canonical score names. Missing names are omitted."""
    if not isinstance(data, dict):
        return {}
    scores = {}
    for metric in data.get('metrics') or []:
        if not isinstance(metric, dict):
            continue
        name = _metric_name(metric)
        score = _metric_score(metric)
        if name and score is not None:
            scores[name] = score
    return scores


def _ordered(scores: dict) -> dict:
    ordered = {name: scores[name] for name in _SCORE_ORDER if name in scores}
    for name in sorted(scores):
        if name not in ordered:
            ordered[name] = scores[name]
    return ordered


def newest_reports_dir(eval_output_dir) -> Path | None:
    root = Path(eval_output_dir) if eval_output_dir else None
    if root is None or not root.is_dir():
        return None
    found = []
    for path in root.rglob('reports'):
        if not path.is_dir():
            continue
        # Dual-eval writes 基座 under ``eval_output/base/``; do not mix it into
        # the stage-checkpoint report when scanning the parent eval_output.
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == 'base':
            continue
        found.append(path)
    if not found:
        return None
    # Prefer wall-clock mtime so a freshly written evalscope report wins over an
    # older stub (e.g. mock ``native/mock/reports`` vs ``native/<timestamp>/reports``).
    return max(found, key=lambda path: (path.stat().st_mtime, path.parent.name))


def _load_json(path: Path) -> dict:
    try:
        if path.stat().st_size > _MAX_REPORT_BYTES:
            return {}
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def inside_directory(root, candidate) -> Path | None:
    """Resolved file inside root, or None."""
    if not root or not candidate:
        return None
    try:
        base = Path(root).resolve()
        path = Path(candidate).resolve()
    except OSError:
        return None
    if path != base and base not in path.parents:
        return None
    if not path.is_file():
        return None
    return path


def read_eval_report(eval_output_dir, root=None) -> dict:
    """Scores plus report paths from the newest evalscope ``reports`` directory.

    ``root`` limits which files are returned. Scores are only those written in the JSON.
    """
    reports = newest_reports_dir(eval_output_dir)
    if reports is None:
        return {}
    scores = {}
    json_path = None
    json_files = sorted(path for path in reports.rglob('*.json') if path.is_file())
    for path in json_files:
        found = scores_from_report(_load_json(path))
        if not found:
            continue
        if json_path is None or path.stem == 'general_qa':
            json_path = path
        if len(json_files) == 1 or path.stem == 'general_qa':
            scores.update(found)
        else:
            for name, score in found.items():
                scores.setdefault(name, score)
                scores[f'{path.stem}/{name}'] = score
    html = reports / 'report.html'
    if root is not None:
        if json_path is not None and inside_directory(root, json_path) is None:
            json_path = None
            scores = {}
        if inside_directory(root, html) is None:
            html = None
    result = {}
    ordered = _ordered({key: value for key, value in scores.items() if '/' not in key})
    prefixed = _ordered({key: value for key, value in scores.items() if '/' in key})
    if ordered or prefixed:
        result['scores'] = {**ordered, **prefixed}
    if 'bleu-1' in scores:
        result['bleu_1'] = scores['bleu-1']
    if 'rouge-l-f' in scores:
        result['rouge_l_f'] = scores['rouge-l-f']
    if html is not None and html.is_file():
        result['report_html'] = str(html.resolve())
    if json_path is not None:
        result['report_json'] = str(json_path.resolve())
    return result


def merge_eval_report(metrics: dict, eval_output_dir, root=None) -> dict:
    """Add report scores without replacing eval_rows or a score already recorded.

    If a newer on-disk ``report.html`` appears (different path), refresh the
    score fields from that report so mock stubs can be overwritten by real
    evalscope output.
    """
    merged = dict(metrics or {})
    found = read_eval_report(eval_output_dir, root=root)
    if not found:
        return merged
    disk_html = found.get('report_html')
    if disk_html and disk_html != merged.get('report_html'):
        for key, value in found.items():
            if key in {'scores', 'bleu_1', 'rouge_l_f', 'report_html', 'report_json'}:
                merged[key] = value
            elif key not in merged:
                merged[key] = value
        return merged
    if merged.get('scores') and merged.get('report_html'):
        return merged
    for key, value in found.items():
        if key not in merged:
            merged[key] = value
    return merged


_COMPARE_METRIC_KEYS = ('accuracy', 'bleu_1', 'rouge_l_f', 'eval_rows', 'mock_score')


def compact_eval_metrics(metrics: dict) -> dict:
    """Keep score fields for the compare table; drop bulky paths from the payload."""
    if not isinstance(metrics, dict):
        return {}
    out = {}
    for key in _COMPARE_METRIC_KEYS:
        if key in metrics and metrics[key] is not None:
            out[key] = metrics[key]
    scores = metrics.get('scores')
    if isinstance(scores, dict) and scores:
        out['scores'] = dict(scores)
    return out


def base_eval_output_dir(eval_output_dir) -> Path | None:
    """Sibling ``base/`` under the stage eval root (``…/eval_output/base``)."""
    if not eval_output_dir:
        return None
    root = Path(str(eval_output_dir))
    if root.name == 'base':
        return root
    if root.name == 'adapter':
        return root.parent / 'base'
    return root / 'base'
