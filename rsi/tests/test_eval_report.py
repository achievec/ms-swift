# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from pathlib import Path

from fastapi.testclient import TestClient

from rsi.eval_report import inside_directory, read_eval_report, scores_from_report


def _report(bleu_1=0.11, rouge_l_f=0.32):
    names = (
        ('bleu-1', bleu_1),
        ('bleu-2', 0.05),
        ('bleu-3', 0.01),
        ('bleu-4', 0.0),
        ('Rouge-1-R', 0.42),
        ('Rouge-1-P', 0.4),
        ('Rouge-1-F', 0.41),
        ('Rouge-2-R', 0.2),
        ('Rouge-2-P', 0.19),
        ('Rouge-2-F', 0.18),
        ('Rouge-L-R', 0.33),
        ('Rouge-L-P', 0.31),
        ('Rouge-L-F', rouge_l_f),
    )
    return {
        'dataset_name': 'general_qa',
        'model_name': 'Qwen2.5-7B-Instruct',
        'metrics': [
            {
                'name': name,
                'score': score,
                'categories': [{'name': ['default'], 'subsets': [{'name': 'itops', 'score': score, 'num': 4}]}],
            }
            for name, score in names
        ],
    }


def _write_report(eval_dir: Path) -> Path:
    reports = eval_dir / 'native' / '20261020_144133' / 'reports'
    dataset = reports / 'Qwen2.5-7B-Instruct'
    dataset.mkdir(parents=True)
    (dataset / 'general_qa.json').write_text(json.dumps(_report()), encoding='utf-8')
    html = reports / 'report.html'
    html.write_text('<html><body>general_qa report</body></html>', encoding='utf-8')
    return html


def test_scores_from_named_metrics_and_v2_identity():
    scores = scores_from_report(_report())
    assert scores['bleu-1'] == 0.11
    assert scores['rouge-l-f'] == 0.32
    assert scores['rouge-1-r'] == 0.42
    identity = scores_from_report({
        'metrics': [
            {'score': 0.21, 'identity': {'name': 'bleu', 'aggregation': 'mean', 'dimensions': {'ngram': 1}}},
            {
                'score': 0.44,
                'identity': {'name': 'rouge', 'aggregation': 'mean', 'dimensions': {'variant': 'l', 'statistic': 'f1'}},
            },
        ],
    })
    assert identity == {'bleu-1': 0.21, 'rouge-l-f': 0.44}


def test_read_eval_report_uses_newest_directory(tmp_path):
    older = tmp_path / 'native' / '20261020_010000' / 'reports' / 'model'
    older.mkdir(parents=True)
    (older / 'general_qa.json').write_text(json.dumps(_report(bleu_1=0.01, rouge_l_f=0.02)), encoding='utf-8')
    html = _write_report(tmp_path)
    found = read_eval_report(tmp_path, root=tmp_path)
    assert found['bleu_1'] == 0.11
    assert found['rouge_l_f'] == 0.32
    assert found['scores']['bleu-4'] == 0.0
    assert found['scores']['rouge-l-r'] == 0.33
    assert found['report_html'] == str(html.resolve())
    assert found['report_json'].endswith('general_qa.json')
    outside = tmp_path.parent / f'{tmp_path.name}-outside'
    outside.mkdir()
    leaked = outside / 'report.html'
    leaked.write_text('no', encoding='utf-8')
    assert inside_directory(tmp_path, leaked) is None


def test_stage_view_shows_report_scores_and_serves_html(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'rsi-home')
    with TestClient(app) as api:
        run_id = api.post('/api/runs', json={
            'name': '报告',
            'scenario': '评测报告要出现在结果里',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
            'model_type': 'qwen2',
            'template': 'qwen2_5',
        }).json()['run']['id']
        assert api.post(f'/api/runs/{run_id}/stages/data/run', json={
            'params': {'source': 'bundled', 'bundled': 'itops_mix'},
        }).status_code == 200
        assert api.post(f'/api/runs/{run_id}/stages/sft/run', json={'params': {}}).status_code == 200
        evaluated = api.post(f'/api/runs/{run_id}/stages/sft_eval/run', json={'params': {}})
        assert evaluated.status_code == 200, evaluated.text
        assert evaluated.json()['result']['metrics']['eval_rows'] == 4
        missing = api.get(f'/api/runs/{run_id}/stages/sft_eval/report')
        assert missing.status_code == 404

        view = api.get(f'/api/runs/{run_id}').json()
        eval_dir = Path(view['forms']['sft_eval']['values']['eval_output_dir'])
        html = _write_report(eval_dir)
        shown = api.get(f'/api/runs/{run_id}').json()
        metrics = shown['forms']['sft_eval']['result']['metrics']
        assert metrics['eval_rows'] == 4
        assert metrics['bleu_1'] == 0.11
        assert metrics['rouge_l_f'] == 0.32
        assert metrics['scores']['bleu-1'] == 0.11
        assert metrics['scores']['rouge-l-f'] == 0.32
        assert shown['forms']['sft_eval']['result']['report_path'] == str(html.resolve())
        body = api.get(f'/api/runs/{run_id}/stages/sft_eval/report')
        assert body.status_code == 200, body.text
        assert 'general_qa report' in body.text
        assert '查看完整报告' in (Path(__file__).resolve().parents[1] / 'static' / 'app.js').read_text(encoding='utf-8')


def test_fullscreen_exit_clears_scroll_lock():
    root = Path(__file__).resolve().parents[1] / 'static'
    script = (root / 'app.js').read_text(encoding='utf-8')
    css = (root / 'app.css').read_text(encoding='utf-8')
    assert 'function syncLogScrollLock()' in script
    assert script.count('syncLogScrollLock()') >= 4
    assert 'new MutationObserver' in script
    assert "document.documentElement.classList.toggle('log-lock', locked)" in script
    assert 'html.log-lock, body.log-lock { overflow: hidden; }' in css
