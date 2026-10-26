# Copyright (c) ModelScope Contributors. All rights reserved.
import json

from fastapi.testclient import TestClient

from rsi.workbench import export_argv, export_checkpoint, latest_dataset, probe_service, score_row


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv('RSI_BACKEND', 'mock')
    monkeypatch.setenv('RSI_SERVE_BACKEND', 'mock')
    from rsi.app import create_app
    app = create_app(tmp_path / 'home')
    return TestClient(app)


SAMPLE = '\n'.join([
    json.dumps({
        'messages': [
            {'role': 'user', 'content': '工单 A'},
            {'role': 'assistant', 'content': '先看证书缓存，再收日志。'},
        ],
        'solution': '证书缓存|日志',
    }, ensure_ascii=False),
    json.dumps({
        'messages': [
            {'role': 'user', 'content': '工单 B'},
            {'role': 'assistant', 'content': '重启电脑即可。'},
        ],
        'solution': '证书缓存|日志',
    }, ensure_ascii=False),
    json.dumps({
        'messages': [
            {'role': 'user', 'content': '工单 C'},
            {'role': 'assistant', 'content': '收集日志后交给值班。'},
        ],
        'solution': '日志',
    }, ensure_ascii=False),
])


def test_inspect_split_and_next_pack(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    inspected = client.post('/api/workbench/inspect', json={'text': SAMPLE})
    assert inspected.status_code == 200, inspected.text
    body = inspected.json()
    assert body['coverage']['num_rows'] == 3
    assert body['coverage']['full_hit'] == 2

    split = client.post('/api/workbench/split', json={'text': SAMPLE, 'eval_ratio': 0.34, 'seed': 1})
    assert split.status_code == 200, split.text
    saved = split.json()
    assert saved['train_rows'] + saved['eval_rows'] == 3
    assert saved['eval_rows'] == 1
    train = open(saved['train_path'], encoding='utf-8').read()
    heldout = open(saved['eval_path'], encoding='utf-8').read()
    assert '工单' in train and '工单' in heldout
    assert set(train.splitlines()).isdisjoint(heldout.splitlines())

    packed = client.post('/api/workbench/next-pack', json={'text': SAMPLE})
    assert packed.status_code == 200, packed.text
    assert packed.json()['kept_rows'] == 2
    assert packed.json()['retry_rows'] == 1
    retry = open(packed.json()['retry_path'], encoding='utf-8').read()
    assert 'assistant' not in retry
    assert '工单 B' in retry


def test_split_files_can_be_chosen_as_bundled(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    split = client.post('/api/workbench/split', json={'text': SAMPLE, 'eval_ratio': 0.34, 'seed': 1})
    assert split.status_code == 200, split.text
    listed = client.get('/api/workbench/files')
    assert listed.status_code == 200, listed.text
    files = listed.json()['files']
    train = next(item for item in files if item['id'].endswith('/train.jsonl'))
    heldout = next(item for item in files if item['id'].endswith('/eval.jsonl'))
    assert train['kind'] == 'train' and heldout['kind'] == 'eval'
    assert train['rows'] == split.json()['train_rows']
    created = client.post('/api/runs', json={
        'name': '选用拆分数据',
        'scenario': '训练页直接选工作台生成的文件',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    run_id = created.json()['run']['id']
    started = client.post(f'/api/runs/{run_id}/stages/sft/run', json={
        'params': {'source': 'bundled', 'bundled': train['id']},
    })
    assert started.status_code == 200, started.text
    assert train['path'] in started.json()['command']
    view = client.get(f'/api/runs/{run_id}').json()
    heldout_config = view['forms']['sft_eval']['values']['custom_eval_config']
    assert heldout_config.endswith('/eval.jsonl')
    assert heldout_config.rsplit('/', 1)[0] == train['path'].rsplit('/', 1)[0]
    blocked = client.post(f'/api/runs/{run_id}/stages/sft/run', json={
        'params': {'source': 'bundled', 'bundled': 'wb:../../etc/passwd'},
    })
    assert blocked.status_code == 400


def test_load_dataset_recorded_by_the_data_step(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post('/api/runs', json={
        'name': '已有数据',
        'scenario': '读取本轮训练文件',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    run_id = created.json()['run']['id']
    missing = client.get(f'/api/runs/{run_id}/workbench/dataset')
    assert missing.status_code == 400
    loaded = client.post(f'/api/runs/{run_id}/stages/sft/run', json={
        'params': {'source': 'bundled', 'bundled': 'itops_mix'},
    })
    assert loaded.status_code == 200, loaded.text
    found = client.get(f'/api/runs/{run_id}/workbench/dataset')
    assert found.status_code == 200, found.text
    body = found.json()
    assert body['path'].endswith('itops_mix.jsonl')
    assert 'VPN' in body['text'] or body['inspect']['coverage']['num_rows'] == 6
    assert '逐条看' in body['how_to']


def test_dataset_notice_when_eval_still_uses_itops(tmp_path):
    data = tmp_path / 'train.jsonl'
    data.write_text(SAMPLE, encoding='utf-8')
    view = {
        'steps': [{
            'stage': 'sft',
            'status': 'succeeded',
            'round_index': 1,
            'params': {'dataset': str(data)},
            'result': {},
        }],
        'forms': {'sft_eval': {'values': {'custom_eval_config': 'rsi/examples/itops_eval.jsonl'}}},
        'checkpoints': [],
    }
    body = latest_dataset(view)
    assert '不是一套' in body['notice']


def test_complete_reads_only_workbench_files(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post('/api/runs', json={
        'name': '补全文件',
        'scenario': '只读工具页生成的待补全文件',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    run_id = created.json()['run']['id']
    packed = client.post('/api/workbench/next-pack', json={'text': SAMPLE})
    assert packed.status_code == 200, packed.text
    retry_path = packed.json()['retry_path']

    def fake_chat(_run_id, messages, **_kwargs):
        return {'choices': [{'message': {'content': '先看证书缓存，再收日志。'}}]}

    client.app.state.pipeline.chat = fake_chat
    done = client.post(f'/api/runs/{run_id}/workbench/complete', json={'path': retry_path})
    assert done.status_code == 200, done.text
    assert done.json()['answered'] == 1
    outside = client.post(f'/api/runs/{run_id}/workbench/complete', json={'path': '/etc/passwd'})
    assert outside.status_code == 400


def test_report_and_idle_export(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post('/api/runs', json={
        'name': '报表',
        'scenario': '只读实验记录',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    assert created.status_code == 200, created.text
    run_id = created.json()['run']['id']
    report = client.get(f'/api/runs/{run_id}/report')
    assert report.status_code == 200
    assert '报表' in report.text
    assert 'Qwen/Qwen2.5-7B-Instruct' in report.text
    ready = client.get(f'/api/runs/{run_id}/serve/ready')
    assert ready.status_code == 200
    assert ready.json()['ready'] is False
    status = client.get(f'/api/runs/{run_id}/export')
    assert status.json()['status'] == 'idle'
    blocked = client.post(f'/api/runs/{run_id}/export')
    assert blocked.status_code == 400


def test_score_and_probe_do_not_require_training():
    scored = score_row({
        'messages': [{'role': 'assistant', 'content': '日志'}],
        'solution': '日志|证书',
    })
    assert scored['score'] == 0.5
    assert scored['missing'] == ['证书']
    down = probe_service({'status': 'running', 'host': '127.0.0.1', 'port': 1})
    assert down['ready'] is False


def test_export_uses_sft_checkpoint_before_promotion():
    view = {
        'run': {'round_index': 1, 'base_model': 'Qwen/Qwen2.5-7B-Instruct'},
        'rounds': [{'round_index': 1, 'promoted_checkpoint': ''}],
        'checkpoints': [
            {'round_index': 1, 'kind': 'sft', 'path': '/models/sft', 'active': 1},
            {'round_index': 1, 'kind': 'rl', 'path': '/models/rl', 'active': 1},
            {'round_index': 2, 'kind': 'sft', 'path': '/models/other', 'active': 1},
        ],
    }
    assert export_checkpoint(view) == '/models/rl'
    view['rounds'] = [{'promoted_checkpoint': '/models/promo'}]
    assert export_checkpoint(view) == '/models/promo'


def test_export_lora_includes_base_model(tmp_path):
    adapter = tmp_path / 'lora'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (adapter / 'args.json').write_text(
        json.dumps({'model': 'Qwen/Qwen2.5-7B-Instruct'}), encoding='utf-8')
    argv = export_argv(str(adapter), str(tmp_path / 'out'), 'fallback-base')
    assert argv[argv.index('--model') + 1] == 'Qwen/Qwen2.5-7B-Instruct'
    assert argv[argv.index('--adapters') + 1] == str(adapter)
    assert '--merge_lora' in argv
    full = tmp_path / 'full'
    full.mkdir()
    (full / 'config.json').write_text('{}', encoding='utf-8')
    full_argv = export_argv(str(full), str(tmp_path / 'full-out'))
    assert '--adapters' not in full_argv
    assert full_argv[full_argv.index('--model') + 1] == str(full)


def test_complete_uses_running_chat_and_skips_answered_rows(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post('/api/runs', json={
        'name': '补全',
        'scenario': '用现有推理补题目',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    })
    run_id = created.json()['run']['id']
    seen = []

    def fake_chat(_run_id, messages, **_kwargs):
        seen.append(messages[-1]['content'])
        return {'choices': [{'message': {'content': '先看证书缓存，再收日志。'}}]}

    client.app.state.pipeline.chat = fake_chat
    prompt = json.dumps({
        'messages': [{'role': 'user', 'content': '工单 B'}],
        'solution': '证书缓存|日志',
    }, ensure_ascii=False)
    done = client.post(f'/api/runs/{run_id}/workbench/complete', json={'text': prompt})
    assert done.status_code == 200, done.text
    body = done.json()
    assert body['answered'] == 1
    assert seen == ['工单 B']
    written = open(body['next_path'], encoding='utf-8').read()
    assert '先看证书缓存' in written
    assert '内置示例' in body['how_to']
