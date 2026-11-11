# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from pathlib import Path

from rsi.dataset_validate import validate_public_id, validate_text


def test_messages_and_solution():
    row = {
        'messages': [
            {'role': 'user', 'content': 'VPN 连不上'},
            {'role': 'assistant', 'content': '检查客户端版本'},
        ],
        'solution': '客户端版本',
    }
    report = validate_text(json.dumps(row, ensure_ascii=False) + '\n', 'demo.jsonl')
    assert report['ok']
    assert report['formats']['messages'] == 1
    assert report['num_rows'] == 1


def test_alpaca_sharegpt_and_query():
    alpaca = '{"instruction": "分类", "input": "磁盘满", "output": "清理日志"}\n'
    sharegpt = '{"conversation": [{"human": "你好", "assistant": "在的"}]}\n'
    query = '{"query": "1+1", "response": "2"}\n'
    assert validate_text(alpaca, 'a.jsonl')['formats'] == {'alpaca': 1}
    assert validate_text(sharegpt, 's.jsonl')['formats'] == {'sharegpt': 1}
    assert validate_text(query, 'q.jsonl')['formats'] == {'query': 1}


def test_rejects_unknown_shape_and_bad_role():
    bad = validate_text('{"foo": 1}\n', 'bad.jsonl')
    assert not bad['ok']
    assert bad['errors']
    role = validate_text('{"messages": [{"role": "narrator", "content": "x"}]}\n', 'role.jsonl')
    assert not role['ok']
    assert 'role' in role['errors'][0]['error']


def test_json_array_and_empty():
    array = validate_text('[{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}]',
                          'batch.json')
    assert array['ok']
    assert array['num_rows'] == 1
    empty = validate_text('   ', 'empty.jsonl')
    assert not empty['ok']


def test_prompt_with_solution_is_rl_shaped():
    row = '{"messages": [{"role": "user", "content": "工单"}], "solution": "网关"}\n'
    report = validate_text(row, 'rl.jsonl')
    assert report['ok']
    assert 'messages+solution' in report['formats']


def test_public_id():
    assert validate_public_id('AI-ModelScope/alpaca-gpt4-data-zh#500')['ok']
    assert not validate_public_id('not a dataset')['ok']
    assert not validate_public_id('../secrets')['ok']


def test_bundled_example_files():
    root = Path(__file__).resolve().parents[1] / 'examples'
    train = validate_text((root / 'itops_mix.jsonl').read_text(encoding='utf-8'), 'itops_mix.jsonl')
    heldout = validate_text((root / 'itops_eval.jsonl').read_text(encoding='utf-8'), 'itops_eval.jsonl')
    assert train['ok'] and train['num_rows'] == 6
    assert heldout['ok'] and heldout['num_rows'] == 4
