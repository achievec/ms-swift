# Copyright (c) ModelScope Contributors. All rights reserved.
"""Branching RSI flow: eval is a gate, RL is optional, a round can end."""
import json

# A succeeded eval with accuracy at or above this line is 通过.
# An explicit ``passed`` value on the step overrides the score.
PASS_ACCURACY = 0.5

ACTION_LABELS = {
    'rl': '去做 RL',
    'next': '继续下一轮',
    'stop': '终止',
    'finish': '完成',
}

CANVAS = {'width': 860, 'height': 236}

NODE_LAYOUT = (
    {'id': 'data', 'label': '数据', 'x': 12, 'y': 18, 'w': 116, 'h': 62},
    {'id': 'sft', 'label': 'SFT', 'x': 168, 'y': 18, 'w': 116, 'h': 62},
    {'id': 'sft_eval', 'label': 'SFT 评测', 'x': 324, 'y': 18, 'w': 128, 'h': 62},
    {'id': 'rl', 'label': '在线 RL', 'x': 492, 'y': 18, 'w': 116, 'h': 62},
    {'id': 'rl_eval', 'label': 'RL 评测', 'x': 648, 'y': 18, 'w': 116, 'h': 62},
    {'id': 'promote', 'label': '晋级', 'x': 168, 'y': 156, 'w': 116, 'h': 62},
    {'id': 'next_round', 'label': '下一轮', 'x': 324, 'y': 156, 'w': 116, 'h': 62},
    {'id': 'finish', 'label': '完成', 'x': 492, 'y': 156, 'w': 116, 'h': 62, 'terminal': True},
    {'id': 'stop', 'label': '终止', 'x': 648, 'y': 156, 'w': 116, 'h': 62, 'terminal': True},
)

EDGE_LAYOUT = (
    {'from': 'data', 'to': 'sft', 'label': ''},
    {'from': 'sft', 'to': 'sft_eval', 'label': ''},
    {'from': 'sft_eval', 'to': 'rl', 'label': '通过', 'when': 'sft_pass'},
    {'from': 'rl', 'to': 'rl_eval', 'label': ''},
    {'from': 'sft_eval', 'to': 'promote', 'label': '下一轮', 'when': 'sft_done'},
    {'from': 'sft_eval', 'to': 'finish', 'label': '完成', 'when': 'sft_pass'},
    {'from': 'sft_eval', 'to': 'stop', 'label': '终止', 'when': 'sft_fail'},
    {'from': 'rl_eval', 'to': 'promote', 'label': '下一轮', 'when': 'rl_done'},
    {'from': 'rl_eval', 'to': 'finish', 'label': '完成', 'when': 'rl_pass'},
    {'from': 'rl_eval', 'to': 'stop', 'label': '终止', 'when': 'rl_fail'},
    {'from': 'promote', 'to': 'next_round', 'label': ''},
)


def _loads(value):
    if isinstance(value, dict):
        return value
    if not value or not isinstance(value, str):
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def eval_verdict(step) -> bool | None:
    """True when this eval passed, False when it did not, None when it is not finished."""
    if not step or step.get('status') != 'succeeded':
        return None
    params = _loads(step.get('params_json') if 'params_json' in step else step.get('params'))
    if 'passed' in params:
        return bool(params['passed'])
    result = _loads(step.get('result_json') if 'result_json' in step else step.get('result'))
    metrics = result.get('metrics') or {}
    if 'accuracy' in metrics:
        threshold = params.get('pass_threshold', PASS_ACCURACY)
        try:
            return float(metrics['accuracy']) >= float(threshold)
        except (TypeError, ValueError):
            return False
    return True


def verdict_label(value) -> str:
    if value is True:
        return '通过'
    if value is False:
        return '未通过'
    return ''


def _done(active, name) -> bool:
    return (active.get(name) or {}).get('status') == 'succeeded'


def stage_unlocked(stage: str, active: dict, rnd: dict | None) -> bool:
    if stage == 'data':
        return True
    if stage == 'sft':
        return _done(active, 'data')
    if stage == 'sft_eval':
        return _done(active, 'sft')
    if stage == 'rl':
        return eval_verdict(active.get('sft_eval')) is True
    if stage == 'rl_eval':
        return _done(active, 'rl')
    if stage == 'promote':
        return eval_verdict(active.get('sft_eval')) is not None or eval_verdict(active.get('rl_eval')) is not None
    if stage == 'next_round':
        return bool((rnd or {}).get('promoted_checkpoint'))
    return False


def blocked_message(stage: str, active: dict, rnd: dict | None) -> str:
    if stage == 'sft' and not _done(active, 'data'):
        return '请先完成「数据」'
    if stage == 'sft_eval' and not _done(active, 'sft'):
        return '请先完成「SFT」'
    if stage == 'rl':
        verdict = eval_verdict(active.get('sft_eval'))
        if verdict is False:
            return 'SFT 评测未通过，不能进入在线 RL'
        if verdict is not True:
            return '请先完成「SFT 评测」'
    if stage == 'rl_eval' and not _done(active, 'rl'):
        return '请先完成「在线 RL」'
    if stage == 'promote':
        return '请先完成评测'
    if stage == 'next_round':
        return '还没有晋级结果'
    return '这一步还不能运行'


def _actions(ids) -> list:
    return [{'id': item, 'label': ACTION_LABELS[item], 'danger': item == 'stop'} for item in ids]


def decision_actions(run: dict, rnd: dict | None, active: dict) -> tuple:
    """Buttons for the frontier gate. Empty when this round is not waiting on a choice."""
    if not run or run.get('status') != 'active' or not rnd or rnd.get('status') != 'active':
        return [], ''
    if rnd.get('round_index') != run.get('round_index'):
        return [], ''
    rl_verdict = eval_verdict(active.get('rl_eval'))
    sft_verdict = eval_verdict(active.get('sft_eval'))
    rl_step = active.get('rl') or {}
    if rl_verdict is not None:
        if rl_verdict:
            return _actions(['next', 'finish']), 'RL 评测已通过，可以继续。'
        return _actions(['next', 'stop']), 'RL 评测未通过。'
    if rl_step.get('status') in {'succeeded', 'running', 'syncing'}:
        return [], ''
    if sft_verdict is None:
        return [], ''
    if sft_verdict:
        return _actions(['rl', 'next', 'finish']), 'SFT 评测已通过，可以继续。'
    return _actions(['next', 'stop']), 'SFT 评测未通过。'


def _edge_live(edge, active, sft_verdict, rl_verdict) -> bool:
    kind = edge.get('when')
    if not kind:
        return _done(active, edge['from'])
    if kind == 'sft_pass':
        return sft_verdict is True
    if kind == 'sft_fail':
        return sft_verdict is False
    if kind == 'sft_done':
        return sft_verdict is not None
    if kind == 'rl_pass':
        return rl_verdict is True
    if kind == 'rl_fail':
        return rl_verdict is False
    if kind == 'rl_done':
        return rl_verdict is not None
    return False


def _terminal_status(node_id: str, run: dict) -> str:
    if node_id == 'stop':
        return 'failed' if (run or {}).get('status') == 'stopped' else 'locked'
    if node_id == 'finish':
        return 'succeeded' if (run or {}).get('status') == 'completed' else 'locked'
    return 'locked'


def assemble_flow(run, rnd, active, status_of, metric_of) -> dict:
    sft_verdict = eval_verdict(active.get('sft_eval'))
    rl_verdict = eval_verdict(active.get('rl_eval'))
    nodes = []
    for spec in NODE_LAYOUT:
        terminal = bool(spec.get('terminal'))
        status = _terminal_status(spec['id'], run) if terminal else status_of(spec['id'])
        verdict = ''
        if spec['id'] == 'sft_eval':
            verdict = verdict_label(sft_verdict)
        elif spec['id'] == 'rl_eval':
            verdict = verdict_label(rl_verdict)
        nodes.append({
            'id': spec['id'],
            'label': spec['label'],
            'x': spec['x'],
            'y': spec['y'],
            'w': spec['w'],
            'h': spec['h'],
            'status': status,
            'verdict': verdict,
            'metric': '' if terminal else (metric_of(spec['id']) or ''),
            'terminal': terminal,
            'clickable': not (terminal and status == 'locked'),
        })
    edges = []
    for edge in EDGE_LAYOUT:
        edges.append({
            'from': edge['from'],
            'to': edge['to'],
            'label': edge.get('label') or '',
            'live': _edge_live(edge, active, sft_verdict, rl_verdict),
        })
    actions, prompt = decision_actions(run, rnd, active)
    return {'canvas': CANVAS, 'nodes': nodes, 'edges': edges, 'actions': actions, 'prompt': prompt}
