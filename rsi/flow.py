# Copyright (c) ModelScope Contributors. All rights reserved.
"""Branching RSI flow: eval is a gate, RL is optional, a round can end."""
import json

# A succeeded eval with accuracy at or above this line is 通过.
# An explicit ``passed`` value on the step overrides the score.
PASS_ACCURACY = 0.5

ACTION_LABELS = {
    'rl': 'RL',
    'next': '下一轮数据',
    'finish': '完成',
}

# One horizontal trunk. Decision exits hang straight under the active eval
# so the lines do not cross. 晋级 is the form behind 「下一轮数据」, not its own node.
NODE_W = 124
NODE_H = 64
TRUNK_Y = 16
OUT_Y = 156
TRUNK = (
    ('data', '数据', 12),
    ('sft', 'SFT', 164),
    ('sft_eval', 'SFT 评测', 316),
    ('rl', '在线 RL', 488),
    ('rl_eval', 'RL 评测', 640),
)
CANVAS = {'width': 860, 'height': 248}


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
        return '不通过'
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
    return [{'id': item, 'label': ACTION_LABELS[item], 'danger': False} for item in ids]


def _round_open(run, rnd) -> bool:
    return bool(run and rnd and run.get('status') == 'active' and rnd.get('status') == 'active'
                and rnd.get('round_index') == run.get('round_index'))


def decision_actions(run: dict, rnd: dict | None, active: dict) -> tuple:
    """Buttons for the frontier gate. Empty when this round is not waiting on a choice."""
    if not _round_open(run, rnd):
        return [], ''
    rl_verdict = eval_verdict(active.get('rl_eval'))
    sft_verdict = eval_verdict(active.get('sft_eval'))
    rl_step = active.get('rl') or {}
    if rl_verdict is not None:
        if rl_verdict:
            return _actions(['next', 'finish']), 'RL 评测已通过。'
        return _actions(['next']), 'RL 评测不通过。'
    if rl_step.get('status') in {'succeeded', 'running', 'syncing'}:
        return [], ''
    if sft_verdict is None:
        return [], ''
    if sft_verdict:
        return _actions(['rl', 'next', 'finish']), 'SFT 评测已通过。'
    return _actions(['next']), 'SFT 评测不通过。'


def _node(node_id, label, x, y, status, verdict='', metric='', terminal=False, clickable=True) -> dict:
    return {
        'id': node_id,
        'label': label,
        'x': x,
        'y': y,
        'w': NODE_W,
        'h': NODE_H,
        'status': status,
        'verdict': verdict,
        'metric': metric or '',
        'terminal': terminal,
        'clickable': clickable,
    }


def _edge(source, target, label, live, kind) -> dict:
    return {'from': source, 'to': target, 'label': label, 'live': bool(live), 'kind': kind}


def _outcome_slots(anchor_x, count) -> list:
    total = count * NODE_W + (count - 1) * 20
    start = anchor_x + NODE_W / 2 - total / 2
    return [int(start + index * (NODE_W + 20)) for index in range(count)]


def assemble_flow(run, rnd, active, status_of, metric_of) -> dict:
    sft_verdict = eval_verdict((active or {}).get('sft_eval'))
    rl_verdict = eval_verdict((active or {}).get('rl_eval'))
    rl_step = (active or {}).get('rl') or {}
    actions, prompt = decision_actions(run, rnd, active)
    action_ids = {item['id'] for item in actions}
    trunk = {node_id: x for node_id, _label, x in TRUNK}
    nodes = []
    for node_id, label, x in TRUNK:
        verdict = ''
        if node_id == 'sft_eval':
            verdict = verdict_label(sft_verdict)
        elif node_id == 'rl_eval':
            verdict = verdict_label(rl_verdict)
        nodes.append(_node(node_id, label, x, TRUNK_Y, status_of(node_id), verdict, metric_of(node_id)))

    edges = [
        _edge('data', 'sft', '', _done(active, 'data'), 'trunk'),
        _edge('sft', 'sft_eval', '', _done(active, 'sft'), 'trunk'),
    ]
    # The RL trunk stays visible until an eval has failed. A failed eval has no RL edge.
    if sft_verdict is not False:
        edges.append(_edge('sft_eval', 'rl', 'RL' if sft_verdict else '', sft_verdict is True, 'trunk'))
        edges.append(_edge('rl', 'rl_eval', '', _done(active, 'rl'), 'trunk'))

    # A closed round shows only the exit that was taken. An open round shows the
    # choices that the latest eval still allows. Lines hang under that one gate.
    exits = []
    gate = None
    if _done(active, 'next_round'):
        gate = 'rl_eval' if rl_verdict is not None else 'sft_eval' if sft_verdict is not None else None
        if gate:
            failed = rl_verdict is False or (rl_verdict is None and sft_verdict is False)
            exits = [('next_data', '下一轮数据', '不通过' if failed else '下一轮数据')]
    elif ((run or {}).get('status') == 'completed' and rnd
          and rnd.get('round_index') == (run or {}).get('round_index')):
        gate = 'rl_eval' if rl_verdict is not None else 'sft_eval' if sft_verdict is not None else None
        if gate:
            exits = [('finish', '完成', '完成')]
    elif _round_open(run, rnd):
        rl_busy = rl_step.get('status') in {'succeeded', 'running', 'syncing'}
        if rl_verdict is False:
            gate, exits = 'rl_eval', [('next_data', '下一轮数据', '不通过')]
        elif rl_verdict is True:
            gate = 'rl_eval'
            exits = [('next_data', '下一轮数据', '下一轮数据'), ('finish', '完成', '完成')]
        elif sft_verdict is False and not rl_busy:
            gate, exits = 'sft_eval', [('next_data', '下一轮数据', '不通过')]
        elif sft_verdict is True and not rl_busy:
            gate = 'sft_eval'
            exits = [('next_data', '下一轮数据', '下一轮数据'), ('finish', '完成', '完成')]

    if gate and exits:
        slots = _outcome_slots(trunk[gate], len(exits))
        right = max(node['x'] + node['w'] for node in nodes)
        for (exit_id, label, edge_label), x in zip(exits, slots):
            if exit_id == 'finish':
                status = 'succeeded' if (run or {}).get('status') == 'completed' else 'ready'
                terminal = True
                clickable = status == 'succeeded'
            else:
                nxt = (active or {}).get('next_round') or {}
                if nxt.get('status') == 'succeeded':
                    status = 'succeeded'
                elif 'next' in action_ids or nxt.get('status') in {'ready', 'running', 'failed', 'syncing'}:
                    status = 'ready' if nxt.get('status') not in {'running', 'failed', 'syncing'} else nxt['status']
                else:
                    status = 'locked'
                terminal = False
                clickable = status != 'locked'
            nodes.append(_node(exit_id, label, x, OUT_Y, status, terminal=terminal, clickable=clickable))
            edges.append(_edge(gate, exit_id, edge_label, True, 'down'))
            right = max(right, x + NODE_W)
        canvas = {'width': max(CANVAS['width'], right + 16), 'height': CANVAS['height']}
    else:
        canvas = {'width': CANVAS['width'], 'height': TRUNK_Y + NODE_H + 28}
    return {'canvas': canvas, 'nodes': nodes, 'edges': edges, 'actions': actions, 'prompt': prompt}
