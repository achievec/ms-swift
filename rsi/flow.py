# Copyright (c) ModelScope Contributors. All rights reserved.
"""Branching RSI flow: eval is a gate, RL is optional, a round can end."""
import json

# A succeeded eval with accuracy at or above this line is 通过.
# An explicit ``passed`` value on the step overrides the score.
PASS_ACCURACY = 0.5

ACTION_LABELS = {
    'rl': 'RL',
    'next': '开启下一轮',
    'finish': '完成',
}

# Stage jobs that can put an experiment card into 运行中 / 失败.
# ``syncing`` is an in-progress model or dataset fetch on one of these stages.
LIST_JOB_STAGES = frozenset({'sft', 'sft_eval', 'rl', 'rl_eval', 'export'})
_BUSY_STEP_STATUS = frozenset({'running', 'syncing'})

# The guide is a fixed template. Every branch is drawn on every round.
# Unreached nodes stay gray; reachable and taken edges are highlighted.
NODE_W = 108
NODE_H = 52
TRUNK_Y = 16
EXIT_Y = 132
CANVAS = {'width': 856, 'height': 280}


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


def _went_next(active: dict) -> bool:
    """True after the merged「开启下一轮」stage, or a legacy split next_round step."""
    return _done(active, 'promote') or _done(active, 'next_round')


def stage_unlocked(stage: str, active: dict, rnd: dict | None) -> bool:
    if stage == 'sft':
        return True
    if stage == 'sft_eval':
        return _done(active, 'sft')
    if stage == 'rl':
        return eval_verdict(active.get('sft_eval')) is True
    if stage == 'rl_eval':
        return _done(active, 'rl')
    if stage == 'promote':
        return eval_verdict(active.get('sft_eval')) is not None or eval_verdict(active.get('rl_eval')) is not None
    return False


def blocked_message(stage: str, active: dict, rnd: dict | None) -> str:
    if stage == 'sft_eval' and not _done(active, 'sft'):
        return '请先完成「SFT」'
    if stage == 'rl':
        verdict = eval_verdict(active.get('sft_eval'))
        if verdict is False:
            return 'SFT 评测未通过，不能进入 RL'
        if verdict is not True:
            return '请先完成「SFT 评测」'
    if stage == 'rl_eval' and not _done(active, 'rl'):
        return '请先完成「RL」'
    if stage == 'promote':
        return '请先完成评测'
    return '这一步还不能运行'


def _actions(ids) -> list:
    return [{'id': item, 'label': ACTION_LABELS[item], 'danger': False} for item in ids]


def _round_open(run, rnd) -> bool:
    return bool(run and rnd and run.get('status') == 'active' and rnd.get('status') == 'active'
                and rnd.get('round_index') == run.get('round_index'))


def _round_finished(run, rnd) -> bool:
    """True when this run ended on the round we are viewing (「完成」)."""
    return bool(run and rnd and run.get('status') == 'completed'
                and rnd.get('round_index') == run.get('round_index'))


def decision_actions(run: dict, rnd: dict | None, active: dict) -> tuple:
    """Buttons for the frontier gate. Empty when this round is not waiting on a choice.

    After 「完成」, the run is closed but 「开启下一轮」stays available so a later
    round can still start from prior checkpoints. Fail-only gates are unchanged.
    """
    rl_verdict = eval_verdict(active.get('rl_eval'))
    sft_verdict = eval_verdict(active.get('sft_eval'))
    rl_step = active.get('rl') or {}
    if _round_finished(run, rnd):
        # Finish only opens on a pass; keep offering next so completion is not a dead end.
        if rl_verdict is True or (rl_verdict is None and sft_verdict is True):
            return _actions(['next']), '实验已完成，仍可开启下一轮。'
        return [], ''
    if not _round_open(run, rnd):
        return [], ''
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


def _node(node_id, label, x, y, status, verdict='', metric='', terminal=False, clickable=True,
          width=None, height=None, minor=False) -> dict:
    return {
        'id': node_id,
        'label': label,
        'x': x,
        'y': y,
        'w': NODE_W if width is None else width,
        'h': NODE_H if height is None else height,
        'status': status,
        'verdict': verdict,
        'metric': metric or '',
        'terminal': terminal,
        'clickable': clickable,
        'minor': minor,
    }


def _points(pairs) -> list:
    return [[int(round(x)), int(round(y))] for x, y in pairs]


def _label_at(points) -> tuple:
    best_len = -1
    best = (points[0][0], points[0][1], 'middle')
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = abs(x2 - x1) + abs(y2 - y1)
        if length <= best_len:
            continue
        best_len = length
        horizontal = abs(x2 - x1) >= abs(y2 - y1)
        best = ((x1 + x2) / 2, (y1 + y2) / 2 - (8 if horizontal else 0), 'middle' if horizontal else 'start')
        if not horizontal:
            best = (x1 + 10, (y1 + y2) / 2, 'start')
    return best


def _edge(source, target, label, live, kind, points, taken=False, show_label=True) -> dict:
    label_x, label_y, anchor = _label_at(points)
    return {
        'from': source,
        'to': target,
        'label': label,
        'live': bool(live),
        'taken': bool(taken),
        'kind': kind,
        'points': _points(points),
        'show_label': bool(show_label and label),
        'label_x': int(round(label_x)),
        'label_y': int(round(label_y)),
        'label_anchor': anchor,
    }


def _box(node_id, label, x, y, width=NODE_W, height=NODE_H):
    return {'id': node_id, 'label': label, 'x': x, 'y': y, 'w': width, 'h': height}


def _down_points(src, dst):
    x1 = src['x'] + src['w'] / 2
    y1 = src['y'] + src['h']
    x2 = dst['x'] + dst['w'] / 2
    y2 = dst['y']
    if abs(x1 - x2) < 6:
        return [(x1, y1), (x2, y2)]
    mid = (y1 + y2) / 2
    return [(x1, y1), (x1, mid), (x2, mid), (x2, y2)]


def _side_points(src, dst):
    y = src['y'] + src['h'] / 2
    return [(src['x'] + src['w'], y), (dst['x'], y)]


def assemble_flow(run, rnd, active, status_of, metric_of) -> dict:
    """Fixed map: SFT eval always has four exits, RL eval always has three.

    「开启下一轮」is both the exit choice and the work stage (pick checkpoint + open
    the next round). There is no separate 晋级 / 下一轮数据 pair on the board.
    """
    active = active or {}
    run = run or {}
    sft_verdict = eval_verdict(active.get('sft_eval'))
    rl_verdict = eval_verdict(active.get('rl_eval'))
    rl_step = active.get('rl') or {}
    rl_started = rl_step.get('status') in {'succeeded', 'running', 'syncing'}
    actions, prompt = decision_actions(run, rnd, active)
    action_ids = {item['id'] for item in actions}
    open_round = _round_open(run, rnd)
    went_next = _went_next(active)
    finished = _round_finished(run, rnd)
    # A finished round still exposes the same eval frontier so sibling exits stay lit.
    frontier_open = open_round or (finished and not went_next)
    finished_after_rl = bool(finished and rl_verdict is not None)
    finished_from_sft = bool(finished and rl_verdict is None)
    next_from_rl = bool(went_next and rl_verdict is not None)
    next_from_sft = bool(went_next and rl_verdict is None)

    if frontier_open and rl_verdict is not None:
        frontier = 'rl'
    elif frontier_open and not rl_started and sft_verdict is not None:
        frontier = 'sft'
    else:
        frontier = None

    sft = _box('sft', 'SFT', 20, TRUNK_Y)
    sft_eval = _box('sft_eval', 'SFT 评测', 168, TRUNK_Y)
    sft_fail = _box('sft_fail', '不通过', 28, EXIT_Y)
    sft_next = _box('sft_next', '开启下一轮', 168, EXIT_Y)
    sft_finish = _box('sft_finish', '完成', 308, EXIT_Y)
    rl = _box('rl', 'RL', 440, TRUNK_Y)
    rl_eval = _box('rl_eval', 'RL 评测', 588, TRUNK_Y)
    rl_fail = _box('rl_fail', '不通过', 448, EXIT_Y)
    rl_next = _box('rl_next', '开启下一轮', 588, EXIT_Y)
    rl_finish = _box('rl_finish', '完成', 728, EXIT_Y)
    placed = {item['id']: item for item in (
        sft, sft_eval, sft_fail, sft_next, sft_finish, rl, rl_eval, rl_fail, rl_next, rl_finish)}

    def branch_status(ready, taken, failed=False):
        if taken:
            return 'succeeded'
        if failed:
            return 'failed'
        if ready:
            return 'ready'
        return 'locked'

    sft_fail_on = sft_verdict is False
    rl_fail_on = rl_verdict is False
    choosing_next = run.get('current_stage') == 'promote'
    statuses = {
        'sft_fail': branch_status(False, False, sft_fail_on),
        'sft_next': branch_status(
            (frontier == 'sft' and 'next' in action_ids) or (choosing_next and frontier == 'sft'),
            next_from_sft),
        'sft_finish': branch_status(frontier == 'sft' and 'finish' in action_ids, finished_from_sft),
        'rl_fail': branch_status(False, False, rl_fail_on),
        'rl_next': branch_status(
            (frontier == 'rl' and 'next' in action_ids) or (choosing_next and frontier == 'rl'),
            next_from_rl),
        'rl_finish': branch_status(frontier == 'rl' and 'finish' in action_ids, finished_after_rl),
    }
    nodes = []
    for node_id, label, _x in (
        ('sft', 'SFT', 0), ('sft_eval', 'SFT 评测', 0), ('rl', 'RL', 0), ('rl_eval', 'RL 评测', 0),
    ):
        box = placed[node_id]
        verdict = ''
        if node_id == 'sft_eval':
            verdict = verdict_label(sft_verdict)
        elif node_id == 'rl_eval':
            verdict = verdict_label(rl_verdict)
        nodes.append(_node(node_id, label, box['x'], box['y'], status_of(node_id), verdict, metric_of(node_id)))
    for node_id in ('sft_fail', 'sft_next', 'sft_finish', 'rl_fail', 'rl_next', 'rl_finish'):
        box = placed[node_id]
        status = statuses[node_id]
        nodes.append(_node(
            node_id, box['label'], box['x'], box['y'], status,
            terminal=node_id.endswith('finish'),
            clickable=status in {'ready', 'succeeded'},
            width=box['w'], height=box['h'],
        ))

    rl_edge_live = (frontier == 'sft' and sft_verdict is True) or rl_started or rl_verdict is not None
    rl_edge_taken = rl_started or rl_verdict is not None

    def down(src, dst, label, live, taken=False, show_label=True):
        return _edge(src, dst, label, live, 'down', _down_points(placed[src], placed[dst]), taken, show_label)

    edges = [
        _edge('sft', 'sft_eval', '', _done(active, 'sft'), 'trunk', _side_points(sft, sft_eval), _done(active, 'sft'), False),
        _edge('sft_eval', 'rl', 'RL', rl_edge_live, 'trunk', _side_points(sft_eval, rl), rl_edge_taken, True),
        _edge('rl', 'rl_eval', '', _done(active, 'rl'), 'trunk', _side_points(rl, rl_eval), _done(active, 'rl'), False),
        down('sft_eval', 'sft_fail', '不通过', sft_fail_on, sft_fail_on, True),
        down('sft_eval', 'sft_finish', '完成', statuses['sft_finish'] in {'ready', 'succeeded'}, finished_from_sft, True),
        down('sft_eval', 'sft_next', '开启下一轮', statuses['sft_next'] in {'ready', 'succeeded'}, next_from_sft, False),
        _edge('sft_fail', 'sft_next', '', sft_fail_on and statuses['sft_next'] in {'ready', 'succeeded'}, 'join',
              _side_points(sft_fail, sft_next), next_from_sft and sft_fail_on, False),
        down('rl_eval', 'rl_fail', '不通过', rl_fail_on, rl_fail_on, True),
        down('rl_eval', 'rl_finish', '完成', statuses['rl_finish'] in {'ready', 'succeeded'}, finished_after_rl, True),
        down('rl_eval', 'rl_next', '开启下一轮', statuses['rl_next'] in {'ready', 'succeeded'}, next_from_rl, False),
        _edge('rl_fail', 'rl_next', '', rl_fail_on and statuses['rl_next'] in {'ready', 'succeeded'}, 'join',
              _side_points(rl_fail, rl_next), next_from_rl and rl_fail_on, False),
    ]
    # RL「开启下一轮」may loop back to SFT on the board. The SFT-side return
    # (sft_next → sft 「下一轮」) is omitted — the node and promote behavior remain.
    if next_from_rl:
        bottom = EXIT_Y + NODE_H + 26
        mid_y = sft['y'] + sft['h'] / 2
        src = placed['rl_next']
        edges.append(_edge(
            'rl_next', 'sft', '下一轮', True, 'return',
            [(src['x'] + src['w'] / 2, src['y'] + src['h']),
             (src['x'] + src['w'] / 2, bottom),
             (8, bottom),
             (8, mid_y),
             (sft['x'], mid_y)],
            True, True,
        ))
    return {'canvas': dict(CANVAS), 'nodes': nodes, 'edges': edges, 'actions': actions, 'prompt': prompt}


def experiment_list_status(run: dict, steps: list | None = None, round_status: str | None = None) -> str:
    """Compact experiment-list status: ``running``, ``completed``, ``failed``, or ``pending``.

    ``steps`` are lightweight rows for this run (``id``, ``round_index``, ``stage``,
    ``status``). Logs and params are not read. A ``syncing`` step is a fetch still
    in progress. Busy work on any round wins. Otherwise a failed latest job on the
    current round is 失败. A completed run or a closed current round is 已完成.
    """
    jobs = []
    for step in steps or []:
        if step.get('stage') not in LIST_JOB_STAGES:
            continue
        if step.get('status') == 'superseded':
            continue
        jobs.append(step)
    if (run or {}).get('status') == 'running' or any(step.get('status') in _BUSY_STEP_STATUS for step in jobs):
        return 'running'
    current = (run or {}).get('round_index')
    latest = None
    for step in jobs:
        if current is not None and step.get('round_index') != current:
            continue
        if latest is None or (step.get('id') or 0) >= (latest.get('id') or 0):
            latest = step
    if latest and latest.get('status') == 'failed':
        return 'failed'
    if (run or {}).get('status') == 'completed' or round_status == 'closed':
        return 'completed'
    return 'pending'
