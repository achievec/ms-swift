# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stage machine for one RSI run: data → SFT → eval → RL → eval → promote → next round."""
import json
import re
import shutil
import threading
import uuid
from pathlib import Path

import yaml

from rsi.backends import BackendError, swift_launch
from rsi.commands import STAGE_COMMAND, build_argv, format_shell
from rsi.dataset_validate import MAX_BYTES, validate_file, validate_public_id
from rsi.schemas import STAGE_META, STAGE_ORDER, presets
from rsi.webui_catalog import match_model
from rsi.serving import find_free_port, loopback, port_is_free
from rsi.store import Store

LABELS = {item['id']: item['label'] for item in STAGE_META}
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent / 'examples'
SWIFT_STAGES = {'sft', 'sft_eval', 'rl', 'rl_eval'}
CHECKPOINT_RESET = {
    'data': ['sft', 'rl'],
    'sft': ['sft', 'rl'],
    'sft_eval': ['rl'],
    'rl': ['rl'],
    'rl_eval': [],
    'promote': [],
    'next_round': [],
}


class PipelineError(Exception):

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _loads(text, fallback=None):
    if not text:
        return {} if fallback is None else fallback
    return json.loads(text)


def _merge(defaults: dict, user: dict) -> dict:
    incoming = dict(user or {})
    extra_yaml = incoming.pop('extra_yaml', None)
    merged = dict(defaults)
    for key, value in incoming.items():
        if value is None or value == '':
            continue
        merged[key] = value
    if extra_yaml and str(extra_yaml).strip():
        try:
            loaded = yaml.safe_load(extra_yaml)
        except yaml.YAMLError as exc:
            raise PipelineError(f'额外 YAML 无法解析：{exc}') from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise PipelineError('额外 YAML 必须是键值映射')
        for key, value in loaded.items():
            if value is not None:
                merged[key] = value
        merged['extra_yaml'] = extra_yaml
    return merged


def _metric(step, name):
    if not step or step.get('status') != 'succeeded' or not step.get('result_json'):
        return None
    metrics = _loads(step['result_json']).get('metrics') or {}
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _format_errors(report: dict) -> str:
    parts = []
    for item in report.get('errors') or []:
        if item.get('row'):
            parts.append(f"第 {item['row']} 行：{item['error']}")
        else:
            parts.append(item['error'])
    return '数据格式不符合 ms-swift：' + ('；'.join(parts) if parts else '未知错误')


def _slug(name: str, round_index: int, kind: str) -> str:
    ascii_name = re.sub(r'[^A-Za-z0-9]+', '-', name or '').strip('-').lower()
    if not ascii_name:
        ascii_name = 'rsi-model'
    return f'{ascii_name}-r{round_index}-{kind or "ckpt"}'[:64]


def materialize_general_qa(source: Path, dest_dir: Path):
    """Turn a messages/query jsonl into the folder layout `swift eval --eval_dataset general_qa` expects."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open(encoding='utf-8') as handle, (dest_dir / 'itops.jsonl').open('w', encoding='utf-8') as target:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('query') and item.get('response'):
                query, response = item['query'], item['response']
            else:
                query = ''
                response = ''
                for message in item.get('messages') or []:
                    if not isinstance(message, dict):
                        continue
                    role = message.get('role')
                    content = message.get('content') or ''
                    if role == 'user' and not query:
                        query = content
                    elif role == 'assistant':
                        response = content
            if not query or not response:
                continue
            target.write(json.dumps({'query': query, 'response': response}, ensure_ascii=False) + '\n')
            written += 1
    if written == 0:
        raise PipelineError(f'评测文件没有可交给 swift eval 的问答：{source}')
    return dest_dir, written


def _resolve_existing(value):
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value)
    if path.is_file():
        return str(path.resolve())
    candidate = REPO_ROOT / value
    if candidate.is_file():
        return str(candidate.resolve())
    return value


class Pipeline:

    def __init__(self, home: Path, backend):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.store = Store(self.home / 'rsi.sqlite')
        self.store.recover_interrupted()
        self.examples = EXAMPLES
        from rsi.serving import ServeManager
        self.serve = ServeManager()

    def stage_dir(self, run_id: str, round_index: int, stage: str) -> Path:
        path = self.home / 'runs' / run_id / f'round_{round_index}' / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_run(self, name: str, scenario: str, base_model: str, model_type: str = '', template: str = '') -> dict:
        name = (name or '').strip()
        scenario = (scenario or '').strip()
        base_model = (base_model or '').strip()
        matched = match_model(base_model) or {}
        model_type = (model_type or matched.get('model_type') or '').strip()
        template = (template or matched.get('template') or '').strip()
        if not name or len(name) > 80:
            raise PipelineError('实验名称需要 1 到 80 个字符')
        if not scenario or len(scenario) > 500:
            raise PipelineError('场景说明需要 1 到 500 个字符')
        if not base_model or len(base_model) > 300:
            raise PipelineError('基座模型需要 1 到 300 个字符')
        run_id = uuid.uuid4().hex[:8]
        self.store.create_run(run_id, name, scenario, base_model, model_type, template)
        return self.view(run_id)

    def _apply_model_meta(self, preset: dict, run: dict):
        if run.get('model_type') and not preset.get('model_type'):
            preset['model_type'] = run['model_type']
        if run.get('template') and not preset.get('template'):
            preset['template'] = run['template']

    def list_runs(self) -> list:
        return self.store.list_runs()

    def delete_run(self, run_id: str) -> dict:
        if not self.store.get_run(run_id):
            raise PipelineError('实验不存在', 404)
        self.serve.stop(run_id)
        deleted = self.store.delete_run(run_id)
        if not deleted:
            raise PipelineError('实验不存在', 404)
        return {'ok': True, 'id': run_id, 'workspace_removed': self._remove_run_workspace(run_id)}

    def _remove_run_workspace(self, run_id: str) -> bool:
        """Remove only this experiment's directory under RSI home/runs."""
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id or ''):
            return False
        root = (self.home / 'runs').resolve()
        target = (root / run_id).resolve()
        if target.parent != root or not target.is_dir():
            return False
        shutil.rmtree(target)
        return True

    def _require_run(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        if not run:
            raise PipelineError('实验不存在', 404)
        return run

    def save_upload(self, filename: str, data: bytes) -> dict:
        if len(data) > MAX_BYTES:
            raise PipelineError('文件超过 20MB')
        suffix = Path(filename or 'data.jsonl').suffix.lower()
        if suffix not in {'.jsonl', '.json', '.txt'}:
            suffix = '.jsonl'
        dest_dir = self.home / 'uploads'
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f'{uuid.uuid4().hex}{suffix}'
        dest.write_bytes(data)
        report = validate_file(dest)
        return {'path': str(dest), 'filename': filename, 'validation': report}

    def _upload_path(self, raw: str) -> Path:
        uploads = (self.home / 'uploads').resolve()
        path = Path(raw).resolve()
        if path != uploads and uploads not in path.parents:
            raise PipelineError('上传文件必须位于 RSI 上传目录')
        if not path.is_file():
            raise PipelineError('上传文件不存在')
        return path

    def resolve_dataset(self, params: dict):
        source = params.get('source') or 'bundled'
        if source == 'bundled':
            name = params.get('bundled') or 'itops_mix'
            path = self.examples / f'{name}.jsonl'
            if not path.is_file():
                raise PipelineError(f'未知内置数据集：{name}')
            return str(path.resolve()), validate_file(path)
        if source == 'public':
            report = validate_public_id(params.get('dataset') or '')
            return report.get('dataset') or '', report
        if source == 'upload':
            path = self._upload_path(params.get('dataset') or '')
            return str(path), validate_file(path)
        if source == 'path':
            raw = params.get('dataset') or ''
            path = Path(raw)
            if not path.is_file():
                raise PipelineError(f'本地文件不存在：{raw}')
            return str(path.resolve()), validate_file(path)
        raise PipelineError('未知数据来源')

    def validate_request(self, params: dict) -> dict:
        _dataset, report = self.resolve_dataset(params or {})
        return report

    def _data_context(self, run_id: str, round_index: int):
        active = self.store.latest_active_steps(run_id, round_index)
        step = active.get('data')
        if not step or step['status'] != 'succeeded':
            return {}, {}
        return _loads(step['params_json']), _loads(step['result_json'])

    def _defaults_for(self, run: dict, rnd: dict, stage: str) -> dict:
        preset = dict(presets().get(stage) or {})
        if stage == 'data':
            if rnd['round_index'] > 1:
                previous = self.store.latest_active_steps(run['id'], rnd['round_index'] - 1)
                nxt = previous.get('next_round')
                reuse = True
                if nxt and nxt['status'] == 'succeeded':
                    reuse = bool(_loads(nxt['params_json']).get('reuse_dataset', True))
                prev_data = previous.get('data')
                if reuse and prev_data and prev_data['status'] == 'succeeded':
                    carried = _loads(prev_data['params_json'])
                    carried.pop('extra_yaml', None)
                    preset.update(carried)
            return preset

        data_params, data_result = self._data_context(run['id'], rnd['round_index'])
        if stage == 'sft':
            preset['model'] = rnd['base_model']
            self._apply_model_meta(preset, run)
            if data_result.get('dataset'):
                preset['dataset'] = data_result['dataset']
            for key in ('max_length', 'split_dataset_ratio', 'dataset_num_proc', 'dataloader_num_workers',
                        'padding_free', 'val_dataset'):
                if data_params.get(key) not in (None, ''):
                    preset[key] = data_params[key]
            preset['output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], 'sft'))
            return preset

        if stage == 'rl':
            sft_ckpt = self.store.active_checkpoint(run['id'], rnd['round_index'], 'sft')
            preset['model'] = sft_ckpt['path'] if sft_ckpt else ''
            self._apply_model_meta(preset, run)
            if data_result.get('dataset'):
                preset['dataset'] = data_result['dataset']
            preset['output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], 'rl'))
            preset['external_plugins'] = _resolve_existing(preset.get('external_plugins'))
            return preset

        if stage in {'sft_eval', 'rl_eval'}:
            kind = 'sft' if stage == 'sft_eval' else 'rl'
            checkpoint = self.store.active_checkpoint(run['id'], rnd['round_index'], kind)
            preset['model'] = checkpoint['path'] if checkpoint else ''
            self._apply_model_meta(preset, run)
            preset['eval_output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], stage) / 'eval_output')
            preset['custom_eval_config'] = _resolve_existing(preset.get('custom_eval_config'))
            return preset

        if stage == 'promote':
            _candidates, recommendation = self.candidates(run, rnd)
            preset['choice'] = recommendation
            preset['reason'] = preset.get('reason') or ''
            return preset

        if stage == 'serve':
            promo = self.store.latest_promotion(run['id'])
            if promo:
                preset['model'] = promo['promoted_checkpoint']
                preset['served_model_name'] = _slug(run['name'], promo['round_index'], promo['promoted_kind'])
            self._apply_model_meta(preset, run)
            return preset
        return preset

    def candidates(self, run: dict, rnd: dict):
        active = self.store.latest_active_steps(run['id'], rnd['round_index'])
        sft = self.store.active_checkpoint(run['id'], rnd['round_index'], 'sft')
        rl = self.store.active_checkpoint(run['id'], rnd['round_index'], 'rl')
        base_score = None
        if rnd['round_index'] > 1:
            previous = self.store.get_round(run['id'], rnd['round_index'] - 1)
            if previous and previous.get('promoted_score') is not None:
                base_score = previous['promoted_score']
        rows = [
            {
                'kind': 'base',
                'label': '当前基座',
                'path': rnd['base_model'],
                'score': base_score,
            },
            {
                'kind': 'sft',
                'label': 'SFT checkpoint',
                'path': sft['path'] if sft else None,
                'score': _metric(active.get('sft_eval'), 'accuracy'),
            },
            {
                'kind': 'rl',
                'label': 'RL checkpoint',
                'path': rl['path'] if rl else None,
                'score': _metric(active.get('rl_eval'), 'accuracy'),
            },
        ]
        preference = {'rl': 3, 'sft': 2, 'base': 1}
        best = None
        best_key = None
        for row in rows:
            if not row['path'] or row['score'] is None:
                continue
            key = (row['score'], preference[row['kind']])
            if best_key is None or key > best_key:
                best_key = key
                best = row['kind']
        if best is None:
            if rl:
                best = 'rl'
            elif sft:
                best = 'sft'
            else:
                best = 'base'
        return rows, best

    def _prepare(self, run, rnd, stage, user, strict: bool):
        merged = _merge(self._defaults_for(run, rnd, stage), user)
        side = {}
        if stage == 'data':
            dataset, report = self.resolve_dataset(merged)
            if strict and not report['ok']:
                raise PipelineError(_format_errors(report))
            merged['dataset'] = dataset
            side['validation'] = report
            return merged, side

        if stage in {'sft', 'rl'}:
            dataset = merged.get('dataset') or ''
            if strict:
                if not dataset:
                    raise PipelineError('缺少数据集，请先完成数据阶段')
                path = Path(dataset)
                if path.is_file():
                    report = validate_file(path)
                    if not report['ok']:
                        raise PipelineError(_format_errors(report))
                else:
                    report = validate_public_id(dataset)
                    if not report['ok']:
                        raise PipelineError(_format_errors(report))
                if not merged.get('model'):
                    raise PipelineError('缺少模型路径')
                if not merged.get('output_dir'):
                    merged['output_dir'] = str(self.stage_dir(run['id'], rnd['round_index'], stage))
            elif not dataset:
                merged['dataset'] = '<dataset>'
            return merged, side

        if stage in {'sft_eval', 'rl_eval'}:
            custom = merged.get('custom_eval_config')
            if custom:
                path = Path(str(custom))
                if path.is_file():
                    report = validate_file(path)
                    if strict and not report['ok']:
                        raise PipelineError(_format_errors(report))
                    side['eval_rows'] = report.get('num_rows')
                    merged['custom_eval_config'] = str(path.resolve())
                elif strict:
                    raise PipelineError(f'自定义评测文件不存在：{custom}')
            if strict and not merged.get('model'):
                raise PipelineError('缺少待评测的 checkpoint')
            extra = merged.get('extra_eval_args') or {}
            if isinstance(extra, str) and extra.strip():
                extra = json.loads(extra)
            if not isinstance(extra, dict):
                extra = {}
            if self.backend.name == 'swift' and str(merged.get('eval_dataset') or 'itops-heldout') == 'itops-heldout':
                source = Path(str(merged.get('custom_eval_config') or ''))
                if not source.is_file():
                    source = EXAMPLES / 'itops_eval.jsonl'
                qa_dir, rows = materialize_general_qa(
                    source, self.stage_dir(run['id'], rnd['round_index'], stage) / 'general_qa')
                side['eval_rows'] = rows
                merged['eval_dataset'] = 'general_qa'
                merged['eval_dataset_args'] = {
                    'general_qa': {
                        'local_path': str(qa_dir),
                        'subset_list': ['itops'],
                    },
                }
                extra.pop('custom_eval_config', None)
            elif merged.get('custom_eval_config'):
                extra['custom_eval_config'] = merged['custom_eval_config']
            if extra:
                merged['extra_eval_args'] = extra
            else:
                merged.pop('extra_eval_args', None)
            if not merged.get('eval_dataset'):
                if strict:
                    raise PipelineError('请填写 eval_dataset')
                merged['eval_dataset'] = 'itops-heldout'
            return merged, side

        if stage == 'promote':
            rows, _recommendation = self.candidates(run, rnd)
            choice = (merged.get('choice') or '').strip()
            if strict and choice not in {'base', 'sft', 'rl'}:
                raise PipelineError('晋级选择必须是 base、sft 或 rl')
            picked = next((row for row in rows if row['kind'] == choice), None)
            if strict and (not picked or not picked.get('path')):
                raise PipelineError(f'没有可用的 {choice} 检查点')
            side['candidates'] = rows
            side['picked'] = picked
            return merged, side

        if stage == 'next_round':
            if strict and not rnd.get('promoted_checkpoint'):
                raise PipelineError('还没有晋级结果')
            return merged, side
        raise PipelineError('未知阶段')

    def _status_for(self, stage: str, active: dict) -> str:
        step = active.get(stage)
        if step and step['status'] in {'succeeded', 'failed', 'running'}:
            return step['status']
        index = STAGE_ORDER.index(stage)
        for previous in STAGE_ORDER[:index]:
            if active.get(previous, {}).get('status') != 'succeeded':
                return 'locked'
        return 'ready'

    def _form_state(self, run, rnd, stage, active, recommendation):
        defaults = self._defaults_for(run, rnd, stage)
        if stage == 'promote':
            defaults['choice'] = recommendation
        step = active.get(stage)
        if step and step['status'] == 'succeeded':
            values = _loads(step['params_json'])
        else:
            values = defaults
        result = _loads(step['result_json']) if step and step.get('result_json') else None
        return {
            'status': self._status_for(stage, active),
            'values': values,
            'preset': defaults,
            'result': result,
            'logs': step['logs'] if step else '',
            'command': step['command'] if step else '',
            'started_at': step['started_at'] if step else None,
            'ended_at': step['ended_at'] if step else None,
            'step_id': step['id'] if step else None,
            'backend': step['backend'] if step else None,
        }

    def view(self, run_id: str, round_index=None) -> dict:
        run = self._require_run(run_id)
        if round_index is None:
            round_index = run['round_index']
        else:
            round_index = int(round_index)
        rnd = self.store.get_round(run_id, round_index)
        if not rnd:
            raise PipelineError('轮次不存在', 404)
        active = self.store.latest_active_steps(run_id, round_index)
        candidates, recommendation = self.candidates(run, rnd)
        forms = {}
        board = []
        for meta in STAGE_META:
            stage = meta['id']
            state = self._form_state(run, rnd, stage, active, recommendation)
            forms[stage] = state
            metric = None
            metrics = (state.get('result') or {}).get('metrics') or {}
            if 'accuracy' in metrics:
                metric = f"acc {metrics['accuracy']}"
            elif 'train_loss' in metrics:
                metric = f"loss {metrics['train_loss']}"
            elif 'reward' in metrics:
                metric = f"reward {metrics['reward']}"
            board.append({'id': stage, 'label': meta['label'], 'summary': meta['summary'], 'status': state['status'],
                          'metric': metric})
        service = self.store.latest_service(run_id)
        serve_values = self._defaults_for(run, rnd, 'serve')
        if service and service.get('status') == 'running' and service.get('params_json'):
            serve_values = _loads(service['params_json'])
        if service and service.get('api_key'):
            service = dict(service)
            service['api_key_set'] = True
            service['api_key'] = ''
        steps = []
        for row in self.store.list_steps(run_id):
            item = dict(row)
            item['params'] = _loads(row['params_json'])
            item['result'] = _loads(row['result_json']) if row.get('result_json') else None
            steps.append(item)
        return {
            'run': run,
            'round': rnd,
            'rounds': self.store.list_rounds(run_id),
            'mutable': rnd['round_index'] == run['round_index'] and rnd['status'] == 'active',
            'board': board,
            'forms': forms,
            'candidates': candidates,
            'recommendation': recommendation,
            'serve_form': serve_values,
            'steps': steps,
            'checkpoints': self.store.list_checkpoints(run_id),
            'service': service,
            'services': self._public_services(run_id),
            'backend': self.backend.name,
        }

    def _public_services(self, run_id: str) -> list:
        rows = []
        for service in self.store.list_services(run_id):
            item = dict(service)
            item['api_key_set'] = bool(item.get('api_key'))
            item['api_key'] = ''
            rows.append(item)
        return rows

    def _assert_can_run(self, run: dict, stage: str):
        if stage not in STAGE_ORDER:
            raise PipelineError('未知阶段')
        rnd = self.store.get_round(run['id'], run['round_index'])
        if not rnd or rnd['status'] != 'active':
            raise PipelineError('当前轮次已结束，不能再改')
        active = self.store.latest_active_steps(run['id'], run['round_index'])
        index = STAGE_ORDER.index(stage)
        for previous in STAGE_ORDER[:index]:
            if active.get(previous, {}).get('status') != 'succeeded':
                raise PipelineError(f'请先完成「{LABELS[previous]}」')

    def preview(self, run_id: str, stage: str, params: dict) -> dict:
        if stage == 'serve':
            run = self._require_run(run_id)
            merged = self._prepare_serve(run, params or {}, strict=False)
            return {'params': merged, 'command': format_shell(build_argv('serve', merged))}
        if stage not in STAGE_ORDER:
            raise PipelineError('未知阶段')
        run = self._require_run(run_id)
        rnd = self.store.get_round(run_id, run['round_index'])
        merged, _side = self._prepare(run, rnd, stage, params or {}, strict=False)
        command = format_shell(build_argv(stage, merged)) if stage in STAGE_COMMAND else ''
        return {'params': merged, 'command': command}

    def run_stage(self, run_id: str, stage: str, params: dict) -> dict:
        run = self._require_run(run_id)
        self._assert_can_run(run, stage)
        rnd = self.store.get_round(run_id, run['round_index'])
        merged, side = self._prepare(run, rnd, stage, params or {}, strict=True)
        index = STAGE_ORDER.index(stage)
        with self.store.transaction() as conn:
            self.store.supersede(conn, run_id, rnd['round_index'], STAGE_ORDER[index:])
            self.store.deactivate_checkpoints(conn, run_id, rnd['round_index'], CHECKPOINT_RESET.get(stage, []))
            if index <= STAGE_ORDER.index('promote') and stage != 'next_round':
                self.store.clear_promotion(conn, run_id, rnd['round_index'])
            self.store.set_current_stage(conn, run_id, stage)
            step_id = self.store.insert_step(conn, run_id, rnd['round_index'], stage, merged, self.backend.name)

        if stage in SWIFT_STAGES and self.backend.name != 'mock':
            thread = threading.Thread(
                target=self._run_swift_stage,
                args=(run_id, rnd['round_index'], stage, step_id, merged, side),
                daemon=True)
            thread.start()
            return {
                'ok': True,
                'status': 'running',
                'step_id': step_id,
                'command': format_shell(build_argv(stage, merged)),
                'result': None,
                'view': self.view(run_id),
            }

        try:
            outcome = self._execute(run, rnd, stage, merged, side)
        except (PipelineError, BackendError) as exc:
            logs = getattr(exc, 'logs', '') or str(exc)
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step_id, 'failed', {'error': str(exc), 'backend': self.backend.name}, logs,
                                       format_shell(build_argv(stage, merged)) if stage in STAGE_COMMAND else '')
            raise PipelineError(str(exc)) from exc

        self._commit_success(run_id, rnd['round_index'], stage, step_id, merged, outcome)
        refreshed = self.store.get_run(run_id)
        return {
            'ok': True,
            'status': 'succeeded',
            'step_id': step_id,
            'command': outcome.get('command') or '',
            'result': outcome.get('result'),
            'view': self.view(refreshed['id'], refreshed['round_index'] if stage == 'next_round' else rnd['round_index']),
        }

    def _run_swift_stage(self, run_id, round_index, stage, step_id, params, side):
        run = self.store.get_run(run_id)
        rnd = self.store.get_round(run_id, round_index)
        try:
            outcome = self._execute(run, rnd, stage, params, side)
        except (PipelineError, BackendError) as exc:
            logs = getattr(exc, 'logs', '') or str(exc)
            with self.store.transaction() as conn:
                self.store.finish_step(conn, step_id, 'failed', {'error': str(exc), 'backend': self.backend.name}, logs,
                                       format_shell(build_argv(stage, params)))
            return
        self._commit_success(run_id, round_index, stage, step_id, params, outcome)

    def _execute(self, run, rnd, stage, params, side) -> dict:
        if stage == 'data':
            report = side['validation']
            formats = ', '.join(f'{name}×{count}' for name, count in (report.get('formats') or {}).items()) or '无'
            rows = report.get('num_rows')
            logs = (f'[rsi] 数据校验通过。来源={params.get("source")} 行数={rows} 格式={formats}\n'
                    f'[rsi] dataset={params.get("dataset")}\n')
            for warning in report.get('warnings') or []:
                logs += f'[rsi] 注意：{warning}\n'
            return {
                'command': '',
                'logs': logs,
                'result': {
                    'backend': self.backend.name,
                    'mock': self.backend.name == 'mock',
                    'dataset': params.get('dataset'),
                    'validation': report,
                },
            }

        if stage in SWIFT_STAGES:
            context = {
                'round_index': rnd['round_index'],
                'run_id': run['id'],
                'workdir': params.get('output_dir') or params.get('eval_output_dir'),
                'eval_rows': side.get('eval_rows'),
                'scenario': run['scenario'],
            }
            if context['workdir']:
                Path(context['workdir']).mkdir(parents=True, exist_ok=True)
            produced = self.backend.execute(stage, params, context)
            result = {
                'backend': self.backend.name,
                'mock': produced.get('mock', self.backend.name == 'mock'),
                'metrics': produced.get('metrics') or {},
                'checkpoint': produced.get('checkpoint'),
            }
            outcome = {
                'command': produced.get('command') or '',
                'logs': produced.get('logs') or '',
                'result': result,
            }
            if produced.get('checkpoint'):
                outcome['checkpoint'] = produced['checkpoint']
                outcome['checkpoint_kind'] = 'sft' if stage == 'sft' else 'rl' if stage == 'rl' else None
                outcome['metrics'] = produced.get('metrics') or {}
            return outcome

        if stage == 'promote':
            picked = side['picked']
            score = picked.get('score')
            score_text = '未评测' if score is None else str(score)
            reason = (params.get('reason') or '').strip()
            logs = (f"[rsi] 晋级选择 {picked['kind']}，分数 {score_text}。\n"
                    f"[rsi] 路径：{picked['path']}\n")
            if reason:
                logs += f'[rsi] 说明：{reason}\n'
            return {
                'command': '',
                'logs': logs,
                'result': {
                    'backend': self.backend.name,
                    'mock': self.backend.name == 'mock',
                    'choice': picked['kind'],
                    'promoted_checkpoint': picked['path'],
                    'score': score,
                    'candidates': side['candidates'],
                },
                'promotion': {
                    'checkpoint': picked['path'],
                    'kind': picked['kind'],
                    'score': score,
                },
            }

        if stage == 'next_round':
            nxt = rnd['round_index'] + 1
            logs = (f"[rsi] 第 {rnd['round_index']} 轮结束。下一轮基座：{rnd['promoted_checkpoint']}\n")
            if params.get('note'):
                logs += f"[rsi] 备注：{params['note']}\n"
            return {
                'command': '',
                'logs': logs,
                'result': {
                    'backend': self.backend.name,
                    'mock': self.backend.name == 'mock',
                    'next_round': nxt,
                    'base_model': rnd['promoted_checkpoint'],
                    'reuse_dataset': bool(params.get('reuse_dataset', True)),
                },
                'next_base': rnd['promoted_checkpoint'],
                'note': params.get('note') or '',
            }
        raise PipelineError('未知阶段')

    def _commit_success(self, run_id, round_index, stage, step_id, params, outcome):
        with self.store.transaction() as conn:
            self.store.finish_step(conn, step_id, 'succeeded', outcome.get('result'), outcome.get('logs'),
                                   outcome.get('command') or '')
            if outcome.get('checkpoint') and outcome.get('checkpoint_kind'):
                self.store.add_checkpoint(conn, run_id, round_index, outcome['checkpoint_kind'], outcome['checkpoint'],
                                          outcome.get('metrics') or {})
            if stage == 'promote' and outcome.get('promotion'):
                promo = outcome['promotion']
                self.store.set_promotion(conn, run_id, round_index, promo['checkpoint'], promo['kind'], promo['score'])
                self.store.set_current_stage(conn, run_id, 'next_round')
            elif stage == 'next_round':
                self.store.close_round_and_open_next(conn, run_id, round_index, outcome.get('note') or '',
                                                     outcome['next_base'])
            else:
                nxt = STAGE_ORDER[STAGE_ORDER.index(stage) + 1]
                self.store.set_current_stage(conn, run_id, nxt)

    def _prepare_serve(self, run: dict, user: dict, strict: bool) -> dict:
        merged = _merge(self._defaults_for(run, {'round_index': run['round_index'], 'base_model': run['base_model']},
                                           'serve'), user)
        # _defaults_for serve uses latest promotion and ignores the dummy round except for signature.
        if strict and not merged.get('model'):
            raise PipelineError('完成至少一次晋级后才能部署')
        if not merged.get('served_model_name'):
            merged['served_model_name'] = _slug(run['name'], run['round_index'], 'model')
        try:
            port = int(merged.get('port') or 0)
        except (TypeError, ValueError) as exc:
            raise PipelineError('端口必须是整数') from exc
        if port == 0:
            port = find_free_port()
        merged['port'] = port
        merged['host'] = merged.get('host') or '127.0.0.1'
        return merged

    def _serve_backend_name(self) -> str:
        import os
        return os.environ.get('RSI_SERVE_BACKEND', self.backend.name).strip().lower() or 'mock'

    def start_serve(self, run_id: str, params: dict) -> dict:
        run = self._require_run(run_id)
        promo = self.store.latest_promotion(run_id)
        if not promo:
            raise PipelineError('完成至少一次晋级后才能部署')
        merged = self._prepare_serve(run, params or {}, strict=True)
        host = merged['host']
        port = int(merged['port'])
        previous = self.store.latest_service(run_id)
        if previous and previous['status'] == 'running':
            self.serve.stop(run_id)
            with self.store.transaction() as conn:
                self.store.stop_service_row(conn, previous['id'], (previous.get('logs') or '') + '\n[rsi] 被新的部署替换。\n')
        if not port_is_free(host, port):
            raise PipelineError(f'端口 {port} 已被占用，换一个端口再启动')
        command = format_shell(build_argv('serve', merged))
        backend_name = self._serve_backend_name()
        if backend_name == 'swift':
            self._start_swift_serve(run, merged, command)
            logs = f'[rsi] 已启动 swift deploy（后台进程）。\n{command}\n'
        else:
            try:
                self.serve.start_mock(run_id, host, port, merged['served_model_name'], merged.get('api_key') or '',
                                      run['scenario'], merged['model'])
            except TimeoutError as exc:
                raise PipelineError(f'推理服务没有在预期时间内起来：{exc}') from exc
            logs = (f'[mock] OpenAI 兼容服务已监听 http://{loopback(host)}:{port}/v1\n'
                    '[mock] 没有加载权重。/v1/models 与 /v1/chat/completions 可直接调用。\n'
                    f'{command}\n')
        result = {
            'backend': backend_name,
            'mock': backend_name != 'swift',
            'base_url': f'http://{loopback(host)}:{port}/v1',
            'served_model_name': merged['served_model_name'],
            'checkpoint': merged['model'],
        }
        with self.store.transaction() as conn:
            self.store.insert_service(
                conn, {
                    'run_id': run_id,
                    'checkpoint': merged['model'],
                    'served_model_name': merged['served_model_name'],
                    'host': host,
                    'port': port,
                    'api_key': merged.get('api_key') or '',
                    'backend': backend_name,
                    'params': merged,
                    'logs': logs,
                    'command': command,
                })
            step_id = self.store.insert_step(conn, run_id, run['round_index'], 'serve', merged, backend_name)
            self.store.finish_step(conn, step_id, 'succeeded', result, logs, command)
        return {'ok': True, 'result': result, 'command': command, 'view': self.view(run_id)}

    def _start_swift_serve(self, run, params, command):
        import subprocess
        logical = build_argv('serve', params)
        argv, env = swift_launch(logical)
        log_dir = self.stage_dir(run['id'], run['round_index'], 'serve')
        log_path = log_dir / 'deploy.log'
        log_file = log_path.open('w', encoding='utf-8')
        log_file.write(command + '\n')
        log_file.write('exec: ' + ' '.join(argv) + '\n')
        log_file.flush()
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        self.serve.track(run['id'], process)

    def stop_serve(self, run_id: str) -> dict:
        run = self._require_run(run_id)
        service = self.store.latest_service(run_id)
        if not service or service['status'] != 'running':
            raise PipelineError('没有正在运行的推理服务')
        self.serve.stop(run_id)
        with self.store.transaction() as conn:
            self.store.stop_service_row(conn, service['id'], (service.get('logs') or '') + '\n[rsi] 服务已停止。\n')
            step_id = self.store.insert_step(conn, run_id, run['round_index'], 'serve_stop',
                                             {'service_id': service['id']}, service['backend'])
            self.store.finish_step(conn, step_id, 'succeeded', {'stopped': service['id']}, '[rsi] 推理服务已停止。\n', '')
        return {'ok': True, 'view': self.view(run_id)}

    def chat(self, run_id: str, messages: list, temperature=None) -> dict:
        self._require_run(run_id)
        service = self.store.latest_service(run_id)
        if not service or service['status'] != 'running':
            raise PipelineError('推理服务未启动')
        if not messages:
            raise PipelineError('messages 不能为空')
        import urllib.error
        import urllib.request
        host = loopback(service['host'])
        url = f"http://{host}:{service['port']}/v1/chat/completions"
        payload = {'model': service['served_model_name'], 'messages': messages}
        if temperature is not None:
            payload['temperature'] = temperature
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        if service.get('api_key'):
            request.add_header('Authorization', f"Bearer {service['api_key']}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise PipelineError(f'推理服务返回 {exc.code}: {detail}', exc.code if exc.code in {401, 404} else 502) from exc
        except urllib.error.URLError as exc:
            raise PipelineError(f'无法连接推理服务：{exc.reason}', 502) from exc
