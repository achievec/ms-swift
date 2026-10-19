# Copyright (c) ModelScope Contributors. All rights reserved.
"""Model, template, and dataset ids for the RSI forms.

The lists match what the Gradio Web-UI shows (`get_model_list`,
`ModelType.get_model_name_list`, template names, `get_dataset_list`) but are
parsed from the registry sources so this package still does not import torch.
"""
from functools import lru_cache
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / 'swift' / 'model' / 'models'
MODEL_CONSTANT = REPO_ROOT / 'swift' / 'model' / 'constant.py'
TEMPLATE_CONSTANT = REPO_ROOT / 'swift' / 'template' / 'constant.py'
DATASET_DIR = REPO_ROOT / 'swift' / 'dataset'

_CALLS = ('ModelMeta', 'ModelGroup', 'Model')
_TYPE_CLASSES = ('LLMModelType', 'MLLMModelType', 'BertModelType', 'RMModelType')
_TEMPLATE_CLASSES = ('LLMTemplateType', 'MLLMTemplateType', 'RMTemplateType', 'TemplateType')
_STRING = re.compile(r'''['\"]([^'\"]+)['\"]''')
_TEMPLATE = re.compile(r'TemplateType\.(\w+)')
_META_TEMPLATE = re.compile(r'template\s*=\s*TemplateType\.(\w+)')
_ASSIGN = re.compile(r'''^\s+([A-Za-z_]\w*)\s*=\s*['\"]([^'\"]+)['\"]\s*(?:#.*)?$''', re.M)
_DATASET = re.compile(r'''ms_dataset_id\s*=\s*['\"]([^'\"]+)['\"]''')


def _skip_string(text, index):
    quote = text[index]
    if text.startswith(quote * 3, index):
        end = text.find(quote * 3, index + 3)
        return len(text) if end < 0 else end + 3
    index += 1
    while index < len(text):
        char = text[index]
        if char == '\\':
            index += 2
            continue
        if char == quote:
            return index + 1
        if char == '\n':
            return index
        index += 1
    return index


def _find_calls(text):
    """Return (name, body) for Model / ModelGroup / ModelMeta calls."""
    found = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == '#':
            newline = text.find('\n', index)
            index = length if newline < 0 else newline + 1
            continue
        if char in {'"', "'"}:
            index = _skip_string(text, index)
            continue
        if char != 'M':
            index += 1
            continue
        matched = None
        for name in _CALLS:
            if not text.startswith(name + '(', index):
                continue
            previous = text[index - 1] if index else ''
            if previous.isalnum() or previous == '_':
                continue
            matched = name
            break
        if not matched:
            index += 1
            continue
        body_start = index + len(matched) + 1
        depth = 1
        cursor = body_start
        while cursor < length and depth:
            char = text[cursor]
            if char == '#':
                newline = text.find('\n', cursor)
                cursor = length if newline < 0 else newline + 1
                continue
            if char in {'"', "'"}:
                cursor = _skip_string(text, cursor)
                continue
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    found.append((matched, text[body_start:cursor]))
                    cursor += 1
                    break
            cursor += 1
        index = cursor
    return found


def _class_strings(path: Path, class_names):
    text = path.read_text(encoding='utf-8')
    values = []
    seen = set()
    for class_name in class_names:
        marker = f'class {class_name}'
        start = text.find(marker)
        if start < 0:
            continue
        next_class = text.find('\nclass ', start + len(marker))
        body = text[start:next_class if next_class > 0 else None]
        for _name, value in _ASSIGN.findall(body):
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
    return values


def _mask(body, spans):
    chars = list(body)
    for start, end in spans:
        for index in range(start, min(end, len(chars))):
            chars[index] = ' '
    return ''.join(chars)


def _first_string(body):
    # Skip strings that sit inside comments. _find_calls already dropped comments
    # from the call scanner; the body itself may still contain '#' comments.
    match = _STRING.search(body)
    return match.group(1) if match else ''


def _parse_models(type_names):
    known_types = set(type_names)
    models = []
    mapping = {}
    seen = set()
    if not MODEL_DIR.is_dir():
        return models, mapping
    for path in sorted(MODEL_DIR.glob('*.py')):
        try:
            calls = _find_calls(path.read_text(encoding='utf-8'))
        except OSError:
            continue
        for name, body in calls:
            if name != 'ModelMeta':
                continue
            head = body.lstrip()
            model_type = None
            typed = re.match(r'[A-Za-z_]\w*\.([A-Za-z_]\w*)', head)
            if typed and typed.group(1) in known_types:
                model_type = typed.group(1)
            elif head[:1] in {'"', "'"}:
                literal = _first_string(head)
                if literal in known_types:
                    model_type = literal
            group_spans = []
            for group_name, group_body in _find_calls(body):
                if group_name != 'ModelGroup':
                    continue
                # Locate this group inside the meta body so a meta-level template
                # is not confused with a group template.
                snippet = 'ModelGroup(' + group_body + ')'
                at = body.find(snippet)
                if at >= 0:
                    group_spans.append((at, at + len(snippet)))
                templates = _TEMPLATE.findall(group_body)
                template = templates[0] if templates else None
                for inner_name, inner_body in _find_calls(group_body):
                    if inner_name != 'Model':
                        continue
                    model_id = _first_string(inner_body)
                    if not model_id or model_id in seen:
                        continue
                    seen.add(model_id)
                    models.append(model_id)
                    mapping[model_id] = {'model_type': model_type, 'template': template}
            masked = _mask(body, group_spans)
            meta_template = _META_TEMPLATE.search(masked)
            fallback = meta_template.group(1) if meta_template else None
            if not fallback:
                continue
            # Groups with no TemplateType of their own inherit the ModelMeta template.
            for group_name, group_body in _find_calls(body):
                if group_name != 'ModelGroup' or _TEMPLATE.search(group_body):
                    continue
                for inner_name, inner_body in _find_calls(group_body):
                    if inner_name != 'Model':
                        continue
                    model_id = _first_string(inner_body)
                    current = mapping.get(model_id)
                    if current and current['template'] is None:
                        current['template'] = fallback
    return models, mapping


def _parse_datasets():
    found = []
    seen = set()
    if not DATASET_DIR.is_dir():
        return found
    for path in sorted(DATASET_DIR.rglob('*.py')):
        try:
            text = path.read_text(encoding='utf-8')
        except OSError:
            continue
        for dataset_id in _DATASET.findall(text):
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            found.append(dataset_id)
    return found


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    model_types = _class_strings(MODEL_CONSTANT, _TYPE_CLASSES) if MODEL_CONSTANT.is_file() else []
    templates = _class_strings(TEMPLATE_CONSTANT, _TEMPLATE_CLASSES) if TEMPLATE_CONSTANT.is_file() else []
    models, mapping = _parse_models(model_types)
    return {
        'models': models,
        'model_types': model_types,
        'templates': templates,
        'datasets': _parse_datasets(),
        'by_model': mapping,
    }


def match_model(model_id: str) -> dict | None:
    if not model_id:
        return None
    return load_catalog()['by_model'].get(model_id.strip())


# Documented in docs/source/Instruction/Evaluation.md. Fixed here so the form
# does not import EvalScope or need a network call to discover dataset names.
EVAL_DATASETS_BY_BACKEND = {
    'Native': (
        'arc',
        'bbh',
        'ceval',
        'cmmlu',
        'competition_math',
        'general_qa',
        'gpqa',
        'gsm8k',
        'hellaswag',
        'humaneval',
        'ifeval',
        'iquiz',
        'mmlu',
        'mmlu_pro',
        'race',
        'trivia_qa',
        'truthful_qa',
    ),
    'OpenCompass': (
        'obqa',
        'cmb',
        'AX_b',
        'siqa',
        'nq',
        'mbpp',
        'winogrande',
        'mmlu',
        'BoolQ',
        'cluewsc',
        'ocnli',
        'lambada',
        'CMRC',
        'ceval',
        'csl',
        'cmnli',
        'bbh',
        'ReCoRD',
        'math',
        'humaneval',
        'eprstmt',
        'WSC',
        'storycloze',
        'MultiRC',
        'RTE',
        'chid',
        'gsm8k',
        'AX_g',
        'bustm',
        'afqmc',
        'piqa',
        'lcsts',
        'strategyqa',
        'Xsum',
        'agieval',
        'ocnli_fc',
        'C3',
        'tnews',
        'race',
        'triviaqa',
        'CB',
        'WiC',
        'hellaswag',
        'summedits',
        'GaokaoBench',
        'ARC_e',
        'COPA',
        'ARC_c',
        'DRCD',
    ),
    'VLMEvalKit': (
        'COCO_VAL',
        'MME',
        'HallusionBench',
        'POPE',
        'MMBench_DEV_EN',
        'MMBench_TEST_EN',
        'MMBench_DEV_CN',
        'MMBench_TEST_CN',
        'MMBench',
        'MMBench_CN',
        'MMBench_DEV_EN_V11',
        'MMBench_TEST_EN_V11',
        'MMBench_DEV_CN_V11',
        'MMBench_TEST_CN_V11',
        'MMBench_V11',
        'MMBench_CN_V11',
        'SEEDBench_IMG',
        'SEEDBench2',
        'SEEDBench2_Plus',
        'ScienceQA_VAL',
        'ScienceQA_TEST',
        'MMT-Bench_ALL_MI',
        'MMT-Bench_ALL',
        'MMT-Bench_VAL_MI',
        'MMT-Bench_VAL',
        'AesBench_VAL',
        'AesBench_TEST',
        'CCBench',
        'AI2D_TEST',
        'MMStar',
        'RealWorldQA',
        'MLLMGuard_DS',
        'BLINK',
        'OCRVQA_TEST',
        'OCRVQA_TESTCORE',
        'TextVQA_VAL',
        'DocVQA_VAL',
        'DocVQA_TEST',
        'InfoVQA_VAL',
        'InfoVQA_TEST',
        'ChartQA_TEST',
        'MathVision',
        'MathVision_MINI',
        'MMMU_DEV_VAL',
        'MMMU_TEST',
        'OCRBench',
        'MathVista_MINI',
        'LLaVABench',
        'MMVet',
        'MTVQA_TEST',
        'MMLongBench_DOC',
        'VCR_EN_EASY_500',
        'VCR_EN_EASY_100',
        'VCR_EN_EASY_ALL',
        'VCR_EN_HARD_500',
        'VCR_EN_HARD_100',
        'VCR_EN_HARD_ALL',
        'VCR_ZH_EASY_500',
        'VCR_ZH_EASY_100',
        'VCR_ZH_EASY_ALL',
        'VCR_ZH_HARD_500',
        'VCR_ZH_HARD_100',
        'VCR_ZH_HARD_ALL',
        'MMDU',
        'MMBench-Video',
        'Video-MME',
    ),
}


def eval_datasets_by_backend() -> dict:
    return {backend: list(names) for backend, names in EVAL_DATASETS_BY_BACKEND.items()}


def eval_dataset_choices(backend: str = 'Native') -> list:
    names = EVAL_DATASETS_BY_BACKEND.get(backend)
    if names is None:
        names = EVAL_DATASETS_BY_BACKEND['Native']
    return list(names)
