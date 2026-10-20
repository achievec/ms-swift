# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import stat
import subprocess
import sys
from pathlib import Path

from rsi.backends import swift_launch, training_python


def _executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_fetch_yaml_train_python_is_used_when_present(tmp_path, monkeypatch):
    configured = tmp_path / 'conda' / 'bin' / 'python'
    _executable(configured)
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text(f'train_python: {configured}\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.delenv('RSI_TRAIN_PYTHON', raising=False)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    argv, _env = swift_launch(['swift', 'sft', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[0] == str(configured)
    assert argv[1:4] == ['-m', 'swift.cli.main', 'sft']
    assert training_python() == str(configured)


def test_missing_config_python_uses_the_current_interpreter(tmp_path, monkeypatch):
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text('train_python: /tmp/rsi-missing-train-python/bin/python\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.delenv('RSI_TRAIN_PYTHON', raising=False)
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    argv, _env = swift_launch(['swift', 'eval', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[0] == sys.executable
    assert argv[1:4] == ['-m', 'swift.cli.main', 'eval']


def test_train_python_and_swift_bin_override_fetch_yaml(tmp_path, monkeypatch):
    configured = tmp_path / 'conda' / 'bin' / 'python'
    chosen = tmp_path / 'override' / 'bin' / 'python'
    _executable(configured)
    _executable(chosen)
    cfg = tmp_path / 'fetch.yaml'
    cfg.write_text(f'train_python: {configured}\n', encoding='utf-8')
    monkeypatch.setenv('RSI_FETCH_CONFIG', str(cfg))
    monkeypatch.delenv('RSI_SWIFT_BIN', raising=False)
    monkeypatch.setenv('RSI_TRAIN_PYTHON', str(chosen))
    argv, _env = swift_launch(['swift', 'rlhf', '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    assert argv[0] == str(chosen)
    assert argv[1:4] == ['-m', 'swift.cli.main', 'rlhf']
    script = tmp_path / 'fake-swift'
    script.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    script.chmod(0o755)
    monkeypatch.setenv('RSI_SWIFT_BIN', str(script))
    argv, _env = swift_launch(['swift', 'sft', '--model', 'x'])
    assert argv[0] == str(script)
    assert argv[1] == 'sft'


def test_bundled_train_python_is_the_gpu_conda_env():
    text = Path(__file__).resolve().parents[1].joinpath('configs', 'fetch.yaml').read_text(encoding='utf-8')
    assert 'conda_env: model-train' in text
    assert 'train_python: /usr/local/conda/envs/model-train/bin/python' in text
    assert 'rsi_home: /opt/meituan/dolphinfs_liubo108/ms-swift/workspace' in text


def _source_run_sh(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    script = (root / 'run.sh').read_text(encoding='utf-8').replace('main "$@"', ':')
    harness = tmp_path / 'run.sh'
    harness.write_text(script, encoding='utf-8')
    config = tmp_path / 'rsi' / 'configs'
    config.mkdir(parents=True, exist_ok=True)
    (config / 'fetch.yaml').write_text(
        (root / 'rsi' / 'configs' / 'fetch.yaml').read_text(encoding='utf-8'), encoding='utf-8')
    return harness


def _run_conda_hook(tmp_path: Path, env: dict, body: str) -> subprocess.CompletedProcess:
    harness = _source_run_sh(tmp_path)
    return subprocess.run(
        ['bash', '-c', f'source "{harness}"; {body}'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15, check=False)


def test_linux_run_sh_activates_conda_env_and_warns_when_missing(tmp_path):
    base = tmp_path / 'conda'
    (base / 'bin').mkdir(parents=True)
    (base / 'etc' / 'profile.d').mkdir(parents=True)
    conda = base / 'bin' / 'conda'
    conda.write_text(
        '#!/bin/bash\n'
        'if [[ "$1" == "info" && "$2" == "--base" ]]; then\n'
        '  printf "%s\\n" "$CONDA_BASE"\n'
        '  exit 0\n'
        'fi\n'
        'echo "conda stub: $*" >&2\n'
        'exit 1\n',
        encoding='utf-8')
    conda.chmod(0o755)
    (base / 'etc' / 'profile.d' / 'conda.sh').write_text(
        'conda() {\n'
        '  if [[ "${1:-}" == "activate" ]]; then\n'
        '    if [[ "${2:-}" == "model-train" ]]; then\n'
        '      export VIRTUAL_ENV="${CONDA_BASE}/envs/model-train"\n'
        '      mkdir -p "${VIRTUAL_ENV}/bin"\n'
        '      printf "#!/bin/sh\\n" > "${VIRTUAL_ENV}/bin/python"\n'
        '      chmod +x "${VIRTUAL_ENV}/bin/python"\n'
        '      return 0\n'
        '    fi\n'
        '    echo "EnvironmentNameNotFound: ${2:-}" >&2\n'
        '    return 1\n'
        '  fi\n'
        '  return 1\n'
        '}\n',
        encoding='utf-8')
    uname_dir = tmp_path / 'darwin-bin'
    uname_dir.mkdir()
    uname = uname_dir / 'uname'
    uname.write_text('#!/bin/sh\nif [ "$1" = "-s" ]; then echo Darwin; exit 0; fi\nexec /usr/bin/uname "$@"\n', encoding='utf-8')
    uname.chmod(0o755)

    env = os.environ.copy()
    env.pop('RSI_CONDA_ENV', None)
    env.pop('CONDA_EXE', None)
    env['CONDA_BASE'] = str(base)
    env['HOME'] = str(tmp_path / 'home')
    env['PATH'] = '/usr/bin:/bin'

    missing = _run_conda_hook(tmp_path, env, 'activate_linux_conda')
    missing_out = missing.stdout + missing.stderr
    assert missing.returncode == 0, missing_out
    assert '未找到 conda' in missing_out
    assert 'model-train' in missing_out
    assert '已激活' not in missing_out

    env['CONDA_EXE'] = str(conda)
    via_exe = _run_conda_hook(tmp_path, env, 'activate_linux_conda')
    via_exe_out = via_exe.stdout + via_exe.stderr
    assert via_exe.returncode == 0, via_exe_out
    assert '已激活 conda 环境：model-train' in via_exe_out
    env.pop('CONDA_EXE')

    env['RSI_CONDA_ENV'] = ''
    disabled = _run_conda_hook(tmp_path, env, 'activate_linux_conda; printf "PY=%s\\n" "${PY:-}"')
    disabled_out = disabled.stdout + disabled.stderr
    assert disabled.returncode == 0, disabled_out
    assert disabled_out.strip() == 'PY='

    env.pop('RSI_CONDA_ENV')
    env['PATH'] = f'{base / "bin"}:/usr/bin:/bin'
    activated = _run_conda_hook(
        tmp_path, env, 'activate_linux_conda; resolve_python; printf "PY=%s\\n" "$PY"')
    activated_out = activated.stdout + activated.stderr
    assert activated.returncode == 0, activated_out
    assert '已激活 conda 环境：model-train' in activated_out
    assert '使用已激活的虚拟环境：' in activated_out
    assert activated_out.strip().endswith(f'{base}/envs/model-train/bin/python')

    env['RSI_CONDA_ENV'] = 'no-such-env'
    absent = _run_conda_hook(tmp_path, env, 'activate_linux_conda; printf "VENV=%s\\n" "${VIRTUAL_ENV:-}"')
    absent_out = absent.stdout + absent.stderr
    assert absent.returncode == 0, absent_out
    assert '无法激活 conda 环境 no-such-env' in absent_out
    assert 'VENV=' in absent_out.splitlines()

    env['RSI_CONDA_ENV'] = 'model-train'
    env['PATH'] = f'{uname_dir}:{base / "bin"}:/usr/bin:/bin'
    macos = _run_conda_hook(tmp_path, env, 'activate_linux_conda; printf "VENV=%s\\n" "${VIRTUAL_ENV:-}"')
    macos_out = macos.stdout + macos.stderr
    assert macos.returncode == 0, macos_out
    assert macos_out.strip() == 'VENV='

    venv = tmp_path / 'rsi' / '.venv' / 'bin' / 'python'
    _executable(venv)
    env['RSI_CONDA_ENV'] = ''
    env['PATH'] = '/usr/bin:/bin'
    fallback = _run_conda_hook(tmp_path, env, 'activate_linux_conda; resolve_python; printf "PY=%s\\n" "$PY"')
    fallback_out = fallback.stdout + fallback.stderr
    assert fallback.returncode == 0, fallback_out
    assert '未找到 conda' not in fallback_out
    assert '使用已有虚拟环境：rsi/.venv' in fallback_out
    assert fallback_out.strip().endswith(str(venv))
