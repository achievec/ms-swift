# Copyright (c) ModelScope Contributors. All rights reserved.
import subprocess
from pathlib import Path


def test_sync_script_parses_and_does_not_force_by_default():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'sync-to-sankuai.sh'
    completed = subprocess.run(['bash', '-n', str(script)], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    text = script.read_text(encoding='utf-8')
    assert 'cursor/rsi-diag-card-ui-20261019-4ed3' in text
    assert 'git@github.com:achievec/ms-swift.git' in text
    assert 'ssh://git@git.sankuai.com/~liyuhao03/ms-swift.git' in text
    assert '--force-sankuai' in text
    assert '--force-with-lease' in text
    assert 'git push --force ' not in text
    assert 'git push -f ' not in text
    assert '--ff-only' in text
    assert 'cherry-pick' in text
    assert '工作区有未提交改动' in text
    assert '/Users/liyuhao03/' not in text
