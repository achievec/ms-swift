async function api(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body = {};
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (_err) {
      body = { detail: text };
    }
  }
  if (!response.ok) {
    throw new Error(body.detail || response.statusText);
  }
  return body;
}

function show(id, value) {
  const box = document.getElementById(id);
  box.replaceChildren();
  const pre = document.createElement('pre');
  pre.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  box.appendChild(pre);
}

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  const input = document.createElement('textarea');
  input.value = text;
  document.body.appendChild(input);
  input.select();
  document.execCommand('copy');
  input.remove();
  return Promise.resolve();
}

function showPaths(id, note, pairs) {
  const box = document.getElementById(id);
  box.replaceChildren();
  const hint = document.createElement('p');
  hint.className = 'muted';
  hint.textContent = note;
  box.appendChild(hint);
  pairs.forEach((pair) => {
    if (!pair[1]) return;
    const row = document.createElement('div');
    row.className = 'path-row';
    const code = document.createElement('code');
    code.textContent = pair[0] + '  ' + pair[1];
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = '复制路径';
    button.addEventListener('click', () => {
      copyText(pair[1]).then(() => {
        button.textContent = '已复制';
      }).catch(() => {
        button.textContent = '复制失败';
      });
    });
    row.append(code, button);
    box.appendChild(row);
  });
}

function dataBody() {
  return {
    text: document.getElementById('data-text').value,
    filename: 'data.jsonl',
    eval_ratio: Number(document.getElementById('eval-ratio').value),
  };
}

document.getElementById('pick-file').addEventListener('click', () => {
  document.getElementById('file').click();
});

document.getElementById('file').addEventListener('change', async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  document.getElementById('data-text').value = await file.text();
});

document.getElementById('inspect').addEventListener('click', async () => {
  try {
    show('data-out', await api('/api/workbench/inspect', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(dataBody()),
    }));
  } catch (err) {
    show('data-out', err.message);
  }
});

document.getElementById('split').addEventListener('click', async () => {
  try {
    const saved = await api('/api/workbench/split', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(dataBody()),
    });
    showPaths('data-out', `训练 ${saved.train_rows} 条，留出 ${saved.eval_rows} 条。${saved.how_to}`, [
      ['训练集', saved.train_path],
      ['留出集', saved.eval_path],
    ]);
  } catch (err) {
    show('data-out', err.message);
  }
});

document.getElementById('pack').addEventListener('click', async () => {
  try {
    const saved = await api('/api/workbench/next-pack', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(dataBody()),
    });
    showPaths('data-out', `可继续用 ${saved.kept_rows} 条，待补全 ${saved.retry_rows} 条。${saved.how_to}`, [
      ['可继续用', saved.kept_path],
      ['待补全', saved.retry_path],
    ]);
  } catch (err) {
    show('data-out', err.message);
  }
});

document.getElementById('complete').addEventListener('click', async () => {
  try {
    const id = selectedRun();
    const saved = await api(`/api/runs/${id}/workbench/complete`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(dataBody()),
    });
    const note = `本次补了 ${saved.answered} 条，还剩 ${saved.pending_rows} 条。${saved.how_to}`;
    showPaths('data-out', note, [
      ['下一轮数据', saved.next_path],
      ['仍未回答', saved.waiting_path],
    ]);
  } catch (err) {
    show('data-out', err.message);
  }
});

async function loadRuns() {
  const select = document.getElementById('run');
  const data = await api('/api/runs');
  select.innerHTML = (data.runs || []).map((run) => {
    const option = document.createElement('option');
    option.value = run.id;
    option.textContent = run.name;
    return option.outerHTML;
  }).join('');
  if (!select.innerHTML) select.innerHTML = '<option value="">还没有实验</option>';
}

function selectedRun() {
  const id = document.getElementById('run').value;
  if (!id) throw new Error('先创建或选择一个实验');
  return id;
}

document.getElementById('load-data').addEventListener('click', async () => {
  try {
    const id = selectedRun();
    const saved = await api(`/api/runs/${id}/workbench/dataset`);
    document.getElementById('data-text').value = saved.text || '';
    const coverage = saved.inspect && saved.inspect.coverage ? saved.inspect.coverage : {};
    const pairs = [['本轮数据', saved.path]];
    (saved.checkpoints || []).forEach((row) => {
      pairs.push([`第${row.round_index}轮 ${row.kind}`, row.path]);
    });
    showPaths(
      'data-out',
      `第 ${saved.round_index} 轮，${coverage.num_rows || 0} 条，关键词全中 ${coverage.full_hit || 0} 条。${saved.how_to}`,
      pairs,
    );
  } catch (err) {
    show('data-out', err.message);
  }
});

document.getElementById('report').addEventListener('click', async () => {
  try {
    const id = selectedRun();
    const response = await fetch(`/api/runs/${id}/report`);
    show('run-out', await response.text());
  } catch (err) {
    show('run-out', err.message);
  }
});

document.getElementById('ready').addEventListener('click', async () => {
  try {
    const id = selectedRun();
    show('run-out', await api(`/api/runs/${id}/serve/ready`));
  } catch (err) {
    show('run-out', err.message);
  }
});

document.getElementById('export').addEventListener('click', async () => {
  try {
    const id = selectedRun();
    show('run-out', await api(`/api/runs/${id}/export`, { method: 'POST' }));
  } catch (err) {
    show('run-out', err.message);
  }
});

document.getElementById('export-status').addEventListener('click', async () => {
  try {
    const id = selectedRun();
    show('run-out', await api(`/api/runs/${id}/export`));
  } catch (err) {
    show('run-out', err.message);
  }
});

loadRuns().catch((err) => show('run-out', err.message));
