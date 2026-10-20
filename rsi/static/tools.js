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
  document.getElementById(id).textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
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
    show('data-out', await api('/api/workbench/split', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(dataBody()),
    }));
  } catch (err) {
    show('data-out', err.message);
  }
});

document.getElementById('pack').addEventListener('click', async () => {
  try {
    show('data-out', await api('/api/workbench/next-pack', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(dataBody()),
    }));
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
