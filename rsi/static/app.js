const state = {
  schema: null,
  runs: [],
  view: null,
  stage: 'data',
  tab: 'flow',
  round: null,
  expanded: null,
  chat: [],
  busy: false,
};

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

function toast(message) {
  const el = document.getElementById('toast');
  el.hidden = false;
  el.textContent = message;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { el.hidden = true; }, 4200);
}

async function api(url, options = {}) {
  const opts = { ...options };
  if (typeof opts.body === 'string') {
    opts.headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  }
  const response = await fetch(url, opts);
  let data = {};
  try {
    data = await response.json();
  } catch (_err) {
    data = {};
  }
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : '请求失败');
  }
  return data;
}

function labelOf(stage) {
  const found = (state.schema.stages || []).find((item) => item.id === stage);
  if (found) return found.label;
  if (stage === 'serve') return '推理服务';
  if (stage === 'serve_stop') return '停止推理';
  return stage;
}

function statusText(status) {
  return {
    succeeded: '已完成',
    ready: '可运行',
    locked: '未解锁',
    failed: '失败',
    running: '运行中',
    superseded: '已作废',
  }[status] || status;
}

async function refreshRuns(selectId) {
  const payload = await api('/api/runs');
  state.runs = payload.runs || [];
  renderRuns();
  if (selectId) {
    await openRun(selectId, null, false);
  }
}

function renderRuns() {
  const root = document.getElementById('run-list');
  if (!state.runs.length) {
    root.innerHTML = '<p class="muted">还没有实验。</p>';
    return;
  }
  root.innerHTML = state.runs.map((run) => `
    <button type="button" class="run-card ${state.view && state.view.run.id === run.id ? 'active' : ''}" data-run="${run.id}">
      <b>${escapeHtml(run.name)}</b>
      <small>第 ${run.round_index} 轮 · ${escapeHtml(labelOf(run.current_stage))}</small>
    </button>`).join('');
}

async function openRun(id, round, keepStage) {
  const query = round ? `?round=${round}` : '';
  state.view = await api(`/api/runs/${id}${query}`);
  state.round = state.view.round.round_index;
  if (!keepStage) {
    state.stage = state.view.mutable ? state.view.run.current_stage : 'data';
    state.tab = 'flow';
  }
  render();
}

function fieldValue(field, values) {
  const raw = values ? values[field.key] : undefined;
  if (raw === undefined || raw === null) return '';
  if (typeof raw === 'object') return JSON.stringify(raw);
  return raw;
}

function renderField(field, values) {
  const value = fieldValue(field, values);
  const help = `${escapeHtml(field.help || '')} ${field.arg ? `<code>${escapeHtml(field.arg)}</code>` : ''}`;
  if (field.type === 'checkbox') {
    const checked = value === true || value === 'true' ? 'checked' : '';
    return `<div class="field" data-key="${field.key}"><label class="check"><input type="checkbox" data-field="${field.key}" ${checked}/> ${escapeHtml(field.label)}</label><small>${help}</small></div>`;
  }
  if (field.type === 'select') {
    const options = (field.choices || []).map((choice) => {
      const optionValue = typeof choice === 'string' ? choice : choice.value;
      const optionLabel = typeof choice === 'string' ? (choice || '（空）') : choice.label;
      const selected = String(value) === String(optionValue) ? 'selected' : '';
      return `<option value="${escapeHtml(optionValue)}" ${selected}>${escapeHtml(optionLabel)}</option>`;
    }).join('');
    return `<label class="field" data-key="${field.key}"><span>${escapeHtml(field.label)}</span><select data-field="${field.key}">${options}</select><small>${help}</small></label>`;
  }
  if (field.type === 'textarea') {
    return `<label class="field" data-key="${field.key}"><span>${escapeHtml(field.label)}</span><textarea data-field="${field.key}" rows="4">${escapeHtml(value)}</textarea><small>${help}</small></label>`;
  }
  const inputType = field.type === 'number' ? 'number' : 'text';
  const step = field.step ? `step="${field.step}"` : '';
  return `<label class="field" data-key="${field.key}"><span>${escapeHtml(field.label)}</span><input type="${inputType}" data-field="${field.key}" value="${escapeHtml(value)}" ${step}/><small>${help}</small></label>`;
}

function renderGroups(fields, values) {
  const primary = fields.filter((field) => !field.advanced);
  const advanced = fields.filter((field) => field.advanced);
  const groups = [];
  for (const field of primary) {
    const last = groups[groups.length - 1];
    if (!last || last.name !== field.group) groups.push({ name: field.group, fields: [] });
    groups[groups.length - 1].fields.push(field);
  }
  const blocks = groups.map((group) => `
    <h3>${escapeHtml(group.name)}</h3>
    <div class="grid">${group.fields.map((field) => renderField(field, values)).join('')}</div>`).join('');
  const extra = advanced.length ? `
    <details>
      <summary>更多参数（与 swift Web-UI / 命令行同名）</summary>
      <div class="grid">${advanced.map((field) => renderField(field, values)).join('')}</div>
    </details>` : '';
  return blocks + extra;
}

function collectParams() {
  const params = {};
  document.querySelectorAll('[data-field]').forEach((el) => {
    if (el.type === 'radio') {
      if (el.checked) params[el.dataset.field] = el.value;
      return;
    }
    if (el.type === 'checkbox') {
      params[el.dataset.field] = el.checked;
      return;
    }
    if (el.type === 'number') {
      params[el.dataset.field] = el.value === '' ? null : Number(el.value);
      return;
    }
    params[el.dataset.field] = el.value;
  });
  const yaml = document.getElementById('extra-yaml');
  if (yaml) params.extra_yaml = yaml.value;
  return params;
}

function syncSource() {
  const source = document.querySelector('[data-field="source"]');
  if (!source) return;
  const value = source.value;
  document.querySelectorAll('[data-key="bundled"]').forEach((el) => { el.hidden = value !== 'bundled'; });
  document.querySelectorAll('[data-key="dataset"]').forEach((el) => { el.hidden = value === 'bundled'; });
  const upload = document.getElementById('upload-row');
  if (upload) upload.hidden = value !== 'upload';
}

function renderPromote(values) {
  const candidates = state.view.candidates || [];
  const selected = values.choice || state.view.recommendation;
  const cards = candidates.map((cand) => {
    const disabled = cand.path ? '' : 'disabled';
    const checked = selected === cand.kind ? 'checked' : '';
    const best = cand.kind === state.view.recommendation ? 'best' : '';
    const score = cand.score === null || cand.score === undefined ? '—' : cand.score;
    return `<label class="candidate ${best}">
      <input type="radio" name="choice" data-field="choice" value="${cand.kind}" ${checked} ${disabled}/>
      <b>${escapeHtml(cand.label)}</b>
      ${cand.kind === state.view.recommendation ? '<span class="badge succeeded">建议</span>' : ''}
      <div class="score">${escapeHtml(score)}</div>
      <small class="mono">${escapeHtml(cand.path || '尚无检查点')}</small>
    </label>`;
  }).join('');
  return `<div class="candidates">${cards}</div>`;
}

function renderStage(stage) {
  const view = state.view;
  const form = view.forms[stage];
  const fields = state.schema.fields[stage] || [];
  const meta = state.schema.stages.find((item) => item.id === stage);
  const mutable = view.mutable && form.status !== 'locked' && form.status !== 'running';
  const buttonLabel = {
    ready: '运行本阶段',
    failed: '重试本阶段',
    succeeded: '重新运行并作废后续阶段',
    running: '运行中',
    locked: '请先完成上一阶段',
  }[form.status] || '运行本阶段';
  const promote = stage === 'promote' ? renderPromote(form.values || {}) : '';
  const upload = stage === 'data' ? `
    <div id="upload-row" class="field">
      <span>上传 jsonl / json</span>
      <input id="file-input" type="file" accept=".jsonl,.json,.txt" />
      <small>上传后会立刻校验，并把路径填进数据集字段。</small>
    </div>` : '';
  const validate = stage === 'data' ? '<button type="button" class="ghost" data-action="validate">校验格式</button>' : '';
  const result = form.result ? `<h3>结果</h3><pre class="log">${escapeHtml(JSON.stringify(form.result, null, 2))}</pre>` : '';
  const logs = form.logs ? `<h3>日志</h3><pre class="log">${escapeHtml(form.logs)}</pre>` : '';
  const command = form.command ? `<h3>将执行 / 已记录的命令</h3><pre class="command">${escapeHtml(form.command)}</pre>` : '';
  return `
    <section class="panel">
      <p class="kicker">${escapeHtml(meta.label)} · ${escapeHtml(statusText(form.status))}</p>
      <h2>${escapeHtml(meta.summary)}</h2>
      ${view.mutable ? '' : '<p class="muted">这是历史轮次，只能查看。</p>'}
      ${promote}
      ${renderGroups(fields, form.values || {})}
      ${upload}
      <details class="yaml">
        <summary>高级参数 YAML</summary>
        <p class="muted">键名就是 swift 的命令行参数。同名键会覆盖上面的表单。</p>
        <textarea id="extra-yaml">${escapeHtml((form.values && form.values.extra_yaml) || '')}</textarea>
      </details>
      <div class="row-actions">
        <button type="button" class="primary" data-action="run" ${mutable ? '' : 'disabled'}>${buttonLabel}</button>
        ${validate}
        <button type="button" class="ghost" data-action="preview" ${view.mutable ? '' : 'disabled'}>预览命令</button>
        <button type="button" class="ghost" data-action="preset" ${view.mutable ? '' : 'disabled'}>恢复 A100 预设</button>
      </div>
      <div id="live-command"></div>
      ${result}
      ${command}
      ${logs}
    </section>`;
}

function renderFlow() {
  const view = state.view;
  const rail = view.board.map((item) => `
    <button type="button" data-stage="${item.id}" class="${item.status} ${state.stage === item.id ? 'selected' : ''}">
      <b>${escapeHtml(item.label)}</b>
      <small>${escapeHtml(statusText(item.status))}${item.metric ? ' · ' + escapeHtml(item.metric) : ''}</small>
    </button>`).join('');
  return `<div class="rail">${rail}</div>${renderStage(state.stage)}`;
}

function renderAudit() {
  const steps = state.view.steps || [];
  const rows = steps.map((step) => {
    const open = state.expanded === step.id;
    const detail = open ? `<tr class="expand"><td colspan="7">
      <p><b>开始</b> ${escapeHtml(step.started_at)} · <b>结束</b> ${escapeHtml(step.ended_at || '')}</p>
      <h3>参数</h3><pre class="log">${escapeHtml(JSON.stringify(step.params, null, 2))}</pre>
      <h3>结果</h3><pre class="log">${escapeHtml(JSON.stringify(step.result, null, 2))}</pre>
      <h3>日志</h3><pre class="log">${escapeHtml(step.logs || '')}</pre>
      ${step.command ? `<h3>命令</h3><pre class="command">${escapeHtml(step.command)}</pre>` : ''}
    </td></tr>` : '';
    return `<tr data-step="${step.id}">
      <td>${step.id}</td>
      <td>${step.round_index}</td>
      <td>${escapeHtml(labelOf(step.stage))}</td>
      <td><span class="badge ${step.status}">${escapeHtml(statusText(step.status))}</span></td>
      <td>${escapeHtml(step.backend)}</td>
      <td>${escapeHtml(step.started_at || '')}</td>
      <td>${escapeHtml(step.ended_at || '')}</td>
    </tr>${detail}`;
  }).join('');
  return `<section class="panel">
    <h2>审计</h2>
    <p class="muted">每次运行都会留下开始时间、参数、结束时间、结果和日志。重新运行不会删旧记录，只把后续步骤标成已作废。</p>
    <table>
      <thead><tr><th>ID</th><th>轮</th><th>阶段</th><th>状态</th><th>后端</th><th>开始</th><th>结束</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="7">还没有步骤。</td></tr>'}</tbody>
    </table>
  </section>`;
}

function renderServe() {
  const view = state.view;
  const fields = state.schema.fields.serve;
  const values = view.serve_form || {};
  const service = view.service;
  const promoted = (view.rounds || []).some((round) => round.promoted_checkpoint);
  const running = service && service.status === 'running';
  const bubbles = state.chat.map((item) => `<div class="bubble ${item.role}">${escapeHtml(item.content)}</div>`).join('');
  const endpoint = running ? `http://${service.host === '0.0.0.0' ? '127.0.0.1' : service.host}:${service.port}/v1` : '';
  return `<section class="panel">
    <h2>OpenAI 兼容推理</h2>
    <p class="muted">晋级之后可以启动服务。默认执行 <code>swift deploy</code>，在 GPU 上提供 <code>/v1/models</code> 和 <code>/v1/chat/completions</code>。不想加载权重时，设置 <code>RSI_SERVE_BACKEND=mock</code>。</p>
    ${promoted ? '' : '<p>还没有晋级检查点，部署按钮会保持关闭。</p>'}
    ${renderGroups(fields, values)}
    <div class="row-actions">
      <button type="button" class="primary" data-action="serve" ${promoted ? '' : 'disabled'}>${running ? '重新启动服务' : '启动推理 API'}</button>
      <button type="button" class="danger" data-action="serve-stop" ${running ? '' : 'disabled'}>停止</button>
    </div>
    ${running ? `<p>服务中：<code>${escapeHtml(endpoint)}</code> · 模型名 <code>${escapeHtml(service.served_model_name)}</code> · 检查点 <span class="mono">${escapeHtml(service.checkpoint)}</span></p>
      <pre class="command">curl ${escapeHtml(endpoint)}/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"${escapeHtml(service.served_model_name)}","messages":[{"role":"user","content":"VPN 连不上"}]}'</pre>` : ''}
    <h3>试一句</h3>
    <div class="chat-log">${bubbles}</div>
    <div class="row-actions">
      <input id="chat-input" type="text" placeholder="例如：会议室 Wi-Fi 门户证书不受信任" ${running ? '' : 'disabled'} />
      <button type="button" class="primary" data-action="chat" ${running ? '' : 'disabled'}>发送</button>
    </div>
  </section>`;
}

function render() {
  renderRuns();
  const main = document.getElementById('main');
  if (!state.view) {
    main.innerHTML = `<div class="empty"><h1>从一轮可点击的后训练闭环开始</h1>
      <p>行业里常见的迭代是：场景 SFT 建立行为，评测做门禁，在线 RL（这里默认 GRPO）用奖励把策略从 SFT 上再推进一步，只有胜出的检查点才能当下一轮基座，最后用 OpenAI 兼容接口接走。</p>
      <p>左边新建实验。默认会在 GPU 0 上执行 swift sft / eval / rlhf / deploy。没有 GPU 时设置 RSI_BACKEND=mock。</p></div>`;
    return;
  }
  const view = state.view;
  const rounds = (view.rounds || []).map((round) => `
    <button type="button" data-round="${round.round_index}" class="${round.round_index === view.round.round_index ? 'active' : ''}">第 ${round.round_index} 轮${round.promoted_kind ? ' · ' + round.promoted_kind : ''}</button>`).join('');
  const body = state.tab === 'audit' ? renderAudit() : state.tab === 'serve' ? renderServe() : renderFlow();
  main.innerHTML = `
    <p class="kicker">Round ${view.round.round_index} · ${escapeHtml(view.backend)}</p>
    <h1>${escapeHtml(view.run.name)}</h1>
    <p>${escapeHtml(view.run.scenario)}</p>
    <p class="muted">最初基座 <span class="mono">${escapeHtml(view.run.base_model)}</span><br/>本轮基座 <span class="mono">${escapeHtml(view.round.base_model)}</span></p>
    <div class="rounds">${rounds}</div>
    <div class="tabs">
      <button type="button" data-tab="flow" class="${state.tab === 'flow' ? 'active' : ''}">流程</button>
      <button type="button" data-tab="audit" class="${state.tab === 'audit' ? 'active' : ''}">审计</button>
      <button type="button" data-tab="serve" class="${state.tab === 'serve' ? 'active' : ''}">推理服务</button>
    </div>
    ${body}`;
  syncSource();
}

async function runCurrent() {
  const stage = state.stage;
  const params = collectParams();
  const payload = await api(`/api/runs/${state.view.run.id}/stages/${stage}/run`, {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  state.view = payload.view;
  state.round = payload.view.round.round_index;
  if (payload.status === 'succeeded' && payload.view.mutable) {
    state.stage = payload.view.run.current_stage;
  }
  toast(`${labelOf(stage)} ${payload.status === 'running' ? '已在后台启动' : '已完成'}`);
  render();
}

async function previewCurrent() {
  const params = collectParams();
  const payload = await api(`/api/runs/${state.view.run.id}/stages/${state.stage}/preview`, {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  const slot = document.getElementById('live-command');
  slot.innerHTML = `<h3>命令预览</h3><pre class="command">${escapeHtml(payload.command || '这一阶段不调用 swift。')}</pre>`;
}

async function validateCurrent() {
  const params = collectParams();
  const report = await api('/api/datasets/validate', {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  const slot = document.getElementById('live-command');
  slot.innerHTML = `<h3>校验结果</h3><pre class="log">${escapeHtml(JSON.stringify(report, null, 2))}</pre>`;
  toast(report.ok ? `格式通过，${report.num_rows ?? '公开'} 条` : '格式未通过');
}

function applyPreset() {
  const preset = state.view.forms[state.stage].preset || {};
  document.querySelectorAll('[data-field]').forEach((el) => {
    if (!(el.dataset.field in preset)) return;
    const value = preset[el.dataset.field];
    if (el.type === 'checkbox') el.checked = value === true || value === 'true';
    else if (el.type === 'radio') el.checked = el.value === value;
    else if (value !== undefined && value !== null && typeof value !== 'object') el.value = value;
  });
  const yaml = document.getElementById('extra-yaml');
  if (yaml) yaml.value = preset.extra_yaml || '';
  syncSource();
  toast('已恢复本阶段的 A100 预设');
}

async function uploadFile(file) {
  const body = new FormData();
  body.append('file', file);
  const response = await fetch('/api/datasets/upload', { method: 'POST', body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '上传失败');
  const dataset = document.querySelector('[data-field="dataset"]');
  if (dataset) dataset.value = data.path;
  const source = document.querySelector('[data-field="source"]');
  if (source) source.value = 'upload';
  syncSource();
  const slot = document.getElementById('live-command');
  if (slot) slot.innerHTML = `<h3>上传校验</h3><pre class="log">${escapeHtml(JSON.stringify(data.validation, null, 2))}</pre>`;
  toast(data.validation.ok ? '上传通过校验' : '上传文件格式未通过');
}

async function startServe() {
  const params = collectParams();
  const payload = await api(`/api/runs/${state.view.run.id}/serve`, {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  state.view = payload.view;
  toast('推理服务已启动');
  render();
}

async function stopServe() {
  const payload = await api(`/api/runs/${state.view.run.id}/serve/stop`, { method: 'POST' });
  state.view = payload.view;
  toast('推理服务已停止');
  render();
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const content = input.value.trim();
  if (!content) return;
  state.chat.push({ role: 'user', content });
  input.value = '';
  render();
  const payload = await api(`/api/runs/${state.view.run.id}/chat`, {
    method: 'POST',
    body: JSON.stringify({ messages: [{ role: 'user', content }] }),
  });
  const reply = payload.choices && payload.choices[0] && payload.choices[0].message
    ? payload.choices[0].message.content
    : JSON.stringify(payload);
  state.chat.push({ role: 'assistant', content: reply });
  render();
}

document.addEventListener('click', async (event) => {
  const target = event.target.closest('button');
  if (!target || state.busy) return;
  try {
    if (target.id === 'new-run') {
      document.getElementById('create-dialog').showModal();
      return;
    }
    if (target.id === 'create-cancel') {
      document.getElementById('create-dialog').close();
      return;
    }
    if (target.dataset.run) {
      await openRun(target.dataset.run, null, false);
      return;
    }
    if (target.dataset.round) {
      await openRun(state.view.run.id, target.dataset.round, true);
      return;
    }
    if (target.dataset.tab) {
      state.tab = target.dataset.tab;
      render();
      return;
    }
    if (target.dataset.stage) {
      state.stage = target.dataset.stage;
      state.tab = 'flow';
      render();
      return;
    }
    if (target.dataset.action === 'run') {
      state.busy = true;
      await runCurrent();
      return;
    }
    if (target.dataset.action === 'preview') {
      await previewCurrent();
      return;
    }
    if (target.dataset.action === 'validate') {
      await validateCurrent();
      return;
    }
    if (target.dataset.action === 'preset') {
      applyPreset();
      return;
    }
    if (target.dataset.action === 'serve') {
      state.busy = true;
      await startServe();
      return;
    }
    if (target.dataset.action === 'serve-stop') {
      await stopServe();
      return;
    }
    if (target.dataset.action === 'chat') {
      state.busy = true;
      await sendChat();
    }
  } catch (err) {
    toast(err.message);
  } finally {
    state.busy = false;
  }
});

document.addEventListener('click', (event) => {
  const row = event.target.closest('tr[data-step]');
  if (!row || event.target.closest('button')) return;
  const id = Number(row.dataset.step);
  state.expanded = state.expanded === id ? null : id;
  render();
});

document.addEventListener('change', async (event) => {
  if (event.target.dataset && event.target.dataset.field === 'source') syncSource();
  if (event.target.id === 'file-input' && event.target.files[0]) {
    try {
      await uploadFile(event.target.files[0]);
    } catch (err) {
      toast(err.message);
    }
  }
});

document.getElementById('create-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = event.target;
  try {
    const created = await api('/api/runs', {
      method: 'POST',
      body: JSON.stringify({
        name: form.name.value,
        scenario: form.scenario.value,
        base_model: form.base_model.value,
      }),
    });
    document.getElementById('create-dialog').close();
    await refreshRuns(created.run.id);
    toast('实验已创建，从数据阶段开始');
  } catch (err) {
    toast(err.message);
  }
});

async function boot() {
  state.schema = await api('/api/schema');
  const health = await api('/api/health');
  const gpuReady = health.gpu && health.gpu.available;
  document.getElementById('backend-pill').textContent = health.backend === 'mock'
    ? 'Mock 后端'
    : (gpuReady ? 'ms-swift · GPU' : 'ms-swift · 无 GPU');
  if (health.hardware && health.hardware.label) {
    document.getElementById('hardware-pill').textContent = health.hardware.label;
  }
  await refreshRuns();
  render();
}

boot().catch((err) => toast(err.message));
