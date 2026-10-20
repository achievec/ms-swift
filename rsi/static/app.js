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
  liveTimer: null,
  liveBusy: false,
};

function visibleResult(result) {
  if (!result || typeof result !== 'object' || Array.isArray(result)) return result;
  const copy = { ...result };
  delete copy.mock;
  delete copy.backend;
  return copy;
}

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
  if (stage === 'decide') return '选择';
  if (stage === 'finish') return '完成';
  return stage;
}

function runStatusLabel(run) {
  if (run.status === 'completed') return '已完成';
  return labelOf(run.current_stage);
}

function statusText(status) {
  return {
    succeeded: '已完成',
    ready: '可运行',
    locked: '未解锁',
    failed: '失败',
    running: '运行中',
    superseded: '已作废',
    syncing: '同步中',
  }[status] || status;
}

function formatElapsed(startedAt) {
  const start = Date.parse(startedAt || '');
  if (!startedAt || Number.isNaN(start)) return '';
  let secs = Math.max(0, Math.floor((Date.now() - start) / 1000));
  const hours = Math.floor(secs / 3600);
  secs -= hours * 3600;
  const mins = Math.floor(secs / 60);
  const rem = secs - mins * 60;
  const clock = `${String(mins).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
  return hours > 0 ? `${hours}:${clock}` : clock;
}

function elapsedLabel(startedAt) {
  const clock = formatElapsed(startedAt);
  return clock ? `运行中 · ${clock}` : '运行中';
}

function runningMarkup(startedAt) {
  return `<i class="spin" aria-hidden="true"></i><span data-started="${escapeHtml(startedAt || '')}">${escapeHtml(elapsedLabel(startedAt))}</span>`;
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
    <div class="run-card ${state.view && state.view.run.id === run.id ? 'active' : ''}">
      <button type="button" class="run-open" data-run="${run.id}">
        <b>${escapeHtml(run.name)}</b>
        <small>第 ${run.round_index} 轮 · ${escapeHtml(runStatusLabel(run))}</small>
      </button>
      <button type="button" class="run-delete" data-delete-run="${run.id}" title="删除实验" aria-label="删除 ${escapeHtml(run.name)}">
        <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M6.2 2.2h3.6l.4.8H13v1.3H3V3h2.8l.4-.8zM4.2 5.2h7.6l-.5 8.1a1 1 0 0 1-1 .9H5.7a1 1 0 0 1-1-.9l-.5-8.1z"/></svg>
      </button>
    </div>`).join('');
}

function askConfirm({ title, name = '', body, okLabel = '确定', danger = false }) {
  const dialog = document.getElementById('confirm-dialog');
  document.getElementById('confirm-title').textContent = title;
  const nameEl = document.getElementById('confirm-name');
  nameEl.textContent = name;
  nameEl.hidden = !name;
  document.getElementById('confirm-body').textContent = body;
  const ok = document.getElementById('confirm-ok');
  const cancel = document.getElementById('confirm-cancel');
  ok.textContent = okLabel;
  ok.className = danger ? 'danger' : 'primary';
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      ok.removeEventListener('click', onOk);
      cancel.removeEventListener('click', onCancel);
      dialog.removeEventListener('cancel', onEscape);
      if (dialog.open) dialog.close();
      resolve(value);
    };
    const onOk = () => finish(true);
    const onCancel = () => finish(false);
    const onEscape = (event) => {
      event.preventDefault();
      finish(false);
    };
    ok.addEventListener('click', onOk);
    cancel.addEventListener('click', onCancel);
    dialog.addEventListener('cancel', onEscape);
    dialog.showModal();
  });
}

async function deleteRun(id) {
  const run = state.runs.find((item) => item.id === id);
  const name = run ? run.name : id;
  const index = state.runs.findIndex((item) => item.id === id);
  const confirmed = await askConfirm({
    title: '删除实验',
    name,
    body: '审计里的轮次、阶段和检查点记录会去掉，该实验目录下的阶段产物也会清理。上传的数据文件会保留。',
    okLabel: '删除',
    danger: true,
  });
  if (!confirmed) return;
  await api(`/api/runs/${id}`, { method: 'DELETE' });
  const wasCurrent = state.view && state.view.run.id === id;
  const neighbor = state.runs[index + 1] || state.runs[index - 1];
  if (wasCurrent) {
    state.chat = [];
    state.view = null;
    if (neighbor) await refreshRuns(neighbor.id);
    else {
      await refreshRuns();
      render();
    }
  } else {
    await refreshRuns();
    render();
  }
  toast('实验已删除');
}

async function openRun(id, round, keepStage) {
  const query = round ? `?round=${round}` : '';
  state.view = await api(`/api/runs/${id}${query}`);
  state.round = state.view.round.round_index;
  if (!keepStage) {
    const current = state.view.run.current_stage;
    state.stage = state.view.mutable || current === 'finish' ? current : 'data';
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

function choiceEntries(field) {
  if (field.choices && field.choices.length) {
    return field.choices.map((choice) => {
      if (typeof choice === 'string') return { value: choice, label: choice || '（空）' };
      return { value: String(choice.value), label: choice.label };
    });
  }
  const raw = field.choices_from && state.schema.catalog ? state.schema.catalog[field.choices_from] : [];
  return (raw || []).map((item) => ({ value: String(item), label: String(item) }));
}

function renderField(field, values) {
  const value = fieldValue(field, values);
  if (field.type === 'checkbox') {
    const checked = value === true || value === 'true' ? 'checked' : '';
    return `<div class="field" data-key="${field.key}"><label class="check"><input type="checkbox" data-field="${field.key}" ${checked}/> ${escapeHtml(field.label)}</label></div>`;
  }
  if (field.type === 'select' || field.type === 'multiselect') {
    const multiple = field.type === 'multiselect';
    const selected = new Set(multiple ? String(value ?? '').split(/\s+/).filter(Boolean) : [String(value)]);
    const entries = choiceEntries(field);
    if (multiple) {
      selected.forEach((token) => {
        if (token && !entries.some((item) => item.value === token)) entries.push({ value: token, label: token });
      });
    }
    const options = entries.map((choice) => {
      const mark = selected.has(String(choice.value)) ? 'selected' : '';
      return `<option value="${escapeHtml(choice.value)}" ${mark}>${escapeHtml(choice.label)}</option>`;
    }).join('');
    const multi = multiple ? 'multiple data-multi size="5"' : '';
    return `<label class="field" data-key="${field.key}"><span>${escapeHtml(field.label)}</span><select data-field="${field.key}" ${multi}>${options}</select></label>`;
  }
  if (field.type === 'combo' || field.type === 'multicombo') {
    const source = field.choices_from ? `data-choices-from="${escapeHtml(field.choices_from)}"` : '';
    const multi = field.type === 'multicombo' ? 'data-multi="1"' : '';
    return `<label class="field combo" data-key="${field.key}"><span>${escapeHtml(field.label)}</span><div class="combo-anchor"><input type="text" data-field="${field.key}" data-combo="1" ${multi} ${source} value="${escapeHtml(value)}" autocomplete="off" /><div class="combo-list" hidden></div></div></label>`;
  }
  if (field.type === 'textarea') {
    return `<label class="field" data-key="${field.key}"><span>${escapeHtml(field.label)}</span><textarea data-field="${field.key}" rows="4">${escapeHtml(value)}</textarea></label>`;
  }
  const inputType = field.type === 'number' ? 'number' : 'text';
  const step = field.step ? `step="${field.step}"` : '';
  return `<label class="field" data-key="${field.key}"><span>${escapeHtml(field.label)}</span><input type="${inputType}" data-field="${field.key}" value="${escapeHtml(value)}" ${step}/></label>`;
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
  const blocks = groups.map((group) => {
    const head = group.fields.slice(0, 3);
    const modelRow = head.length === 3
      && head.every((field) => field.type === 'combo')
      && head.map((field) => field.key).join(',') === 'model,model_type,template';
    const body = modelRow ? group.fields.slice(3) : group.fields;
    const modelHtml = modelRow
      ? `<div class="grid model-grid">${head.map((field) => renderField(field, values)).join('')}</div>`
      : '';
    const bodyHtml = body.length
      ? `<div class="grid">${body.map((field) => renderField(field, values)).join('')}</div>`
      : '';
    return `<h3>${escapeHtml(group.name)}</h3>${modelHtml}${bodyHtml}`;
  }).join('');
  const extra = advanced.length ? `
    <details>
      <summary>更多参数</summary>
      <div class="grid">${advanced.map((field) => renderField(field, values)).join('')}</div>
    </details>` : '';
  return blocks + extra;
}

function collectParams() {
  const params = {};
  document.querySelectorAll('#main [data-field]').forEach((el) => {
    if (el.type === 'radio') {
      if (el.checked) params[el.dataset.field] = el.value;
      return;
    }
    if (el.multiple) {
      params[el.dataset.field] = Array.from(el.selectedOptions).map((option) => option.value).filter(Boolean).join(' ');
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

function logPanel(text, liveAttrs) {
  return `<div class="log-panel">
    <div class="log-head"><h3>日志</h3><button type="button" class="ghost" data-log-fullscreen>全屏</button></div>
    <pre class="log live-log" data-pinned="1"${liveAttrs || ''}>${escapeHtml(text || '')}</pre>
  </div>`;
}

function setLogFullscreen(panel, on) {
  panel.classList.toggle('log-fullscreen', on);
  const button = panel.querySelector('[data-log-fullscreen]');
  if (button) button.textContent = on ? '退出全屏' : '全屏';
  document.body.classList.toggle('log-lock', Boolean(document.querySelector('.log-panel.log-fullscreen')));
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
    </div>` : '';
  const validate = stage === 'data' ? '<button type="button" class="ghost" data-action="validate">校验格式</button>' : '';
  const result = form.result ? `<h3>结果</h3><pre class="log">${escapeHtml(JSON.stringify(visibleResult(form.result), null, 2))}</pre>` : '';
  const live = form.status === 'running';
  const liveAttrs = live ? ` data-live-log="1" data-live-stage="${escapeHtml(stage)}" data-live-round="${view.round.round_index}"` : '';
  const logs = (form.logs || live) ? logPanel(form.logs || '', liveAttrs) : '';
  const command = form.command ? `<h3>命令</h3><pre class="command">${escapeHtml(form.command)}</pre>` : '';
  const kicker = live
    ? `${escapeHtml(meta.label)} · ${runningMarkup(form.started_at)}`
    : `${escapeHtml(meta.label)} · ${escapeHtml(statusText(form.status))}`;
  return `
    <section class="panel">
      <p class="kicker">${kicker}</p>
      <h2>${escapeHtml(meta.summary)}</h2>
      ${view.mutable ? '' : '<p class="muted">这是历史轮次，只能查看。</p>'}
      ${promote}
      ${renderGroups(fields, form.values || {})}
      ${upload}
      <details class="yaml">
        <summary>高级参数 YAML</summary>
        <textarea id="extra-yaml">${escapeHtml((form.values && form.values.extra_yaml) || '')}</textarea>
      </details>
      ${fetchBar()}
      <div class="row-actions">
        <button type="button" class="primary" data-action="run" ${mutable ? '' : 'disabled'}>${buttonLabel}</button>
        ${validate}
        <button type="button" class="ghost" data-action="preview" ${view.mutable ? '' : 'disabled'}>预览命令</button>
        <button type="button" class="ghost" data-action="preset" ${view.mutable ? '' : 'disabled'}>恢复预设</button>
      </div>
      <div id="live-command"></div>
      ${result}
      ${command}
      ${logs}
    </section>`;
}

function flowEdgeGeometry(edge, from, to) {
  if (edge.points && edge.points.length) {
    const d = edge.points.map((pt, index) => `${index ? 'L' : 'M'} ${pt[0]} ${pt[1]}`).join(' ');
    return { d, x: edge.label_x, y: edge.label_y, anchor: edge.label_anchor || 'middle' };
  }
  if (edge.kind === 'down') {
    const x1 = from.x + from.w / 2;
    const y1 = from.y + from.h;
    const x2 = to.x + to.w / 2;
    const y2 = to.y;
    if (Math.abs(x1 - x2) < 6) {
      return { d: `M ${x1} ${y1} L ${x2} ${y2}`, x: x1 + 10, y: (y1 + y2) / 2, anchor: 'start' };
    }
    const mid = Math.round((y1 + y2) / 2);
    return {
      d: `M ${x1} ${y1} L ${x1} ${mid} L ${x2} ${mid} L ${x2} ${y2}`,
      x: (x1 + x2) / 2,
      y: mid - 7,
      anchor: 'middle',
    };
  }
  const y = from.y + from.h / 2;
  const x1 = from.x + from.w;
  const x2 = to.x;
  return { d: `M ${x1} ${y} L ${x2} ${y}`, x: (x1 + x2) / 2, y: y - 8, anchor: 'middle' };
}

function flowNodeSelected(node) {
  const choosingNext = state.stage === 'promote' || state.stage === 'next_round';
  if (node.id === 'promote') return choosingNext;
  if (node.id === 'sft_next' || node.id === 'rl_next' || node.id === 'next_data') {
    return choosingNext && node.status !== 'locked';
  }
  if (node.id === 'sft_finish' || node.id === 'rl_finish' || node.id === 'finish') return state.stage === 'finish';
  return state.stage === node.id;
}

function renderGraph() {
  const flow = state.view.flow || { canvas: { width: 860, height: 108 }, nodes: [], edges: [], actions: [], prompt: '' };
  const byId = {};
  flow.nodes.forEach((node) => { byId[node.id] = node; });
  const paths = flow.edges.map((edge) => {
    const from = byId[edge.from];
    const to = byId[edge.to];
    if (!from || !to) return '';
    const geom = flowEdgeGeometry(edge, from, to);
    const marker = edge.live ? 'url(#flow-arrow-live)' : 'url(#flow-arrow)';
    const label = edge.label && edge.show_label !== false
      ? `<text class="flow-edge-label ${edge.live ? 'live' : ''}" x="${geom.x}" y="${geom.y}" text-anchor="${geom.anchor}">${escapeHtml(edge.label)}</text>`
      : '';
    const taken = edge.taken ? ' taken' : '';
    const flowing = to.status === 'running' ? ' flowing' : '';
    return `<path class="flow-edge ${edge.live ? 'live' : ''}${taken}${flowing}" d="${geom.d}" marker-end="${marker}"/>${label}`;
  }).join('');
  const nodes = flow.nodes.map((node) => {
    const form = (state.view.forms || {})[node.id] || {};
    const caption = node.status === 'running'
      ? runningMarkup(form.started_at)
      : escapeHtml(node.verdict || statusText(node.status));
    const disabled = node.clickable ? '' : 'disabled';
    const selected = flowNodeSelected(node) ? 'selected' : '';
    const minor = node.minor ? 'minor' : '';
    const busy = node.status === 'running' ? ' aria-busy="true"' : '';
    return `<button type="button" class="flow-node ${node.status} ${selected} ${minor}" data-stage="${node.id}" ${disabled}${busy} style="left:${node.x}px;top:${node.y}px;width:${node.w}px;height:${node.h}px"><b>${escapeHtml(node.label)}</b><small>${caption}</small></button>`;
  }).join('');
  const actions = (flow.actions || []).map((action) => `<button type="button" class="${action.danger ? 'danger' : 'primary'}" data-decision="${action.id}">${escapeHtml(action.label)}</button>`).join('');
  const actionBar = actions
    ? `<div class="flow-actions">${flow.prompt ? `<p class="muted">${escapeHtml(flow.prompt)}</p>` : ''}<div class="row-actions">${actions}</div></div>`
    : '';
  const canvas = flow.canvas || { width: 860, height: 236 };
  return `<div class="flow-scroll"><div class="flow-canvas" style="width:${canvas.width}px;height:${canvas.height}px">
    <svg viewBox="0 0 ${canvas.width} ${canvas.height}" aria-hidden="true">
      <defs>
        <marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#d1d5db"/></marker>
        <marker id="flow-arrow-live" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6366f1"/></marker>
      </defs>
      ${paths}
    </svg>
    ${nodes}
  </div></div>${actionBar}`;
}

function renderTerminal() {
  const completed = state.view.run.status === 'completed';
  return `<section class="panel"><h2>${completed ? '实验已完成' : '实验已结束'}</h2><p>本轮到此结束。已有检查点仍留在记录里。</p></section>`;
}

function renderFlow() {
  const graph = renderGraph();
  if (state.stage === 'finish') return graph + renderTerminal();
  if (!state.view.forms[state.stage]) state.stage = 'data';
  return graph + renderStage(state.stage);
}

function renderAudit() {
  const steps = state.view.steps || [];
  const rows = steps.map((step) => {
    const open = state.expanded === step.id;
    const detail = open ? `<tr class="expand"><td colspan="6">
      <p><b>开始</b> ${escapeHtml(formatShanghai(step.started_at))} · <b>结束</b> ${escapeHtml(formatShanghai(step.ended_at || ''))}</p>
      <h3>参数</h3><pre class="log">${escapeHtml(JSON.stringify(step.params, null, 2))}</pre>
      <h3>结果</h3><pre class="log">${escapeHtml(JSON.stringify(visibleResult(step.result), null, 2))}</pre>
      ${logPanel(step.logs || '', step.status === 'running' ? ` data-live-log="1" data-live-stage="${escapeHtml(step.stage)}" data-live-round="${step.round_index}"` : '')}
      ${step.command ? `<h3>命令</h3><pre class="command">${escapeHtml(step.command)}</pre>` : ''}
    </td></tr>` : '';
    return `<tr data-step="${step.id}">
      <td>${step.id}</td>
      <td>${step.round_index}</td>
      <td>${escapeHtml(labelOf(step.stage))}</td>
      <td><span class="badge ${step.status}">${escapeHtml(statusText(step.status))}</span></td>
      <td>${escapeHtml(formatShanghai(step.started_at || ''))}</td>
      <td>${escapeHtml(formatShanghai(step.ended_at || ''))}</td>
    </tr>${detail}`;
  }).join('');
  return `<section class="panel">
    <h2>审计</h2>
    <table>
      <thead><tr><th>ID</th><th>轮</th><th>阶段</th><th>状态</th><th>开始</th><th>结束</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="6">还没有步骤。</td></tr>'}</tbody>
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
    ${promoted ? '' : '<p>还没有晋级检查点。</p>'}
    ${renderGroups(fields, values)}
    ${fetchBar()}
    <div class="row-actions">
      <button type="button" class="primary" data-action="serve" ${promoted ? '' : 'disabled'}>${running ? '重新启动服务' : '启动推理 API'}</button>
      <button type="button" class="danger" data-action="serve-stop" ${running ? '' : 'disabled'}>停止</button>
    </div>
    ${running ? `<p>服务中：<code>${escapeHtml(endpoint)}</code> · 模型名 <code>${escapeHtml(service.served_model_name)}</code> · 检查点 <span class="mono">${escapeHtml(service.checkpoint)}</span></p>
      <pre class="command">curl ${escapeHtml(endpoint)}/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"${escapeHtml(service.served_model_name)}","messages":[{"role":"user","content":"你好"}]}'</pre>` : ''}
    <h3>试一句</h3>
    <div class="chat-log">${bubbles}</div>
    <div class="row-actions">
      <input id="chat-input" type="text" placeholder="输入内容" ${running ? '' : 'disabled'} />
      <button type="button" class="primary" data-action="chat" ${running ? '' : 'disabled'}>发送</button>
    </div>
  </section>`;
}

function render() {
  renderRuns();
  const main = document.getElementById('main');
  if (!state.view) {
    main.innerHTML = `<div class="empty"><h1>从一轮可点击的后训练闭环开始</h1>
      <p>从左边新建实验开始。</p></div>`;
    syncLive();
    return;
  }
  const view = state.view;
  const rounds = (view.rounds || []).map((round) => `
    <button type="button" data-round="${round.round_index}" class="${round.round_index === view.round.round_index ? 'active' : ''}">第 ${round.round_index} 轮${round.promoted_kind ? ' · ' + round.promoted_kind : ''}</button>`).join('');
  const body = state.tab === 'audit' ? renderAudit() : state.tab === 'serve' ? renderServe() : renderFlow();
  main.innerHTML = `
    <p class="kicker">第 ${view.round.round_index} 轮</p>
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
  syncLive();
}

async function runCurrent() {
  const stage = state.stage;
  const params = collectParams();
  state.busy = true;
  try {
    const payload = await api(`/api/runs/${state.view.run.id}/stages/${stage}/run`, {
      method: 'POST',
      body: JSON.stringify({ params }),
    });
    state.view = payload.view;
    state.round = payload.view.round.round_index;
    if (payload.status === 'succeeded' && payload.view.mutable) {
      state.stage = payload.view.run.current_stage;
    }
    const outcome = payload.status === 'syncing' ? '模型同步中' : payload.status === 'running' ? '已在后台启动' : '已完成';
    toast(`${labelOf(stage)} ${outcome}`);
    render();
  } finally {
    state.busy = false;
  }
}

async function previewCurrent() {
  const params = collectParams();
  const payload = await api(`/api/runs/${state.view.run.id}/stages/${state.stage}/preview`, {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  const slot = document.getElementById('live-command');
  slot.innerHTML = `<h3>命令预览</h3><pre class="command">${escapeHtml(payload.command || '这一阶段没有可预览的命令。')}</pre>`;
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
  toast('已恢复本阶段预设');
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
  const payload = await api(`/api/runs/${state.view.run.id}/serve/stop`, { method: 'POST', body: '{}' });
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
      const form = document.getElementById('create-form');
      form.name.value = '';
      form.scenario.value = '';
      document.getElementById('create-dialog').showModal();
      return;
    }
    if (target.id === 'create-cancel') {
      document.getElementById('create-dialog').close();
      return;
    }
    if (target.dataset.deleteRun) {
      await deleteRun(target.dataset.deleteRun);
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
    if (target.dataset.decision) {
      const payload = await api(`/api/runs/${state.view.run.id}/decide`, {
        method: 'POST',
        body: JSON.stringify({ action: target.dataset.decision }),
      });
      state.view = payload.view;
      state.stage = payload.stage || state.stage;
      state.tab = 'flow';
      toast(payload.view.flow && payload.view.flow.prompt ? payload.view.flow.prompt : '已记录');
      render();
      return;
    }
    if (target.dataset.stage) {
      if (target.disabled) return;
      let stage = target.dataset.stage;
      if (stage === 'sft_finish' || stage === 'rl_finish' || stage === 'finish') {
        if (state.view.run.status !== 'completed') {
          const payload = await api(`/api/runs/${state.view.run.id}/decide`, {
            method: 'POST',
            body: JSON.stringify({ action: 'finish' }),
          });
          state.view = payload.view;
          state.stage = payload.stage || 'finish';
          state.tab = 'flow';
          render();
          return;
        }
        stage = 'finish';
      }
      if (stage === 'next_data' || stage === 'sft_next' || stage === 'rl_next' || stage === 'promote') {
        const promoted = state.view.round && state.view.round.promoted_checkpoint;
        stage = promoted ? 'next_round' : 'promote';
      }
      state.stage = stage;
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
    if (target.dataset.action === 'refetch') {
      const params = collectParams();
      const payload = await api(`/api/runs/${state.view.run.id}/fetch`, {
        method: 'POST',
        body: JSON.stringify({
          model: params.model || state.view.run.base_model,
          dataset: params.dataset || '',
        }),
      });
      state.view = payload.view;
      toast(fetchNotice() || '已提交检查');
      render();
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
  if (event.target.dataset && event.target.dataset.field === 'eval_backend') {
    const scope = event.target.closest('section, form') || document;
    scope.querySelectorAll('[data-choices-from="eval_datasets_by_backend"]').forEach((input) => {
      const list = input.parentElement.querySelector('.combo-list');
      if (list && !list.hidden) showCombo(input, true);
    });
  }
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
        model_type: form.model_type.value,
        template: form.template.value,
      }),
    });
    document.getElementById('create-dialog').close();
    await refreshRuns(created.run.id);
    toast('实验已创建，从数据阶段开始');
  } catch (err) {
    toast(err.message);
  }
});

function fetchRows() {
  return (state.view && state.view.fetches) || [];
}

function fetchNotice() {
  const rows = fetchRows();
  const syncing = rows.find((row) => row.state === '同步中');
  if (syncing) {
    if (syncing.message && syncing.message !== '同步中') return `${syncing.repo_id} ${syncing.message}`;
    return `${syncing.repo_id} 同步中，进度见进程日志`;
  }
  const failed = rows.find((row) => row.state === '失败');
  if (failed) return failed.message || '失败';
  if (rows.some((row) => row.state === '就绪')) return '已就绪';
  return '';
}

function fetchProgressText() {
  const parts = [];
  fetchRows().forEach((row) => {
    if (row.state === '同步中') {
      const detail = row.message && row.message !== '同步中' ? row.message : '同步中，进度见进程日志';
      parts.push(`${row.repo_id} ${detail}`);
    } else if (row.state === '失败') {
      parts.push(`${row.repo_id} ${row.message || '失败'}`);
    }
  });
  return parts.join(' · ');
}

function fetchBar() {
  if (!state.schema || state.schema.model_source !== 'hope') return '';
  const text = fetchProgressText();
  const badge = `<span class="badge fetch-progress" data-fetch-note${text ? '' : ' hidden'}>${escapeHtml(text)}</span>`;
  return `<div class="row-actions">${badge}<button type="button" class="ghost" data-action="refetch">重新拉取</button></div>`;
}

function paintFetchNotes() {
  const text = fetchProgressText();
  document.querySelectorAll('[data-fetch-note]').forEach((el) => {
    el.textContent = text;
    el.hidden = !text;
  });
}

function needsFetchPoll() {
  return Boolean(state.schema && state.schema.model_source === 'hope' && state.view
    && fetchRows().some((row) => row.state === '同步中'));
}

function catalogNames(source, input) {
  if (source === 'datasets' && state.schema && state.schema.model_source === 'hope') {
    return (state.schema.catalog && state.schema.catalog.hf_datasets) || [];
  }
  if (source === 'eval_datasets_by_backend') {
    const scope = (input && input.closest('section, form')) || document;
    const backend = scope.querySelector('[data-field="eval_backend"]');
    const key = backend && backend.value ? backend.value : 'Native';
    const table = state.schema && state.schema.catalog ? state.schema.catalog.eval_datasets_by_backend || {} : {};
    return table[key] || [];
  }
  const raw = source && state.schema && state.schema.catalog ? state.schema.catalog[source] : [];
  return raw || [];
}

function comboFilterText(input) {
  if (input.dataset.multi !== '1') return input.value.trim();
  if (/\s$/.test(input.value)) return '';
  const parts = input.value.trim().split(/\s+/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : '';
}

function toggleMultiValue(input, value) {
  const raw = input.value;
  const trailingSpace = /\s$/.test(raw);
  let tokens = raw.trim().split(/\s+/).filter(Boolean);
  if (!trailingSpace && tokens.length) {
    const last = tokens[tokens.length - 1];
    if (last !== value && value.toLowerCase().includes(last.toLowerCase())) tokens.pop();
  }
  if (tokens.includes(value)) tokens = tokens.filter((token) => token !== value);
  else tokens.push(value);
  return tokens.join(' ');
}

function showCombo(input, browse) {
  const list = input.parentElement.querySelector('.combo-list');
  if (!list) return;
  const multi = input.dataset.multi === '1';
  const query = (browse && multi ? '' : (multi ? comboFilterText(input) : input.value.trim())).toLowerCase();
  const all = catalogNames(input.dataset.choicesFrom, input).map((item) => String(item));
  const cap = input.dataset.choicesFrom === 'eval_datasets_by_backend' ? all.length : 40;
  const exact = !multi && query ? all.findIndex((item) => item.toLowerCase() === query) : -1;
  let shown;
  if (browse && exact >= 0) {
    const start = Math.max(0, exact - 8);
    shown = all.slice(start, start + cap);
  } else if (!query) {
    shown = all.slice(0, cap);
  } else {
    shown = all.filter((item) => item.toLowerCase().includes(query)).slice(0, cap);
  }
  if (!shown.length) {
    list.hidden = true;
    list.innerHTML = '';
    return;
  }
  const selected = new Set(multi ? input.value.trim().split(/\s+/).filter(Boolean) : []);
  list.innerHTML = shown.map((item) => {
    const pressed = selected.has(item) ? ' aria-pressed="true"' : '';
    return `<button type="button" data-combo-value="${escapeHtml(item)}"${pressed}>${escapeHtml(item)}</button>`;
  }).join('');
  list.hidden = false;
}

function applyModelMatch(scope, modelId) {
  const meta = (state.schema.catalog.by_model || {})[modelId];
  if (!meta) return;
  const typeEl = scope.querySelector('[data-field="model_type"], [name="model_type"]');
  const templateEl = scope.querySelector('[data-field="template"], [name="template"]');
  if (typeEl && meta.model_type) typeEl.value = meta.model_type;
  if (templateEl && meta.template) templateEl.value = meta.template;
}

function fillCreateModel() {
  const box = document.getElementById('create-model');
  const model = state.schema.base_model_default || 'Qwen/Qwen2.5-7B-Instruct';
  const meta = (state.schema.catalog.by_model || {})[model] || {};
  box.innerHTML = `<div class="grid model-grid">
    <label class="field combo"><span>模型id或路径(--model)</span>
      <div class="combo-anchor"><input name="base_model" data-combo="1" data-choices-from="models" required maxlength="300" value="${escapeHtml(model)}" autocomplete="off" /><div class="combo-list" hidden></div></div>
    </label>
    <label class="field combo"><span>模型类型(--model_type)</span>
      <div class="combo-anchor"><input name="model_type" data-combo="1" data-choices-from="model_types" maxlength="120" value="${escapeHtml(meta.model_type || '')}" autocomplete="off" /><div class="combo-list" hidden></div></div>
    </label>
    <label class="field combo"><span>模型Prompt模板类型(--template)</span>
      <div class="combo-anchor"><input name="template" data-combo="1" data-choices-from="templates" maxlength="120" value="${escapeHtml(meta.template || '')}" autocomplete="off" /><div class="combo-list" hidden></div></div>
    </label>
  </div>`;
}

document.addEventListener('focusin', (event) => {
  if (event.target.dataset && event.target.dataset.combo) showCombo(event.target, true);
});

document.addEventListener('input', (event) => {
  const el = event.target;
  if (!el.dataset || !el.dataset.combo) return;
  showCombo(el, false);
  if (el.dataset.field === 'model' || el.name === 'base_model') {
    applyModelMatch(el.closest('dialog, section, form') || document, el.value.trim());
  }
});

document.addEventListener('mousedown', (event) => {
  const option = event.target.closest('[data-combo-value]');
  if (!option) return;
  event.preventDefault();
  const input = option.closest('.combo').querySelector('input');
  if (input.dataset.multi === '1') {
    input.value = toggleMultiValue(input, option.dataset.comboValue);
    showCombo(input, true);
  } else {
    input.value = option.dataset.comboValue;
    option.closest('.combo-list').hidden = true;
  }
  if (input.dataset.field === 'model' || input.name === 'base_model') {
    applyModelMatch(input.closest('dialog, section, form') || document, input.value.trim());
  }
});

document.addEventListener('focusout', (event) => {
  const combo = event.target.closest && event.target.closest('.combo');
  if (!combo) return;
  setTimeout(() => {
    if (!combo.contains(document.activeElement)) {
      const list = combo.querySelector('.combo-list');
      if (list) list.hidden = true;
    }
  }, 120);
});

function syncLive() {
  const needs = document.querySelector('[data-started], [data-live-log]') || needsFetchPoll();
  if (!needs) {
    if (state.liveTimer) {
      clearInterval(state.liveTimer);
      state.liveTimer = null;
    }
    return;
  }
  if (!state.liveTimer) state.liveTimer = setInterval(pollLive, 1000);
  if (!state.liveBusy) pollLive();
}

function paintElapsed() {
  document.querySelectorAll('[data-started]').forEach((el) => {
    const label = elapsedLabel(el.dataset.started);
    if (el.textContent !== label) el.textContent = label;
  });
}

function paintLog(el, text) {
  const next = text || '';
  if (el.textContent === next) {
    if (el.dataset.pinned !== '0') el.scrollTop = el.scrollHeight;
    return;
  }
  const pinned = el.dataset.pinned !== '0';
  el.textContent = next;
  if (pinned) el.scrollTop = el.scrollHeight;
}

async function pollFetchProgress() {
  if (!needsFetchPoll()) return;
  const now = Date.now();
  if (state.fetchPollAt && now - state.fetchPollAt < 5000) return;
  state.fetchPollAt = now;
  let view;
  try {
    view = await api(`/api/runs/${state.view.run.id}?round=${state.view.round.round_index}`);
  } catch (_err) {
    return;
  }
  const wasSyncing = fetchRows().some((row) => row.state === '同步中');
  state.view.fetches = view.fetches || [];
  paintFetchNotes();
  if (wasSyncing && !fetchRows().some((row) => row.state === '同步中')) {
    state.view = view;
    const failed = fetchRows().find((row) => row.state === '失败');
    if (failed && failed.message) toast(failed.message);
    else if (fetchRows().some((row) => row.state === '就绪')) toast('模型已在本地');
    render();
  }
}

async function pollLive() {
  paintElapsed();
  if (state.liveBusy || !state.view) return;
  const panels = [...document.querySelectorAll('[data-live-log]')];
  if (!panels.length) {
    if (!needsFetchPoll()) return;
    state.liveBusy = true;
    try {
      await pollFetchProgress();
    } finally {
      state.liveBusy = false;
    }
    return;
  }
  state.liveBusy = true;
  try {
    const groups = new Map();
    panels.forEach((el) => {
      const key = `${el.dataset.liveStage}|${el.dataset.liveRound || ''}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(el);
    });
    for (const [key, els] of groups) {
      const [stage, round] = key.split('|');
      const query = new URLSearchParams();
      if (round) query.set('round', round);
      let payload;
      try {
        payload = await api(`/api/runs/${state.view.run.id}/stages/${encodeURIComponent(stage)}/log?${query}`);
      } catch (_err) {
        continue;
      }
      els.forEach((el) => {
        if (el.isConnected) paintLog(el, payload.text);
      });
      const current = ((state.view.forms || {})[stage] || {}).status;
      if (payload.done && (current === 'running' || current === 'syncing')) {
        const view = await api(`/api/runs/${state.view.run.id}?round=${state.view.round.round_index}`);
        state.view = view;
        state.round = view.round.round_index;
        toast(`${labelOf(stage)} ${statusText(payload.status)}`);
        render();
        return;
      }
    }
    await pollFetchProgress();
  } finally {
    state.liveBusy = false;
  }
}

document.addEventListener('scroll', (event) => {
  const el = event.target;
  if (!el || !el.classList || !el.classList.contains('live-log')) return;
  const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
  el.dataset.pinned = gap < 28 ? '1' : '0';
}, true);

document.addEventListener('click', (event) => {
  const button = event.target.closest('[data-log-fullscreen]');
  if (!button) return;
  const panel = button.closest('.log-panel');
  if (!panel) return;
  event.preventDefault();
  setLogFullscreen(panel, !panel.classList.contains('log-fullscreen'));
});

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  document.querySelectorAll('.log-panel.log-fullscreen').forEach((panel) => setLogFullscreen(panel, false));
});

async function boot() {
  state.schema = await api('/api/schema');
  fillCreateModel();
  await refreshRuns();
  render();
}

boot().catch((err) => toast(err.message));
