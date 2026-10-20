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
  serveDraft: null,
  serveDraftRun: null,
  inferLogVisible: false,
  inferLog: '',
  inferLogTimer: null,
  inferRuntimeOpen: false,
  mediaTab: 'image',
  mediaFiles: { image: null, video: null, audio: null },
};

function evalReportLink(stage, result, roundIndex) {
  if (stage !== 'sft_eval' && stage !== 'rl_eval') return '';
  const metrics = (result && result.metrics) || {};
  const path = (result && result.report_path) || metrics.report_html || '';
  if (!path || !state.view) return '';
  const round = roundIndex || state.view.round.round_index;
  const href = `/api/runs/${encodeURIComponent(state.view.run.id)}/stages/${encodeURIComponent(stage)}/report?round=${round}`;
  return `<a class="ghost" href="${href}" target="_blank" rel="noopener">查看完整报告</a>`;
}

function scoreHeadline(metrics) {
  if (!metrics) return '';
  const bits = [];
  if (metrics.bleu_1 != null && metrics.bleu_1 !== '') bits.push(`BLEU-1 ${metrics.bleu_1}`);
  if (metrics.rouge_l_f != null && metrics.rouge_l_f !== '') bits.push(`ROUGE-L F1 ${metrics.rouge_l_f}`);
  return bits.length ? `<p>${escapeHtml(bits.join(' · '))}</p>` : '';
}

function scoreTable(metrics) {
  const scores = metrics && metrics.scores;
  if (!scores || typeof scores !== 'object') return '';
  const rows = Object.keys(scores).map((key) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(scores[key])}</td></tr>`).join('');
  return rows ? `<table class="score-table"><tbody>${rows}</tbody></table>` : '';
}

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
  state.serveDraft = null;
  state.serveDraftRun = null;
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

function logPanel(text, liveAttrs, title) {
  return `<div class="log-panel">
    <div class="log-head"><h3>${escapeHtml(title || '日志')}</h3>${logTools()}</div>
    <pre class="log live-log" data-pinned="1"${liveAttrs || ''}>${escapeHtml(text || '')}</pre>
  </div>`;
}

function outputPanel(title, text, kind) {
  const cls = kind === 'command' ? 'command' : 'log';
  return `<div class="log-panel"><div class="log-head"><h3>${escapeHtml(title)}</h3>${logTools()}</div><pre class="${cls}">${escapeHtml(text || '')}</pre></div>`;
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
  const reportLink = form.result ? evalReportLink(stage, form.result) : '';
  const result = form.result ? `${scoreHeadline(form.result.metrics)}${scoreTable(form.result.metrics)}${reportLink ? `<p class="row-actions">${reportLink}</p>` : ''}${outputPanel('结果', JSON.stringify(visibleResult(form.result), null, 2))}` : '';
  const live = form.status === 'running';
  const liveAttrs = live ? ` data-live-log="1" data-live-stage="${escapeHtml(stage)}" data-live-round="${view.round.round_index}"` : '';
  const logs = (form.logs || live) ? logPanel(form.logs || '', liveAttrs) : '';
  const command = form.command ? outputPanel('命令', form.command, 'command') : '';
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
      ${fetchNote()}
      <div class="row-actions">
        <button type="button" class="primary" data-action="run" ${mutable ? '' : 'disabled'}>${buttonLabel}</button>
        ${refetchButton()}
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
  return `<section class="panel"><h2>${completed ? '实验已完成' : '实验已结束'}</h2><p>本轮到此结束。已有检查点仍留在记录里。</p>
    <div class="row-actions"><button type="button" class="primary" data-action="open-serve">去部署推理服务</button></div>
  </section>`;
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
      ${outputPanel('参数', JSON.stringify(step.params, null, 2))}
      ${scoreHeadline((step.result || {}).metrics)}${scoreTable((step.result || {}).metrics)}${evalReportLink(step.stage, step.result, step.round_index) ? `<p class="row-actions">${evalReportLink(step.stage, step.result, step.round_index)}</p>` : ''}${outputPanel('结果', JSON.stringify(visibleResult(step.result), null, 2))}
      ${logPanel(step.logs || '', step.status === 'running' ? ` data-live-log="1" data-live-stage="${escapeHtml(step.stage)}" data-live-round="${step.round_index}"` : '')}
      ${step.command ? outputPanel('命令', step.command, 'command') : ''}
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

function serveFormValues(view) {
  const base = state.serveDraft && state.serveDraftRun === view.run.id
    ? { ...state.serveDraft }
    : { ...(view.serve_form || {}) };
  const fill = view.serve_fill || {};
  ['model', 'model_type', 'template', 'adapters'].forEach((key) => {
    if (Object.prototype.hasOwnProperty.call(fill, key)) base[key] = fill[key] == null ? '' : fill[key];
  });
  return base;
}

function inferItemValue(item, values) {
  const raw = values ? values[item.id] : undefined;
  if (raw === false || raw === 0) return raw;
  if (raw !== undefined && raw !== null && raw !== '') return raw;
  if (item.default !== undefined && item.default !== null) return item.default;
  return '';
}

function inferChoices(item, value) {
  let entries = (item.choices || []).map((choice) => String(choice));
  if (!entries.length && item.choices_from && state.schema.catalog) {
    entries = (state.schema.catalog[item.choices_from] || []).map((choice) => String(choice));
  }
  const tokens = item.multiselect ? String(value ?? '').split(/[\s,]+/).filter(Boolean) : [String(value ?? '')];
  const selected = new Set(tokens);
  selected.forEach((token) => {
    if (token && !entries.includes(token)) entries.unshift(token);
  });
  return { entries, selected };
}

function inferAction(id) {
  return {
    reset: 'reset-infer',
    load_checkpoint: 'serve',
    refresh_tasks: 'serve-refresh',
    show_log: 'serve-log',
    stop_show_log: 'serve-log-stop',
    kill_task: 'serve-stop',
    clear_history: 'clear-chat',
    submit: 'chat',
  }[id] || id;
}

function renderInferControl(item, values, running) {
  const value = inferItemValue(item, values);
  const locked = item.locked ? 'disabled' : '';
  const lockTitle = item.locked ? ' title="由本轮实验决定，不能在这里修改"' : '';
  const info = item.info ? `<small>${escapeHtml(item.info)}</small>` : '';
  if (item.widget === 'Button') {
    const action = inferAction(item.id);
    let disabled = '';
    if (item.id === 'load_checkpoint' && !(values.model || values.adapters)) disabled = 'disabled';
    if (item.id === 'kill_task' && !running) disabled = 'disabled';
    if (item.id === 'submit' && !running) disabled = 'disabled';
    const klass = item.primary ? 'primary' : (item.id === 'kill_task' ? 'danger' : 'ghost');
    return `<div class="infer-action"><button type="button" class="${klass}" data-action="${action}" ${disabled}>${escapeHtml(item.label)}</button></div>`;
  }
  if (item.widget === 'Checkbox') {
    const checked = value === true || value === 'true' ? 'checked' : '';
    return `<div class="field check" data-key="${item.id}"${lockTitle}><label class="check"><input type="checkbox" data-field="${item.id}" ${checked} ${locked}/> ${escapeHtml(item.label)}</label>${info}</div>`;
  }
  if (item.widget === 'Slider') {
    const min = item.minimum == null ? 0 : item.minimum;
    const max = item.maximum == null ? 1 : item.maximum;
    const step = item.step == null ? 'any' : item.step;
    return `<div class="field infer-slider" data-key="${item.id}"><div class="slider-head"><span>${escapeHtml(item.label)}</span><input type="number" class="slider-num" data-slider-num="${item.id}" min="${min}" max="${max}" step="${step}" value="${escapeHtml(value)}" /><button type="button" class="ghost icon" data-action="slider-reset" data-reset="${item.id}" title="恢复初始值">↺</button></div>${info}<input type="range" data-field="${item.id}" min="${min}" max="${max}" step="${step}" value="${escapeHtml(value)}" /><div class="slider-ends"><span>${escapeHtml(min)}</span><span>${escapeHtml(max)}</span></div></div>`;
  }
  if (item.widget === 'Dropdown') {
    const choice = inferChoices(item, value);
    if (item.locked) {
      const current = String(value ?? '');
      choice.entries = current ? [current] : [];
      choice.selected = new Set(current ? [current] : []);
    }
    const options = choice.entries.map((entry) => {
      const mark = choice.selected.has(entry) ? 'selected' : '';
      return `<option value="${escapeHtml(entry)}" ${mark}>${escapeHtml(entry || '（空）')}</option>`;
    }).join('');
    const multi = item.multiselect ? 'multiple data-multi size="4"' : '';
    return `<label class="field" data-key="${item.id}"${lockTitle}><span>${escapeHtml(item.label)}</span>${info}<select data-field="${item.id}" ${multi} ${locked}>${options}</select></label>`;
  }
  if (item.id === 'log') {
    return `<div class="log-panel infer-log" data-key="log"><div class="log-head"><div class="log-title"><span>${escapeHtml(item.label)}</span>${info}</div>${logTools()}</div><textarea class="log" readonly rows="8">${escapeHtml(state.inferLog || '')}</textarea></div>`;
  }
  if (item.id === 'running_tasks') {
    const services = (state.view && state.view.services) || [];
    const options = services.map((service) => {
      const mark = service.status === 'running' ? 'selected' : '';
      const text = `${service.status === 'running' ? '运行中' : '已停止'} ${service.command || service.served_model_name || service.id}`;
      return `<option value="${escapeHtml(String(service.id))}" ${mark}>${escapeHtml(text)}</option>`;
    }).join('');
    return `<label class="field" data-key="running_tasks"><span>${escapeHtml(item.label)}</span>${info}<select>${options || '<option value="">没有部署</option>'}</select></label>`;
  }
  const long = Number(item.lines) >= 4;
  const control = long
    ? `<textarea data-field="${item.id}" rows="${item.lines}" ${item.locked ? 'readonly' : ''}>${escapeHtml(value)}</textarea>`
    : `<input type="text" data-field="${item.id}" value="${escapeHtml(value)}" ${item.locked ? 'readonly' : ''}/>`;
  return `<label class="field" data-key="${item.id}"${lockTitle}><span>${escapeHtml(item.label)}</span>${info}${control}</label>`;
}

function renderInferRow(row, values, running, fill) {
  if (row.length === 1 && row[0].id === 'log' && !state.inferLogVisible) return '';
  const fields = [];
  const sides = [];
  row.forEach((item) => {
    let control = renderInferControl(item, values, running);
    if (item.id === 'adapters' && fill.checkpoint) {
      control += '<button type="button" class="ghost fill-serve" data-action="fill-serve">填入本轮晋级 checkpoint</button>';
    }
    if (item.widget === 'Button') sides.push(control);
    else fields.push(control);
  });
  const group = fields.length ? `<div class="infer-group">${fields.join('')}</div>` : '';
  const side = sides.length ? `<div class="infer-sides">${sides.join('')}</div>` : '';
  const onlyButtons = fields.length === 0 ? ' actions' : '';
  return `<div class="infer-row${onlyButtons}${sides.length ? ' has-side' : ''}">${group}${side}</div>`;
}

function renderMediaTabs() {
  const tabs = ['image', 'video', 'audio'];
  const labels = { image: 'Image', video: 'Video', audio: 'Audio' };
  const accept = { image: 'image/*', video: 'video/*', audio: 'audio/*' };
  const bar = tabs.map((tab) => `<button type="button" data-media-tab="${tab}" class="${state.mediaTab === tab ? 'active' : ''}">${labels[tab]}</button>`).join('');
  const files = state.mediaFiles || {};
  const panels = tabs.map((tab) => {
    const file = files[tab];
    const name = file ? escapeHtml(file.name) : '';
    return `<label class="drop" data-drop="${tab}" ${state.mediaTab === tab ? '' : 'hidden'}><input type="file" data-media="${tab}" accept="${accept[tab]}" /><span class="drop-copy"><b>↑</b><br/>Drop ${labels[tab]} Here<br/>- or -<br/>Click to Upload</span><span class="drop-name">${name}</span></label>`;
  }).join('');
  return `<div class="media-tabs"><div class="media-tabbar">${bar}</div>${panels}</div>`;
}

function renderChatbot(label) {
  const bubbles = state.chat.map((item) => `<div class="bubble ${item.role}">${escapeHtml(item.content)}</div>`).join('');
  return `<div class="infer-group infer-chat-wrap"><span class="chat-label">${escapeHtml(label || 'Chatbot')}</span><div class="infer-chat">${bubbles}</div></div>`;
}

function renderServe() {
  const view = state.view;
  const layout = state.schema.llm_infer;
  const values = serveFormValues(view);
  const fill = view.serve_fill || {};
  const running = Boolean(view.service && view.service.status === 'running');
  if (!layout) {
    return '<section class="panel infer-form"><p>推理服务布局还没加载。</p></section>';
  }
  const chunks = layout.sections.map((section) => {
    if (section.kind === 'rows') {
      return (section.rows || []).map((row) => renderInferRow(row, values, running, fill)).join('');
    }
    if (section.kind === 'accordion') {
      const body = (section.rows || []).map((row) => renderInferRow(row, values, running, fill)).join('');
      const open = state.inferRuntimeOpen || state.inferLogVisible ? ' open' : '';
      return `<details class="infer-runtime"${open}><summary>${escapeHtml(section.label || '运行时')}</summary><div class="body">${body}</div></details>`;
    }
    if (section.kind === 'chatbot') {
      return renderChatbot(section.item && section.item.label);
    }
    if (section.kind === 'prompt') {
      const prompt = section.prompt || { label: '请输入：' };
      return `<div class="infer-row prompt-row"><div class="infer-group"><label class="field"><span>${escapeHtml(prompt.label)}</span><textarea id="chat-input" rows="4"></textarea></label></div>${renderMediaTabs()}</div>`;
    }
    if (section.kind === 'actions') {
      return `<div class="infer-row actions">${(section.items || []).map((item) => renderInferControl(item, values, running)).join('')}</div>`;
    }
    return '';
  }).join('');
  return `<section class="panel infer-form">${chunks}${fetchNote()}</section>`;
}

function render() {
  syncLogScrollLock();
  renderRuns();
  const main = document.getElementById('main');
  if (!state.view) {
    main.innerHTML = `<div class="empty"><h1>从一轮可点击的后训练闭环开始</h1>
      <p>从左边新建实验开始。</p></div>`;
    syncLive();
    syncLogScrollLock();
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
  syncLogScrollLock();
  if (state.tab === 'serve' && state.inferLogVisible) ensureInferLogPoll();
  else if (state.tab !== 'serve') stopInferLogPoll();
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
  slot.innerHTML = outputPanel('命令预览', payload.command || '这一阶段没有可预览的命令。', 'command');
}

async function validateCurrent() {
  const params = collectParams();
  const report = await api('/api/datasets/validate', {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  const slot = document.getElementById('live-command');
  slot.innerHTML = outputPanel('校验结果', JSON.stringify(report, null, 2));
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
  if (slot) slot.innerHTML = outputPanel('上传校验', JSON.stringify(data.validation, null, 2));
  toast(data.validation.ok ? '上传通过校验' : '上传文件格式未通过');
}

function stopInferLogPoll() {
  if (!state.inferLogTimer) return;
  clearInterval(state.inferLogTimer);
  state.inferLogTimer = null;
}

async function refreshInferLog() {
  if (!state.inferLogVisible || !state.view) return;
  const payload = await api(`/api/runs/${state.view.run.id}/serve/log`);
  state.inferLog = payload.log || '';
  const box = document.querySelector('[data-key="log"] textarea');
  if (box && box.value !== state.inferLog) {
    const follow = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    box.value = state.inferLog;
    if (follow) box.scrollTop = box.scrollHeight;
  }
}

function ensureInferLogPoll() {
  if (state.inferLogTimer || !state.inferLogVisible) return;
  state.inferLogTimer = setInterval(() => {
    refreshInferLog().catch(() => {});
  }, 1000);
}

async function startServe() {
  const params = collectParams();
  const payload = await api(`/api/runs/${state.view.run.id}/serve`, {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  state.view = payload.view;
  state.serveDraft = null;
  state.serveDraftRun = null;
  const alerts = (state.schema.llm_infer && state.schema.llm_infer.alerts) || {};
  toast(alerts.loaded_alert || '模型加载完成');
  render();
}

async function stopServe() {
  const payload = await api(`/api/runs/${state.view.run.id}/serve/stop`, { method: 'POST', body: '{}' });
  state.view = payload.view;
  toast('推理服务已停止');
  render();
}

function readDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function buildUserContent(text, files) {
  const parts = [];
  if (text) parts.push({ type: 'text', text });
  if (files.image) parts.push({ type: 'image_url', image_url: { url: await readDataUrl(files.image) } });
  if (files.video) parts.push({ type: 'video_url', video_url: { url: await readDataUrl(files.video) } });
  if (files.audio) parts.push({ type: 'input_audio', input_audio: { data: await readDataUrl(files.audio) } });
  if (parts.length === 1 && parts[0].type === 'text') return text;
  return parts.length ? parts : text;
}

async function sendChat() {
  const alerts = (state.schema.llm_infer && state.schema.llm_infer.alerts) || {};
  const running = state.view.service && state.view.service.status === 'running';
  if (!running) {
    toast(alerts.generate_alert || '请先部署模型');
    return;
  }
  const input = document.getElementById('chat-input');
  const content = input.value.trim();
  const files = state.mediaFiles || {};
  if (!content && !files.image && !files.video && !files.audio) return;
  const params = collectParams();
  const userContent = await buildUserContent(content, files);
  const history = state.chat.map((item) => ({ role: item.role, content: item.content }));
  history.push({ role: 'user', content: userContent });
  state.chat.push({ role: 'user', content: content || '（多媒体）' });
  state.chat.push({ role: 'assistant', content: '' });
  input.value = '';
  render();
  const response = await fetch(`/api/runs/${state.view.run.id}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: history,
      stream: true,
      system: params.system || '',
      temperature: params.temperature,
      top_k: params.top_k,
      top_p: params.top_p,
      repetition_penalty: params.repetition_penalty,
      max_new_tokens: params.max_new_tokens,
      infer_model_type: params.infer_model_type || '',
    }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || '发送失败');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const step = await reader.read();
    if (step.done) break;
    buffer += decoder.decode(step.value, { stream: true });
    const pieces = buffer.split('\n\n');
    buffer = pieces.pop();
    pieces.forEach((piece) => {
      const data = piece.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n');
      if (!data || data === '[DONE]') return;
      let payload;
      try {
        payload = JSON.parse(data);
      } catch (err) {
        return;
      }
      const choice = (payload.choices || [])[0] || {};
      const delta = (choice.delta && choice.delta.content) || (choice.message && choice.message.content) || '';
      if (!delta) return;
      const last = state.chat[state.chat.length - 1];
      last.content += delta;
      const node = document.querySelector('.infer-chat .bubble.assistant:last-child');
      if (node) node.textContent = last.content;
    });
  }
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
    if (target.dataset.action === 'open-serve') {
      state.tab = 'serve';
      render();
      return;
    }
    if (target.dataset.action === 'fill-serve') {
      const fill = (state.view && state.view.serve_fill) || {};
      const current = collectParams();
      state.serveDraft = {
        ...current,
        model: fill.model || current.model || '',
        adapters: fill.adapters || '',
        model_type: fill.model_type || current.model_type || '',
        template: fill.template || current.template || '',
      };
      state.serveDraftRun = state.view.run.id;
      render();
      toast('已填入本轮 checkpoint');
      return;
    }
    if (target.dataset.action === 'reset-infer') {
      const layout = state.schema.llm_infer;
      const current = collectParams();
      const locked = new Set((layout && layout.locked) || []);
      const next = { ...current };
      Object.values((layout && layout.by_id) || {}).forEach((item) => {
        if (!item || item.widget === 'Button' || locked.has(item.id)) return;
        next[item.id] = item.default == null ? '' : item.default;
      });
      state.serveDraft = next;
      state.serveDraftRun = state.view.run.id;
      render();
      toast('已恢复初始值');
      return;
    }
    if (target.dataset.action === 'slider-reset') {
      const key = target.dataset.reset;
      const item = state.schema.llm_infer && state.schema.llm_infer.by_id[key];
      const input = document.querySelector(`[data-field="${key}"]`);
      const num = document.querySelector(`[data-slider-num="${key}"]`);
      if (input && item) {
        input.value = item.default;
        if (num) num.value = item.default;
        rememberServeDraft({ target: input });
      }
      return;
    }
    if (target.dataset.mediaTab) {
      state.mediaTab = target.dataset.mediaTab;
      render();
      return;
    }
    if (target.dataset.action === 'clear-chat') {
      state.chat = [];
      state.mediaFiles = { image: null, video: null, audio: null };
      render();
      return;
    }
    if (target.dataset.action === 'serve-refresh') {
      const round = state.view.round.round_index;
      state.view = await api(`/api/runs/${state.view.run.id}?round=${round}`);
      render();
      toast('已找回部署');
      return;
    }
    if (target.dataset.action === 'serve-log') {
      const payload = await api(`/api/runs/${state.view.run.id}/serve/log`);
      state.inferLog = payload.log || '';
      state.inferLogVisible = true;
      state.inferRuntimeOpen = true;
      render();
      ensureInferLogPoll();
      return;
    }
    if (target.dataset.action === 'serve-log-stop') {
      state.inferLogVisible = false;
      stopInferLogPoll();
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

document.addEventListener('toggle', (event) => {
  const node = event.target;
  if (!node || !node.classList || !node.classList.contains('infer-runtime')) return;
  state.inferRuntimeOpen = Boolean(node.open);
}, true);

function rememberServeDraft(event) {
  if (state.tab !== 'serve' || !state.view) return;
  const field = event.target && event.target.dataset && event.target.dataset.field;
  if (!field || !event.target.closest('#main')) return;
  state.serveDraft = collectParams();
  state.serveDraftRun = state.view.run.id;
}

document.addEventListener('input', (event) => {
  const target = event.target;
  if (target && target.type === 'range') {
    const num = document.querySelector(`[data-slider-num="${target.dataset.field}"]`);
    if (num) num.value = target.value;
  }
  if (target && target.dataset && target.dataset.sliderNum) {
    const range = document.querySelector(`[data-field="${target.dataset.sliderNum}"]`);
    if (range) {
      range.value = target.value;
      rememberServeDraft({ target: range });
      return;
    }
  }
  rememberServeDraft(event);
});

document.addEventListener('change', async (event) => {
  const media = event.target.dataset && event.target.dataset.media;
  if (media && event.target.files && event.target.files[0]) {
    state.mediaFiles = state.mediaFiles || { image: null, video: null, audio: null };
    state.mediaFiles[media] = event.target.files[0];
    const name = event.target.parentElement && event.target.parentElement.querySelector('.drop-name');
    if (name) name.textContent = event.target.files[0].name;
  }
  rememberServeDraft(event);
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
    return `${syncing.repo_id} 同步中，进度见 rsi/.rsi.log`;
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
      const detail = row.message && row.message !== '同步中' ? row.message : '同步中，进度见 rsi/.rsi.log';
      parts.push(`${row.repo_id} ${detail}`);
    } else if (row.state === '失败') {
      parts.push(`${row.repo_id} ${row.message || '失败'}`);
    }
  });
  return parts.join(' · ');
}

function fetchNote() {
  if (!state.schema || state.schema.model_source !== 'hope') return '';
  const text = fetchProgressText();
  return `<span class="badge fetch-progress" data-fetch-note${text ? '' : ' hidden'}>${escapeHtml(text)}</span>`;
}

function refetchButton() {
  if (!state.schema || state.schema.model_source !== 'hope') return '';
  return '<button type="button" class="ghost" data-action="refetch">重新拉取</button>';
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

async function boot() {
  state.schema = await api('/api/schema');
  fillCreateModel();
  await refreshRuns();
  render();
}

boot().catch((err) => toast(err.message));
