const state = {
  schema: null,
  runs: [],
  view: null,
  // Master–detail: list = experiment table; workspace = open run.
  screen: 'list',
  runFilter: '',
  stage: 'sft',
  tab: 'flow',
  round: null,
  expanded: null,
  chat: [],
  busy: false,
  liveTimer: null,
  liveBusy: false,
  runListTimer: null,
  runListGen: 0,
  serveDraft: null,
  serveDraftRun: null,
  inferLog: '',
  inferLogTimer: null,
  inferRuntimeOpen: false,
  exportDraft: null,
  exportDraftRun: null,
  exportLog: '',
  exportLogTimer: null,
  exportRuntimeOpen: false,
  // After 「用该模型部署推理」: show exported path as --model (no adapters).
  serveExportOverride: null,
  serveExportOverrideRun: null,
  draftTimer: null,
  mediaTab: 'image',
  mediaFiles: { image: null, video: null, audio: null },
  metricsCache: {},
  // stage|round → { next, start, truncated } for incremental /stages/.../log follow.
  logCursors: {},
  // Flexible same-run eval report picker (GET /api/runs/{id}/eval-reports).
  evalPicker: {
    runId: null,
    round: null,
    scope: 'round', // 'round' | 'all'
    reports: [],
    selected: {},
    tableHtml: '',
    loading: false,
    error: '',
  },
};

// Same keys as swift.ui Runtime.sft_plot / grpo_plot (placeholder until TB arrives).
const STAGE_METRIC_KEYS = {
  sft: ['train/loss', 'train/acc', 'train/learning_rate', 'eval/loss', 'eval/acc'],
  rl: ['train/loss', 'train/reward', 'train/learning_rate', 'train/completions/mean_length', 'train/kl'],
};

// Canonical compare keys (alias top-level bleu_1 / rouge_l_f → scores names).
const COMPARE_METRIC_ALIASES = {
  bleu_1: 'bleu-1',
  bleu_2: 'bleu-2',
  bleu_3: 'bleu-3',
  bleu_4: 'bleu-4',
  rouge_l_f: 'rouge-l-f',
  rouge_1_f: 'rouge-1-f',
  rouge_2_f: 'rouge-2-f',
};

const COMPARE_METRIC_LABELS = {
  accuracy: 'Accuracy',
  'bleu-1': 'BLEU-1',
  'bleu-2': 'BLEU-2',
  'bleu-3': 'BLEU-3',
  'bleu-4': 'BLEU-4',
  'rouge-1-f': 'ROUGE-1 F1',
  'rouge-2-f': 'ROUGE-2 F1',
  'rouge-l-f': 'ROUGE-L F1',
  mock_score: 'Mock score',
};

const COMPARE_METRIC_ORDER = [
  'accuracy',
  'rouge-l-f',
  'rouge-1-f',
  'rouge-2-f',
  'bleu-1',
  'bleu-2',
  'bleu-3',
  'bleu-4',
  'mock_score',
];

function evalReportHref(stage, roundIndex, kind) {
  if (!state.view) return '';
  const round = roundIndex || state.view.round.round_index;
  let href = `/api/runs/${encodeURIComponent(state.view.run.id)}/stages/${encodeURIComponent(stage)}/report?round=${round}`;
  if (kind) href += `&kind=${encodeURIComponent(kind)}`;
  return href;
}

function evalReportLink(stage, result, roundIndex) {
  if (stage !== 'sft_eval' && stage !== 'rl_eval') return '';
  const metrics = (result && result.metrics) || {};
  const path = (result && result.report_path) || metrics.report_html || '';
  if (!path || !state.view) return '';
  const href = evalReportHref(stage, roundIndex);
  return `<a class="ghost" href="${href}" target="_blank" rel="noopener">查看完整报告</a>`;
}

function rowHasReport(row) {
  return Boolean(row && (row.report_path || (row.metrics && row.metrics.report_html)));
}

function compareReportLinks(stage, result, roundIndex) {
  const rows = result && Array.isArray(result.compare) ? result.compare : [];
  const links = rows
    .filter((row) => rowHasReport(row))
    .map((row) => {
      const href = evalReportHref(stage, roundIndex, row.role);
      return `<a class="ghost" href="${href}" target="_blank" rel="noopener">${escapeHtml(row.label || row.role)} 报告</a>`;
    });
  if (!links.length) {
    const single = evalReportLink(stage, result, roundIndex);
    return single ? `<p class="row-actions compare-reports">${single}</p>` : '';
  }
  return `<p class="row-actions compare-reports">${links.join(' · ')}</p>`;
}

function scoreHeadline(metrics) {
  if (!metrics) return '';
  const bits = [];
  if (metrics.accuracy != null && metrics.accuracy !== '') bits.push(`acc ${metrics.accuracy}`);
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

function normalizeCompareKey(key) {
  return COMPARE_METRIC_ALIASES[key] || key;
}

function compareMetricLabel(key) {
  const norm = normalizeCompareKey(key);
  if (COMPARE_METRIC_LABELS[norm]) return COMPARE_METRIC_LABELS[norm];
  const rouge = /^rouge-([12l])-([rpf])$/i.exec(norm);
  if (rouge) {
    const variant = rouge[1].toLowerCase() === 'l' ? 'L' : rouge[1];
    const stat = { r: 'R', p: 'P', f: 'F1' }[rouge[2].toLowerCase()] || rouge[2].toUpperCase();
    return `ROUGE-${variant} ${stat}`;
  }
  const bleu = /^bleu-(\d+)$/i.exec(norm);
  if (bleu) return `BLEU-${bleu[1]}`;
  return String(key);
}

function compareMetricKeys(rows) {
  const available = new Set();
  rows.forEach((row) => {
    const metrics = (row && row.metrics) || {};
    Object.keys(metrics).forEach((key) => {
      if (key === 'scores' || key === 'report_html' || key === 'report_json' || key === 'eval_rows') return;
      if (metrics[key] != null && metrics[key] !== '') available.add(normalizeCompareKey(key));
    });
    const scores = metrics.scores;
    if (scores && typeof scores === 'object') {
      Object.keys(scores).forEach((name) => {
        if (scores[name] != null && scores[name] !== '') available.add(normalizeCompareKey(name));
      });
    }
  });
  const preferred = COMPARE_METRIC_ORDER.filter((key) => available.has(key));
  if (preferred.length) return preferred;
  return Array.from(available)
    .filter((key) => {
      const match = /^rouge-([12l])-[rp]$/i.exec(key);
      if (match && available.has(`rouge-${match[1].toLowerCase()}-f`)) return false;
      return true;
    })
    .sort();
}

function compareCellRaw(row, key) {
  const norm = normalizeCompareKey(key);
  const metrics = (row && row.metrics) || {};
  const candidates = [norm, key];
  Object.keys(COMPARE_METRIC_ALIASES).forEach((alias) => {
    if (COMPARE_METRIC_ALIASES[alias] === norm) candidates.push(alias);
  });
  for (let i = 0; i < candidates.length; i += 1) {
    const candidate = candidates[i];
    if (metrics[candidate] != null && metrics[candidate] !== '') return metrics[candidate];
  }
  const scores = metrics.scores;
  if (scores && typeof scores === 'object') {
    if (scores[norm] != null && scores[norm] !== '') return scores[norm];
    if (scores[key] != null && scores[key] !== '') return scores[key];
  }
  return null;
}

function formatCompareValue(value) {
  if (value == null || value === '') return '—';
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (Number.isInteger(value)) return String(value);
    const rounded = Math.round(value * 10000) / 10000;
    return String(rounded);
  }
  return String(value);
}

function compareHighlightClasses(values) {
  const nums = [];
  values.forEach((value, index) => {
    if (typeof value === 'number' && Number.isFinite(value)) nums.push({ index, value });
  });
  const classes = values.map(() => '');
  if (nums.length < 2) return classes;
  const max = Math.max(...nums.map((item) => item.value));
  const min = Math.min(...nums.map((item) => item.value));
  if (max === min) return classes;
  nums.forEach((item) => {
    if (item.value === max) classes[item.index] = 'compare-best';
    else if (item.value === min) classes[item.index] = 'compare-worst';
  });
  return classes;
}

function buildCompareTableHtml(rows, opts) {
  const options = opts || {};
  const filtered = (rows || []).filter((row) => row && row.label);
  if (filtered.length < 2) return '';
  const keys = compareMetricKeys(filtered);
  if (!keys.length) return '';
  const head = `<tr><th>指标</th>${filtered.map((row) => `<th>${escapeHtml(row.label)}</th>`).join('')}</tr>`;
  const body = keys.map((key) => {
    const values = filtered.map((row) => compareCellRaw(row, key));
    const highlights = compareHighlightClasses(values);
    const cells = values.map((value, index) => {
      const cls = highlights[index] ? ` class="${highlights[index]}"` : '';
      return `<td${cls}>${escapeHtml(formatCompareValue(value))}</td>`;
    }).join('');
    return `<tr><td>${escapeHtml(compareMetricLabel(key))}</td>${cells}</tr>`;
  }).join('');
  const title = options.title || '对比';
  const sub = options.sub || '越高越好（浅色标出最优 / 最弱）';
  const links = options.linksHtml || '';
  return `<div class="compare-block"><p class="compare-title">${escapeHtml(title)}</p><p class="muted compare-sub">${escapeHtml(sub)}</p><table class="score-table compare-table"><thead>${head}</thead><tbody>${body}</tbody></table>${links}</div>`;
}

function compareTable(stage, result, roundIndex) {
  const rows = result && Array.isArray(result.compare) ? result.compare.filter((row) => row && row.label) : [];
  if (rows.length < 2) return '';
  const reports = compareReportLinks(stage, result, roundIndex);
  return buildCompareTableHtml(rows, {
    title: '基座对比',
    sub: '同一评测集 · 越高越好（浅色标出最优 / 最弱）',
    linksHtml: reports,
  });
}

function evalPickerCacheKey(runId, round, scope) {
  return `${runId}|${scope === 'all' ? 'all' : (round == null ? '' : round)}`;
}

function evalPickerMarkup() {
  const picker = state.evalPicker || {};
  return `<div class="compare-block eval-picker" data-eval-picker="1">
    <div class="eval-picker-launch">
      <button type="button" class="primary" data-action="eval-picker-open">对比</button>
      <span class="muted">挑选同实验评测报告做指标对照</span>
    </div>
    <div class="eval-picker-result" data-eval-picker-result="1">${picker.tableHtml || ''}</div>
  </div>`;
}

function syncEvalCompareScopeRadios() {
  const dialog = document.getElementById('eval-compare-dialog');
  if (!dialog) return;
  const scope = (state.evalPicker && state.evalPicker.scope) === 'all' ? 'all' : 'round';
  dialog.querySelectorAll('input[name="eval-picker-scope"]').forEach((input) => {
    input.checked = input.value === scope;
  });
}

function paintEvalPickerList() {
  const root = document.querySelector('#eval-compare-dialog [data-eval-picker-list]');
  if (!root) return;
  const picker = state.evalPicker || {};
  if (picker.loading) {
    root.innerHTML = '<p class="muted">加载评测报告…</p>';
    return;
  }
  if (picker.error) {
    root.innerHTML = `<p class="muted">${escapeHtml(picker.error)}</p>`;
    return;
  }
  const reports = picker.reports || [];
  if (!reports.length) {
    root.innerHTML = '<p class="muted">还没有可对比的评测报告。先跑完 SFT/RL 评测。</p>';
    return;
  }
  const chips = reports.map((report) => {
    const id = report.id;
    const checked = picker.selected && picker.selected[id] ? ' checked' : '';
    const meta = [];
    if (report.metrics && report.metrics.accuracy != null) meta.push(`acc ${report.metrics.accuracy}`);
    if (report.metrics && report.metrics.bleu_1 != null) meta.push(`BLEU-1 ${report.metrics.bleu_1}`);
    const hint = meta.length ? `<span class="eval-picker-meta">${escapeHtml(meta.join(' · '))}</span>` : '';
    const link = report.has_report && report.report_url
      ? `<a class="ghost eval-picker-report" href="${escapeHtml(report.report_url)}" target="_blank" rel="noopener">报告</a>`
      : '';
    return `<label class="eval-picker-chip"><input type="checkbox" data-eval-report-id="${escapeHtml(id)}"${checked}/> <span class="eval-picker-label">${escapeHtml(report.label)}</span>${hint}${link}</label>`;
  }).join('');
  root.innerHTML = `<div class="eval-picker-chips">${chips}</div>`;
}

function paintEvalPickerResult() {
  const slot = document.querySelector('[data-eval-picker-result]');
  if (!slot) return;
  slot.innerHTML = (state.evalPicker && state.evalPicker.tableHtml) || '';
}

async function loadEvalReports(force) {
  if (!state.view || (state.stage !== 'sft_eval' && state.stage !== 'rl_eval')) return;
  const dialog = document.getElementById('eval-compare-dialog');
  if (!dialog || !dialog.open) return;
  const runId = state.view.run.id;
  const round = state.view.round.round_index;
  const picker = state.evalPicker;
  const scope = picker.scope === 'all' ? 'all' : 'round';
  const cacheKey = evalPickerCacheKey(runId, round, scope);
  if (!force && picker._cacheKey === cacheKey && picker.reports && picker.reports.length && !picker.error) {
    paintEvalPickerList();
    return;
  }
  picker.runId = runId;
  picker.round = round;
  picker.loading = true;
  picker.error = '';
  paintEvalPickerList();
  try {
    const query = scope === 'round' ? `?round=${encodeURIComponent(round)}` : '';
    const payload = await api(`/api/runs/${encodeURIComponent(runId)}/eval-reports${query}`);
    const reports = (payload && payload.reports) || [];
    const keep = {};
    reports.forEach((report) => {
      if (picker.selected && picker.selected[report.id]) keep[report.id] = true;
    });
    picker.reports = reports;
    picker.selected = keep;
    picker._cacheKey = cacheKey;
    picker.error = '';
  } catch (err) {
    picker.reports = [];
    picker.error = (err && err.message) || '加载评测报告失败';
  } finally {
    picker.loading = false;
    paintEvalPickerList();
  }
}

function openEvalCompareDialog() {
  const dialog = document.getElementById('eval-compare-dialog');
  if (!dialog) return;
  syncEvalCompareScopeRadios();
  paintEvalPickerList();
  if (!dialog.open) dialog.showModal();
  loadEvalReports(false).catch(() => {});
}

function closeEvalCompareDialog() {
  const dialog = document.getElementById('eval-compare-dialog');
  if (dialog && dialog.open) dialog.close();
}

function runEvalPickerCompare() {
  const picker = state.evalPicker || {};
  const selectedIds = Object.keys(picker.selected || {}).filter((id) => picker.selected[id]);
  if (selectedIds.length < 2) {
    toast('请至少勾选两份报告再对比');
    return;
  }
  const byId = {};
  (picker.reports || []).forEach((report) => { byId[report.id] = report; });
  const rows = selectedIds.map((id) => byId[id]).filter(Boolean).map((report) => ({
    label: report.label,
    role: report.role,
    metrics: report.metrics || {},
    report_path: report.report_path || '',
    report_url: report.report_url || '',
  }));
  if (rows.length < 2) {
    toast('勾选的报告已失效，请刷新列表');
    return;
  }
  const links = rows
    .filter((row) => row.report_url)
    .map((row) => `<a class="ghost" href="${escapeHtml(row.report_url)}" target="_blank" rel="noopener">${escapeHtml(row.label)} 报告</a>`);
  const linksHtml = links.length
    ? `<p class="row-actions compare-reports">${links.join(' · ')}</p>`
    : '';
  picker.tableHtml = buildCompareTableHtml(rows, {
    title: '对比结果',
    sub: `已选 ${rows.length} 份报告 · 越高越好（浅色标出最优 / 最弱）`,
    linksHtml,
  }) || '<p class="muted">选中的报告没有可对比的数值指标。</p>';
  closeEvalCompareDialog();
  paintEvalPickerResult();
}

function evalResultMarkup(stage, result, roundIndex) {
  if (!result) return '';
  const table = compareTable(stage, result, roundIndex);
  // When the side-by-side table is present, skip the duplicate single-column scores.
  const headline = table ? '' : scoreHeadline(result.metrics);
  const scores = table ? '' : scoreTable(result.metrics);
  const report = table ? '' : compareReportLinks(stage, result, roundIndex);
  const baseNote = result.base_error
    ? `<p class="muted">基座评测失败：${escapeHtml(result.base_error)}（本阶段检查点结果已保留）</p>`
    : '';
  const picker = (stage === 'sft_eval' || stage === 'rl_eval') ? evalPickerMarkup() : '';
  return `${table}${baseNote}${picker}${headline}${scores}${report}${outputPanel('结果', JSON.stringify(visibleResult(result), null, 2))}`;
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

function normalizeBase(value) {
  let text = String(value == null ? '' : value).trim().replace(/\\/g, '/');
  while (text.indexOf('//') !== -1) text = text.replace('//', '/');
  while (text.endsWith('/')) text = text.slice(0, -1);
  return text;
}

function sameBase(left, right) {
  return normalizeBase(left) === normalizeBase(right);
}

function baseCaption(initial, current) {
  const first = initial == null ? '' : String(initial);
  const round = current == null ? '' : String(current);
  if (sameBase(first, round)) {
    const shown = normalizeBase(round) ? round : (normalizeBase(first) ? first : '');
    return `实验底座 <span class="mono">${escapeHtml(shown)}</span>`;
  }
  return `实验底座 <span class="mono">${escapeHtml(first)}</span><br/>本轮起点 <span class="mono">${escapeHtml(round)}</span>`;
}

function toast(message) {
  const el = document.getElementById('toast');
  el.hidden = false;
  el.textContent = message;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { el.hidden = true; }, 4200);
}

function setBusy(on) {
  state.busy = Boolean(on);
  document.body.classList.toggle('is-busy', state.busy);
}

function tipAttr(text) {
  return text ? ` title="${escapeHtml(text)}"` : '';
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
  if (stage === 'export') return '模型导出';
  if (stage === 'decide') return '选择';
  if (stage === 'finish') return '完成';
  return stage;
}

function listStatusOf(run) {
  const code = run && run.list_status;
  if (code === 'running' || code === 'completed' || code === 'failed' || code === 'pending') return code;
  if (run && run.status === 'completed') return 'completed';
  if (run && run.status === 'running') return 'running';
  return 'pending';
}

function listStatusLabel(code) {
  if (code === 'running') return '运行中';
  if (code === 'completed') return '已完成';
  if (code === 'failed') return '失败';
  return '待运行';
}

function runListFingerprint(runs) {
  return (runs || []).map((run) => [
    run.id, run.name, run.round_index, run.current_stage, run.status, listStatusOf(run),
    run.deployable ? '1' : '0', run.deploy_kind || '',
  ].join('\t')).join('\n');
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

function ensureRunListPoll() {
  const running = (state.runs || []).some((run) => listStatusOf(run) === 'running');
  if (!running) {
    if (state.runListTimer) {
      clearInterval(state.runListTimer);
      state.runListTimer = null;
    }
    return;
  }
  if (state.runListTimer) return;
  state.runListTimer = setInterval(() => {
    refreshRunList().catch(() => {});
  }, 2000);
}

async function refreshRunList() {
  const gen = ++state.runListGen;
  const payload = await api('/api/runs');
  if (gen !== state.runListGen) return;
  const next = payload.runs || [];
  const changed = runListFingerprint(state.runs) !== runListFingerprint(next);
  state.runs = next;
  if (changed) renderRuns();
  ensureRunListPoll();
}

function touchRunList() {
  refreshRunList().catch(() => {});
}

async function refreshRuns(selectId) {
  await refreshRunList();
  if (selectId) {
    await openRun(selectId, null, false);
  }
}

function filteredRuns() {
  const q = String(state.runFilter || '').trim().toLowerCase();
  if (!q) return state.runs || [];
  return (state.runs || []).filter((run) => {
    const hay = [run.name, run.scenario, run.base_model, run.id, run.current_stage]
      .map((v) => String(v || '').toLowerCase())
      .join(' ');
    return hay.includes(q);
  });
}

function syncScreenAttr() {
  document.body.dataset.screen = state.screen === 'workspace' ? 'workspace' : 'list';
  const nav = document.getElementById('nav-runs');
  if (nav) nav.classList.toggle('active', state.screen !== 'workspace');
}

function renderRuns() {
  // Table lives inside #main when on list screen; partial refresh keeps toolbar state.
  if (state.screen !== 'list') return;
  const tbody = document.querySelector('[data-runs-tbody]');
  if (!tbody) {
    render();
    return;
  }
  const rows = filteredRuns();
  const meta = document.querySelector('[data-runs-meta]');
  if (meta) {
    meta.textContent = rows.length === (state.runs || []).length
      ? `共 ${rows.length} 个实验`
      : `显示 ${rows.length} / ${state.runs.length} 个实验`;
  }
  if (!state.runs.length) {
    tbody.innerHTML = `<tr class="runs-empty-row"><td colspan="7">
      <div class="runs-empty">
        <p>还没有实验。</p>
        <button type="button" class="primary" id="new-run">新建实验</button>
      </div>
    </td></tr>`;
    return;
  }
  if (!rows.length) {
    tbody.innerHTML = `<tr class="runs-empty-row"><td colspan="7"><p class="muted runs-empty">没有匹配「${escapeHtml(state.runFilter)}」的实验。</p></td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((run) => {
    const code = listStatusOf(run);
    const updated = formatShanghai(run.updated_at || run.created_at || '');
    const deployable = Boolean(run.deployable);
    const deployKind = run.deploy_kind === 'rl' ? 'RL' : (run.deploy_kind === 'sft' ? 'SFT' : '');
    const statusBits = [`<span class="badge run-list-status ${escapeHtml(code)}">${escapeHtml(listStatusLabel(code))}</span>`];
    if (deployable) {
      statusBits.push(`<span class="badge deployable" title="有可用 ${escapeHtml(deployKind || '')} checkpoint">可部署</span>`);
    }
    const deployBtn = deployable
      ? `<button type="button" class="ghost compact" data-open-serve="${escapeHtml(run.id)}" aria-label="部署 ${escapeHtml(run.name)}">部署</button>`
      : '';
    return `<tr data-run-row="${escapeHtml(run.id)}">
      <td class="col-name">
        <button type="button" class="linkish" data-run="${escapeHtml(run.id)}">${escapeHtml(run.name)}</button>
        <div class="muted row-sub">${escapeHtml(run.scenario || '')}</div>
      </td>
      <td class="col-status">${statusBits.join(' ')}</td>
      <td class="mono">第 ${escapeHtml(String(run.round_index || 1))} 轮</td>
      <td>${escapeHtml(labelOf(run.current_stage))}</td>
      <td class="mono col-model" title="${escapeHtml(run.base_model || '')}">${escapeHtml(run.base_model || '—')}</td>
      <td class="mono muted">${escapeHtml(updated)}</td>
      <td class="col-actions">
        <button type="button" class="primary compact" data-run="${escapeHtml(run.id)}">打开</button>
        ${deployBtn}
        <button type="button" class="ghost compact" data-delete-run="${escapeHtml(run.id)}" aria-label="删除 ${escapeHtml(run.name)}">删除</button>
      </td>
    </tr>`;
  }).join('');
}

function renderExperimentsPage() {
  const q = escapeHtml(state.runFilter || '');
  return `<section class="runs-page">
    <header class="runs-toolbar">
      <div>
        <p class="kicker">实验管理</p>
        <h1>全部实验</h1>
        <p class="muted" data-runs-meta></p>
      </div>
      <div class="runs-toolbar-actions">
        <label class="runs-search">
          <span class="sr-only">筛选实验</span>
          <input type="search" data-run-filter placeholder="按名称 / 场景 / 底座筛选" value="${q}" />
        </label>
        <button type="button" class="primary" id="new-run">新建实验</button>
      </div>
    </header>
    <div class="runs-table-wrap">
      <table class="runs-table">
        <thead>
          <tr>
            <th scope="col">实验</th>
            <th scope="col">状态</th>
            <th scope="col">轮次</th>
            <th scope="col">当前阶段</th>
            <th scope="col">底座模型</th>
            <th scope="col">更新时间</th>
            <th scope="col">操作</th>
          </tr>
        </thead>
        <tbody data-runs-tbody></tbody>
      </table>
    </div>
  </section>`;
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
  const confirmed = await askConfirm({
    title: '删除实验',
    name,
    body: '实验记录里的轮次、阶段和检查点会去掉，该实验目录下的阶段产物也会清理。上传的数据文件会保留。',
    okLabel: '删除',
    danger: true,
  });
  if (!confirmed) return;
  await api(`/api/runs/${id}`, { method: 'DELETE' });
  const wasCurrent = state.view && state.view.run.id === id;
  if (wasCurrent) {
    state.chat = [];
    state.view = null;
    state.screen = 'list';
  }
  await refreshRuns();
  render();
  toast('实验已删除');
}

function showRunsList() {
  state.screen = 'list';
  render();
}

async function openRun(id, round, keepStage) {
  await flushDraft();
  const query = round ? `?round=${round}` : '';
  const prevId = state.view && state.view.run ? state.view.run.id : null;
  state.view = await api(`/api/runs/${id}${query}`);
  state.screen = 'workspace';
  if (prevId !== id) state.logCursors = {};
  state.serveDraft = null;
  state.serveDraftRun = null;
  state.exportDraft = null;
  state.exportDraftRun = null;
  state.exportLog = '';
  if (prevId !== id) clearServeExportOverride();
  state.round = state.view.round.round_index;
  if (!keepStage) {
    const current = state.view.run.current_stage;
    state.stage = state.view.mutable || current === 'finish' ? current : 'sft';
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
  const lockTitle = field.locked ? ' title="由实验创建时决定，不能在这里修改"' : '';
  if (field.locked && (field.type === 'combo' || field.type === 'multicombo' || field.type === 'text' || field.type === 'number' || field.type === 'textarea')) {
    const control = field.type === 'textarea'
      ? `<textarea data-field="${field.key}" rows="3" readonly${lockTitle}>${escapeHtml(value)}</textarea>`
      : `<input type="text" data-field="${field.key}" value="${escapeHtml(value)}" readonly${lockTitle}/>`;
    return `<label class="field locked" data-key="${field.key}">${inferCaption(field)}${control}</label>`;
  }
  if (field.type === 'checkbox') {
    const checked = value === true || value === 'true' ? 'checked' : '';
    const disabled = field.locked ? 'disabled' : '';
    return `<div class="field${field.locked ? ' locked' : ''}" data-key="${field.key}"${lockTitle}><label class="check"><input type="checkbox" data-field="${field.key}" ${checked} ${disabled}/> ${escapeHtml(field.label)}</label></div>`;
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
    const disabled = field.locked ? 'disabled' : '';
    return `<label class="field${field.locked ? ' locked' : ''}" data-key="${field.key}"${lockTitle}><span>${escapeHtml(field.label)}</span><select data-field="${field.key}" ${multi} ${disabled}>${options}</select></label>`;
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
    if (el.closest('[hidden]')) return;
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
    if (el.type === 'number' || el.type === 'range') {
      params[el.dataset.field] = el.value === '' ? null : Number(el.value);
      return;
    }
    params[el.dataset.field] = el.value;
  });
  const yaml = document.getElementById('extra-yaml');
  if (yaml) params.extra_yaml = yaml.value;
  if (state.tab === 'serve' && state.serveExportOverride
      && state.view && state.serveExportOverrideRun === state.view.run.id) {
    params.deploy_export = true;
    params.model = state.serveExportOverride.model || '';
    params.adapters = '';
  }
  return params;
}

function syncSource() {
  const source = document.querySelector('[data-field="source"]');
  if (!source) return;
  const value = source.value;
  document.querySelectorAll('[data-dataset-mode]').forEach((el) => {
    el.hidden = el.dataset.datasetMode !== value;
  });
}

function datasetTokens(value) {
  return String(value ?? '').trim().split(/\s+/).filter(Boolean);
}

function normalizeChoiceEntry(item) {
  if (item && typeof item === 'object' && !Array.isArray(item) && 'value' in item) {
    const value = String(item.value);
    const label = item.label == null || item.label === '' ? value : String(item.label);
    return { value, label: label || '（空）' };
  }
  const value = String(item ?? '');
  return { value, label: value || '（空）' };
}

function choiceLabelMap(input) {
  const map = {};
  catalogEntries(input && input.dataset ? input.dataset.choicesFrom : '', input).forEach((entry) => {
    map[entry.value] = entry.label;
  });
  return map;
}

function paintDatasetChips(box) {
  if (!box) return;
  const hidden = chipField(box);
  const tokens = datasetTokens(hidden && hidden.value);
  box.querySelectorAll('.chip').forEach((el) => el.remove());
  const query = box.querySelector('.chip-query');
  if (!query) return;
  const labels = choiceLabelMap(query);
  const html = tokens.map((token) => {
    const label = labels[token] || token;
    return `<span class="chip"><span>${escapeHtml(label)}</span><button type="button" data-chip-remove="${escapeHtml(token)}" aria-label="移除">×</button></span>`;
  }).join('');
  query.insertAdjacentHTML('beforebegin', html);
  const clear = box.querySelector('[data-action="chip-clear"], [data-action="dataset-clear"]');
  if (clear) clear.hidden = tokens.length === 0;
}

function datasetChoices(kind) {
  const base = kind === 'eval'
    ? ((state.schema && state.schema.bundled_eval) || [])
    : ((state.schema && state.schema.bundled) || []);
  const extra = (state.workbenchFiles || []).filter((item) => item.kind === kind);
  return extra.concat(base);
}

function choiceFile(item, current) {
  if (item && String(item.id).startsWith('wb:')) return item.path;
  if (item && item.path && String(item.path).indexOf('examples/') >= 0 && String(item.id).endsWith('_eval')) {
    return bundledEvalLocation(item, current);
  }
  return bundledTrainLocation(item, current);
}

function bundledSelect(selected) {
  const items = datasetChoices('train');
  const current = selected || 'helpdesk_train';
  const options = items.map((item) => {
    const mark = item.id === current ? 'selected' : '';
    return `<option value="${escapeHtml(item.id)}" ${mark}>${escapeHtml(item.label)}</option>`;
  }).join('');
  return `<select data-field="bundled">${options}</select>`;
}

function shownDataSource(value, dataset) {
  if (value === 'public' || value === 'upload' || value === 'path' || value === 'bundled') return value;
  const text = String(dataset || '');
  if (text.startsWith('/') || text.includes('\\') || text.startsWith('.')) return 'path';
  return 'public';
}

function renderLockedModelRow(fields, values) {
  const labels = {
    model: '模型id或路径(--model)',
    model_type: '模型类型(--model_type)',
    template: '模型Prompt模板类型(--template)',
  };
  const byKey = {};
  (fields || []).forEach((field) => { byKey[field.key] = field; });
  const cells = ['model', 'model_type', 'template'].map((key) => {
    const field = byKey[key] || { key, label: labels[key], info: '' };
    const value = values && values[key] != null ? values[key] : '';
    return `<div class="field locked" data-key="${key}">${inferCaption(field)}<div class="control"><input type="text" data-field="${key}" value="${escapeHtml(value)}" readonly title="由实验创建时决定，不能在这里修改"/></div><div class="extra"></div></div>`;
  }).join('');
  return `<div class="infer-row infer-grid model-lock"><div class="infer-group" style="--cols:3">${cells}</div></div>`;
}

function renderDatasetSlider(field, values) {
  const min = field.minimum == null ? 0 : field.minimum;
  const max = field.maximum == null ? 1 : field.maximum;
  const step = field.step == null ? 'any' : field.step;
  let value = fieldValue(field, values);
  if (value === '' && field.default != null) value = field.default;
  return `<div class="field infer-slider" data-key="${field.key}">${inferCaption(field)}<div class="control slider-head"><input type="number" class="slider-num" data-slider-num="${field.key}" min="${min}" max="${max}" step="${step}" value="${escapeHtml(value)}" /><button type="button" class="ghost icon" data-action="slider-reset" data-reset="${field.key}" title="恢复初始值">↺</button></div><div class="extra"><input type="range" data-field="${field.key}" min="${min}" max="${max}" step="${step}" value="${escapeHtml(value)}" /><div class="slider-ends"><span>${escapeHtml(min)}</span><span>${escapeHtml(max)}</span></div></div></div>`;
}

function renderDatasetSettings(fields, values) {
  const byKey = {};
  fields.forEach((field) => { byKey[field.key] = field; });
  const source = byKey.source;
  const dataset = byKey.dataset || { key: 'dataset', label: '数据集名称(--dataset)', info: '选择训练的数据集，支持复选/本地路径' };
  const datasetValue = fieldValue(dataset, values);
  const sourceValue = shownDataSource(fieldValue(source, values), datasetValue);
  const options = choiceEntries(source).map((choice) => {
    const mark = String(choice.value) === sourceValue ? 'selected' : '';
    return `<option value="${escapeHtml(choice.value)}" ${mark}>${escapeHtml(choice.label)}</option>`;
  }).join('');
  const tokens = datasetTokens(datasetValue);
  const chips = tokens.map((token) => `<span class="chip"><span>${escapeHtml(token)}</span><button type="button" data-chip-remove="${escapeHtml(token)}" aria-label="移除">×</button></span>`).join('');
  const clearHidden = tokens.length ? '' : 'hidden';
  const hide = (mode) => (mode === sourceValue ? '' : 'hidden');
  const sourceField = `<div class="field" data-key="source">${inferCaption(source)}<div class="control"><select data-field="source">${options}</select></div><div class="extra"></div></div>`;
  const datasetField = `<div class="field" data-key="dataset">${inferCaption(dataset)}<div class="control">
      <div data-dataset-mode="public" ${hide('public')}><div class="combo chip-combo"><div class="combo-anchor chip-box">${chips}<input class="chip-query" data-combo="1" data-chip="1" data-choices-from="datasets" value="" autocomplete="off" placeholder="选择或输入数据集" /><button type="button" class="chip-clear" data-action="dataset-clear" aria-label="清空" ${clearHidden}>×</button><button type="button" class="chip-caret" data-action="dataset-menu" aria-label="展开数据集">▾</button><input type="hidden" data-field="dataset" value="${escapeHtml(tokens.join(' '))}" /><div class="combo-list" hidden></div></div></div></div>
      <div data-dataset-mode="upload" ${hide('upload')}><input id="file-input" type="file" accept=".jsonl,.json,.txt,.zip,.tar,.tar.gz,.tgz" /><input class="dataset-path" type="text" data-field="dataset" value="${escapeHtml(datasetValue)}" readonly placeholder="上传后显示文件路径" /></div>
      <div data-dataset-mode="path" ${hide('path')}><input type="text" data-field="dataset" value="${escapeHtml(datasetValue)}" placeholder="服务器上的文件或目录路径" /></div>
      <div data-dataset-mode="bundled" ${hide('bundled')}>${bundledSelect(values && values.bundled)}</div>
    </div><div class="extra"></div></div>`;
  const split = byKey.split_dataset_ratio ? renderDatasetSlider(byKey.split_dataset_ratio, values) : '';
  const length = byKey.max_length ? renderDatasetSlider(byKey.max_length, values) : '';
  const padding = byKey.padding_free ? (() => {
    const checked = fieldValue(byKey.padding_free, values) === true || fieldValue(byKey.padding_free, values) === 'true' ? 'checked' : '';
    return `<div class="field check" data-key="padding_free">${inferCaption(byKey.padding_free)}<div class="control"><label class="check"><input type="checkbox" data-field="padding_free" ${checked}/></label></div><div class="extra"></div></div>`;
  })() : '';
  return `<details class="infer-runtime dataset-settings" open><summary>数据集设置</summary><div class="body"><div class="infer-row infer-grid"><div class="infer-group dataset-grid">${sourceField}${datasetField}${split}${length}${padding}</div></div></div></details>`;
}

function renderPromote(values) {
  const candidates = state.view.candidates || [];
  let selected = values.choice || state.view.recommendation;
  const selectedRow = candidates.find((cand) => cand.kind === selected);
  if (!selectedRow || !selectedRow.path) {
    const fallback = candidates.find((cand) => cand.path);
    selected = fallback ? fallback.kind : '';
  }
  const cards = candidates.map((cand) => {
    const disabled = cand.path ? '' : 'disabled';
    const checked = selected === cand.kind ? 'checked' : '';
    const best = cand.kind === state.view.recommendation ? 'best' : '';
    const score = cand.score === null || cand.score === undefined ? '—' : cand.score;
    const tip = cand.path ? '' : ' title="还没有检查点，不能选"';
    return `<label class="candidate ${best}"${tip}>
      <input type="radio" name="choice" data-field="choice" value="${cand.kind}" ${checked} ${disabled}${tip}/>
      <b>${escapeHtml(cand.label)}</b>
      ${cand.kind === state.view.recommendation ? '<span class="badge succeeded">建议</span>' : ''}
      <div class="score">${escapeHtml(score)}</div>
      <small class="mono">${escapeHtml(cand.path || '尚无检查点')}</small>
    </label>`;
  }).join('');
  const intro = '<p class="muted">底座模型不变；这里只选下一轮要带上的一份起点（LoRA 或全量 checkpoint），三选一、不会叠加。</p>';
  const missing = candidates.filter((cand) => !cand.path).map((cand) => cand.label);
  const hint = missing.length
    ? `<p class="muted run-reason">${escapeHtml(missing.join('、'))} 还没有检查点，${missing.length > 1 ? '这些选项不能选。' : '这个选项不能选。'}</p>`
    : '';
  return `${intro}<div class="candidates">${cards}</div>${hint}`;
}

function logCursorKey(stage, round) {
  return `${stage}|${round == null ? '' : round}`;
}

function liveLogSeed(stage, formLogs, round) {
  // Full render replaces the <pre>. Prefer text already painted by pollLive over an
  // empty/stale step snapshot (syncing→running often has form.logs === '').
  const roundKey = round == null ? (state.view && state.view.round ? state.view.round.round_index : '') : round;
  const cached = (state.logCursors[logCursorKey(stage, roundKey)] || {}).text || '';
  const existing = [...document.querySelectorAll('[data-live-log], [data-stage-log]')].find(
    (el) => el.dataset.liveStage === String(stage)
      && String(el.dataset.liveRound || '') === String(roundKey || '')
      && el.isConnected);
  const shown = existing ? (existing.textContent || '') : '';
  const best = (shown && cached && shown.length >= cached.length) ? shown
    : (cached || shown);
  const fromForm = formLogs || '';
  if (best && (!fromForm || best.length >= fromForm.length)) return best;
  return fromForm;
}

function logPanel(text, liveAttrs, title) {
  const truncated = liveAttrs && /\bdata-log-truncated="1"/.test(liveAttrs);
  const hint = truncated
    ? '<p class="muted log-earlier-hint">仅显示末尾；超大日志可点「加载更早」分页查看全文。</p>'
    : '';
  return `<div class="log-panel">
    <div class="log-head"><h3>${escapeHtml(title || '日志')}</h3>${logTools({ earlier: truncated })}</div>
    ${hint}
    <pre class="log live-log" data-pinned="1"${liveAttrs || ''}>${escapeHtml(text || '')}</pre>
  </div>`;
}

function stageLogAttrs(stage, round, opts) {
  const options = opts || {};
  const key = logCursorKey(stage, round);
  const cursor = state.logCursors[key] || {};
  const once = options.once ? ' data-log-once="1"' : '';
  const truncated = cursor.truncated || cursor.start > 0 ? ' data-log-truncated="1"' : '';
  return ` data-live-log="1" data-live-stage="${escapeHtml(stage)}" data-live-round="${escapeHtml(round)}"${once}${truncated}`;
}

function syncLogEarlierButton(el) {
  const panel = el && el.closest ? el.closest('.log-panel') : null;
  if (!panel || typeof setLogEarlierVisible !== 'function') return;
  const truncated = el.dataset.logTruncated === '1'
    || (Number(el.dataset.logStart || 0) > 0);
  setLogEarlierVisible(panel, truncated);
}

function outputPanel(title, text, kind) {
  const cls = kind === 'command' ? 'command' : 'log';
  return `<div class="log-panel"><div class="log-head"><h3>${escapeHtml(title)}</h3>${logTools()}</div><pre class="${cls}">${escapeHtml(text || '')}</pre></div>`;
}

function metricEmptySvg() {
  return `<svg viewBox="0 0 48 48" aria-hidden="true"><path fill="currentColor" d="M8 36V12h4v20h28v4H8zm6-6 8-10 6 7 8-12 4 3-11 16-6-7-5 6z"/><path fill="currentColor" opacity=".45" d="M10 10 38 38l-2.1 2.1L7.9 12.1z"/></svg>`;
}

function metricLineSvg(series) {
  const steps = (series && series.steps) || [];
  const values = (series && series.values) || [];
  const smoothed = (series && series.smoothed) || null;
  if (!steps.length || !values.length) {
    return `<div class="metric-empty">${metricEmptySvg()}</div>`;
  }
  const w = 240;
  const h = 110;
  const pad = 8;
  const xs = steps;
  const ys = smoothed && smoothed.length === values.length
    ? values.concat(smoothed)
    : values;
  let minX = xs[0];
  let maxX = xs[0];
  let minY = ys[0];
  let maxY = ys[0];
  xs.forEach((x) => { if (x < minX) minX = x; if (x > maxX) maxX = x; });
  ys.forEach((y) => { if (y < minY) minY = y; if (y > maxY) maxY = y; });
  if (minX === maxX) { minX -= 1; maxX += 1; }
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const sx = (x) => pad + ((x - minX) / (maxX - minX)) * (w - pad * 2);
  const sy = (y) => h - pad - ((y - minY) / (maxY - minY)) * (h - pad * 2);
  const pathFor = (vals, color, width) => {
    if (vals.length === 1) {
      const cx = sx(xs[0]);
      const cy = sy(vals[0]);
      return `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="3" fill="${color}"/>`;
    }
    const d = vals.map((v, i) => `${i ? 'L' : 'M'}${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round"/>`;
  };
  const raw = smoothed ? pathFor(values, '#FFE2D9', 1.5) : '';
  const main = pathFor(smoothed || values, '#FF7043', 2);
  return `<svg class="metric-plot" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="${escapeHtml((series && series.name) || '')}">${raw}${main}</svg>`;
}

function metricsCacheKey(stage, round) {
  return `${stage}|${round || ''}`;
}

function metricsPanel(stage, round, live) {
  const keys = STAGE_METRIC_KEYS[stage];
  if (!keys) return '';
  const cached = state.metricsCache[metricsCacheKey(stage, round)];
  const byName = {};
  ((cached && cached.series) || []).forEach((item) => { byName[item.name] = item; });
  const cards = keys.map((name) => {
    const series = byName[name] || { name, steps: [], values: [] };
    return `<div class="metric-card" data-metric="${escapeHtml(name)}">
      <div class="metric-head"><svg class="icon" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M1 13h14v1H1zm2-1 3-4 2.5 3L12 5l2 2.5-.9.7L12 7.2 8.4 12 5.8 9 3 12.5z"/></svg><span>${escapeHtml(name)}</span></div>
      ${metricLineSvg(series)}
    </div>`;
  }).join('');
  const liveAttrs = live
    ? ` data-live-metrics="1" data-live-stage="${escapeHtml(stage)}" data-live-round="${escapeHtml(String(round || ''))}"`
    : ` data-metrics-once="1" data-live-stage="${escapeHtml(stage)}" data-live-round="${escapeHtml(String(round || ''))}"`;
  return `<div class="metric-charts"${liveAttrs}>${cards}</div>`;
}

function paintMetrics(el, payload) {
  if (!el || !el.isConnected || !payload) return;
  const stage = el.dataset.liveStage || payload.stage;
  const round = el.dataset.liveRound || payload.round;
  state.metricsCache[metricsCacheKey(stage, round)] = payload;
  const byName = {};
  (payload.series || []).forEach((item) => { byName[item.name] = item; });
  el.querySelectorAll('.metric-card').forEach((card) => {
    const name = card.dataset.metric;
    const series = byName[name] || { name, steps: [], values: [] };
    const head = card.querySelector('.metric-head');
    const next = metricLineSvg(series);
    const old = card.querySelector('.metric-plot, .metric-empty');
    if (old) old.outerHTML = next;
    else if (head) head.insertAdjacentHTML('afterend', next);
  });
}

async function fetchStageMetrics(stage, round) {
  if (!state.view || !STAGE_METRIC_KEYS[stage]) return null;
  const query = new URLSearchParams();
  if (round) query.set('round', round);
  try {
    return await api(`/api/runs/${state.view.run.id}/stages/${encodeURIComponent(stage)}/metrics?${query}`);
  } catch (_err) {
    return null;
  }
}

function nextRoundAfter(view) {
  const index = view && view.round ? view.round.round_index : null;
  if (index == null) return null;
  return (view.rounds || []).find((round) => round.round_index === index + 1) || null;
}

function renderStage(stage) {
  const view = state.view;
  const form = view.forms[stage];
  const fields = state.schema.fields[stage] || [];
  const meta = state.schema.stages.find((item) => item.id === stage);
  const mutable = view.mutable && form.status !== 'locked' && form.status !== 'running';
  const blockReason = !view.mutable
    ? '历史轮次只能查看'
    : (form.status === 'running' ? '运行中' : (form.status === 'locked' ? '请先完成上一阶段' : ''));
  const nextRound = stage === 'promote' ? nextRoundAfter(view) : null;
  const promoteDone = stage === 'promote' && form.status === 'succeeded';
  const buttonLabel = stage === 'promote'
    ? ({
      ready: '开启下一轮',
      failed: '重试开启下一轮',
      succeeded: '再次开启下一轮',
      running: '开启中',
      locked: '请先完成评测',
    }[form.status] || '开启下一轮')
    : ({
      ready: '运行本阶段',
      failed: '重试本阶段',
      succeeded: '重新运行并作废后续阶段',
      running: '运行中',
      locked: '请先完成上一阶段',
    }[form.status] || '运行本阶段');
  const values = form.values || {};
  const promote = stage === 'promote' ? renderPromote(values) : '';
  const viewTip = tipAttr(view.mutable ? '' : '历史轮次只能查看');
  const validate = stage === 'sft'
    ? `<button type="button" class="ghost" data-action="validate"${viewTip} ${view.mutable ? '' : 'disabled'}>校验格式</button>`
    : '';
  const result = form.result
    ? evalResultMarkup(stage, form.result)
    : ((stage === 'sft_eval' || stage === 'rl_eval') ? evalPickerMarkup() : '');
  const streaming = form.status === 'running' || form.status === 'syncing';
  const finished = form.status === 'succeeded' || form.status === 'failed';
  // Completed stages still fetch /stages/.../log so「加载更早」can reach full swift.log
  // without stuffing multi‑MB text into every /api/runs/{id} poll.
  const wantLog = streaming || finished || Boolean(form.logs);
  const liveAttrs = wantLog
    ? stageLogAttrs(stage, view.round.round_index, { once: !streaming })
    : '';
  const logText = wantLog
    ? liveLogSeed(stage, form.logs || '', view.round.round_index)
    : '';
  const logs = wantLog ? logPanel(logText, liveAttrs) : '';
  const charts = STAGE_METRIC_KEYS[stage]
    ? metricsPanel(stage, view.round.round_index, streaming || form.status === 'succeeded' || form.status === 'failed')
    : '';
  const command = form.command ? outputPanel('命令', form.command, 'command') : '';
  const stop = form.status === 'running'
    ? '<button type="button" class="danger" data-action="stage-stop">停止</button>'
    : '';
  const kicker = form.status === 'running'
    ? `${escapeHtml(meta.label)} · ${runningMarkup(form.started_at)}`
    : `${escapeHtml(meta.label)} · ${escapeHtml(statusText(form.status))}`;
  const gotoNext = promoteDone && nextRound
    ? `<button type="button" class="primary" data-action="goto-round" data-round="${nextRound.round_index}">前往第 ${nextRound.round_index} 轮</button>`
    : '';
  const runClass = promoteDone && nextRound ? 'ghost' : 'primary';
  const promoteHint = promoteDone
    ? `<p class="muted run-reason">${nextRound
      ? `下一轮已开。点「前往第 ${nextRound.round_index} 轮」继续；「再次开启下一轮」会作废本轮后续并重新晋级。`
      : '会作废本轮后续并重新晋级。'}</p>`
    : '';
  const formBody = `
      ${view.mutable ? '' : '<p class="muted">这是历史轮次，只能查看。</p>'}
      ${promote}
      ${state.schema.stage_layouts && state.schema.stage_layouts[stage]
        ? `<div class="infer-form stage-form">${renderBlocks(state.schema.stage_layouts[stage].blocks, values)}</div>`
        : renderGroups(fields, values)}
      ${state.schema.stage_layouts && state.schema.stage_layouts[stage] ? '' : `<details class="yaml">
        <summary>高级参数 YAML</summary>
        <textarea id="extra-yaml">${escapeHtml((values && values.extra_yaml) || '')}</textarea>
      </details>`}
      ${fetchNote()}
      <div id="live-command"></div>
      ${result}
      ${command}
      ${charts}
      ${logs}`;
  const foot = `
      <div class="stage-card-foot">
        ${gotoNext}
        <button type="button" class="${runClass}" data-action="run"${tipAttr(mutable ? '' : blockReason)} ${mutable ? '' : 'disabled'}>${buttonLabel}</button>
        ${stop}
        ${refetchButton()}
        ${validate}
        <button type="button" class="ghost" data-action="preview"${viewTip} ${view.mutable ? '' : 'disabled'}>预览命令</button>
        <button type="button" class="ghost" data-action="preset"${viewTip} ${view.mutable ? '' : 'disabled'}>恢复预设</button>
        ${promoteHint}
        ${blockReason ? `<p class="muted run-reason">${escapeHtml(blockReason)}</p>` : ''}
      </div>`;
  return `
    <article class="stage-card">
      <header class="stage-card-head">
        <div>
          <p class="kicker">${escapeHtml(meta.label)}</p>
          <h2>${escapeHtml(meta.summary)}</h2>
        </div>
        <span class="stage-card-status badge ${escapeHtml(form.status)}">${kicker}</span>
      </header>
      <div class="stage-card-body">${formBody}</div>
      ${foot}
    </article>`;
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

function flowNodeTitle(node) {
  if (node.status === 'locked') return '未解锁';
  if (node.status === 'failed') return '这一步只展示结果';
  if (node.status === 'running') return '运行中';
  return '当前不能点开';
}

function flowNodeSelected(node) {
  const choosingNext = state.stage === 'promote';
  if (node.id === 'sft_next' || node.id === 'rl_next' || node.id === 'next_data' || node.id === 'promote') {
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
    const tip = node.clickable ? '' : tipAttr(flowNodeTitle(node));
    return `<button type="button" class="flow-node ${node.status} ${selected} ${minor}" data-stage="${node.id}"${tip} ${disabled}${busy} style="left:${node.x}px;top:${node.y}px;width:${node.w}px;height:${node.h}px"><b>${escapeHtml(node.label)}</b><small>${caption}</small></button>`;
  }).join('');
  const canvas = flow.canvas || { width: 860, height: 236 };
  return `<div class="flow-scroll"><div class="flow-canvas" style="width:${canvas.width}px;height:${canvas.height}px">
    <svg viewBox="0 0 ${canvas.width} ${canvas.height}" aria-hidden="true">
      <defs>
        <marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#cbd5e1"/></marker>
        <marker id="flow-arrow-live" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#0f766e"/></marker>
      </defs>
      ${paths}
    </svg>
    ${nodes}
  </div></div>`;
}

function flowDecisionBar() {
  const flow = (state.view && state.view.flow) || {};
  const promoteDone = ((state.view.forms || {}).promote || {}).status === 'succeeded';
  const pending = promoteDone ? [] : (flow.actions || []);
  if (!pending.length) return '';
  const actions = pending.map((action) => (
    `<button type="button" class="${action.danger ? 'danger' : 'primary'}" data-decision="${action.id}">${escapeHtml(action.label)}</button>`
  )).join('');
  return `<div class="stage-rail-decisions">
    ${flow.prompt ? `<p class="muted">${escapeHtml(flow.prompt)}</p>` : ''}
    <div class="row-actions">${actions}</div>
  </div>`;
}

function renderStageRail() {
  const flow = (state.view && state.view.flow) || { nodes: [] };
  const forms = (state.view && state.view.forms) || {};
  // Prefer main (non-minor) flow nodes for the stepper; fall back to schema stages.
  let nodes = (flow.nodes || []).filter((node) => !node.minor);
  if (!nodes.length && state.schema && state.schema.stages) {
    nodes = state.schema.stages.map((item) => ({
      id: item.id,
      label: item.label,
      status: (forms[item.id] || {}).status || 'locked',
      clickable: Boolean(forms[item.id]),
      verdict: '',
    }));
  }
  const items = nodes.map((node, index) => {
    const form = forms[node.id] || {};
    const caption = node.status === 'running'
      ? runningMarkup(form.started_at)
      : escapeHtml(node.verdict || statusText(node.status));
    const disabled = node.clickable ? '' : 'disabled';
    const selected = flowNodeSelected(node) ? 'selected' : '';
    const tip = node.clickable ? '' : tipAttr(flowNodeTitle(node));
    const busy = node.status === 'running' ? ' aria-busy="true"' : '';
    return `<li><button type="button" class="stage-step ${node.status} ${selected}" data-stage="${node.id}"${tip} ${disabled}${busy}>
      <span class="stage-idx">${index + 1}</span>
      <span><b>${escapeHtml(node.label)}</b><small>${caption}</small></span>
    </button></li>`;
  }).join('');
  return `<nav class="stage-rail" aria-label="本轮阶段">
    <p class="stage-rail-title">本轮阶段</p>
    <ol class="stage-rail-list">${items}</ol>
    ${flowDecisionBar()}
  </nav>`;
}

function renderTerminal() {
  const completed = state.view.run.status === 'completed';
  const run = state.view.run || {};
  const round = state.view.round || {};
  const values = {
    model: round.base_model || run.base_model || '',
    model_type: run.model_type || '',
    template: run.template || '',
  };
  const canNext = completed && (state.view.flow.actions || []).some((item) => item.id === 'next');
  return `<article class="stage-card">
    <header class="stage-card-head">
      <div>
        <p class="kicker">终点</p>
        <h2>${completed ? '实验已完成' : '实验已结束'}</h2>
      </div>
    </header>
    <div class="stage-card-body">
      <p>${canNext ? '本轮已标记完成。仍可点「开启下一轮」选出检查点并开下一轮，检查点都还在。' : '本轮到此结束。已有检查点仍留在记录里。'}</p>
      <div class="infer-form">${renderLockedModelRow([], values)}</div>
    </div>
    <div class="stage-card-foot">
      ${canNext ? '<button type="button" class="primary" data-decision="next">开启下一轮</button>' : ''}
      <button type="button" class="${canNext ? 'ghost' : 'primary'}" data-action="open-serve">去部署推理服务</button>
      <button type="button" class="ghost" data-action="open-export">去模型导出</button>
    </div>
  </article>`;
}

function renderFlow() {
  if (!state.view.forms[state.stage] && state.stage !== 'finish') state.stage = 'sft';
  const pane = state.stage === 'finish' ? renderTerminal() : renderStage(state.stage);
  return `<div class="flow-layout">
    ${renderStageRail()}
    <div class="stage-pane">
      ${pane}
      <details class="flow-map">
        <summary>流程全景图（可选）</summary>
        ${renderGraph()}
      </details>
    </div>
  </div>`;
}

function durationCell(step) {
  const text = formatDurationZh(step.started_at, step.ended_at);
  if (!step.started_at || step.ended_at) return escapeHtml(text);
  return `<span data-duration-start="${escapeHtml(step.started_at)}">${escapeHtml(text)}</span>`;
}

function renderAudit() {
  const steps = state.view.steps || [];
  const rows = steps.map((step) => {
    const open = state.expanded === step.id;
    const detail = open ? `<tr class="expand"><td colspan="7">
      <p><b>开始</b> ${escapeHtml(formatShanghai(step.started_at))} · <b>结束</b> ${escapeHtml(formatShanghai(step.ended_at || ''))} · <b>耗时</b> ${escapeHtml(formatDurationZh(step.started_at, step.ended_at))}</p>
      ${outputPanel('参数', JSON.stringify(step.params, null, 2))}
      ${evalResultMarkup(step.stage, step.result, step.round_index)}
      ${(() => {
        const live = step.status === 'running' || step.status === 'syncing';
        const done = step.status === 'succeeded' || step.status === 'failed';
        if (!live && !done && !(step.logs || '')) return '';
        return logPanel(
          liveLogSeed(step.stage, step.logs || '', step.round_index),
          stageLogAttrs(step.stage, step.round_index, { once: !live }));
      })()}
      ${step.command ? outputPanel('命令', step.command, 'command') : ''}
    </td></tr>` : '';
    return `<tr data-step="${step.id}">
      <td>${step.id}</td>
      <td>${step.round_index}</td>
      <td>${escapeHtml(labelOf(step.stage))}</td>
      <td><span class="badge ${step.status}">${escapeHtml(statusText(step.status))}</span></td>
      <td>${escapeHtml(formatShanghai(step.started_at || ''))}</td>
      <td>${escapeHtml(formatShanghai(step.ended_at || ''))}</td>
      <td>${durationCell(step)}</td>
    </tr>${detail}`;
  }).join('');
  return `<section class="panel">
    <h2>实验记录</h2>
    <table>
      <thead><tr><th>ID</th><th>轮</th><th>阶段</th><th>状态</th><th>开始</th><th>结束</th><th>耗时</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="7">还没有步骤。</td></tr>'}</tbody>
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
  if (state.serveExportOverride && state.serveExportOverrideRun === view.run.id) {
    base.model = state.serveExportOverride.model || '';
    base.adapters = '';
  }
  return base;
}

function clearServeExportOverride() {
  state.serveExportOverride = null;
  state.serveExportOverrideRun = null;
}

function exportResultMarkup(job) {
  if (!job || !job.status || job.status === 'idle' || job.status === 'running') return '';
  const ok = job.status === 'succeeded';
  const badge = `<span class="badge ${escapeHtml(job.status)}">${escapeHtml(statusText(job.status))}</span>`;
  const code = job.returncode == null ? '' : ` · exit ${escapeHtml(String(job.returncode))}`;
  const when = job.ended_at
    ? ` · ${escapeHtml(formatShanghai(job.ended_at))}`
    : '';
  const path = job.output_dir
    ? `<p class="export-result-path mono" title="${escapeHtml(job.output_dir)}">${escapeHtml(job.output_dir)}</p>`
    : '<p class="muted">没有记录输出路径</p>';
  const err = !ok && job.error
    ? `<pre class="export-result-error">${escapeHtml(job.error)}</pre>`
    : '';
  const deploy = ok && job.output_dir
    ? '<button type="button" class="primary" data-action="deploy-export">用该模型部署推理</button>'
    : '';
  const artifacts = (job.artifacts || []).filter((row) => row && row.output_dir);
  const list = artifacts.length
    ? `<details class="export-artifacts"><summary>本轮导出记录（${artifacts.length}）</summary><ul>${artifacts.map((row) => {
      const mark = `<span class="badge ${escapeHtml(row.status)}">${escapeHtml(statusText(row.status))}</span>`;
      const stamp = row.ended_at ? escapeHtml(formatShanghai(row.ended_at)) : '';
      const deployOne = row.status === 'succeeded'
        ? `<button type="button" class="ghost" data-action="deploy-export" data-output="${escapeHtml(row.output_dir)}">部署</button>`
        : '';
      return `<li>${mark}<span class="mono" title="${escapeHtml(row.output_dir)}">${escapeHtml(row.output_dir)}</span><span class="muted">${stamp}</span>${deployOne}</li>`;
    }).join('')}</ul></details>`
    : '';
  return `<div class="export-result ${ok ? 'ok' : 'bad'}">
    <div class="export-result-head"><strong>最近一次导出</strong>${badge}<span class="muted">${code}${when}</span></div>
    ${path}
    ${err}
    <div class="row-actions">${deploy}</div>
    ${list}
  </div>`;
}

function inferItemValue(item, values) {
  const raw = values ? values[item.id] : undefined;
  if (raw === false || raw === 0) return raw;
  if (item.locked) return raw == null ? '' : raw;
  if (raw !== undefined && raw !== null && raw !== '') return raw;
  if (item.default !== undefined && item.default !== null) return item.default;
  return '';
}

function inferChoices(item, value) {
  let entries = (item.choices || []).map((choice) => normalizeChoiceEntry(choice).value);
  if (!entries.length && item.choices_from && state.schema.catalog) {
    entries = (state.schema.catalog[item.choices_from] || []).map((choice) => normalizeChoiceEntry(choice).value);
  }
  const tokens = item.multiselect ? String(value ?? '').split(/[\s,]+/).filter(Boolean) : [String(value ?? '')];
  const selected = new Set(tokens);
  selected.forEach((token) => {
    if (token && !entries.includes(token)) entries.unshift(token);
  });
  return { entries, selected };
}

function chatCanSend() {
  const input = document.getElementById('chat-input');
  const text = input ? input.value.trim() : '';
  const files = state.mediaFiles || {};
  return Boolean(text || files.image || files.video || files.audio);
}

function syncSendButton() {
  const button = document.querySelector('[data-action="chat"]');
  if (!button) return;
  const ready = chatCanSend();
  button.disabled = !ready;
  button.classList.toggle('primary', ready);
  button.classList.toggle('ghost', !ready);
}

function runningTaskText(service) {
  const cmd = String(service.command || '').replace(/\s+/g, ' ').trim();
  const where = service.port ? `${service.host || '127.0.0.1'}:${service.port}` : '';
  const bits = ['运行中', service.served_model_name || '', where, cmd];
  return bits.filter(Boolean).join(' · ');
}

function inferAction(id) {
  if (state.tab === 'export') {
    return {
      export: 'export-run',
      export_preview: 'export-preview',
      refresh_tasks: 'export-refresh',
      kill_task: 'export-stop',
    }[id] || id;
  }
  return {
    reset: 'reset-infer',
    load_checkpoint: 'serve',
    refresh_tasks: 'serve-refresh',
    kill_task: 'serve-stop',
    clear_history: 'clear-chat',
    submit: 'chat',
  }[id] || id;
}

function inferLabelParts(item) {
  const raw = String(item.label || '');
  const match = raw.match(/^(.*?)\(--([A-Za-z0-9_]+)\)\s*$/);
  if (!match) return { name: raw, flag: '' };
  return { name: match[1].trim(), flag: `--${match[2]}` };
}

function inferCaption(item) {
  const parts = inferLabelParts(item);
  const hint = item.info || '';
  const flag = parts.flag
    ? `<span class="flag" title="${escapeHtml(parts.flag)}">${escapeHtml(parts.flag)}</span>`
    : '<span class="flag"></span>';
  return `<span class="lab" title="${escapeHtml(parts.name)}">${escapeHtml(parts.name)}</span>${flag}<small title="${escapeHtml(hint)}">${escapeHtml(hint)}</small>`;
}

function inferPathControl(item, value) {
  const text = value == null ? '' : String(value);
  const read = item.locked ? 'readonly' : '';
  return `<textarea class="path" data-field="${item.id}" rows="2" title="${escapeHtml(text)}" ${read}>${escapeHtml(text)}</textarea>`;
}

function chipField(box) {
  return box ? box.querySelector('input[type="hidden"][data-field]') : null;
}

function renderChipBox(item, value) {
  const entries = (item.choices || []).map(normalizeChoiceEntry);
  const labels = Object.fromEntries(entries.map((entry) => [entry.value, entry.label]));
  const tokens = datasetTokens(value);
  const chips = tokens.map((token) => {
    const label = labels[token] || token;
    return `<span class="chip"><span>${escapeHtml(label)}</span><button type="button" data-chip-remove="${escapeHtml(token)}" aria-label="移除">×</button></span>`;
  }).join('');
  const clearHidden = tokens.length ? '' : 'hidden';
  const from = item.choices_from ? ` data-choices-from="${escapeHtml(item.choices_from)}"` : '';
  const inline = !item.choices_from && entries.length
    ? ` data-choice-list="${escapeHtml(JSON.stringify(entries))}"` : '';
  const custom = item.allow_custom ? '' : ' data-allow-custom="0"';
  const read = item.locked ? 'readonly' : '';
  const placeholder = item.allow_custom ? '选择或输入' : '选择';
  return `<div class="combo chip-combo"><div class="combo-anchor chip-box">${chips}<input class="chip-query" data-combo="1" data-chip="1"${from}${inline}${custom} value="" autocomplete="off" placeholder="${placeholder}" ${read}/><button type="button" class="chip-clear" data-action="chip-clear" aria-label="清空" ${clearHidden}>×</button><button type="button" class="chip-caret" data-action="chip-menu" aria-label="展开">▾</button><input type="hidden" data-field="${escapeHtml(item.id)}" value="${escapeHtml(tokens.join(' '))}" /><div class="combo-list" hidden></div></div></div>`;
}

function evalPathMatches(path, item) {
  const text = String(path || '').replace(/\\/g, '/');
  return text.endsWith(`/${item.id}.jsonl`) || text === item.path || text.endsWith(`/${item.path}`);
}

function bundledEvalLocation(item, current) {
  const text = String(current || '').replace(/\\/g, '/');
  const marker = '/rsi/examples/';
  const at = text.lastIndexOf(marker);
  if (at >= 0) return text.slice(0, at) + marker + `${item.id}.jsonl`;
  return item.path;
}

function renderEvalDatasetField(item, value, values) {
  const items = datasetChoices('eval');
  const custom = values && values.custom_eval_config ? String(values.custom_eval_config) : '';
  const publicTokens = datasetTokens(value).filter((token) => token && token !== 'itops-heldout');
  let selected = '__path__';
  if (publicTokens.length) selected = '__public__';
  else {
    const matched = items.find((entry) => evalPathMatches(custom, entry));
    if (matched) selected = matched.id;
    else if (!custom) selected = (items[0] && items[0].id) || '__path__';
    else if (String(custom).replace(/\\/g, '/').includes('/uploads/')) selected = '__upload__';
    else selected = '__path__';
  }
  const bundledOptions = items.map((entry) => {
    const mark = entry.id === selected ? 'selected' : '';
    return `<option value="${escapeHtml(entry.id)}" ${mark}>${escapeHtml(entry.label)}</option>`;
  }).join('');
  const publicMark = selected === '__public__' ? 'selected' : '';
  const uploadMark = selected === '__upload__' ? 'selected' : '';
  const pathMark = selected === '__path__' ? 'selected' : '';
  const options = `${bundledOptions}<option value="__upload__" ${uploadMark}>从本机上传</option><option value="__path__" ${pathMark}>服务器上的文件</option><option value="__public__" ${publicMark}>公开评测集（ceval、gsm8k 等）</option>`;
  const chipValue = selected === '__public__' ? publicTokens.join(' ') : 'itops-heldout';
  const publicStyle = selected === '__public__' ? '' : ' style="display:none"';
  const uploadStyle = selected === '__upload__' ? '' : ' style="display:none"';
  const shown = Object.assign({}, item, { info: '直接选留出题，或从电脑上传一份 jsonl。' });
  return `<div class="field" data-key="${item.id}">${inferCaption(shown)}<div class="control"><select data-eval-pick>${options}</select><div data-eval-upload${uploadStyle}><input id="eval-file-input" type="file" accept=".jsonl,.json,.txt,.zip,.tar,.tar.gz,.tgz" /></div><div data-eval-public${publicStyle}>${renderChipBox(item, chipValue)}</div></div><div class="extra"><small data-eval-note></small></div></div>`;
}

function syncEvalPick(fromUser) {
  const pick = document.querySelector('[data-eval-pick]');
  if (!pick) return;
  const mode = pick.value;
  const items = datasetChoices('eval');
  const bundled = items.find((item) => item.id === mode);
  const publicBox = document.querySelector('[data-eval-public]');
  const uploadBox = document.querySelector('[data-eval-upload]');
  const pathField = document.querySelector('#main [data-key="custom_eval_config"]');
  const config = document.querySelector('#main [data-field="custom_eval_config"]');
  const box = document.querySelector('[data-key="eval_dataset"] .chip-box');
  const hidden = chipField(box);
  if (publicBox) publicBox.style.display = mode === '__public__' ? '' : 'none';
  if (uploadBox) uploadBox.style.display = mode === '__upload__' ? '' : 'none';
  if (pathField) pathField.classList.toggle('eval-collapsed', mode !== '__path__' && mode !== '__upload__');
  if (bundled) {
    if (hidden) hidden.value = 'itops-heldout';
    if (config) config.value = choiceFile(bundled, config.value);
    if (fromUser) {
      document.querySelectorAll('#main [data-field="eval_limit"]').forEach((el) => {
        el.value = String(bundled.rows);
      });
    }
  } else if (mode === '__public__') {
    if (config) config.value = '';
    if (hidden && hidden.value.trim() === 'itops-heldout') {
      hidden.value = '';
      paintDatasetChips(box);
    }
  } else if (hidden) {
    hidden.value = 'itops-heldout';
  }
  const note = document.querySelector('[data-eval-note]');
  if (!note) return;
  if (bundled) note.textContent = `共 ${bundled.rows} 条留出题。切换这一项时，取样数会改成 ${bundled.rows}。`;
  else if (mode === '__upload__') note.textContent = '从电脑选择 jsonl。上传后出现服务器路径，取样数改成文件条数。';
  else if (mode === '__public__') note.textContent = '在下面选择公开基准。';
  else note.textContent = '填写这份 jsonl 在服务器上的路径。';
}

function bundledTrainLocation(item, current) {
  const text = String(current || '').replace(/\\/g, '/');
  const marker = '/rsi/examples/';
  const at = text.lastIndexOf(marker);
  if (at >= 0) return text.slice(0, at) + marker + `${item.id}.jsonl`;
  return `rsi/examples/${item.id}.jsonl`;
}

function rlDatasetMode(value) {
  const text = String(value || '').trim();
  const items = datasetChoices('train');
  const matched = items.find((entry) => evalPathMatches(text, {
    id: entry.id,
    path: entry.path || `rsi/examples/${entry.id}.jsonl`,
  }));
  if (matched) return matched.id;
  if (!text) return '__public__';
  const normalized = text.replace(/\\/g, '/');
  if (normalized.includes('/uploads/')) return '__upload__';
  if (normalized.startsWith('/') || normalized.includes('.jsonl')) return '__path__';
  return '__public__';
}

function renderRlDatasetField(item, value) {
  const items = datasetChoices('train');
  const text = value == null ? '' : String(value);
  const selected = rlDatasetMode(text);
  const bundledOptions = items.map((entry) => {
    const mark = entry.id === selected ? 'selected' : '';
    return `<option value="${escapeHtml(entry.id)}" ${mark}>${escapeHtml(entry.label)}</option>`;
  }).join('');
  const uploadMark = selected === '__upload__' ? 'selected' : '';
  const pathMark = selected === '__path__' ? 'selected' : '';
  const publicMark = selected === '__public__' ? 'selected' : '';
  const options = `${bundledOptions}<option value="__upload__" ${uploadMark}>从本机上传</option><option value="__path__" ${pathMark}>服务器上的文件</option><option value="__public__" ${publicMark}>公开数据集</option>`;
  const publicStyle = selected === '__public__' ? '' : ' style="display:none"';
  const uploadStyle = selected === '__upload__' ? '' : ' style="display:none"';
  const pathStyle = selected === '__path__' || selected === '__upload__' ? '' : ' style="display:none"';
  const readOnly = selected === '__upload__' ? 'readonly' : '';
  const shown = Object.assign({}, item, { info: '默认沿用本轮 SFT 训练集。要换，在这里选。' });
  return `<div class="field" data-key="${item.id}">${inferCaption(shown)}<div class="control"><select data-rl-dataset-pick>${options}</select><div data-rl-dataset-path${pathStyle}><input type="text" data-rl-path value="${escapeHtml(text)}" placeholder="服务器上的 jsonl 路径" ${readOnly}/></div><div data-rl-dataset-upload${uploadStyle}><input id="rl-file-input" type="file" accept=".jsonl,.json,.txt,.zip,.tar,.tar.gz,.tgz" /></div><div data-rl-dataset-public${publicStyle}>${renderChipBox(item, selected === '__public__' ? text : text)}</div></div><div class="extra"><small data-rl-dataset-note></small></div></div>`;
}

function syncRlDataset(fromUser) {
  const pick = document.querySelector('[data-rl-dataset-pick]');
  if (!pick) return;
  const mode = pick.value;
  const items = datasetChoices('train');
  const bundled = items.find((item) => item.id === mode);
  const publicBox = document.querySelector('[data-rl-dataset-public]');
  const uploadBox = document.querySelector('[data-rl-dataset-upload]');
  const pathBox = document.querySelector('[data-rl-dataset-path]');
  const pathInput = document.querySelector('[data-rl-path]');
  const box = document.querySelector('[data-key="dataset"] .chip-box');
  const hidden = chipField(box);
  if (publicBox) publicBox.style.display = mode === '__public__' ? '' : 'none';
  if (uploadBox) uploadBox.style.display = mode === '__upload__' ? '' : 'none';
  if (pathBox) pathBox.style.display = mode === '__path__' || mode === '__upload__' ? '' : 'none';
  if (pathInput) pathInput.readOnly = mode === '__upload__';
  if (bundled && hidden) {
    if (fromUser || !evalPathMatches(hidden.value, { id: bundled.id, path: `rsi/examples/${bundled.id}.jsonl` })) {
      hidden.value = choiceFile(bundled, hidden.value);
    }
  } else if (mode === '__public__' && hidden && /[/\\]/.test(hidden.value)) {
    hidden.value = '';
    paintDatasetChips(box);
  } else if ((mode === '__path__' || mode === '__upload__') && hidden && pathInput && fromUser) {
    hidden.value = pathInput.value;
  }
  if (pathInput && hidden && mode !== '__public__' && hidden.value && pathInput.value !== hidden.value && !fromUser) {
    pathInput.value = hidden.value;
  }
  const note = document.querySelector('[data-rl-dataset-note]');
  if (!note) return;
  if (bundled) note.textContent = 'RL 使用这份训练集，和 SFT 相同即可。';
  else if (mode === '__upload__') note.textContent = '从电脑选择 jsonl。上传成功后才会换成你的文件。';
  else if (mode === '__path__') note.textContent = '填写训练机上的 jsonl 路径。';
  else note.textContent = '从下面选择公开数据集。';
}

function renderInferControl(item, values, running, fill) {
  const value = inferItemValue(item, values);
  const locked = item.locked ? 'disabled' : '';
  const lockTitle = item.locked ? ' title="由本轮实验决定，不能在这里修改"' : '';
  const info = item.info ? `<small>${escapeHtml(item.info)}</small>` : '';
  if (item.widget === 'Button') {
    const action = inferAction(item.id);
    let disabled = '';
    let why = '';
    if (item.id === 'load_checkpoint' && !(values.model || values.adapters)) {
      disabled = 'disabled';
      why = '还没有可部署的模型';
    }
    if (item.id === 'export' && !(values.model || values.adapters)) {
      disabled = 'disabled';
      why = '还没有可导出的模型';
    }
    if (item.id === 'export' && running) {
      disabled = 'disabled';
      why = '导出进行中';
    }
    if (item.id === 'kill_task' && !running) {
      disabled = 'disabled';
      why = state.tab === 'export' ? '当前没有导出任务' : '当前没有运行中的推理';
    }
    if (item.id === 'submit' && !chatCanSend()) {
      disabled = 'disabled';
      why = '请先输入内容';
    }
    let klass = item.primary ? 'primary' : (item.id === 'kill_task' ? 'danger' : 'ghost');
    if (item.id === 'submit') klass = chatCanSend() ? 'primary' : 'ghost';
    const tip = why ? ` title="${escapeHtml(why)}"` : '';
    return `<div class="infer-action"><button type="button" class="${klass}" data-action="${action}" ${disabled}${tip}>${escapeHtml(item.label)}</button></div>`;
  }
  if (item.id === 'log') {
    const logText = state.tab === 'export' ? (state.exportLog || '') : (state.inferLog || '');
    return `<div class="log-panel infer-log" data-key="log"><div class="log-head"><div class="log-title"><span>${escapeHtml(item.label)}</span>${info}</div>${logTools()}</div><textarea class="log" readonly rows="18">${escapeHtml(logText)}</textarea></div>`;
  }
  let captionItem = item;
  if (state.stage === 'rl' && item.id === 'external_plugins') {
    captionItem = Object.assign({}, item, { info: '内置奖励脚本。验收保持这项，清空后关键词奖励不会生效。' });
  } else if (state.stage === 'rl' && item.id === 'reward_funcs') {
    captionItem = Object.assign({}, item, { info: 'itops_checklist 按题目关键词打分，format 检查格式。验收保持这两项。' });
  }
  const caption = inferCaption(captionItem);
  if (item.widget === 'Checkbox') {
    const checked = value === true || value === 'true' ? 'checked' : '';
    return `<div class="field check" data-key="${item.id}"${lockTitle}>${caption}<div class="control"><label class="check"><input type="checkbox" data-field="${item.id}" ${checked} ${locked}/></label></div><div class="extra"></div></div>`;
  }
  if (item.widget === 'Slider') {
    const min = item.minimum == null ? 0 : item.minimum;
    const max = item.maximum == null ? 1 : item.maximum;
    const step = item.step == null ? 'any' : item.step;
    // Empty value on <input type="range"> becomes the browser midpoint (e.g. 129 / 10).
    // Prefer Gradio default, else the slider minimum (ms-swift-safe for batch / NEFTune).
    let shown = value;
    if (shown === '' || shown === undefined || shown === null) {
      shown = item.default != null ? item.default : min;
    }
    return `<div class="field infer-slider" data-key="${item.id}">${caption}<div class="control slider-head"><input type="number" class="slider-num" data-slider-num="${item.id}" min="${min}" max="${max}" step="${step}" value="${escapeHtml(shown)}" /><button type="button" class="ghost icon" data-action="slider-reset" data-reset="${item.id}" title="恢复初始值">↺</button></div><div class="extra"><input type="range" data-field="${item.id}" min="${min}" max="${max}" step="${step}" value="${escapeHtml(shown)}" /><div class="slider-ends"><span>${escapeHtml(min)}</span><span>${escapeHtml(max)}</span></div></div></div>`;
  }
  if (item.id === 'running_tasks' && state.tab === 'export') {
    const job = (state.view && state.view.export_job) || {};
    const active = job.status === 'running';
    let empty = '没有导出任务';
    if (job.status === 'succeeded') empty = `上次导出已完成 · ${job.output_dir || ''}`.trim();
    else if (job.status === 'failed') empty = `上次导出失败 · ${job.output_dir || ''}`.trim();
    const label = active ? `导出进行中 · ${job.output_dir || ''}`.trim() : '';
    const options = active
      ? `<option value="export" selected>${escapeHtml(label)}</option>`
      : `<option value="">${escapeHtml(empty)}</option>`;
    return `<div class="field" data-key="running_tasks">${caption}<div class="control"><select>${options}</select></div><div class="extra"></div></div>`;
  }
  if (item.id === 'running_tasks') {
    const services = ((state.view && state.view.services) || []).filter((service) => service.status === 'running');
    const current = services[services.length - 1];
    const options = services.map((service) => {
      const mark = service === current ? 'selected' : '';
      return `<option value="${escapeHtml(String(service.id))}" ${mark}>${escapeHtml(runningTaskText(service))}</option>`;
    }).join('');
    const empty = '<option value="">没有部署</option>';
    return `<div class="field" data-key="running_tasks">${caption}<div class="control"><select>${options || empty}</select></div><div class="extra"></div></div>`;
  }
  if (item.id === 'model' || item.id === 'adapters') {
    const fillAction = state.tab === 'export' ? 'fill-export' : 'fill-serve';
    const fillText = state.tab === 'export' ? '填入本轮权重' : '填入本轮晋级 checkpoint';
    const extra = item.id === 'adapters' && fill && fill.checkpoint
      ? `<button type="button" class="ghost fill-serve" data-action="${fillAction}">${fillText}</button>`
      : '';
    return `<div class="field" data-key="${item.id}"${lockTitle}>${caption}<div class="control">${inferPathControl(item, value)}</div><div class="extra">${extra}</div></div>`;
  }
  if (item.id === 'eval_dataset' && (state.stage === 'sft_eval' || state.stage === 'rl_eval')) {
    return renderEvalDatasetField(item, value, values);
  }
  if (item.id === 'dataset' && state.stage === 'rl') {
    return renderRlDatasetField(item, value);
  }
  if (item.widget === 'Dropdown' && item.multiselect) {
    return `<div class="field" data-key="${item.id}"${lockTitle}>${caption}<div class="control">${renderChipBox(item, value)}</div><div class="extra"></div></div>`;
  }
  if (item.widget === 'Dropdown') {
    const choice = inferChoices(item, value);
    if (item.locked) {
      const current = String(value ?? '');
      choice.entries = current ? [current] : [];
      choice.selected = new Set(current ? [current] : []);
    } else if (item.default == null || item.default === '') {
      // Gradio value=None / RSI blank default: keep an empty first option so the
      // browser does not submit the first catalog id (e.g. quant_method → awq,
      // or ref_model → 01ai/Yi-1.5-34B).
      if (!choice.entries.includes('')) {
        choice.entries = ['', ...choice.entries];
      }
      const current = value === false || value === 0 ? String(value) : String(value ?? '').trim();
      if (!current) {
        choice.selected = new Set(['']);
      }
    }
    const options = choice.entries.map((entry) => {
      const mark = choice.selected.has(entry) ? 'selected' : '';
      return `<option value="${escapeHtml(entry)}" ${mark}>${escapeHtml(entry || '（空）')}</option>`;
    }).join('');
    const multi = item.multiselect ? 'multiple data-multi size="4"' : '';
    return `<div class="field" data-key="${item.id}"${lockTitle}>${caption}<div class="control"><select data-field="${item.id}" ${multi} ${locked}>${options}</select></div><div class="extra"></div></div>`;
  }
  const long = Number(item.lines) >= 4;
  const control = long
    ? `<textarea class="tall" data-field="${item.id}" rows="${item.lines}" ${item.locked ? 'readonly' : ''}>${escapeHtml(value)}</textarea>`
    : `<input type="text" data-field="${item.id}" value="${escapeHtml(value)}" title="${escapeHtml(value == null ? '' : String(value))}" ${item.locked ? 'readonly' : ''}/>`;
  return `<div class="field" data-key="${item.id}"${lockTitle}>${caption}<div class="control">${control}</div><div class="extra"></div></div>`;
}

function renderBlocks(blocks, values) {
  return (blocks || []).map((block) => renderBlock(block, values)).join('');
}

function renderBlock(block, values) {
  if (!block) return '';
  if (block.kind === 'row') return renderInferRow(block.widgets || [], values, false, null);
  if (block.kind === 'accordion' || block.kind === 'tab') {
    if (block.dataset_panel) {
      const layout = state.schema.stage_layouts && state.schema.stage_layouts[state.stage];
      return renderDatasetSettings((layout && layout.dataset_fields) || [], values);
    }
    const open = block.open ? ' open' : '';
    return `<details class="infer-runtime"${open}><summary>${escapeHtml(block.label || '设置')}</summary><div class="body">${renderBlocks(block.blocks, values)}</div></details>`;
  }
  if (block.kind === 'tabs') {
    let opened = false;
    return (block.tabs || []).map((tab) => {
      const visible = tabVisible(tab, values);
      const open = visible && !opened ? ' open' : '';
      if (visible) opened = true;
      const gate = tab.show_key
        ? ` data-show-key="${escapeHtml(tab.show_key)}" data-show-values="${escapeHtml((tab.show_values || []).join(','))}"`
        : '';
      const hidden = visible ? '' : ' hidden';
      return `<details class="infer-runtime"${open}${gate}${hidden}><summary>${escapeHtml(tab.label || '参数')}</summary><div class="body">${renderBlocks(tab.blocks, values)}</div></details>`;
    }).join('');
  }
  return '';
}

function gpuCountFrom(raw) {
  if (raw == null || raw === '') return 1;
  const items = Array.isArray(raw) ? raw : String(raw).split(/[\s,]+/);
  return items.map((item) => String(item).trim()).filter((item) => item && item !== 'cpu').length;
}

function withGpuGate(html, item, values) {
  if (!html || !item || !item.min_gpus) return html;
  const hidden = gpuCountFrom(values ? values.gpu_id : '') < Number(item.min_gpus) ? ' hidden' : '';
  return html.replace('>', ` data-min-gpus="${Number(item.min_gpus)}"${hidden}>`);
}

function stageFieldValue(key, fallback) {
  const el = document.querySelector(`#main [data-field="${key}"]`);
  if (!el) return fallback == null ? '' : fallback;
  if (el.type === 'checkbox') return el.checked ? 'true' : 'false';
  return el.value;
}

function stageChoice(key, raw) {
  const value = raw == null ? '' : String(raw);
  if (value) return value;
  if (key === 'tuner_type') return 'lora';
  if (key === 'task_type') return 'causal_lm';
  if (key === 'vllm_mode') return 'colocate';
  return '';
}

function tabVisible(tab, values) {
  if (!tab || !tab.show_key) return true;
  const current = stageChoice(tab.show_key, values ? values[tab.show_key] : '');
  return (tab.show_values || []).map(String).includes(current);
}

function syncStageFields() {
  const gpus = gpuCountFrom(stageFieldValue('gpu_id', ''));
  document.querySelectorAll('#main [data-min-gpus]').forEach((el) => {
    el.hidden = gpus < Number(el.dataset.minGpus);
  });
  document.querySelectorAll('#main [data-show-key]').forEach((el) => {
    const allowed = (el.dataset.showValues || '').split(',').filter(Boolean);
    el.hidden = !allowed.includes(stageChoice(el.dataset.showKey, stageFieldValue(el.dataset.showKey, '')));
  });
}

function renderInferRow(row, values, running, fill) {
  if (row.length === 1 && row[0].id === 'log' && !running && !(state.tab === 'export' && state.exportLog)) return '';
  const fields = [];
  const sides = [];
  row.forEach((item) => {
    const control = withGpuGate(renderInferControl(item, values, running, fill), item, values);
    if (item.widget === 'Button') sides.push(control);
    else fields.push(control);
  });
  const grid = fields.length && !row.some((item) => item.id === 'log') ? ' infer-grid' : '';
  const group = fields.length ? `<div class="infer-group" style="--cols:${fields.length}">${fields.join('')}</div>` : '';
  const side = sides.length ? `<div class="infer-sides">${sides.join('')}</div>` : '';
  const onlyButtons = fields.length === 0 ? ' actions' : '';
  return `<div class="infer-row${grid}${onlyButtons}${sides.length ? ' has-side' : ''}">${group}${side}</div>`;
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
      if (running) state.inferRuntimeOpen = true;
      const open = state.inferRuntimeOpen ? ' open' : '';
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

function exportFormValues(view) {
  const drafted = state.exportDraft && state.exportDraftRun === view.run.id;
  const base = drafted ? { ...state.exportDraft } : { ...(view.export_form || {}) };
  const fill = view.export_fill || {};
  ['model', 'model_type', 'template'].forEach((key) => {
    if (Object.prototype.hasOwnProperty.call(fill, key)) base[key] = fill[key] == null ? '' : fill[key];
  });
  return base;
}

function renderExport() {
  const view = state.view;
  const layout = state.schema.llm_export;
  const values = exportFormValues(view);
  const fill = view.export_fill || {};
  const job = view.export_job || {};
  const running = job.status === 'running';
  const finished = job.status === 'succeeded' || job.status === 'failed';
  if (!layout) {
    return '<section class="panel infer-form"><p>模型导出布局还没加载。</p></section>';
  }
  if (running || finished) state.exportRuntimeOpen = true;
  const chunks = (layout.sections || []).map((section) => {
    if (section.kind === 'rows') {
      return (section.rows || []).map((row) => renderInferRow(row, values, running, fill)).join('');
    }
    if (section.kind === 'accordion') {
      const body = (section.rows || []).map((row) => renderInferRow(row, values, running, fill)).join('');
      const open = state.exportRuntimeOpen ? ' open' : '';
      return `<details class="infer-runtime" data-runtime="export"${open}><summary>${escapeHtml(section.label || '运行时')}</summary><div class="body">${body}${exportResultMarkup(job)}</div></details>`;
    }
    return '';
  }).join('');
  const reason = running ? '导出进行中' : (!(values.model || values.adapters) ? '还没有可导出的模型' : '');
  const note = reason ? `<p class="muted run-reason">${escapeHtml(reason)}</p>` : '';
  return `<section class="panel infer-form export-form">${chunks}${note}<div id="export-command"></div></section>`;
}

function render() {
  syncLogScrollLock();
  syncScreenAttr();
  const main = document.getElementById('main');
  if (state.screen !== 'workspace' || !state.view) {
    state.screen = 'list';
    syncScreenAttr();
    main.innerHTML = renderExperimentsPage();
    renderRuns();
    syncLive();
    syncLogScrollLock();
    return;
  }
  const view = state.view;
  const rounds = (view.rounds || []).map((round) => `
    <button type="button" data-round="${round.round_index}" class="${round.round_index === view.round.round_index ? 'active' : ''}">第 ${round.round_index} 轮${round.promoted_kind ? ' · ' + round.promoted_kind : ''}</button>`).join('');
  const body = state.tab === 'audit' ? renderAudit() : state.tab === 'serve' ? renderServe() : state.tab === 'export' ? renderExport() : renderFlow();
  const baseHtml = baseCaption(view.run.base_model, view.round.base_model);
  main.innerHTML = `
    <div class="ws">
      <header class="ws-head">
        <div class="ws-title">
          <button type="button" class="ghost compact back-runs" data-action="back-runs">← 实验列表</button>
          <p class="kicker">第 ${view.round.round_index} 轮</p>
          <h1>${escapeHtml(view.run.name)}</h1>
          <p class="scenario">${escapeHtml(view.run.scenario)}</p>
        </div>
        <div class="ws-side">
          <div class="rounds" aria-label="轮次">${rounds}</div>
          <div class="ws-base muted">${baseHtml}</div>
        </div>
      </header>
      <div class="tabs" role="tablist" aria-label="工作区">
        <button type="button" data-tab="flow" class="${state.tab === 'flow' ? 'active' : ''}">流程</button>
        <button type="button" data-tab="audit" class="${state.tab === 'audit' ? 'active' : ''}">实验记录</button>
        <button type="button" data-tab="serve" class="${state.tab === 'serve' ? 'active' : ''}">推理服务</button>
        <button type="button" data-tab="export" class="${state.tab === 'export' ? 'active' : ''}">模型导出</button>
      </div>
      <div class="ws-body">${body}</div>
    </div>`;
  syncSource();
  syncEvalPick(false);
  syncRlDataset(false);
  syncStageFields();
  syncLive();
  syncLogScrollLock();
  syncSendButton();
  if (inferDeployRunning()) ensureInferLogPoll();
  else stopInferLogPoll();
  if (exportJobRunning()) ensureExportLogPoll();
  else stopExportLogPoll();
  if (state.tab === 'export' && !state.exportLog && state.view.export_job && state.view.export_job.status && state.view.export_job.status !== 'idle') {
    refreshExportLog().catch(() => {});
  }
  paintEvalPickerResult();
}

async function runCurrent() {
  const stage = state.stage;
  if (stage === 'sft_eval' || stage === 'rl_eval') {
    const pick = document.querySelector('[data-eval-pick]');
    const config = document.querySelector('#main [data-field="custom_eval_config"]');
    const path = config ? String(config.value || '') : '';
    if (pick && pick.value === '__upload__' && !path.replace(/\\/g, '/').includes('/uploads/')) {
      toast('请先从电脑选择评测文件');
      return;
    }
  }
  if (stage === 'rl') {
    const pick = document.querySelector('[data-rl-dataset-pick]');
    const hidden = chipField(document.querySelector('[data-key="dataset"] .chip-box'));
    const path = hidden ? String(hidden.value || '') : '';
    if (pick && pick.value === '__upload__' && !path.replace(/\\/g, '/').includes('/uploads/')) {
      toast('请先从电脑选择训练集');
      return;
    }
  }
  const params = collectParams();
  setBusy(true);
  try {
    const payload = await api(`/api/runs/${state.view.run.id}/stages/${stage}/run`, {
      method: 'POST',
      body: JSON.stringify({ params }),
    });
    // New stage run rewrites swift.log — drop the follow cursor so the panel tails afresh.
    delete state.logCursors[logCursorKey(stage, payload.view.round.round_index)];
    state.view = payload.view;
    state.round = payload.view.round.round_index;
    if (payload.status === 'succeeded' && payload.view.mutable) {
      state.stage = payload.view.run.current_stage;
    }
    let outcome = payload.status === 'syncing' ? '模型同步中' : payload.status === 'running' ? '已在后台启动' : '已完成';
    if (stage === 'promote' && payload.status === 'succeeded') {
      const opened = payload.view.round && payload.view.round.round_index;
      outcome = opened ? `已开启第 ${opened} 轮` : outcome;
    }
    toast(`${labelOf(stage)} ${outcome}`);
    render();
    touchRunList();
  } finally {
    setBusy(false);
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
  document.querySelectorAll('[data-slider-num]').forEach((num) => {
    const range = document.querySelector(`[data-field="${num.dataset.sliderNum}"]`);
    if (range) num.value = range.value;
  });
  document.querySelectorAll('.chip-box').forEach((box) => paintDatasetChips(box));
  syncSource();
  flushDraft();
  toast('已恢复本阶段预设');
}

async function uploadRlFile(file) {
  const body = new FormData();
  body.append('file', file);
  const response = await fetch('/api/datasets/upload', { method: 'POST', body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '上传失败');
  const hidden = chipField(document.querySelector('[data-key="dataset"] .chip-box'));
  const pathInput = document.querySelector('[data-rl-path]');
  if (hidden) hidden.value = data.path || '';
  if (pathInput) pathInput.value = data.path || '';
  scheduleDraftSave();
  const ok = Boolean(data.validation && data.validation.ok);
  toast(ok ? '训练集已上传' : '上传文件格式未通过');
}

async function uploadEvalFile(file) {
  const body = new FormData();
  body.append('file', file);
  const response = await fetch('/api/datasets/upload', { method: 'POST', body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '上传失败');
  const config = document.querySelector('#main [data-field="custom_eval_config"]');
  if (config) config.value = data.path || '';
  const box = document.querySelector('[data-key="eval_dataset"] .chip-box');
  const hidden = chipField(box);
  if (hidden) hidden.value = 'itops-heldout';
  const rows = data.validation && Number(data.validation.num_rows);
  if (rows) {
    document.querySelectorAll('#main [data-field="eval_limit"]').forEach((el) => {
      el.value = String(rows);
    });
  }
  scheduleDraftSave();
  const ok = Boolean(data.validation && data.validation.ok);
  toast(ok ? '评测文件已上传' : '上传文件格式未通过');
}

async function uploadFile(file) {
  const body = new FormData();
  body.append('file', file);
  const response = await fetch('/api/datasets/upload', { method: 'POST', body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '上传失败');
  const dataset = document.querySelector('[data-dataset-mode="upload"] [data-field="dataset"]');
  if (dataset) dataset.value = data.path;
  const source = document.querySelector('[data-field="source"]');
  if (source) source.value = 'upload';
  syncSource();
  const slot = document.getElementById('live-command');
  if (slot) slot.innerHTML = outputPanel('上传校验', JSON.stringify(data.validation, null, 2));
  scheduleDraftSave();
  toast(data.validation.ok ? '上传通过校验' : '上传文件格式未通过');
}

function stopInferLogPoll() {
  if (!state.inferLogTimer) return;
  clearInterval(state.inferLogTimer);
  state.inferLogTimer = null;
}

function inferDeployRunning() {
  const view = state.view;
  if (!view || state.tab !== 'serve') return false;
  if (view.service && view.service.status === 'running') return true;
  return (view.services || []).some((service) => service.status === 'running');
}

async function refreshInferLog() {
  if (!inferDeployRunning()) return;
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
  if (state.inferLogTimer || !inferDeployRunning()) return;
  refreshInferLog().catch(() => {});
  state.inferLogTimer = setInterval(() => {
    refreshInferLog().catch(() => {});
  }, 1000);
}

function stopExportLogPoll() {
  if (!state.exportLogTimer) return;
  clearInterval(state.exportLogTimer);
  state.exportLogTimer = null;
}

function exportJobRunning() {
  const view = state.view;
  if (!view || state.tab !== 'export') return false;
  return Boolean(view.export_job && view.export_job.status === 'running');
}

async function refreshExportLog() {
  if (!state.view || state.tab !== 'export') return;
  const payload = await api(`/api/runs/${state.view.run.id}/export/log`);
  const was = state.view.export_job && state.view.export_job.status;
  state.exportLog = payload.log || '';
  if (!state.view.export_job) state.view.export_job = {};
  const job = state.view.export_job;
  job.status = payload.status || job.status;
  if (payload.output_dir != null) job.output_dir = payload.output_dir;
  if (payload.returncode !== undefined) job.returncode = payload.returncode;
  if (payload.ended_at !== undefined) job.ended_at = payload.ended_at;
  if (payload.error !== undefined) job.error = payload.error;
  if (payload.artifacts) job.artifacts = payload.artifacts;
  if (payload.command != null) job.command = payload.command;
  const box = document.querySelector('[data-key="log"] textarea');
  if (box && box.value !== state.exportLog) {
    const follow = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    box.value = state.exportLog;
    if (follow) box.scrollTop = box.scrollHeight;
  }
  if (was === 'running' && payload.status && payload.status !== 'running') {
    stopExportLogPoll();
    await flushDraft();
    try {
      state.view = await api(`/api/runs/${state.view.run.id}`);
    } catch (_err) {
      /* keep patched export_job from the log payload */
    }
    render();
  }
}

function ensureExportLogPoll() {
  if (state.exportLogTimer || !exportJobRunning()) return;
  refreshExportLog().catch(() => {});
  state.exportLogTimer = setInterval(() => {
    refreshExportLog().catch(() => {});
  }, 1000);
}

async function previewExport() {
  const params = collectParams();
  const payload = await api(`/api/runs/${state.view.run.id}/export/preview`, {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  const slot = document.getElementById('export-command');
  if (slot) slot.innerHTML = outputPanel('命令预览', payload.command || '没有可预览的命令。', 'command');
}

async function startExport() {
  const params = collectParams();
  const payload = await api(`/api/runs/${state.view.run.id}/export/run`, {
    method: 'POST',
    body: JSON.stringify({ params }),
  });
  state.view = payload.view;
  state.exportDraft = null;
  state.exportDraftRun = null;
  state.exportLog = payload.log || '';
  state.exportRuntimeOpen = true;
  toast(payload.status === 'running' ? '导出已开始' : '导出已完成');
  render();
  touchRunList();
}

async function stopStage() {
  const stage = state.stage;
  const payload = await api(`/api/runs/${state.view.run.id}/stages/${stage}/stop`, { method: 'POST', body: '{}' });
  state.view = payload.view;
  toast('本阶段已停止');
  render();
  touchRunList();
}

async function stopExport() {
  const payload = await api(`/api/runs/${state.view.run.id}/export/stop`, { method: 'POST', body: '{}' });
  state.view = payload.view;
  state.exportRuntimeOpen = true;
  toast('导出已停止');
  render();
  touchRunList();
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
  state.inferRuntimeOpen = true;
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
  captureDraft();
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

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  const input = event.target;
  if (!input.dataset || input.dataset.chip !== '1') return;
  if (input.dataset.allowCustom === '0') return;
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  const box = input.closest('.chip-box');
  const hidden = chipField(box);
  const tokens = datasetTokens(hidden.value);
  if (!tokens.includes(text)) tokens.push(text);
  hidden.value = tokens.join(' ');
  input.value = '';
  paintDatasetChips(box);
  showCombo(input, true);
  scheduleDraftSave();
  syncStageFields();
});

document.addEventListener('click', async (event) => {
  const navRuns = event.target.closest('#nav-runs, [data-nav="runs"]');
  if (navRuns) {
    event.preventDefault();
    showRunsList();
    return;
  }
  const backRuns = event.target.closest('[data-action="back-runs"]');
  if (backRuns) {
    showRunsList();
    return;
  }
  const chip = event.target.closest('[data-chip-remove]');
  if (chip) {
    event.preventDefault();
    const box = chip.closest('.chip-box');
    const hidden = chipField(box);
    const drop = chip.dataset.chipRemove;
    hidden.value = datasetTokens(hidden.value).filter((item) => item !== drop).join(' ');
    paintDatasetChips(box);
    scheduleDraftSave();
    syncStageFields();
    return;
  }
  const clearAll = event.target.closest('[data-action="chip-clear"], [data-action="dataset-clear"]');
  if (clearAll) {
    event.preventDefault();
    const box = clearAll.closest('.chip-box');
    const hidden = chipField(box);
    hidden.value = '';
    const query = box.querySelector('.chip-query');
    if (query) query.value = '';
    paintDatasetChips(box);
    scheduleDraftSave();
    syncStageFields();
    return;
  }
  const caret = event.target.closest('[data-action="chip-menu"], [data-action="dataset-menu"]');
  if (caret) {
    event.preventDefault();
    const input = caret.closest('.chip-box').querySelector('.chip-query');
    if (input) {
      input.focus();
      showCombo(input, true);
    }
    return;
  }
  const target = event.target.closest('button');
  if (!target || target.disabled || state.busy) return;
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
    if (target.dataset.openServe) {
      await openRun(target.dataset.openServe, null, false);
      state.tab = 'serve';
      render();
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
      await flushDraft();
      state.tab = target.dataset.tab;
      render();
      return;
    }
    if (target.dataset.decision) {
      await flushDraft();
      const payload = await api(`/api/runs/${state.view.run.id}/decide`, {
        method: 'POST',
        body: JSON.stringify({ action: target.dataset.decision }),
      });
      state.view = payload.view;
      state.stage = payload.stage || state.stage;
      state.tab = 'flow';
      toast(payload.view.flow && payload.view.flow.prompt ? payload.view.flow.prompt : '已记录');
      render();
      touchRunList();
      return;
    }
    if (target.dataset.stage) {
      if (target.disabled) return;
      await flushDraft();
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
          touchRunList();
          return;
        }
        stage = 'finish';
      }
      if (stage === 'next_data' || stage === 'sft_next' || stage === 'rl_next' || stage === 'promote') {
        // 「完成」closes the run; 「开启下一轮」reopens it, then shows the promote form.
        const canNext = (state.view.flow.actions || []).some((item) => item.id === 'next');
        if (canNext && (state.view.run.status === 'completed' || state.stage !== 'promote')) {
          const payload = await api(`/api/runs/${state.view.run.id}/decide`, {
            method: 'POST',
            body: JSON.stringify({ action: 'next' }),
          });
          state.view = payload.view;
          state.stage = payload.stage || 'promote';
          state.tab = 'flow';
          toast(payload.view.flow && payload.view.flow.prompt ? payload.view.flow.prompt : '已开启下一轮入口');
          render();
          touchRunList();
          return;
        }
        stage = 'promote';
      }
      state.stage = stage;
      state.tab = 'flow';
      render();
      return;
    }
    if (target.dataset.action === 'stage-stop') {
      const meta = (state.schema.stages || []).find((item) => item.id === state.stage);
      const name = meta ? meta.label : '当前阶段';
      const confirmed = await askConfirm({
        title: '停止本阶段',
        body: `${name} 会停掉。已经写到一半的日志会保留，这一步记为失败，可以再重试。`,
        okLabel: '停止',
        danger: true,
      });
      if (!confirmed) return;
      await stopStage();
      return;
    }
    if (target.dataset.action === 'goto-round') {
      const round = Number(target.dataset.round);
      if (!round || !state.view) return;
      await openRun(state.view.run.id, round, false);
      toast(`已切换到第 ${round} 轮`);
      return;
    }
    if (target.dataset.action === 'run') {
      const form = (state.view.forms || {})[state.stage] || {};
      if (form.status === 'succeeded') {
        const isPromote = state.stage === 'promote';
        const confirmed = await askConfirm({
          title: isPromote ? '再次开启下一轮' : '重新运行并作废后续阶段',
          body: isPromote
            ? '会作废本轮后续并重新晋级；若下一轮已存在，也会清空并重开。'
            : '这一阶段会重跑。本轮后面已经完成的阶段和晋级结果会作废，需要重新再跑。',
          okLabel: isPromote ? '再次开启' : '重新运行',
          danger: true,
        });
        if (!confirmed) return;
      }
      setBusy(true);
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
    if (target.dataset.action === 'eval-picker-open') {
      openEvalCompareDialog();
      return;
    }
    if (target.dataset.action === 'eval-picker-close') {
      closeEvalCompareDialog();
      return;
    }
    if (target.dataset.action === 'eval-picker-refresh') {
      await loadEvalReports(true);
      return;
    }
    if (target.dataset.action === 'eval-picker-compare') {
      runEvalPickerCompare();
      return;
    }
    if (target.dataset.action === 'refetch') {
      await flushDraft();
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
      touchRunList();
      return;
    }
    if (target.dataset.action === 'open-serve') {
      await flushDraft();
      clearServeExportOverride();
      state.tab = 'serve';
      render();
      return;
    }
    if (target.dataset.action === 'open-export') {
      await flushDraft();
      state.tab = 'export';
      render();
      return;
    }
    if (target.dataset.action === 'deploy-export') {
      const job = (state.view && state.view.export_job) || {};
      const output = target.dataset.output || job.output_dir || '';
      if (!output) {
        toast('还没有可部署的导出路径');
        return;
      }
      await flushDraft();
      const current = state.tab === 'serve' ? collectParams() : { ...(state.view.serve_form || {}) };
      state.serveExportOverride = { model: output, adapters: '' };
      state.serveExportOverrideRun = state.view.run.id;
      state.serveDraft = {
        ...current,
        model: output,
        adapters: '',
        model_type: current.model_type || state.view.run.model_type || '',
        template: current.template || state.view.run.template || '',
      };
      state.serveDraftRun = state.view.run.id;
      await saveDraftSnapshot('serve', state.serveDraft);
      state.tab = 'serve';
      state.inferRuntimeOpen = true;
      render();
      toast('已填入导出模型，可直接部署推理');
      return;
    }
    if (target.dataset.action === 'fill-export') {
      const fill = (state.view && state.view.export_fill) || {};
      const current = collectParams();
      state.exportDraft = {
        ...current,
        model: fill.model || '',
        model_type: fill.model_type || '',
        template: fill.template || '',
        adapters: fill.adapters || '',
        output_dir: fill.output_dir || current.output_dir || '',
        merge_lora: Boolean(fill.merge_lora),
      };
      state.exportDraftRun = state.view.run.id;
      await saveDraftSnapshot('export', state.exportDraft);
      render();
      toast('已填入本轮权重');
      return;
    }
    if (target.dataset.action === 'fill-serve') {
      clearServeExportOverride();
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
      await saveDraftSnapshot('serve', state.serveDraft);
      render();
      toast('已填入本轮 checkpoint');
      return;
    }
    if (target.dataset.action === 'reset-infer') {
      clearServeExportOverride();
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
      await saveDraftSnapshot('serve', state.serveDraft);
      render();
      toast('已恢复初始值');
      return;
    }
    if (target.dataset.action === 'slider-reset') {
      const key = target.dataset.reset;
      const stageLayout = state.schema.stage_layouts && state.schema.stage_layouts[state.stage];
      const item = state.tab === 'serve'
        ? (state.schema.llm_infer && state.schema.llm_infer.by_id[key])
        : state.tab === 'export'
          ? (state.schema.llm_export && state.schema.llm_export.by_id[key])
          : ((stageLayout && stageLayout.by_id && stageLayout.by_id[key])
            || (state.schema.llm_infer && state.schema.llm_infer.by_id[key]));
      let fallback = item && item.default;
      if (fallback == null && state.schema && state.schema.fields) {
        const field = (state.schema.fields[state.stage] || []).find((row) => row.key === key);
        if (field && field.default != null) fallback = field.default;
      }
      const input = document.querySelector(`[data-field="${key}"]`);
      const num = document.querySelector(`[data-slider-num="${key}"]`);
      if (input && fallback != null) {
        input.value = fallback;
        if (num) num.value = fallback;
        rememberServeDraft({ target: input });
        scheduleDraftSave();
      }
      return;
    }
    if (target.dataset.mediaTab) {
      captureDraft();
      state.mediaTab = target.dataset.mediaTab;
      render();
      return;
    }
    if (target.dataset.action === 'clear-chat') {
      captureDraft();
      state.chat = [];
      state.mediaFiles = { image: null, video: null, audio: null };
      render();
      return;
    }
    if (target.dataset.action === 'serve-refresh') {
      await flushDraft();
      const round = state.view.round.round_index;
      state.view = await api(`/api/runs/${state.view.run.id}?round=${round}`);
      render();
      toast('已找回部署');
      return;
    }
    if (target.dataset.action === 'serve') {
      setBusy(true);
      await startServe();
      return;
    }
    if (target.dataset.action === 'serve-stop') {
      const confirmed = await askConfirm({
        title: '停止推理',
        body: '当前推理服务会停掉，已打开的对话不再继续。',
        okLabel: '停止',
        danger: true,
      });
      if (!confirmed) return;
      await stopServe();
      return;
    }
    if (target.dataset.action === 'export-preview') {
      await previewExport();
      return;
    }
    if (target.dataset.action === 'export-run') {
      setBusy(true);
      await startExport();
      return;
    }
    if (target.dataset.action === 'export-stop') {
      const confirmed = await askConfirm({
        title: '停止导出',
        body: '正在进行的导出会停掉。已经写到一半的文件会留在导出目录里。',
        okLabel: '停止',
        danger: true,
      });
      if (!confirmed) return;
      await stopExport();
      return;
    }
    if (target.dataset.action === 'export-refresh') {
      await flushDraft();
      const round = state.view.round.round_index;
      state.view = await api(`/api/runs/${state.view.run.id}?round=${round}`);
      state.exportRuntimeOpen = true;
      await refreshExportLog().catch(() => {});
      render();
      toast('已找回导出任务');
      return;
    }
    if (target.dataset.action === 'chat') {
      setBusy(true);
      await sendChat();
    }
  } catch (err) {
    toast(err.message);
  } finally {
    setBusy(false);
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
  if (node.dataset.runtime === 'export') state.exportRuntimeOpen = Boolean(node.open);
  else state.inferRuntimeOpen = Boolean(node.open);
}, true);

const DRAFT_STAGES = ['sft', 'sft_eval', 'rl', 'rl_eval', 'promote'];

function currentDraftStage() {
  if (!state.view) return null;
  if (state.tab === 'serve') return 'serve';
  if (state.tab === 'export') return 'export';
  if (state.tab !== 'flow') return null;
  return DRAFT_STAGES.includes(state.stage) ? state.stage : null;
}

function captureDraft() {
  const stage = currentDraftStage();
  if (!stage || !state.view) return null;
  const params = collectParams();
  if (stage === 'serve') {
    state.serveDraft = params;
    state.serveDraftRun = state.view.run.id;
  } else if (stage === 'export') {
    state.exportDraft = params;
    state.exportDraftRun = state.view.run.id;
  }
  return {
    stage,
    params,
    runId: state.view.run.id,
    round: state.view.round.round_index,
  };
}

function scheduleDraftSave() {
  captureDraft();
  if (state.draftTimer) clearTimeout(state.draftTimer);
  state.draftTimer = setTimeout(() => {
    state.draftTimer = null;
    flushDraft();
  }, 400);
}

async function saveDraftSnapshot(stage, params) {
  if (!state.view || !stage) return;
  if (state.draftTimer) {
    clearTimeout(state.draftTimer);
    state.draftTimer = null;
  }
  const runId = state.view.run.id;
  const round = state.view.round.round_index;
  try {
    await api(`/api/runs/${runId}/drafts`, {
      method: 'POST',
      body: JSON.stringify({ stage, round, params }),
    });
  } catch (_err) {
    /* 草稿没写上时，页面上的值还在。 */
  }
}

async function flushDraft(keepalive) {
  if (state.draftTimer) {
    clearTimeout(state.draftTimer);
    state.draftTimer = null;
  }
  const snap = captureDraft();
  if (!snap) return;
  const url = `/api/runs/${snap.runId}/drafts`;
  const body = JSON.stringify({ stage: snap.stage, round: snap.round, params: snap.params });
  if (keepalive) {
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      keepalive: true,
    }).catch(() => {});
    return;
  }
  await saveDraftSnapshot(snap.stage, snap.params);
}

function draftEvent(target) {
  if (!target || !state.view) return false;
  if (target.id === 'chat-input') return false;
  if (target.id === 'extra-yaml') return true;
  if (!target.dataset || !target.dataset.field) return false;
  return Boolean(target.closest('#main'));
}

function rememberServeDraft(event) {
  if ((state.tab !== 'serve' && state.tab !== 'export') || !state.view) return;
  const field = event.target && event.target.dataset && event.target.dataset.field;
  if (!field || !event.target.closest('#main')) return;
  if (state.tab === 'export') {
    state.exportDraft = collectParams();
    state.exportDraftRun = state.view.run.id;
    return;
  }
  state.serveDraft = collectParams();
  state.serveDraftRun = state.view.run.id;
}

document.addEventListener('input', (event) => {
  const target = event.target;
  if (target && target.dataset && Object.prototype.hasOwnProperty.call(target.dataset, 'runFilter')) {
    state.runFilter = target.value || '';
    renderRuns();
    return;
  }
  if (target && target.type === 'range') {
    const num = document.querySelector(`[data-slider-num="${target.dataset.field}"]`);
    if (num) num.value = target.value;
  }
  if (target && target.dataset && target.dataset.sliderNum) {
    const range = document.querySelector(`[data-field="${target.dataset.sliderNum}"]`);
    if (range) {
      range.value = target.value;
      rememberServeDraft({ target: range });
      scheduleDraftSave();
      return;
    }
  }
  if (target && target.id === 'chat-input') syncSendButton();
  rememberServeDraft(event);
  if (draftEvent(target)) scheduleDraftSave();
  if (target && target.dataset && Object.prototype.hasOwnProperty.call(target.dataset, 'rlPath')) {
    const hidden = chipField(document.querySelector('[data-key="dataset"] .chip-box'));
    if (hidden) hidden.value = target.value;
    scheduleDraftSave();
  }
  if (target && target.dataset && target.dataset.field) syncStageFields();
});

document.addEventListener('change', async (event) => {
  const media = event.target.dataset && event.target.dataset.media;
  if (media && event.target.files && event.target.files[0]) {
    state.mediaFiles = state.mediaFiles || { image: null, video: null, audio: null };
    state.mediaFiles[media] = event.target.files[0];
    const name = event.target.parentElement && event.target.parentElement.querySelector('.drop-name');
    if (name) name.textContent = event.target.files[0].name;
    syncSendButton();
  }
  if (event.target.dataset && event.target.dataset.action === 'eval-picker-scope') {
    const next = event.target.value === 'all' ? 'all' : 'round';
    if (state.evalPicker.scope !== next) {
      state.evalPicker.scope = next;
      state.evalPicker.tableHtml = '';
      await loadEvalReports(true);
    }
    return;
  }
  if (event.target.dataset && event.target.dataset.evalReportId) {
    const id = event.target.dataset.evalReportId;
    if (!state.evalPicker.selected) state.evalPicker.selected = {};
    if (event.target.checked) state.evalPicker.selected[id] = true;
    else delete state.evalPicker.selected[id];
    return;
  }
  rememberServeDraft(event);
  if (draftEvent(event.target)) scheduleDraftSave();
  if (event.target.dataset && event.target.dataset.field === 'source') syncSource();
  if (event.target.dataset && Object.prototype.hasOwnProperty.call(event.target.dataset, 'evalPick')) {
    syncEvalPick(true);
    scheduleDraftSave();
  }
  if (event.target.dataset && Object.prototype.hasOwnProperty.call(event.target.dataset, 'rlDatasetPick')) {
    syncRlDataset(true);
    scheduleDraftSave();
  }
  if (event.target.dataset && event.target.dataset.field) syncStageFields();
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
  if (event.target.id === 'rl-file-input' && event.target.files[0]) {
    const chosen = event.target.files[0];
    event.target.value = '';
    try {
      await uploadRlFile(chosen);
    } catch (err) {
      toast(err.message);
    }
  }
  if (event.target.id === 'eval-file-input' && event.target.files[0]) {
    const chosen = event.target.files[0];
    event.target.value = '';
    try {
      await uploadEvalFile(chosen);
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
    toast('实验已创建，从 SFT 开始');
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
  if (input && input.dataset.choiceList) {
    try {
      const parsed = JSON.parse(input.dataset.choiceList);
      return Array.isArray(parsed) ? parsed : [];
    } catch (_err) {
      return [];
    }
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

function catalogEntries(source, input) {
  return catalogNames(source, input).map(normalizeChoiceEntry);
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
  const chip = input.dataset.chip === '1';
  const query = (browse && (multi || chip) ? '' : (chip ? input.value.trim() : (multi ? comboFilterText(input) : input.value.trim()))).toLowerCase();
  const all = catalogEntries(input.dataset.choicesFrom, input);
  // Model / type / template catalogs are large; match Gradio and show the full
  // list when browsing. Keep a filter cap so typing stays snappy.
  const source = input.dataset.choicesFrom || '';
  const fullCatalog = source === 'eval_datasets_by_backend'
    || source === 'models' || source === 'model_types' || source === 'templates' || source === 'datasets';
  const filterCap = fullCatalog ? 80 : 40;
  const exact = !multi && !chip && query
    ? all.findIndex((item) => item.value.toLowerCase() === query || item.label.toLowerCase() === query)
    : -1;
  let shown;
  if (browse && exact >= 0) {
    const window = fullCatalog ? 40 : filterCap;
    const start = Math.max(0, exact - 8);
    shown = all.slice(start, start + window);
  } else if (!query) {
    shown = fullCatalog ? all : all.slice(0, filterCap);
  } else {
    shown = all.filter((item) => (
      item.value.toLowerCase().includes(query) || item.label.toLowerCase().includes(query)
    )).slice(0, filterCap);
  }
  if (!shown.length) {
    list.hidden = true;
    list.innerHTML = '';
    return;
  }
  let selected = new Set();
  if (chip) {
    const hidden = input.parentElement.querySelector('input[type="hidden"][data-field]');
    selected = new Set(datasetTokens(hidden && hidden.value));
  } else if (multi) {
    selected = new Set(input.value.trim().split(/\s+/).filter(Boolean));
  }
  list.innerHTML = shown.map((item) => {
    const pressed = selected.has(item.value) ? ' aria-pressed="true"' : '';
    return `<button type="button" data-combo-value="${escapeHtml(item.value)}"${pressed}>${escapeHtml(item.label)}</button>`;
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
  if (input.dataset.chip === '1') {
    const box = input.closest('.chip-box');
    const hidden = chipField(box);
    const tokens = datasetTokens(hidden && hidden.value);
    const value = option.dataset.comboValue;
    hidden.value = (tokens.includes(value) ? tokens.filter((item) => item !== value) : tokens.concat(value)).join(' ');
    input.value = '';
    paintDatasetChips(box);
    showCombo(input, true);
    syncStageFields();
    return;
  }
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
  const needs = document.querySelector(
    '[data-started], [data-duration-start], [data-live-log], [data-live-metrics], [data-metrics-once]')
    || needsFetchPoll();
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
  document.querySelectorAll('[data-duration-start]').forEach((el) => {
    const label = formatDurationZh(el.dataset.durationStart, '');
    if (el.textContent !== label) el.textContent = label;
  });
}

function paintLog(el, text, mode) {
  const next = text || '';
  const pinned = el.dataset.pinned !== '0';
  if (mode === 'append') {
    if (!next) {
      if (pinned) el.scrollTop = el.scrollHeight;
      return;
    }
    el.textContent = (el.textContent || '') + next;
    if (pinned) el.scrollTop = el.scrollHeight;
    return;
  }
  if (mode === 'prepend') {
    if (!next) return;
    const prevTop = el.scrollTop;
    const prevHeight = el.scrollHeight;
    el.textContent = next + (el.textContent || '');
    el.scrollTop = prevTop + (el.scrollHeight - prevHeight);
    return;
  }
  if (el.textContent === next) {
    if (pinned) el.scrollTop = el.scrollHeight;
    return;
  }
  el.textContent = next;
  if (pinned) el.scrollTop = el.scrollHeight;
}

function rememberLogCursor(stage, round, payload, text) {
  const key = logCursorKey(stage, round);
  const prev = state.logCursors[key] || {};
  const start = payload.offset == null ? (prev.start || 0) : payload.offset;
  state.logCursors[key] = {
    next: payload.next_offset,
    start,
    truncated: Boolean(payload.truncated) || start > 0,
    text: text == null ? (prev.text || '') : text,
  };
  return state.logCursors[key];
}

async function fetchStageLog(stage, round, params) {
  const query = new URLSearchParams(params || {});
  if (round) query.set('round', round);
  return api(`/api/runs/${state.view.run.id}/stages/${encodeURIComponent(stage)}/log?${query}`);
}

async function loadEarlierLogs(el) {
  if (!el || !state.view || el.dataset.logLoading === '1') return;
  const stage = el.dataset.liveStage;
  const round = el.dataset.liveRound || '';
  const key = logCursorKey(stage, round);
  const cursor = state.logCursors[key] || {};
  const start = Number(el.dataset.logStart != null ? el.dataset.logStart : (cursor.start || 0));
  if (!(start > 0)) {
    syncLogEarlierButton(el);
    return;
  }
  const page = Math.min(start, 256 * 1024);
  const offset = Math.max(0, start - page);
  el.dataset.logLoading = '1';
  try {
    const payload = await fetchStageLog(stage, round, {
      offset: String(offset),
      limit: String(start - offset),
    });
    paintLog(el, payload.text || '', 'prepend');
    const next = cursor.next != null ? cursor.next : payload.next_offset;
    el.dataset.logStart = String(payload.offset);
    el.dataset.logNext = String(next);
    el.dataset.logTruncated = payload.offset > 0 ? '1' : '0';
    state.logCursors[key] = {
      next,
      start: payload.offset,
      truncated: payload.offset > 0,
      history: true,
      text: el.textContent || '',
    };
    const panel = el.closest('.log-panel');
    if (panel) {
      const hint = panel.querySelector('.log-earlier-hint');
      if (hint && payload.offset <= 0) hint.remove();
    }
    syncLogEarlierButton(el);
  } catch (_err) {
    toast('加载更早日志失败');
  } finally {
    delete el.dataset.logLoading;
  }
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
    await flushDraft();
    // Keep painted live-log text; view.forms.*.logs stays empty while streaming.
    state.view = view;
    const failed = fetchRows().find((row) => row.state === '失败');
    if (failed && failed.message) toast(failed.message);
    else if (fetchRows().some((row) => row.state === '就绪')) toast('模型已在本地');
    render();
    touchRunList();
  }
}

async function pollStageMetrics() {
  const live = [...document.querySelectorAll('[data-live-metrics]')];
  const once = [...document.querySelectorAll('[data-metrics-once]')];
  const targets = live.concat(once);
  if (!targets.length) return;
  const groups = new Map();
  targets.forEach((el) => {
    const key = `${el.dataset.liveStage}|${el.dataset.liveRound || ''}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(el);
  });
  for (const [key, els] of groups) {
    const [stage, round] = key.split('|');
    const payload = await fetchStageMetrics(stage, round);
    if (!payload) continue;
    els.forEach((el) => {
      if (!el.isConnected) return;
      paintMetrics(el, payload);
      if (el.hasAttribute('data-metrics-once')) el.removeAttribute('data-metrics-once');
    });
  }
}

async function pollLive() {
  paintElapsed();
  if (state.liveBusy || !state.view) return;
  const panels = [...document.querySelectorAll('[data-live-log]')];
  const wantMetrics = document.querySelector('[data-live-metrics], [data-metrics-once]');
  if (!panels.length && !wantMetrics) {
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
      const cursor = state.logCursors[key] || {};
      // Fresh tail each poll keeps tqdm CR collapse correct. After「加载更早」we
      // append from next_offset so the prepended history is not wiped.
      let follow = Boolean(cursor.history) && cursor.next != null && cursor.next !== '';
      let payload;
      try {
        payload = await fetchStageLog(stage, round, follow
          ? { offset: String(cursor.next) }
          : {});
      } catch (_err) {
        continue;
      }
      // If the log was truncated/rewritten (re-run), restart from a fresh tail.
      if (follow && payload.size < Number(cursor.next || 0)) {
        delete state.logCursors[key];
        try {
          payload = await fetchStageLog(stage, round, {});
        } catch (_err) {
          continue;
        }
        follow = false;
      }
      let painted = '';
      els.forEach((el) => {
        if (!el.isConnected) return;
        paintLog(el, payload.text || '', follow ? 'append' : 'replace');
        painted = el.textContent || '';
        const remembered = rememberLogCursor(stage, round, follow
          ? {
            offset: cursor.start != null ? cursor.start : payload.offset,
            next_offset: payload.next_offset,
            truncated: cursor.truncated || payload.truncated,
          }
          : payload, painted);
        if (cursor.history) remembered.history = true;
        el.dataset.logStart = String(remembered.start || 0);
        el.dataset.logNext = String(remembered.next);
        el.dataset.logTruncated = remembered.truncated ? '1' : '0';
        syncLogEarlierButton(el);
        if (el.hasAttribute('data-log-once')) el.removeAttribute('data-log-once');
      });
      if (!painted && payload) {
        const remembered = rememberLogCursor(stage, round, follow
          ? {
            offset: cursor.start != null ? cursor.start : payload.offset,
            next_offset: payload.next_offset,
            truncated: cursor.truncated || payload.truncated,
          }
          : payload, payload.text || '');
        if (cursor.history) remembered.history = true;
      }
      const current = ((state.view.forms || {})[stage] || {}).status;
      if ((current === 'running' || current === 'syncing') && (payload.done || payload.status !== current)) {
        await flushDraft();
        const view = await api(`/api/runs/${state.view.run.id}?round=${state.view.round.round_index}`);
        state.view = view;
        state.round = view.round.round_index;
        toast(`${labelOf(stage)} ${statusText(payload.status)}`);
        render();
        touchRunList();
        return;
      }
    }
    await pollStageMetrics();
    await pollFetchProgress();
  } finally {
    state.liveBusy = false;
    // After a one-shot fetch, demote finished panels so the timer can stop while
    // keeping stage/round attrs for「加载更早」.
    document.querySelectorAll('[data-live-log]').forEach((el) => {
      if (el.hasAttribute('data-log-once')) return;
      const stage = el.dataset.liveStage;
      const status = ((state.view.forms || {})[stage] || {}).status;
      if (status !== 'running' && status !== 'syncing') {
        el.removeAttribute('data-live-log');
        el.setAttribute('data-stage-log', '1');
      }
    });
    // One-shot metrics drop their attribute after the first paint; stop the timer
    // when nothing else needs live updates.
    if (!document.querySelector(
      '[data-started], [data-duration-start], [data-live-log], [data-live-metrics], [data-metrics-once]')
      && !needsFetchPoll() && state.liveTimer) {
      clearInterval(state.liveTimer);
      state.liveTimer = null;
    }
  }
}

document.addEventListener('scroll', (event) => {
  const el = event.target;
  if (!el || !el.classList || !el.classList.contains('live-log')) return;
  const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
  el.dataset.pinned = gap < 28 ? '1' : '0';
}, true);

window.addEventListener('pagehide', () => {
  flushDraft(true);
  if (state.runListTimer) {
    clearInterval(state.runListTimer);
    state.runListTimer = null;
  }
});

async function boot() {
  state.schema = await api('/api/schema');
  try {
    state.workbenchFiles = (await api('/api/workbench/files')).files || [];
  } catch (_err) {
    state.workbenchFiles = [];
  }
  fillCreateModel();
  await refreshRuns();
  render();
}

boot().catch((err) => toast(err.message));
