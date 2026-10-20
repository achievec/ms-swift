function logTools() {
  return '<span class="log-tools"><button type="button" class="ghost" data-log-copy>复制</button><button type="button" class="ghost" data-log-fullscreen>全屏</button></span>';
}

function logPanelText(panel) {
  const node = panel.querySelector('.result') || panel.querySelector('textarea, pre');
  if (!node) return '';
  return node.tagName === 'TEXTAREA' ? node.value : node.innerText || node.textContent;
}

function syncLogScrollLock() {
  const locked = Boolean(document.querySelector('.log-panel.log-fullscreen'));
  document.body.classList.toggle('log-lock', locked);
  document.documentElement.classList.toggle('log-lock', locked);
}

function setLogFullscreen(panel, on) {
  if (panel && panel.isConnected) {
    panel.classList.toggle('log-fullscreen', Boolean(on));
    const button = panel.querySelector('[data-log-fullscreen]');
    if (button) button.textContent = on ? '退出全屏' : '全屏';
  }
  syncLogScrollLock();
}

function legacyCopy(text) {
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.top = '0';
  area.style.left = '0';
  area.style.opacity = '0';
  document.body.appendChild(area);
  area.focus();
  area.select();
  area.setSelectionRange(0, area.value.length);
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch (_err) {
    ok = false;
  }
  area.remove();
  return ok;
}

function flashCopyButton(button, ok) {
  button.textContent = ok ? '已复制' : '复制失败';
  window.setTimeout(() => {
    if (button.isConnected) button.textContent = '复制';
  }, 1200);
}

document.addEventListener('click', (event) => {
  const copy = event.target.closest('[data-log-copy]');
  if (copy) {
    event.preventDefault();
    const panel = copy.closest('.log-panel');
    if (!panel) return;
    const text = logPanelText(panel);
    const legacy = legacyCopy(text);
    const finish = (ok) => flashCopyButton(copy, ok);
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(() => finish(true)).catch(() => finish(legacy));
    } else {
      finish(legacy);
    }
    return;
  }
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
  syncLogScrollLock();
});

new MutationObserver(() => syncLogScrollLock()).observe(document.documentElement, {
  subtree: true,
  childList: true,
  attributes: true,
  attributeFilter: ['class'],
});
