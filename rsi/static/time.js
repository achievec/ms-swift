function parseInstant(value) {
  if (value === null || value === undefined || value === '') return null;
  const text = String(value).trim();
  if (!text) return null;
  const zoned = /(?:Z|[+-]\d{2}:?\d{2})$/.test(text) ? text : `${text.replace(' ', 'T')}Z`;
  const date = new Date(zoned);
  return Number.isNaN(date.getTime()) ? null : date.getTime();
}

function formatDurationZh(startedAt, endedAt, now) {
  const start = parseInstant(startedAt);
  if (start === null) return '—';
  const endedText = endedAt === null || endedAt === undefined ? '' : String(endedAt).trim();
  const end = endedText ? parseInstant(endedText) : (now == null ? Date.now() : now);
  if (end === null) return '—';
  let secs = Math.max(0, Math.floor((end - start) / 1000));
  const hours = Math.floor(secs / 3600);
  secs -= hours * 3600;
  const mins = Math.floor(secs / 60);
  const rem = secs - mins * 60;
  const parts = [];
  if (hours) parts.push(`${hours} 小时`);
  if (mins) parts.push(`${mins} 分`);
  if (rem || !parts.length) parts.push(`${rem} 秒`);
  return parts.join(' ');
}

function formatShanghai(value) {
  if (value === null || value === undefined || value === '') return '';
  const text = String(value).trim();
  if (!text) return '';
  const zoned = /(?:Z|[+-]\d{2}:?\d{2})$/.test(text) ? text : `${text.replace(' ', 'T')}Z`;
  const date = new Date(zoned);
  if (Number.isNaN(date.getTime())) return text;
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  }).formatToParts(date);
  const get = (type) => parts.find((part) => part.type === type).value;
  return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}:${get('second')}`;
}
