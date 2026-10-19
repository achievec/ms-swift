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
