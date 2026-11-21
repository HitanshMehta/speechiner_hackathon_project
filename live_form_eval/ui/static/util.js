// Small helpers shared by the page's modules. No library: the page works without a network.

export const $ = (id) => document.getElementById(id);

/** An element: el('li', {class: 'turn', title: '…'}, child, 'some text', …). Strings always become text nodes,
 *  so nothing that comes from a conversation, a form or an extractor is ever read as HTML. */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

/** Seconds as a clock: 75.3 → "1:15", or "1:15.3" with tenths. */
export function clock(seconds, tenths = false) {
  const s = Math.max(0, seconds || 0);
  const whole = Math.floor(s);
  const base = `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
  return tenths ? `${base}.${Math.floor((s - whole) * 10)}` : base;
}

/** A number with a fixed count of decimals; an en dash when there is none. */
export const figure = (value, digits = 3) =>
  value === null || value === undefined || Number.isNaN(value) ? '–' : Number(value).toFixed(digits);

/** The index of the last item whose time is at most t, or -1. `at` reads an item's time; items are in time order. */
export function lastAt(items, t, at) {
  let low = 0, high = items.length - 1, found = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (at(items[middle]) <= t) { found = middle; low = middle + 1; } else { high = middle - 1; }
  }
  return found;
}

/** Does a field show nothing? The evaluation's rule: no value, a blank text, an empty list. */
export const isEmpty = (value) =>
  value === null || value === undefined || (typeof value === 'string' && !value.trim())
  || (Array.isArray(value) && value.length === 0)
  || (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0);

export const sameValue = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** A field's value as one line of text: yes or no, options by commas, the facts or lines of a note in a row. */
export function plain(value) {
  if (value === null || value === undefined) return '';
  if (value === true) return 'yes';
  if (value === false) return 'no';
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
export const pause = (ms) => new Promise((done) => setTimeout(done, ms));

async function answer(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

export const getJSON = (url, signal) => fetch(url, { signal }).then(answer);

/** A request that changes something (starts or stops a run, fetches a recording). The header marks it as the
 *  page's own: the server refuses such a request without it, so no other site can make one. */
export const postJSON = (url, body = {}) => fetch(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Lfe-Page': '1' }, body: JSON.stringify(body),
}).then(answer);

/** Follow a feed of the server (a live run, or the score over a reviewed run): `on(event)` for each event, in
 *  order; `lost()` when the server can't be reached. Returns the connection; close() ends it. */
export function follow(feedId, on, lost) {
  const source = new EventSource(`/api/feed?id=${encodeURIComponent(feedId)}`);
  source.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.event === 'done') source.close();  // the feed is over: don't ask for it again
    on(event);
  };
  source.onerror = () => { if (source.readyState === EventSource.CLOSED && lost) lost(); };
  return source;
}
