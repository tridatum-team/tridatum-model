'use strict';
// Admin page. Tables come as server-rendered, autoescaped htmx fragments; everything this script
// writes itself goes in with textContent (no HTML strings).
const el = id => document.getElementById(id);
const token = document.querySelector('meta[name="csrf-token"]')?.content || '';
const OFFLINE = '서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.';
const FAILED = '요청에 실패했습니다. 잠시 후 다시 시도해 주세요.';
const EXPIRED = '세션이 만료됐습니다. 다시 로그인해 주세요.';

function setMessage(id, text, error = false) {
  el(id).textContent = text || '';
  el(id).className = error ? 'msg error' : 'msg';
}
function parse(text) {
  try { const value = JSON.parse(text); return value && typeof value === 'object' ? value : {}; } catch { return {}; }
}
function messageFor(elt) { return elt?.closest('[data-messages]')?.dataset.messages || 'users-message'; }

// ---- usage log filters ---------------------------------------------------
// Dates become Unix seconds in the browser's local time, as in the old page; empty fields are left out.
function dateFilters() {
  const out = {};
  const since = el('f-since').value, until = el('f-until').value;
  if (since) out.since = String(new Date(`${since}T00:00:00`).getTime() / 1000);
  if (until) out.until = String(new Date(`${until}T23:59:59`).getTime() / 1000);
  return out;
}
function formatTime(ts) {
  const d = new Date(ts * 1000);
  const two = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${two(d.getMonth() + 1)}-${two(d.getDate())} ${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
}
// The server renders times in its own zone; show them in the browser's, as the old page did.
function localTimes(root) {
  for (const cell of root.querySelectorAll('td[data-ts]')) {
    const ts = Number(cell.dataset.ts);
    if (Number.isFinite(ts)) cell.textContent = formatTime(ts);
  }
}
// CSV: the form's fields (token included) go in a POST body, never in a URL. It is sent with fetch,
// not as a form navigation: under Referrer-Policy no-referrer a navigation sends "Origin: null",
// which the server rejects.
function filenameOf(response) {
  const match = /filename="([^"]+)"/.exec(response.headers.get('Content-Disposition') || '');
  return match ? match[1] : 'trid-usage.csv';
}
el('csv-form').addEventListener('submit', event => {
  event.preventDefault();
  const form = el('csv-form');
  const values = {since: '', until: '', ...dateFilters(),
                  username: el('f-username').value.trim(), event: el('f-event').value};
  for (const [name, value] of Object.entries(values)) form.elements[name].value = value;
  void (async () => {
    setMessage('events-message', '');
    let response;
    try {
      response = await fetch(form.action, {method: 'POST', body: new URLSearchParams(new FormData(form))});
    } catch { setMessage('events-message', OFFLINE, true); return; }
    if (!response.ok) {
      const data = parse(await response.text().catch(() => ''));
      if (response.status === 401) { setMessage('events-message', EXPIRED, true); location.replace('/#expired'); return; }
      if (response.status === 403 && data.must_change) { location.replace('/'); return; }
      setMessage('events-message', data.error || FAILED, true);
      return;
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = filenameOf(response);
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  })();
});

// ---- htmx requests -------------------------------------------------------
let resetPassword = null;
document.body.addEventListener('htmx:confirm', event => {
  const elt = event.detail.elt;
  const username = elt?.dataset?.reset;
  if (username === undefined) return;
  // Asked here and sent in the form body: a prompt header (HX-Prompt) cannot carry non-Latin-1 text.
  event.preventDefault();
  const password = window.prompt(`${username}의 임시 비밀번호를 입력해 주세요 (10자 이상). 다음 로그인 때 바꾸게 됩니다.`);
  if (!password) return;
  resetPassword = password;
  event.detail.issueRequest(true);
});
document.body.addEventListener('htmx:configRequest', event => {
  const elt = event.detail.elt;
  const params = event.detail.parameters;
  if (elt?.dataset?.reset !== undefined && resetPassword !== null) {
    params.reset_password = resetPassword;
    resetPassword = null;
  }
  if (event.detail.verb === 'get' && event.detail.path === '/ui/admin/events') {
    for (const [name, value] of Object.entries(dateFilters())) params[name] = value;
    if (typeof params.username === 'string') params.username = params.username.trim();
  }
});
document.body.addEventListener('htmx:beforeRequest', event => {
  setMessage(messageFor(event.detail.elt), '');
});
document.body.addEventListener('htmx:afterRequest', event => {
  const elt = event.detail.elt;
  if (event.detail.successful) {
    if (elt?.id === 'create-form') { el('new-username').value = ''; el('new-password').value = ''; }
    return;
  }
  const xhr = event.detail.xhr;
  if (!xhr || !xhr.status) { setMessage(messageFor(elt), OFFLINE, true); return; }
  const data = parse(xhr.responseText);
  if (xhr.status === 401) { setMessage(messageFor(elt), EXPIRED, true); location.replace('/#expired'); return; }
  if (xhr.status === 403 && data.must_change) { location.replace('/'); return; }
  setMessage(messageFor(elt), data.error || FAILED, true);
});
document.body.addEventListener('htmx:afterSwap', event => localTimes(event.detail.target.isConnected ? event.detail.target : document));

el('logout').addEventListener('click', () => void (async () => {
  try {
    await fetch('/api/logout', {method: 'POST', headers: {'X-Trid-Token': token, 'Content-Type': 'application/json'}, body: '{}'});
  } catch { /* leave anyway */ }
  location.replace('/');
})());
localTimes(document);
