'use strict';
// Staff page. Answers, extracted pages and the password form come as server-rendered, autoescaped
// htmx fragments; everything this script writes itself goes in with textContent (no HTML strings).
const el = id => document.getElementById(id);
const meta = name => document.querySelector(`meta[name="${name}"]`)?.content || '';
const token = meta('csrf-token');
const MAX_QUESTION = 2000;
// Upload limit comes from the server setting (meta, digits only); 2 MiB if absent or invalid.
const MAX_UPLOAD = /^\d{1,12}$/.test(meta('trid-max-bytes')) && Number(meta('trid-max-bytes')) > 0 ? Number(meta('trid-max-bytes')) : 2 * 1024 * 1024;
const MAX_UPLOAD_LABEL = MAX_UPLOAD % 1048576 === 0 ? `${MAX_UPLOAD / 1048576}MB` : `${Math.round(MAX_UPLOAD / 1024)}KB`;
const OFFLINE = '서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.';
const FAILED = '요청에 실패했습니다. 잠시 후 다시 시도해 주세요.';
const EXPIRED = '세션이 만료됐습니다. 다시 로그인해 주세요.';
let documentId = null, busy = false;

function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}
function setMessage(id, text, error = false) {
  el(id).textContent = text || '';
  el(id).className = error ? 'msg error' : 'msg';
}
function note(parent, text, kind = '') { parent.append(node('div', `note${kind ? ' ' + kind : ''}`, text)); }
function toLogin() {
  // The server answers "/" without a session with the login page; login.js shows the expiry text for #expired.
  if (location.pathname === '/') { location.hash = 'expired'; location.reload(); } else location.replace('/#expired');
}
function parse(text) {
  try { const value = JSON.parse(text); return value && typeof value === 'object' ? value : {}; } catch { return {}; }
}

class MustChange extends Error {}
// Shared handling of an error status from fetch or htmx: login page, password view, or the message to show.
function failure(status, data) {
  if (status === 401) { toLogin(); return Error(EXPIRED); }
  if (status === 403 && data.must_change) { showPasswordView(); return new MustChange(data.error); }
  return Error(data.error || FAILED);
}
async function api(path, body, headers = {}) {
  let response;
  try {
    response = await fetch(path, {method: 'POST', headers: {'X-Trid-Token': token, ...headers}, body});
  } catch {
    throw Error(OFFLINE);
  }
  let data = {};
  try { data = await response.json(); } catch { /* keep the generic message */ }
  if (!response.ok) throw failure(response.status, data);
  return data;
}

function setBusy(value) {
  busy = value;
  for (const id of ['load', 'file', 'ask', 'question']) el(id).disabled = value;
  el('clear').disabled = value || !documentId;
  if (!value) el('ask').disabled = !documentId;
  el('ask').setAttribute('aria-busy', String(value));
}

// ---- views ---------------------------------------------------------------
function showPasswordView() {
  el('password-view').hidden = false;
  el('work-view').hidden = true;
  el('current')?.focus();
}
function setDocumentId(value) {
  documentId = value;
  el('document_id').value = value || '';
  el('clear').disabled = !documentId;
  el('ask').disabled = !documentId;
}
function resetDocumentView() {
  const view = el('doc-view');
  view.hidden = true;
  view.open = false;
  view.replaceChildren(node('summary', '', '추출 문서 보기'));
  view.firstChild.id = 'doc-summary';
  const pagesBox = node('div');
  pagesBox.id = 'doc-pages';
  view.append(pagesBox);
}
async function showDocument(name) {
  // The extracted pages come from the server as an escaped fragment; the local file name stays here.
  resetDocumentView();
  await htmx.ajax('GET', '/ui/document', '#doc-view');
  const pagesBox = el('doc-pages'), summary = el('doc-summary');
  if (!pagesBox || !summary || !pagesBox.children.length) { resetDocumentView(); return false; }
  summary.textContent = `추출 문서 보기 · ${name} (${pagesBox.children.length}쪽)`;
  el('doc-view').hidden = false;
  // Wide screens show the document beside the answer, so open the extracted text for checking quotes.
  el('doc-view').open = window.matchMedia('(min-width: 1024px)').matches;
  return true;
}

// ---- actions -------------------------------------------------------------
async function run(statusId, action) {
  if (busy) return;
  setBusy(true);
  try { await action(); } catch (error) {
    if (!(error instanceof MustChange)) setMessage(statusId, error.message, true);
  } finally { setBusy(false); }
}
function load() {
  const file = el('file').files[0];
  if (!file) { setMessage('doc-status', 'PDF 파일을 먼저 선택해 주세요.', true); return; }
  if (file.size > MAX_UPLOAD) { setMessage('doc-status', `PDF는 ${MAX_UPLOAD_LABEL} 이하로 넣어 주세요.`, true); return; }
  setMessage('doc-status', '문서를 읽는 중입니다.');
  void run('doc-status', async () => {
    // The file name never leaves the browser (R14): only the bytes are sent; the page shows the local name.
    const data = await api('/api/load', file, {'Content-Type': 'application/pdf'});
    const name = file.name;
    setDocumentId(data.document_id);
    el('result').replaceChildren();
    setMessage('ask-status', '');
    if (!await showDocument(name)) {
      setDocumentId(null);
      // An error status was already shown by the htmx handler; otherwise the document is gone.
      if (el('doc-status').className !== 'msg error') setMessage('doc-status', FAILED, true);
      return;
    }
    setMessage('doc-status', `${name}을(를) 읽었습니다 (${data.pages.length}쪽). 이제 이 문서에서만 찾습니다.`);
    const status = el('doc-status');
    for (const warning of data.warnings || []) note(status, warning, 'warn');
  });
}
function documentCleared() {
  setDocumentId(null);
  resetDocumentView();
  el('file').value = ''; el('question').value = ''; updateCount();
  el('result').replaceChildren();
  setMessage('ask-status', '');
  setMessage('doc-status', '자료를 지웠습니다.');
}
function askProblem() {
  const question = el('question').value.trim();
  if (!documentId) return '문서를 먼저 읽어 주세요.';
  if (!question) return '질문을 입력해 주세요.';
  if (Array.from(question).length > MAX_QUESTION) return `질문은 ${MAX_QUESTION.toLocaleString('ko-KR')}자 이하로 입력해 주세요.`;
  return null;
}
async function logout() {
  try { await api('/api/logout', '{}', {'Content-Type': 'application/json'}); } catch { /* leave anyway */ }
  location.replace('/');
}
function updateCount() { el('count').textContent = Array.from(el('question').value).length.toLocaleString('ko-KR'); }

// ---- htmx requests (/ui/ask, /ui/clear, /ui/password) ---------------------
const STATUS_FOR = {ask: 'ask-status', clear: 'doc-status', 'doc-view': 'doc-status', 'password-form': 'password-message'};
const sourceOf = event => event.detail.elt?.id;

document.body.addEventListener('htmx:confirm', event => {
  const id = sourceOf(event);
  if (id === 'ask' || id === 'clear') {
    if (busy) { event.preventDefault(); return; }
    const problem = id === 'ask' ? askProblem() : null;
    if (problem) { event.preventDefault(); setMessage('ask-status', problem, true); return; }
  }
  if (id === 'ask') { setMessage('ask-status', '답변을 만드는 중입니다.'); el('result').replaceChildren(); }
  if (id === 'password-form') setMessage('password-message', '');
});
// Values are collected before this event, so disabling the inputs here does not empty the request.
document.body.addEventListener('htmx:beforeSend', event => {
  const id = sourceOf(event);
  if (id === 'ask' || id === 'clear') setBusy(true);
});
document.body.addEventListener('htmx:afterRequest', event => {
  const id = sourceOf(event);
  if (!(id in STATUS_FOR)) return;
  if (id === 'ask' || id === 'clear') setBusy(false);
  const xhr = event.detail.xhr;
  if (event.detail.successful) {
    if (id === 'ask') setMessage('ask-status', '');
    if (id === 'clear') documentCleared();
    return;
  }
  // Network failure (no status) or an error status with the API's JSON body.
  const error = xhr && xhr.status ? failure(xhr.status, parse(xhr.responseText)) : Error(OFFLINE);
  if (!(error instanceof MustChange)) setMessage(STATUS_FOR[id], error.message, true);
});

el('who').textContent = meta('username');
el('admin-link').hidden = meta('role') !== 'admin';
el('load').addEventListener('click', load);
el('logout').addEventListener('click', () => void logout());
el('question').addEventListener('input', updateCount);
if (meta('must-change') === '1') showPasswordView();
