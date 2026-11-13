'use strict';
// Staff page. Every server value is placed with textContent (no HTML from responses).
const el = id => document.getElementById(id);
const meta = name => document.querySelector(`meta[name="${name}"]`)?.content || '';
const token = meta('csrf-token');
const MAX_QUESTION = 2000;
const MAX_UPLOAD = 2 * 1024 * 1024;
let documentId = null, pages = [], busy = false;

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
function toLogin() {
  // The server answers "/" without a session with the login page; login.js shows the expiry text for #expired.
  if (location.pathname === '/') { location.hash = 'expired'; location.reload(); } else location.replace('/#expired');
}

class MustChange extends Error {}
async function api(path, body, headers = {}) {
  let response;
  try {
    response = await fetch(path, {method: 'POST', headers: {'X-Trid-Token': token, ...headers}, body});
  } catch {
    throw Error('서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.');
  }
  let data = {};
  try { data = await response.json(); } catch { /* keep the generic message below */ }
  if (response.status === 401) { toLogin(); throw Error('세션이 만료됐습니다. 다시 로그인해 주세요.'); }
  if (response.status === 403 && data.must_change) { showPasswordView(); throw new MustChange(data.error); }
  if (!response.ok) throw Error(data.error || '요청에 실패했습니다. 잠시 후 다시 시도해 주세요.');
  return data;
}
const jsonApi = (path, data) => api(path, JSON.stringify(data), {'Content-Type': 'application/json'});

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
  el('current').focus();
}
function setDocument(data) {
  documentId = data ? data.document_id : null;
  pages = data ? data.pages : [];
  el('clear').disabled = !documentId;
  el('ask').disabled = !documentId;
  el('doc-view').hidden = !documentId;
  el('doc-pages').replaceChildren();
  if (!data) return;
  el('doc-summary').textContent = `추출 문서 보기 · ${data.name} (${pages.length}쪽)`;
  pages.forEach((text, i) => el('doc-pages').append(block(`${i + 1}쪽`, text)));
}
function block(label, text) {
  const b = node('div', 'block', text);
  if (label) b.prepend(node('span', 'label', label));
  return b;
}
function note(parent, text, kind = '') { parent.append(node('div', `note${kind ? ' ' + kind : ''}`, text)); }

// ---- answer rendering (case results) -------------------------------------
// Status shown on every evidence answer: never hide that an answer is unverified or that the model failed.
const STATUS = {
  quoted: '원문 일치 · 정답 검증 아님', answer: '원문 위치 일치 · 정답 검증 아님', unverified: '검증 안 됨 · 해설만',
  review: '검증 보류 · 원문 확인 필요', not_found: '근거 못 찾음', clarify: '되묻기',
  model_failure: '모델 응답 실패',
};
const LIMIT_PARTS = [['table_rows', n => `표 행 ${n}행`], ['tables', () => '검토된 표 행 전부'],
                     ['linked_passages', n => `다른 별표 구간 ${n}개`], ['own_passages', n => `관련도 낮은 구간 ${n}개`]];
function contextLimitLine(result) {
  const counts = result.context_limits || {};
  const parts = LIMIT_PARTS.filter(([key]) => Number(counts[key]) > 0).map(([key, label]) => label(counts[key]));
  if (parts.length) return `일부 근거만 반영 (${parts.join(' · ')} 제외)`;
  return result.context_notes?.length ? '일부 근거만 반영 (문맥 한도로 일부 제외)' : null;
}
function highlighted(text, span) {
  const b = node('div', 'block');
  const chars = Array.from(text);  // the server returns code-point offsets
  const [a, z] = span || [0, 0];
  b.append(document.createTextNode(chars.slice(0, a).join('')));
  if (z > a) b.append(node('mark', '', chars.slice(a, z).join('')));
  b.append(document.createTextNode(chars.slice(z).join('')));
  return b;
}
function placeOf(c) { return c.source_label || (c.page ? `${c.page}쪽` : ''); }

function renderAnswer(result) {
  const out = el('result');
  out.replaceChildren();
  const wrap = node('div', 'answer');
  wrap.append(node('div', 'text', result.answer || '(빈 응답)'));
  const tags = node('div', 'tags');
  tags.append(node('span', `tag${result.status === 'model_failure' ? ' bad' : ''}`,
                   STATUS[result.status] || result.status || '근거 검색'));
  wrap.append(tags);
  const limitLine = contextLimitLine(result);
  if (limitLine) note(wrap, limitLine, 'warn');
  const limits = [];
  if (result.source_limits?.length) limits.push(result.source_limits.some(t => t.includes('그림')) ? '원본 그림 제외 추출본' : '원본 추출 한계 있음');
  if (result.table_warning) limits.push('표 행·열 대응 미검증');
  if (limits.length) note(wrap, `자료 한계: ${limits.join(' · ')} — 자세한 내용은 아래에서 확인하세요`, 'warn');
  for (const text of result.commentary_notes || []) if (text.includes('숨겼습니다')) note(wrap, text, 'warn');

  // Confirmed quotes: exact text found in the uploaded PDF, with its page and the highlighted span.
  const quoted = (result.candidates || []).filter(c => {
    const [a, z] = c.prediction?.source_character_span || [0, 0];
    return z > a;
  });
  for (const c of quoted) {
    const [a, z] = c.prediction.source_character_span;
    const where = placeOf(c);
    const quote = block(`확인된 인용${where ? ' · ' + where : ''}`, '');
    quote.append(node('mark', '', Array.from(c.text).slice(a, z).join('')));
    wrap.append(quote);
  }
  if (quoted.length) {
    const context = node('details', 'card');
    context.append(node('summary', '', `인용 앞뒤 원문 보기 (${quoted.length}개)`));
    for (const c of quoted) {
      const where = placeOf(c);
      const b = highlighted(c.text, c.prediction.source_character_span);
      b.prepend(node('span', 'label', `원문 일치 · 정답 검증 아님${where ? ' · ' + where : ''}`));
      context.append(b);
    }
    wrap.append(context);
  }
  if (result.commentary) wrap.append(block('검증되지 않은 해설 — 원문과 대조하세요', result.commentary));
  if (result.table_warning) wrap.append(block('자료 한계 · 표', result.table_warning));

  const info = [];
  for (const text of result.commentary_notes || []) if (!text.includes('숨겼습니다')) info.push(text);
  if (result.scope) info.push(result.scope);
  for (const text of result.context_notes || []) if (!result.scope?.includes(text)) info.push(text);
  for (const text of result.warnings || []) info.push(text);
  if (result.checked_chunks !== undefined) info.push(`${result.checked_chunks}/${result.total_chunks ?? '?'}개 구간 확인`);
  if (result.model) info.push(`${result.model}${result.elapsed_ms ? ` · ${(result.elapsed_ms / 1000).toFixed(1)}초` : ''}`);
  if (info.length) note(wrap, info.join('\n'));
  out.append(wrap);
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
  if (file.size > MAX_UPLOAD) { setMessage('doc-status', 'PDF는 2MB 이하로 넣어 주세요.', true); return; }
  setMessage('doc-status', '문서를 읽는 중입니다.');
  void run('doc-status', async () => {
    const data = await api('/api/load', file, {'Content-Type': 'application/pdf', 'X-File-Name': encodeURIComponent(file.name)});
    setDocument(data);
    el('result').replaceChildren();
    setMessage('doc-status', `${data.name}을(를) 읽었습니다 (${data.pages.length}쪽). 이제 이 문서에서만 찾습니다.`);
    const status = el('doc-status');
    for (const warning of data.warnings || []) note(status, warning, 'warn');
  });
}
function clearDocument() {
  void run('doc-status', async () => {
    await jsonApi('/api/clear', {});
    setDocument(null);
    el('file').value = ''; el('question').value = ''; updateCount();
    el('result').replaceChildren();
    setMessage('ask-status', '');
    setMessage('doc-status', '자료를 지웠습니다.');
  });
}
function ask() {
  const question = el('question').value.trim();
  if (!documentId) { setMessage('ask-status', '문서를 먼저 읽어 주세요.', true); return; }
  if (!question) { setMessage('ask-status', '질문을 입력해 주세요.', true); return; }
  if (Array.from(question).length > MAX_QUESTION) {
    setMessage('ask-status', `질문은 ${MAX_QUESTION.toLocaleString('ko-KR')}자 이하로 입력해 주세요.`, true); return;
  }
  setMessage('ask-status', '답변을 만드는 중입니다.');
  el('result').replaceChildren();
  void run('ask-status', async () => {
    renderAnswer(await jsonApi('/api/ask', {question, document_id: documentId}));
    setMessage('ask-status', '');
  });
}
async function logout() {
  try { await jsonApi('/api/logout', {}); } catch { /* leave anyway */ }
  location.replace('/');
}
function changePassword(event) {
  event.preventDefault();
  void (async () => {
    setMessage('password-message', '');
    try {
      await jsonApi('/api/password', {current: el('current').value, new: el('new-password').value});
      el('current').value = ''; el('new-password').value = '';
      location.replace('/');
    } catch (error) {
      if (!(error instanceof MustChange)) setMessage('password-message', error.message, true);
    }
  })();
}
function updateCount() { el('count').textContent = Array.from(el('question').value).length.toLocaleString('ko-KR'); }

el('who').textContent = meta('username');
el('admin-link').hidden = meta('role') !== 'admin';
el('load').addEventListener('click', load);
el('clear').addEventListener('click', clearDocument);
el('ask').addEventListener('click', ask);
el('logout').addEventListener('click', () => void logout());
el('password-form').addEventListener('submit', changePassword);
el('question').addEventListener('input', updateCount);
if (meta('must-change') === '1') showPasswordView();
