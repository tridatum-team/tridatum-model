'use strict';
// Chat UI over the A-UI-API-CONTRACT-001 API. Every server value is placed with textContent (no HTML from responses).
const el = id => document.getElementById(id);
const token = document.querySelector('meta[name="trid-token"]').content;
const defaultModel = document.querySelector('meta[name="trid-default-model"]')?.content || '';
const MAX_QUESTION = 2000;
let documentId = null, pages = [], documentName = '', busy = false, lastApplied = null;

function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}
function message(role, text, extraClass = '') {
  document.body.classList.remove('empty');
  const wrap = node('div', `msg ${role}${extraClass ? ' ' + extraClass : ''}`);
  const body = node('div', 'text', text);
  body.prepend(node('span', 'sr-only', role === 'user' ? '나: ' : role === 'bot' ? 'trid: ' : '안내: '));
  wrap.append(body);
  el('output').append(wrap);
  requestAnimationFrame(() => wrap.scrollIntoView({block: 'end', behavior: 'smooth'}));
  return wrap;
}
const system = text => message('system', text);
function failure(text) { message('system', text, 'error').setAttribute('role', 'alert'); }
function tag(text, kind = '') { return node('span', `tag${kind ? ' ' + kind : ''}`, text); }
function note(parent, text, kind = '') { parent.append(node('div', `note${kind ? ' ' + kind : ''}`, text)); }
function block(label, text) {
  const b = node('div', 'block', text);
  if (label) b.prepend(node('span', 'label', label));
  return b;
}
function sourceLink(url) {
  const a = node('a', '', url);
  if (/^https?:\/\//.test(url)) { a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; }
  return a;
}

function setBusy(value) {
  busy = value;
  for (const id of ['send', 'attach', 'attach-pdf', 'attach-paste', 'attach-sample', 'new-chat', 'save-text', 'cancel-text',
                    'version', 'back-workspace']) el(id).disabled = value;
  el('command').setAttribute('aria-busy', String(value));
  el('pending').textContent = value ? '답변을 만드는 중' : '';
  if (!value && el('paste-panel').hidden) el('command').focus();
}
async function api(path, body, headers = {}) {
  let response;
  try {
    response = await fetch(path, {method: 'POST', headers: {'X-Trid-Token': token, ...headers}, body});
  } catch {
    throw Error('서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.');
  }
  let data;
  try { data = await response.json(); } catch { throw Error('서버 응답을 읽지 못했습니다. 잠시 후 다시 시도해 주세요.'); }
  if (!response.ok) throw Error(data.error || '요청에 실패했습니다. 잠시 후 다시 시도해 주세요.');
  return data;
}
const jsonApi = (path, data) => api(path, JSON.stringify(data), {'Content-Type': 'application/json'});
async function task(action) {
  if (busy) return false;
  setBusy(true);
  try { await action(); return true; } catch (error) { failure(error.message); return false; } finally { setBusy(false); }
}

function describe(d) {
  return [d.institution, d.title, d.kind, d.date ? `${d.date_type || '기준일'} ${d.date}` : null,
          d.effective_date ? `시행 ${d.effective_date}` : null].filter(Boolean).join(' · ');
}
function setDocument(data) {
  documentId = data ? data.document_id : null;
  pages = data ? data.pages : [];
  documentName = data ? data.name : '';
  el('doc-chip').hidden = !documentId;
  el('doc-name').textContent = documentId ? `${documentName} (${pages.length}쪽)` : '';
}
function loaded(data) {
  setDocument(data);
  const m = system(`${documentName}을(를) 읽었습니다. 이제 이 문서에서만 찾습니다.`);
  for (const warning of data.warnings || []) note(m, warning.replace('추출 문서 보기에서 확인해 주세요.', '‘문서 보기’로 원문을 확인해 주세요.'), 'warn');
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

// Status shown on every evidence answer: never hide that an answer is unverified or that the model failed.
// Colour stays neutral; the wording carries the status. Only a model failure is drawn as an error.
const STATUS = {
  quoted: '원문 일치 · 정답 검증 아님', answer: '원문 위치 일치 · 정답 검증 아님', unverified: '검증 안 됨 · 해설만',
  review: '검증 보류 · 원문 확인 필요', not_found: '근거 못 찾음', clarify: '되묻기',
  unknown_institution: '해당 기관 자료 없음', institution_no_match: '해당 기관 자료에 근거 없음', model_failure: '모델 응답 실패',
};

// Evidence the server left out to fit the model context (UI-SCOPE-004): the answer covers only what went in, so one
// short line sits right under the answer on every such response; the full wording stays in the fold. `context_limits`
// gives the counts; a server without it still sends `context_notes`, and then the line carries no counts.
const LIMIT_PARTS = [['table_rows', n => `표 행 ${n}행`], ['tables', () => '검토된 표 행 전부'],
                     ['linked_passages', n => `다른 별표 구간 ${n}개`], ['own_passages', n => `관련도 낮은 구간 ${n}개`]];
function contextLimitLine(result) {
  const counts = result.context_limits || {};
  const parts = LIMIT_PARTS.filter(([key]) => Number(counts[key]) > 0).map(([key, label]) => label(counts[key]));
  if (parts.length) return `일부 근거만 반영 (${parts.join(' · ')} 제외)`;
  return result.context_notes?.length ? '일부 근거만 반영 (문맥 한도로 일부 제외)' : null;
}

function renderAnswer(result, started) {
  const m = message('bot', result.answer || '(빈 응답)');
  const limitLine = contextLimitLine(result);
  if (limitLine) note(m, limitLine, 'warn limit');
  const tags = node('div', 'tags');
  if (result.intent === 'chat') tags.append(tag('일반 대화 · 문서 근거 없음 · 검증 대상 아님'));
  else tags.append(tag(STATUS[result.status] || result.status || '근거 검색', result.status === 'model_failure' ? 'bad' : ''));
  m.append(tags);
  for (const text of result.commentary_notes || []) if (text.includes('숨겼습니다')) note(m, text, 'warn');
  // Serious source limits get one short line in the answer body; their full wording stays in the fold below.
  const limits = [];
  if (result.source_limits?.length) limits.push(result.source_limits.some(t => t.includes('그림')) ? '원본 그림 제외 추출본' : '원본 추출 한계 있음');
  if (result.table_warning) limits.push('표 행·열 대응 미검증');
  if (limits.length) note(m, `자료 한계: ${limits.join(' · ')} — 자세한 내용은 아래 출처에서 확인하세요`, 'warn');
  // A verified quote is the answer itself: show the exact span; its surrounding text stays in the fold.
  for (const c of result.candidates || []) {
    const [a, z] = c.prediction?.source_character_span || [0, 0];
    if (z <= a) continue;
    const where = c.source_label || (c.page ? `${c.page}쪽` : '');
    const quote = block(`원문 인용${where ? ' · ' + where : ''}`, '');
    quote.append(node('mark', '', Array.from(c.text).slice(a, z).join('')));
    m.append(quote);
  }

  const ws = result.workspace;
  if (result.intent !== 'chat' && ws) lastApplied = ws.applied || null;
  const details = node('details', 'more');
  const parts = [];
  if (ws?.applied) parts.push(ws.applied.institution || '적용 문서');
  if (result.candidates?.length) parts.push(`인용 ${result.candidates.length}`);
  details.append(node('summary', '', parts.length ? `출처 · ${parts.join(' · ')}` : '자세히'));
  if (ws?.applied) {
    const a = ws.applied;
    const b = block(`적용 문서${ws.pinned ? ' · 고정됨' : ''}`, describe(a));
    if (a.currency) b.append(document.createTextNode(`\n${a.currency}`));
    if (a.source) { b.append(document.createTextNode('\n원문 ')); b.append(sourceLink(a.source)); }
    if (a.license) b.append(document.createTextNode(` · ${a.license}`));
    details.append(b);
  }
  for (const c of result.candidates || []) {
    const b = highlighted(c.text, c.prediction?.source_character_span);
    const where = c.source_label || (c.page ? `${c.page}쪽` : '');
    b.prepend(node('span', 'label', `${result.answer_mode === 'case' ? '원문 일치 · 정답 검증 아님' : '근거'}${where ? ' · ' + where : ''}`));
    details.append(b);
  }
  if (result.table_warning) details.append(block('자료 한계 · 표', result.table_warning));
  if (result.commentary) details.append(block('모델 해설 · 검증 안 됨 — 원문과 대조하세요', result.commentary));
  if (ws?.selection === 'answer' && ws.others?.length) {
    details.append(block('검색된 다른 문서 (이 답변에는 쓰지 않음 · 다른 기관 기준이면 질문에 기관명을 넣어 주세요)',
                         ws.others.slice(0, 3).map(d => `- ${describe(d)}`).join('\n')));
  }
  const info = [];
  if (result.search_query) info.push(`검색어: ${result.search_query}`);
  for (const text of result.commentary_notes || []) if (!text.includes('숨겼습니다')) info.push(text);
  if (result.scope) info.push(result.scope);
  for (const text of result.context_notes || []) if (!result.scope?.includes(text)) info.push(text);
  if (result.checked_chunks !== undefined && result.intent !== 'chat') info.push(`${result.checked_chunks}/${result.total_chunks ?? '?'}개 구간 확인`);
  info.push(`${result.model} · ${(result.elapsed_ms / 1000).toFixed(1)}초 (왕복 ${((performance.now() - started) / 1000).toFixed(1)}초)`);
  details.append(node('div', 'note', info.join('\n')));
  details.lastChild.style.whiteSpace = 'pre-wrap';
  m.append(details);
}

async function ask(question) {
  message('user', question);
  await task(async () => {
    const started = performance.now();
    const request = documentId ? {question, document_id: documentId, version: el('version').value}
                               : {question, scope: 'workspace', version: el('version').value};
    renderAnswer(await jsonApi('/api/ask', request), started);
  });
}
function showSource() {
  if (documentId) {
    const m = system(`${documentName} 원문`);
    const details = node('details', 'more');
    details.open = true;
    details.append(node('summary', '', `${pages.length}쪽`));
    pages.forEach((p, i) => details.append(block(`${i + 1}쪽`, p)));
    m.append(details);
  } else if (lastApplied) {
    const m = system(`직전 답변의 적용 문서: ${describe(lastApplied)}`);
    if (lastApplied.source) {
      const n = node('div', 'note');
      n.append(document.createTextNode('원문 '), sourceLink(lastApplied.source));
      m.append(n);
    }
  } else {
    system('먼저 질문하거나 문서를 첨부해 주세요.');
  }
}
async function newChat() {
  await task(async () => {
    await jsonApi('/api/clear', {});
    setDocument(null); lastApplied = null;
    el('file').value = ''; el('document-text').value = ''; el('output').replaceChildren();
    pastePanel(false);
    document.body.classList.add('empty');
  });
}
function pastePanel(show) {
  el('paste-panel').hidden = !show;
  toggleAttachMenu(false);
  (show ? el('document-text') : el('command')).focus();
}
function toggleAttachMenu(show = el('attach-menu').hidden) {
  el('attach-menu').hidden = !show;
  el('attach').setAttribute('aria-expanded', String(show));
}
function openPdf() { toggleAttachMenu(false); el('file').value = ''; el('file').click(); }
async function sample() { toggleAttachMenu(false); await task(async () => loaded(await jsonApi('/api/sample', {}))); }
function backToWorkspace() { setDocument(null); system('공개 자료 검색으로 돌아왔습니다.'); }
function openInfo() { el('info').showModal(); }

const COMMANDS = {
  '/help': openInfo, '/pdf': openPdf, '/paste': () => pastePanel(true), '/source': showSource, '/sample': sample,
  '/workspace': backToWorkspace, '/clear': newChat, '/reset': newChat,
};
function submit() {
  const input = el('command');
  const value = input.value.trim();
  if (busy || !value) return;
  if (value.startsWith('/') && !value.includes('\n')) {
    const command = COMMANDS[value.split(/\s+/)[0]];
    input.value = ''; resize();
    if (command) void command(); else system('알 수 없는 명령어입니다. i 버튼에서 사용법을 확인하세요.');
    return;
  }
  if (Array.from(value).length > MAX_QUESTION) { failure(`질문은 ${MAX_QUESTION}자 이하로 입력해 주세요. 긴 문서는 ＋ → 문서 붙여넣기로 넣어 주세요.`); return; }
  input.value = ''; resize();
  void ask(value);
}
function resize() {
  const input = el('command');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, window.innerHeight * 0.4)}px`;
}
function updateFlag() {
  const label = el('version').selectedOptions[0]?.textContent || '';
  el('model-flag').hidden = !/회귀/.test(label);
  for (const card of document.querySelectorAll('.version-card')) card.hidden = card.dataset.version !== el('version').value;
}

let composing = false;
el('command').addEventListener('compositionstart', () => { composing = true; });
el('command').addEventListener('compositionend', () => { composing = false; });
el('command').addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.shiftKey) return;
  // IME: Enter that confirms a Korean syllable must not send (isComposing / keyCode 229 on some browsers).
  if (event.isComposing || composing || event.keyCode === 229) return;
  event.preventDefault();
  submit();
});
el('command').addEventListener('input', resize);
el('command-form').addEventListener('submit', event => { event.preventDefault(); submit(); });
el('attach').addEventListener('click', () => toggleAttachMenu());
el('attach-pdf').addEventListener('click', openPdf);
el('attach-paste').addEventListener('click', () => pastePanel(true));
el('attach-sample').addEventListener('click', () => void sample());
el('new-chat').addEventListener('click', () => void newChat());
el('doc-source').addEventListener('click', showSource);
el('back-workspace').addEventListener('click', backToWorkspace);
el('open-info').addEventListener('click', openInfo);
el('open-info-inline').addEventListener('click', openInfo);
el('close-info').addEventListener('click', () => el('info').close());
el('info').addEventListener('click', event => { if (event.target === el('info')) el('info').close(); });
el('version').addEventListener('change', () => {
  updateFlag();
  if (!document.body.classList.contains('empty')) system(`모델: ${el('version').selectedOptions[0].textContent}`);
  el('command').focus();
});
el('version-form').addEventListener('submit', event => event.preventDefault());
el('file').addEventListener('change', () => {
  const file = el('file').files[0];
  toggleAttachMenu(false);
  if (!file || busy) return;
  if (file.size > 2 * 1024 * 1024) { failure('PDF는 2MB 이하로 넣어 주세요.'); return; }
  void task(async () => loaded(await api('/api/load', file, {'Content-Type': 'application/pdf', 'X-File-Name': encodeURIComponent(file.name)})));
});
el('save-text').addEventListener('click', () => void task(async () => {
  loaded(await jsonApi('/api/text', {text: el('document-text').value}));
  el('document-text').value = ''; pastePanel(false);
}));
el('cancel-text').addEventListener('click', () => { el('document-text').value = ''; pastePanel(false); });
el('document-text').addEventListener('keydown', event => {
  if (busy || event.isComposing || event.keyCode === 229) return;
  if (event.key === 'Escape') { event.preventDefault(); el('cancel-text').click(); }
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); el('save-text').click(); }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !el('attach-menu').hidden) { toggleAttachMenu(false); el('attach').focus(); }
});

// Default model: keep the server's selected option; otherwise pick the option named like the default engine.
const select = el('version');
if (!select.querySelector('option[selected]') && defaultModel) {
  const match = Array.from(select.options).find(o => o.textContent === defaultModel);
  if (match) select.value = match.value;
}
updateFlag();
resize();
