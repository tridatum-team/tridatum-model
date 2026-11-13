'use strict';
// Admin page: accounts and the usage log. Every server value is placed with textContent.
const el = id => document.getElementById(id);
const token = document.querySelector('meta[name="csrf-token"]').content;
const PAGE_SIZE = 200;  // the server's page size; a full page means there may be a next one
const EVENT_LABELS = {
  login_ok: '로그인', login_failed: '로그인 실패', login_locked: '로그인 잠김', logout: '로그아웃',
  session_expired: '세션 만료', password_changed: '비밀번호 변경', pdf_upload: 'PDF 올리기', ask: '질문',
  clear: '자료 지우기', admin_create_user: '계정 생성', admin_update_user: '계정 변경',
  admin_reset_password: '비밀번호 초기화', admin_export: '기록 내보내기', retention_purge: '보관 기간 정리',
};
const ROLE_LABELS = {user: '직원', admin: '관리자'};
let offset = 0, lastCount = 0;

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
  location.replace('/#expired');
}

async function post(path, data) {
  let response;
  try {
    response = await fetch(path, {
      method: 'POST', headers: {'X-Trid-Token': token, 'Content-Type': 'application/json'}, body: JSON.stringify(data),
    });
  } catch {
    throw Error('서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.');
  }
  let body = {};
  try { body = await response.json(); } catch { /* generic message below */ }
  if (response.status === 401) { toLogin(); throw Error('세션이 만료됐습니다. 다시 로그인해 주세요.'); }
  if (response.status === 403 && body.must_change) { location.replace('/'); throw Error(body.error); }
  if (!response.ok) throw Error(body.error || '요청에 실패했습니다. 잠시 후 다시 시도해 주세요.');
  return body;
}

// ---- accounts ------------------------------------------------------------
function statusText(user) {
  const parts = [];
  if (user.disabled) parts.push('잠금(관리자)');
  else if (user.locked) parts.push('잠김(로그인 실패)');
  else parts.push('사용 가능');
  if (user.must_change) parts.push('비밀번호 변경 필요');
  return parts.join(' · ');
}
function actionButton(label, handler) {
  const b = node('button', 'btn small', label);
  b.type = 'button';
  b.addEventListener('click', () => void handler());
  return b;
}
async function update(username, changes, done) {
  setMessage('users-message', '');
  try {
    await post('/api/admin/users/update', {username, ...changes});
    await loadUsers();
    setMessage('users-message', done);
  } catch (error) { setMessage('users-message', error.message, true); }
}
function userRow(user) {
  const tr = node('tr');
  tr.append(node('td', '', user.username), node('td', '', ROLE_LABELS[user.role] || user.role),
            node('td', '', statusText(user)));
  const actions = node('div', 'cell-actions');
  if (user.disabled || user.locked) {
    actions.append(actionButton('잠금 해제', () => update(user.username, {disabled: false, unlock: true},
                                                         `${user.username} 계정의 잠금을 풀었습니다.`)));
  }
  if (!user.disabled) {
    actions.append(actionButton('잠금', () => update(user.username, {disabled: true}, `${user.username} 계정을 잠갔습니다.`)));
  }
  actions.append(actionButton('비밀번호 초기화', async () => {
    const password = window.prompt(`${user.username}의 임시 비밀번호를 입력해 주세요 (10자 이상). 다음 로그인 때 바꾸게 됩니다.`);
    if (password) await update(user.username, {reset_password: password}, `${user.username}의 비밀번호를 초기화했습니다.`);
  }));
  const next = user.role === 'admin' ? 'user' : 'admin';
  actions.append(actionButton(`${ROLE_LABELS[next]}로 변경`, async () => {
    if (window.confirm(`${user.username}의 역할을 ${ROLE_LABELS[next]}(으)로 바꿀까요?`)) {
      await update(user.username, {role: next}, `${user.username}의 역할을 바꿨습니다.`);
    }
  }));
  const cell = node('td');
  cell.append(actions);
  tr.append(cell);
  return tr;
}
async function loadUsers() {
  const users = await post('/api/admin/users/list', {});
  el('users').replaceChildren(...users.map(userRow));
}
el('create-form').addEventListener('submit', event => {
  event.preventDefault();
  void (async () => {
    setMessage('users-message', '');
    try {
      const made = await post('/api/admin/users/create', {
        username: el('new-username').value, password: el('new-password').value, role: el('new-role').value,
      });
      el('new-username').value = ''; el('new-password').value = '';
      await loadUsers();
      setMessage('users-message', `${made.username} 계정을 만들었습니다. 첫 로그인 때 비밀번호를 바꿉니다.`);
    } catch (error) { setMessage('users-message', error.message, true); }
  })();
});

// ---- usage log -----------------------------------------------------------
// Empty fields are left out (the API rejects ''); dates become Unix seconds in local time.
function filters() {
  const out = {};
  const since = el('f-since').value, until = el('f-until').value;
  if (since) out.since = new Date(`${since}T00:00:00`).getTime() / 1000;
  if (until) out.until = new Date(`${until}T23:59:59`).getTime() / 1000;
  const username = el('f-username').value.trim();
  if (username) out.username = username;
  if (el('f-event').value) out.event = el('f-event').value;
  return out;
}
function formatTime(ts) {
  const d = new Date(ts * 1000);
  const two = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${two(d.getMonth() + 1)}-${two(d.getDate())} ${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
}
function eventRow(e) {
  const tr = node('tr');
  // A sha256 is 64 characters; show its first 12 in the table (full value in the tooltip and the CSV).
  const entries = Object.entries(e.detail || {});
  const detail = entries.map(([k, v]) => `${k}: ${k === 'sha256' ? String(v).slice(0, 12) + '…' : v}`).join(' · ');
  const cells = [[formatTime(e.ts), 'nowrap'], [EVENT_LABELS[e.event] || e.event, 'nowrap'],
                 [e.username ?? '(알 수 없음)', 'nowrap'], [e.result, 'nowrap'], [e.ip ?? '', 'nowrap'], [detail, 'detail']];
  for (const [text, cls] of cells) tr.append(node('td', cls, text));
  const full = entries.map(([k, v]) => `${k}: ${v}`).join(' · ');
  if (full !== detail) tr.lastChild.title = full;
  return tr;
}
async function loadEvents() {
  setMessage('events-message', '');
  try {
    const rows = await post('/api/admin/events', {...filters(), offset});
    lastCount = rows.length;
    el('events').replaceChildren(...rows.map(eventRow));
    el('page-info').textContent = rows.length ? `${offset + 1}–${offset + rows.length}번째` : '기록이 없습니다.';
  } catch (error) {
    lastCount = 0;
    setMessage('events-message', error.message, true);
  }
  el('prev').disabled = offset === 0;
  el('next').disabled = lastCount < PAGE_SIZE;
  updateCsv();
}
function updateCsv() {
  const params = new URLSearchParams({token, ...filters()});
  el('csv').href = `/api/admin/events.csv?${params}`;
}
el('filter-form').addEventListener('submit', event => { event.preventDefault(); offset = 0; void loadEvents(); });
el('prev').addEventListener('click', () => { offset = Math.max(0, offset - PAGE_SIZE); void loadEvents(); });
el('next').addEventListener('click', () => { offset += PAGE_SIZE; void loadEvents(); });
for (const id of ['f-since', 'f-until', 'f-username', 'f-event']) el(id).addEventListener('input', updateCsv);

el('logout').addEventListener('click', () => void (async () => {
  try { await post('/api/logout', {}); } catch { /* leave anyway */ }
  location.replace('/');
})());

for (const [value, label] of Object.entries(EVENT_LABELS)) {
  const option = node('option', '', label);
  option.value = value;
  el('f-event').append(option);
}
void loadUsers().catch(error => setMessage('users-message', error.message, true));
void loadEvents();
