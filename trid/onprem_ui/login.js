'use strict';
// Login view. Server text goes in with textContent only. A successful login reloads "/",
// which the server answers with the staff page (it handles the forced password change).
const EXPIRED = '세션이 만료됐습니다. 다시 로그인해 주세요.';
const form = document.getElementById('login');
const message = document.getElementById('message');
const submit = document.getElementById('submit');

function show(text, error) {
  message.textContent = text;
  message.className = error ? 'msg error' : 'msg';
}
if (location.hash === '#expired') show(EXPIRED, true);

form.addEventListener('submit', async event => {
  event.preventDefault();
  submit.disabled = true;
  show('', false);
  try {
    const response = await fetch('/api/login', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username: form.elements.username.value, password: form.elements.password.value}),
    });
    let data = {};
    try { data = await response.json(); } catch { /* non-JSON error body */ }
    if (!response.ok) {
      show(data.error || '로그인하지 못했습니다. 잠시 후 다시 시도해 주세요.', true);
      form.elements.password.value = '';
      return;
    }
    form.elements.password.value = '';
    location.replace('/');
  } catch {
    show('서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.', true);
  } finally {
    submit.disabled = false;
  }
});
