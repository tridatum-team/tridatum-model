import test from 'node:test';
import assert from 'node:assert/strict';
import worker from './worker.mjs';

const PUBLIC = 'https://test.tridatum.co';
const ORIGIN = 'https://trid-test-origin.eommydocs.com';
const env = {
  PUBLIC_HOST: 'test.tridatum.co', ORIGIN_URL: ORIGIN, PROXY_SECRET: 'test-only-proxy-secret',
  DEMO_RATE_LIMIT: {limit: async () => ({success: true})},
};

function post(path, body = '{}', headers = {}) {
  return new Request(PUBLIC + path, {
    method: 'POST', body,
    headers: {Origin: PUBLIC, 'Content-Length': String(new TextEncoder().encode(body).byteLength), ...headers},
  });
}

async function withOrigin(handler, run) {
  const old = globalThis.fetch;
  globalThis.fetch = handler;
  try { await run(); } finally { globalThis.fetch = old; }
}

test('serves Django pages, exact static assets, health and read-only HTML fragments', async () => {
  const paths = [
    '/', '/admin', '/healthz', '/ui/document', '/ui/admin/events?username=staff&offset=50',
    '/static/common.css', '/static/app.js', '/static/login.js', '/static/admin.js',
    '/static/vendor/htmx.min.js', '/fonts/PretendardVariable.woff2',
  ];
  const seen = [];
  await withOrigin(async (url, options) => {
    seen.push(url.pathname + url.search);
    assert.equal(url.origin, ORIGIN);
    assert.equal(options.method, 'GET');
    const font = url.pathname.startsWith('/fonts/');
    return new Response('public asset or authenticated page', {
      headers: {'Content-Type': font ? 'font/woff2' : 'text/html', 'Cache-Control': 'public, max-age=999'},
    });
  }, async () => {
    for (const path of paths) {
      const response = await worker.fetch(new Request(PUBLIC + path), env);
      assert.equal(response.status, 200, path);
      assert.equal(response.headers.get('Cache-Control'), path.startsWith('/fonts/') ? 'private, max-age=86400' : 'no-store');
      assert.equal(response.headers.get('Strict-Transport-Security'), 'max-age=86400');
    }
    assert.deepEqual(seen, paths);
  });
});

test('forwards login, authenticated JSON APIs, HTML form actions and admin account mutations', async () => {
  const actions = [
    ['/api/login', '{"username":"staff","password":"test password"}', 'application/json'],
    ['/api/logout', '{}', 'application/json'],
    ['/api/password', '{"current":"old","new":"new"}', 'application/json'],
    ['/api/load', '%PDF-1.7\nfixture', 'application/pdf'],
    ['/api/ask', '{"question":"test"}', 'application/json'],
    ['/api/clear', '{}', 'application/json'],
    ['/ui/ask', 'question=test&document_id=fixture', 'application/x-www-form-urlencoded'],
    ['/ui/clear', '', 'application/x-www-form-urlencoded'],
    ['/ui/password', 'current=old&new=new', 'application/x-www-form-urlencoded'],
    ['/ui/admin/users', 'username=new-user&password=test&role=user', 'application/x-www-form-urlencoded'],
    ['/ui/admin/users/staff.user-01', 'disabled=1', 'application/x-www-form-urlencoded'],
  ];
  const seen = [];
  await withOrigin(async (url, options) => {
    seen.push([url.pathname, new TextDecoder().decode(options.body), options.headers.get('Content-Type')]);
    assert.equal(options.method, 'POST');
    return new Response('<section>updated</section>', {
      headers: {'Content-Type': 'text/html', 'HX-Redirect': '/',
        'Set-Cookie': '__Host-trid=session; Secure; HttpOnly; Path=/; SameSite=Strict'},
    });
  }, async () => {
    for (const [path, body, type] of actions) {
      const response = await worker.fetch(post(path, body, {'Content-Type': type}), env);
      assert.equal(response.status, 200, path);
      assert.equal(response.headers.get('HX-Redirect'), '/');
      assert.match(response.headers.get('Set-Cookie'), /Secure; HttpOnly/);
      assert.equal(response.headers.get('Cache-Control'), 'no-store');
      assert.equal(await response.text(), '<section>updated</section>');
    }
    assert.deepEqual(seen, actions);
  });
});

test('CSV export carries form CSRF and filters in POST body, never a legacy query token', async () => {
  const body = 'csrfmiddlewaretoken=form-token&username=staff&since=1&until=2';
  let calls = 0;
  await withOrigin(async (url, options) => {
    calls++;
    assert.equal(url.href, ORIGIN + '/api/admin/events.csv');
    assert.equal(options.method, 'POST');
    assert.equal(new TextDecoder().decode(options.body), body);
    return new Response('event,username\nlogin_ok,staff\n', {
      headers: {'Content-Type': 'text/csv; charset=utf-8', 'Content-Disposition': 'attachment; filename="trid-usage.csv"'},
    });
  }, async () => {
    const response = await worker.fetch(post('/api/admin/events.csv', body, {
      'Content-Type': 'application/x-www-form-urlencoded', Cookie: '__Host-trid=session',
    }), env);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('Content-Disposition'), 'attachment; filename="trid-usage.csv"');
    assert.equal(response.headers.get('Cache-Control'), 'no-store');
    assert.match(await response.text(), /login_ok,staff/);
    assert.equal((await worker.fetch(new Request(PUBLIC + '/api/admin/events.csv?token=legacy'), env)).status, 404);
    assert.equal(calls, 1);
  });
});

test('preserves CSRF and HTMX metadata while replacing spoofed proxy identity', async () => {
  let captured;
  const browserHeaders = {
    'Content-Type': 'application/x-www-form-urlencoded', Cookie: '__Host-trid=session',
    'X-Trid-Token': 'csrf-token', Accept: 'text/html', Referer: PUBLIC + '/',
    'HX-Request': 'true', 'HX-Target': 'result', 'HX-Trigger': 'ask', 'HX-Trigger-Name': 'question',
    'HX-Current-URL': PUBLIC + '/', 'HX-Boosted': 'false', 'HX-History-Restore-Request': 'false',
  };
  await withOrigin(async (url, options) => {
    captured = {url, options};
    return new Response('answer');
  }, async () => {
    const response = await worker.fetch(post('/ui/ask', 'question=test', {
      ...browserHeaders, 'CF-Connecting-IP': '192.0.2.10',
      'X-Trid-Proxy': 'attacker', 'X-Trid-Host': 'evil.example', 'X-Trid-Client-IP': '192.0.2.99',
      Host: 'evil.example', 'X-Real-IP': '192.0.2.99', 'X-Forwarded-For': '192.0.2.99',
      'X-Forwarded-Host': 'evil.example', 'X-Forwarded-Proto': 'http',
      Forwarded: 'for=192.0.2.99;proto=http', Authorization: 'Bearer attacker',
    }), env);
    assert.equal(response.status, 200);
    assert.equal(captured.url.href, ORIGIN + '/ui/ask');
    for (const [key, value] of Object.entries(browserHeaders))
      assert.equal(captured.options.headers.get(key), value, key);
    assert.equal(captured.options.headers.get('Origin'), PUBLIC);
    assert.equal(captured.options.headers.get('X-Trid-Proxy'), env.PROXY_SECRET);
    assert.equal(captured.options.headers.get('X-Trid-Host'), 'test.tridatum.co');
    assert.equal(captured.options.headers.get('X-Trid-Client-IP'), '192.0.2.10');
    for (const key of ['Host', 'X-Real-IP', 'X-Forwarded-For', 'X-Forwarded-Host', 'X-Forwarded-Proto', 'Forwarded', 'Authorization', 'CF-Connecting-IP'])
      assert.equal(captured.options.headers.get(key), null, key);
    assert.equal(captured.options.redirect, 'manual');
  });
});

test('denies other hosts, origins, internal APIs, old routes and non-allowlisted static paths', async () => {
  let calls = 0;
  await withOrigin(async () => { calls++; return new Response('unexpected'); }, async () => {
    for (const host of ['tridatum.co', 'evil.example', 'test.tridatum.co.evil.example', 'test.tridatum.co:8443'])
      assert.equal((await worker.fetch(new Request(`https://${host}/`), env)).status, 404, host);
    for (const origin of ['https://evil.example', 'http://test.tridatum.co', 'null', ''])
      assert.equal((await worker.fetch(post('/ui/ask', 'question=test', {Origin: origin}), env)).status, 403, origin);
    const missingOrigin = post('/api/login');
    missingOrigin.headers.delete('Origin');
    assert.equal((await worker.fetch(missingOrigin, env)).status, 403);
    for (const path of [
      '/__boom', '/v1/chat/completions', '/runs/model.safetensors', '/onprem.json', '/app.js',
      '/static/settings.py', '/static/vendor/unknown.js', '/static/%2e%2e/onprem.json',
      '/static/..%2fsettings.py', '/static/common.css/extra', '/static//common.css',
      '/fonts/unknown.woff2', '//evil.example/static/app.js',
    ]) assert.equal((await worker.fetch(new Request(PUBLIC + path), env)).status, 404, path);
    for (const path of [
      '/api/admin/users/list', '/api/admin/users/create', '/api/admin/users/update', '/api/admin/events',
      '/ui/admin/users/a', '/ui/admin/users/staff/extra', '/ui/admin/users/staff%2fextra',
      '/ui/admin/users/' + 'a'.repeat(33), '/static/app.js',
    ]) assert.equal((await worker.fetch(post(path), env)).status, 404, path);
    assert.equal((await worker.fetch(new Request(PUBLIC + '/ui/ask'), env)).status, 404);
    assert.equal((await worker.fetch(new Request(PUBLIC + '/', {method: 'DELETE'}), env)).status, 404);
    assert.equal(calls, 0);
  });
});

test('rejects untrusted origin configuration and missing security bindings without fetching', async () => {
  let calls = 0;
  await withOrigin(async () => { calls++; return new Response('unexpected'); }, async () => {
    for (const value of [undefined, 'http://trid-test-origin.eommydocs.com', 'https://evil.example',
      ORIGIN + '.evil.example', ORIGIN + '/other', ORIGIN + '?redirect=evil', 'https://user:pass@trid-test-origin.eommydocs.com']) {
      assert.equal((await worker.fetch(new Request(PUBLIC + '/'), {...env, ORIGIN_URL: value})).status, 503);
    }
    for (const override of [{PUBLIC_HOST: 'evil.example'}, {PUBLIC_HOST: undefined}, {PROXY_SECRET: ''}, {DEMO_RATE_LIMIT: undefined}])
      assert.equal((await worker.fetch(new Request(PUBLIC + '/'), {...env, ...override})).status, 503);
    assert.equal(calls, 0);
  });
});

test('enforces declared and actual body size before reaching the origin', async () => {
  let calls = 0;
  await withOrigin(async () => { calls++; return new Response('unexpected'); }, async () => {
    for (const length of ['-1', 'abc', '2.5', '2097153', '9007199254740992'])
      assert.equal((await worker.fetch(post('/api/load', 'pdf', {'Content-Length': length}), env)).status, 413);
    const missingLength = post('/api/load', 'pdf');
    missingLength.headers.delete('Content-Length');
    assert.equal((await worker.fetch(missingLength, env)).status, 413);
    assert.equal((await worker.fetch(post('/api/load', 'a'.repeat(2 * 1024 * 1024 + 1), {'Content-Length': '1'}), env)).status, 413);
    assert.equal(calls, 0);
  });
});

test('rate limits the Cloudflare client address and fails closed when limiter is unavailable', async (t) => {
  t.mock.method(console, 'error', () => {});
  let calls = 0;
  await withOrigin(async () => { calls++; return new Response('unexpected'); }, async () => {
    const request = new Request(PUBLIC + '/healthz', {headers: {'CF-Connecting-IP': '192.0.2.10', 'X-Trid-Client-IP': '192.0.2.99'}});
    const limited = {...env, DEMO_RATE_LIMIT: {limit: async ({key}) => {
      assert.equal(key, '192.0.2.10');
      return {success: false};
    }}};
    assert.equal((await worker.fetch(request, limited)).status, 429);
    const unavailable = {...env, DEMO_RATE_LIMIT: {limit: async () => { throw new Error('offline'); }}};
    assert.equal((await worker.fetch(request, unavailable)).status, 503);
    assert.equal(calls, 0);
  });
});

test('upgrades HTTP to public TLS without contacting the origin', async () => {
  let calls = 0;
  await withOrigin(async () => { calls++; return new Response('unexpected'); }, async () => {
    const response = await worker.fetch(new Request('http://test.tridatum.co/ui/admin/events?offset=50'), env);
    assert.equal(response.status, 308);
    assert.equal(response.headers.get('Location'), PUBLIC + '/ui/admin/events?offset=50');
    assert.equal(response.headers.get('Cache-Control'), 'no-store');
    assert.equal(calls, 0);
  });
});

test('preserves application authorization errors and hides upstream redirects and HTML failures', async (t) => {
  t.mock.method(console, 'error', () => {});
  for (const status of [401, 403, 429, 500]) {
    await withOrigin(async () => Response.json({error: 'application error'}, {status}), async () => {
      const response = await worker.fetch(post('/ui/ask', 'question=test'), env);
      assert.equal(response.status, status);
      assert.deepEqual(await response.json(), {error: 'application error'});
      assert.equal(response.headers.get('Cache-Control'), 'no-store');
    });
  }
  for (const failure of [
    () => new Response(null, {status: 302, headers: {Location: 'https://evil.example/'}}),
    () => new Response('<html>internal details</html>', {status: 502, headers: {'Content-Type': 'text/html'}}),
    () => { throw new Error('origin unavailable'); },
  ]) {
    await withOrigin(failure, async () => {
      const response = await worker.fetch(new Request(PUBLIC + '/'), env);
      assert.equal(response.status, 503);
      assert.equal(response.headers.get('Location'), null);
      assert.match(await response.text(), /온프레미스 서버에 연결할 수 없습니다/);
      const api = await worker.fetch(post('/api/ask'), env);
      assert.equal(api.status, 503);
      assert.match((await api.json()).error, /온프레미스 서버에 연결할 수 없습니다/);
    });
  }
});
