import test from 'node:test';
import assert from 'node:assert/strict';
import worker from './worker.mjs';

const env = {
  PUBLIC_HOST: 'test.tridatum.co',
  ORIGIN_URL: 'https://origin.example',
  PROXY_SECRET: 'proxy-secret',
  DEMO_RATE_LIMIT: {limit: async () => ({success: true})},
};
const site = `https://${env.PUBLIC_HOST}`;
const getPaths = ['/', '/healthz', '/app.js', '/login.js', '/admin.js', '/common.css', '/admin', '/api/admin/events.csv', '/fonts/PretendardVariable.woff2'];
const postPaths = [
  '/api/login', '/api/logout', '/api/password', '/api/load', '/api/ask', '/api/clear',
  '/api/admin/users/list', '/api/admin/users/create', '/api/admin/users/update', '/api/admin/events',
];

function post(path, headers = {}, body = '{}') {
  return new Request(`${site}${path}`, {
    method: 'POST', body,
    headers: {Origin: site, 'Content-Length': String(Buffer.byteLength(body)), ...headers},
  });
}

function assertPrivate(response) {
  assert.equal(response.headers.get('Cache-Control'), 'no-store');
  assert.equal(response.headers.get('Strict-Transport-Security'), 'max-age=86400');
}

test('allows every explicit onprem page, asset and API route with its supported method', async t => {
  const seen = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    seen.push(`${options.method} ${url.pathname}`);
    return new Response('{"ok":true}', {headers: {'Content-Type': 'application/json'}});
  });
  for (const path of getPaths) {
    assert.equal((await worker.fetch(new Request(`${site}${path}`), env)).status, 200, path);
  }
  for (const path of postPaths) {
    assert.equal((await worker.fetch(post(path), env)).status, 200, path);
  }
  assert.deepEqual(seen, [...getPaths.map(path => `GET ${path}`), ...postPaths.map(path => `POST ${path}`)]);
});

test('rejects wrong hosts, unknown paths, old demo endpoints and wrong methods before origin access', async t => {
  const upstream = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not reach origin'); });
  const requests = [
    new Request('https://evil.example/'),
    ...['/runs/model.safetensors', '/index.html', '/login', '/api/admin/users/delete', '/admin/', '/%61dmin']
      .map(path => new Request(`${site}${path}`)),
    ...['/api/sample', '/api/text', '/api/pin', '/api/admin/events.csv', '/', '/app.js'].map(path => post(path)),
    ...postPaths.map(path => new Request(`${site}${path}`)),
    ...['HEAD', 'OPTIONS', 'PUT', 'DELETE'].map(method => new Request(`${site}/`, {method})),
  ];
  for (const request of requests) {
    const response = await worker.fetch(request, env);
    assert.equal(response.status, 404, `${request.method} ${request.url}`);
    assertPrivate(response);
  }
  assert.equal(upstream.mock.callCount(), 0);
});

test('requires the exact configured HTTPS origin for every POST, including login', async t => {
  const upstream = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not reach origin'); });
  for (const path of postPaths) {
    for (const origin of [null, 'null', 'https://evil.example', `http://${env.PUBLIC_HOST}`, `${site}/`]) {
      const request = post(path);
      if (origin === null) request.headers.delete('Origin');
      else request.headers.set('Origin', origin);
      const response = await worker.fetch(request, env);
      assert.equal(response.status, 403, `${path}: ${origin}`);
      assertPrivate(response);
    }
  }
  assert.equal(upstream.mock.callCount(), 0);
});

test('upgrades plaintext transport to the configured HTTPS host without losing CSV query parameters', async t => {
  const upstream = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not reach origin'); });
  const response = await worker.fetch(new Request(`http://${env.PUBLIC_HOST}/api/admin/events.csv?token=a%2Bb&event=login_ok`), env);
  assert.equal(response.status, 308);
  assert.equal(response.headers.get('Location'), `${site}/api/admin/events.csv?token=a%2Bb&event=login_ok`);
  assertPrivate(response);
  assert.equal(upstream.mock.callCount(), 0);
});

test('fails closed when proxy configuration or the mandatory rate limiter is missing', async t => {
  const upstream = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not reach origin'); });
  for (const key of ['PUBLIC_HOST', 'ORIGIN_URL', 'PROXY_SECRET', 'DEMO_RATE_LIMIT']) {
    const missing = {...env};
    delete missing[key];
    const response = await worker.fetch(post('/api/login'), missing);
    assert.equal(response.status, 503, key);
    assertPrivate(response);
  }
  assert.equal((await worker.fetch(post('/api/login'), {...env, DEMO_RATE_LIMIT: {}})).status, 503);
  assert.equal(upstream.mock.callCount(), 0);
});

test('rate limits every allowed route including login using only the Cloudflare client identity', async t => {
  const keys = [];
  const limited = {...env, DEMO_RATE_LIMIT: {limit: async ({key}) => { keys.push(key); return {success: false}; }}};
  const upstream = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not reach origin'); });
  const requests = [...getPaths.map(path => new Request(`${site}${path}`)), ...postPaths.map(path => post(path))];
  for (const request of requests) {
    request.headers.set('CF-Connecting-IP', '203.0.113.7');
    request.headers.set('X-Real-IP', 'attacker');
    request.headers.set('X-Trid-Client-IP', 'attacker-custom');
    request.headers.set('X-Forwarded-For', 'attacker');
    const response = await worker.fetch(request, limited);
    assert.equal(response.status, 429);
    assertPrivate(response);
  }
  assert.deepEqual(keys, requests.map(() => '203.0.113.7'));
  assert.equal(upstream.mock.callCount(), 0);
});

test('a broken rate limiter cannot bypass protection or contact the origin', async t => {
  t.mock.method(console, 'error', () => {});
  const upstream = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not reach origin'); });
  for (const limit of [async () => { throw new Error('unavailable'); }, async () => null]) {
    const response = await worker.fetch(post('/api/login'), {...env, DEMO_RATE_LIMIT: {limit}});
    assert.equal(response.status, 503);
    assertPrivate(response);
  }
  const response = await worker.fetch(post('/api/login'), {...env, DEMO_RATE_LIMIT: {limit: async () => ({})}});
  assert.equal(response.status, 429);
  assert.equal(upstream.mock.callCount(), 0);
});

test('forwards only required browser headers and replaces all proxy identity from trusted inputs', async t => {
  let captured;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    captured = {url, options};
    return new Response('{"ok":true}', {headers: {'Content-Type': 'application/json'}});
  });
  const result = await worker.fetch(post('/api/load', {
    'Content-Type': 'application/pdf', Cookie: '__Host-trid=session', 'X-Trid-Token': 'csrf-token',
    'CF-Connecting-IP': '2001:db8::7', 'X-Real-IP': 'attacker',
    'X-Trid-Client-IP': 'attacker-custom',
    'X-Trid-Proxy': 'attacker', 'X-Trid-Host': 'evil.example', 'X-File-Name': 'private.pdf',
    Authorization: 'Bearer irrelevant', Forwarded: 'for=attacker', 'X-Forwarded-For': 'attacker',
    'X-Forwarded-Host': 'evil.example', 'X-Forwarded-Proto': 'http', Host: 'evil.example',
    'User-Agent': 'browser', 'Accept-Encoding': 'gzip',
  }, '%PDF'), env);
  assert.equal(result.status, 200);
  assert.equal(captured.url.href, 'https://origin.example/api/load');
  assert.deepEqual(Object.fromEntries(captured.options.headers), {
    'content-length': '4', 'content-type': 'application/pdf', cookie: '__Host-trid=session', origin: site,
    'x-trid-client-ip': '2001:db8::7', 'x-trid-host': env.PUBLIC_HOST,
    'x-trid-proxy': env.PROXY_SECRET, 'x-trid-token': 'csrf-token',
  });
  assert.equal(Buffer.from(captured.options.body).toString(), '%PDF');
  assert.equal(captured.options.redirect, 'manual');
  assert.ok(captured.options.signal instanceof AbortSignal);
  assert.equal(captured.options.signal.aborted, false);
  assertPrivate(result);
});

test('does not fall back to browser-supplied client identity when Cloudflare identity is absent', async t => {
  let key, headers;
  t.mock.method(globalThis, 'fetch', async (_url, options) => { headers = options.headers; return new Response('ok'); });
  const configured = {...env, DEMO_RATE_LIMIT: {limit: async input => { key = input.key; return {success: true}; }}};
  await worker.fetch(post('/api/login', {
    'X-Real-IP': 'attacker', 'X-Trid-Client-IP': 'attacker-custom', 'X-Forwarded-For': 'attacker',
  }), configured);
  assert.equal(key, 'unknown');
  assert.equal(headers.get('X-Trid-Client-IP'), 'unknown');
  assert.equal(headers.get('X-Real-IP'), null);
});

test('preserves CSV authentication and filters exactly, plus download and browser security headers', async t => {
  const query = '?token=a%2Bb%2Fc&since=1&until=2&username=Kim+Lee&event=login_ok';
  const csv = '\ufeffts,username,ip,event,result,detail\n';
  const responseHeaders = {
    'Content-Type': 'text/csv; charset=utf-8',
    'Content-Disposition': 'attachment; filename="trid-usage-20261113.csv"',
    'Content-Security-Policy': "default-src 'self'; frame-ancestors 'none'",
    'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer', 'Permissions-Policy': 'camera=(), microphone=()',
  };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url.href, `https://origin.example/api/admin/events.csv${query}`);
    assert.equal(options.headers.get('Cookie'), '__Host-trid=session');
    assert.equal(options.body, undefined);
    return new Response(csv, {headers: {...responseHeaders, 'Cache-Control': 'public, max-age=3600'}});
  });
  const response = await worker.fetch(new Request(`${site}/api/admin/events.csv${query}`, {
    headers: {Cookie: '__Host-trid=session'},
  }), env);
  assert.equal(response.status, 200);
  for (const [name, value] of Object.entries(responseHeaders)) assert.equal(response.headers.get(name), value);
  assert.deepEqual(new Uint8Array(await response.arrayBuffer()), new TextEncoder().encode(csv));
  assertPrivate(response);
});

test('caches only successful font responses, never pages or font errors', async t => {
  const cases = [
    ['/fonts/PretendardVariable.woff2', 200, 'font/woff2', 'private, max-age=86400'],
    ['/fonts/PretendardVariable.woff2', 404, 'font/woff2', 'no-store'],
    ['/fonts/PretendardVariable.woff2', 200, 'text/html', 'no-store'],
    ['/', 200, 'text/html', 'no-store'],
    ['/admin', 200, 'text/html', 'no-store'],
  ];
  const upstream = t.mock.method(globalThis, 'fetch');
  for (const [path, status, mime, expected] of cases) {
    upstream.mock.mockImplementation(async () => new Response('body', {
      status, headers: {'Content-Type': mime, 'Cache-Control': 'public, max-age=31536000'},
    }));
    const response = await worker.fetch(new Request(`${site}${path}`), env);
    assert.equal(response.headers.get('Cache-Control'), expected, `${path} ${status} ${mime}`);
  }
});

test('preserves login, logout and expired-session Set-Cookie responses and application errors', async t => {
  const cases = [
    ['/api/login', 200, '__Host-trid=session; Path=/; Secure; HttpOnly; SameSite=Strict'],
    ['/api/logout', 200, '__Host-trid=; Max-Age=0; Path=/; Secure; HttpOnly; SameSite=Strict'],
    ['/api/ask', 401, '__Host-trid=; Max-Age=0; Path=/; Secure; HttpOnly; SameSite=Strict'],
    ['/api/login', 503, null],
  ];
  for (const [path, status, cookie] of cases) {
    const headers = {'Content-Type': 'application/json'};
    if (cookie) headers['Set-Cookie'] = cookie;
    const body = JSON.stringify({status: status === 200 ? 'ok' : 'unavailable'});
    t.mock.method(globalThis, 'fetch', async () => new Response(body, {status, headers}));
    const response = await worker.fetch(post(path), env);
    assert.equal(response.status, status);
    assert.equal(response.headers.get('Set-Cookie'), cookie);
    assert.equal(await response.text(), body);
    assertPrivate(response);
  }
});

test('requires a valid declared body length no greater than 2 MiB', async t => {
  const upstream = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Must not reach origin'); });
  for (const length of [null, '', '-1', '1.5', 'NaN', 'Infinity', '2e0', '0x2', '2097153']) {
    const request = post('/api/load');
    if (length === null) request.headers.delete('Content-Length');
    else request.headers.set('Content-Length', length);
    const response = await worker.fetch(request, env);
    assert.equal(response.status, 413, String(length));
    assertPrivate(response);
  }
  assert.equal(upstream.mock.callCount(), 0);
});

test('enforces the actual 2 MiB body limit even with a falsely small declared length', async t => {
  const upstream = t.mock.method(globalThis, 'fetch', async () => new Response('ok'));
  const maximum = 'a'.repeat(2 * 1024 * 1024);
  assert.equal((await worker.fetch(post('/api/load', {}, maximum), env)).status, 200);
  const oversized = await worker.fetch(post('/api/load', {'Content-Length': '1'}, `${maximum}a`), env);
  assert.equal(oversized.status, 413);
  assertPrivate(oversized);
  assert.equal(upstream.mock.callCount(), 1);
});

test('refuses all upstream redirects rather than leaking session cookies or redirect locations', async t => {
  t.mock.method(console, 'error', () => {});
  for (const status of [301, 302, 303, 307, 308]) {
    t.mock.method(globalThis, 'fetch', async (_url, options) => {
      assert.equal(options.redirect, 'manual');
      return new Response(null, {status, headers: {Location: 'https://evil.example/', 'Set-Cookie': 'unsafe=1'}});
    });
    const response = await worker.fetch(post('/api/login'), env);
    assert.equal(response.status, 503);
    assert.equal(response.headers.get('Location'), null);
    assert.equal(response.headers.get('Set-Cookie'), null);
    assertPrivate(response);
  }
});

test('origin outages, timeouts and non-JSON gateway errors give private onprem failures', async t => {
  t.mock.method(console, 'error', () => {});
  for (const fetch of [
    async () => { throw new Error('offline'); },
    async () => { throw new DOMException('expired', 'TimeoutError'); },
    async () => new Response('<html>gateway unavailable</html>', {status: 502, headers: {'Content-Type': 'text/html'}}),
  ]) {
    t.mock.method(globalThis, 'fetch', fetch);
    const home = await worker.fetch(new Request(`${site}/`), env);
    assert.equal(home.status, 503);
    assert.match(home.headers.get('Content-Type'), /text\/html/);
    const page = await home.text();
    assert.match(page, /온프레미스/);
    assert.doesNotMatch(page, /임시 데모|gateway unavailable/);
    assertPrivate(home);
    const api = await worker.fetch(post('/api/ask'), env);
    assert.equal(api.status, 503);
    assert.match((await api.json()).error, /온프레미스/);
    assertPrivate(api);
  }
});
