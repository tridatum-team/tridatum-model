const PUBLIC_HOST = 'test.tridatum.co';
const ORIGIN = 'https://trid-test-origin.eommydocs.com';
const GET_PATHS = new Set([
  '/', '/healthz', '/admin', '/ui/document', '/ui/admin/events',
  '/static/common.css', '/static/app.js', '/static/login.js', '/static/admin.js',
  '/static/vendor/htmx.min.js', '/fonts/PretendardVariable.woff2',
]);
const POST_PATHS = new Set([
  '/api/login', '/api/logout', '/api/password', '/api/load', '/api/ask', '/api/clear',
  '/ui/ask', '/ui/clear', '/ui/password', '/ui/admin/users', '/api/admin/events.csv',
]);
// Account names use the application's exact stored alphabet and length, not arbitrary subpaths.
const UPDATE_USER_PATH = /^\/ui\/admin\/users\/[a-z0-9._-]{2,32}$/;
const MAX_BODY_BYTES = 2 * 1024 * 1024;
const PRIVATE_HEADERS = {
  'Cache-Control': 'no-store',
  'Strict-Transport-Security': 'max-age=86400',
};

function error(status, message) {
  return Response.json({error: message}, {status, headers: PRIVATE_HEADERS});
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (env.PUBLIC_HOST !== PUBLIC_HOST) return error(503, '온프레미스 서버 설정이 준비되지 않았습니다.');
    if (url.hostname !== PUBLIC_HOST || (url.port && url.port !== '443'))
      return error(404, '없는 사이트입니다.');
    if (url.protocol !== 'https:') return new Response(null, {
      status: 308,
      headers: {...PRIVATE_HEADERS, Location: `https://${PUBLIC_HOST}${url.pathname}${url.search}`},
    });
    if (!(request.method === 'GET' && GET_PATHS.has(url.pathname)) &&
        !(request.method === 'POST' && (POST_PATHS.has(url.pathname) || UPDATE_USER_PATH.test(url.pathname))))
      return error(404, '없는 기능입니다.');
    // This Worker is exclusively the test deployment; configuration cannot turn it into an open proxy.
    if (![ORIGIN, `${ORIGIN}/`].includes(env.ORIGIN_URL) || !env.PROXY_SECRET ||
        typeof env.DEMO_RATE_LIMIT?.limit !== 'function')
      return error(503, '온프레미스 서버 설정이 준비되지 않았습니다.');
    if (request.method === 'POST') {
      if (request.headers.get('Origin') !== `https://${PUBLIC_HOST}`)
        return error(403, '허용되지 않은 요청입니다.');
      const declaredLength = request.headers.get('Content-Length');
      const length = Number(declaredLength);
      if (!/^\d+$/.test(declaredLength ?? '') || !Number.isSafeInteger(length) || length > MAX_BODY_BYTES)
        return error(413, '입력은 2MB 이하로 넣어 주세요.');
    }
    const clientIP = request.headers.get('CF-Connecting-IP') || 'unknown';
    try {
      const {success} = await env.DEMO_RATE_LIMIT.limit({key: clientIP});
      if (success !== true) return error(429, '요청이 많습니다. 1분 뒤 다시 시도해 주세요.');
    } catch (failure) {
      console.error('On-prem rate limiter failure:', failure?.name);
      return error(503, '요청 제한 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.');
    }
    // Only session, CSRF, browser UI metadata and body metadata cross the browser boundary.
    // Cloudflare supplies CF-Connecting-IP; never accept browser proxy/forwarded identity headers.
    const headers = new Headers();
    for (const key of [
      'Content-Type', 'Content-Length', 'Cookie', 'Origin', 'Referer', 'Accept', 'X-Trid-Token',
      'HX-Request', 'HX-Target', 'HX-Trigger', 'HX-Trigger-Name', 'HX-Current-URL',
      'HX-Boosted', 'HX-History-Restore-Request',
    ]) {
      if (request.headers.has(key)) headers.set(key, request.headers.get(key));
    }
    headers.set('X-Trid-Client-IP', clientIP);
    headers.set('X-Trid-Proxy', env.PROXY_SECRET);
    headers.set('X-Trid-Host', PUBLIC_HOST);
    try {
      const body = request.method === 'POST' ? await request.arrayBuffer() : undefined;
      if (body && body.byteLength > MAX_BODY_BYTES) return error(413, '입력은 2MB 이하로 넣어 주세요.');
      const upstream = await fetch(new URL(`${url.pathname}${url.search}`, env.ORIGIN_URL), {
        method: request.method, headers, body,
        // CPU generation over retrieved PDF passages can take tens of seconds.
        redirect: 'manual', signal: AbortSignal.timeout(120000),
      });
      // Django renders login directly; its password flow uses HX-Redirect: / on a 200 response.
      // No HTTP redirects are required, and tunnel/login redirects must never escape to clients.
      if (upstream.status >= 300 && upstream.status < 400) throw new Error('On-prem origin redirect refused');
      if (upstream.status >= 500 && !upstream.headers.get('content-type')?.includes('application/json'))
        throw new Error('On-prem origin unavailable');
      const response = new Response(upstream.body, upstream);
      const cacheableFont = url.pathname === '/fonts/PretendardVariable.woff2' &&
        upstream.status === 200 && upstream.headers.get('Content-Type') === 'font/woff2';
      response.headers.set('Cache-Control', cacheableFont ? 'private, max-age=86400' : PRIVATE_HEADERS['Cache-Control']);
      response.headers.set('Strict-Transport-Security', PRIVATE_HEADERS['Strict-Transport-Security']);
      return response;
    } catch (failure) {
      // Never log URLs, cookies, CSRF tokens or request/response bodies.
      console.error('On-prem origin failure:', failure?.name);
      if (request.method === 'GET' && url.pathname === '/') return new Response(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>trid · 연결할 수 없음</title><body style="font-family:system-ui;padding:10vh 8vw;background:#f4f7f5;color:#192d29"><h1>trid 온프레미스 서버에 연결할 수 없습니다.</h1><p>잠시 후 다시 접속하거나 운영자에게 문의해 주세요.</p></body></html>',
        {status: 503, headers: {...PRIVATE_HEADERS, 'Content-Type': 'text/html; charset=utf-8'}});
      return error(503, '온프레미스 서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.');
    }
  },
};
