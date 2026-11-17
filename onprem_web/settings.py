"""Django settings for the on-prem build.

Everything installation-specific comes from the same onprem.json the stdlib server reads
(env TRID_WEB_CONFIG). TRID_WEB_DEV=1 is the insecure local mode (plain http on 127.0.0.1).
"""

import os
import secrets
from pathlib import Path

from trid.onprem.config import load_config

TRID_CONFIG = load_config(Path(os.environ['TRID_WEB_CONFIG']))
TRID_DEV = os.environ.get('TRID_WEB_DEV') == '1'

TRID_TEST_ROUTES = os.environ.get('TRID_TEST_ROUTES') == '1'  # set only by the test support module

BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = TRID_CONFIG.data_dir


def _secret_key() -> str:
    path = _DATA_DIR / 'web-secret-key'
    try:
        return path.read_text(encoding='utf-8').strip()
    except OSError:
        pass
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(64)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another worker won the race
        return path.read_text(encoding='utf-8').strip()
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(key)
    return key


SECRET_KEY = _secret_key()
DEBUG = False  # never True, not even in dev: error pages must not render
if TRID_DEV:
    ALLOWED_HOSTS = ['127.0.0.1', 'localhost']
    CSRF_TRUSTED_ORIGINS = [f'http://{h}:{TRID_CONFIG.port}' for h in ALLOWED_HOSTS]
else:
    ALLOWED_HOSTS = [TRID_CONFIG.hostname]
    CSRF_TRUSTED_ORIGINS = ['https://' + TRID_CONFIG.hostname]
ADMINS = []
MANAGERS = []

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'django.contrib.sessions',
    'onprem_web.audit',
    'onprem_web.accounts',
    'onprem_web.workspace',
    'onprem_web.staffadmin',
]

MIDDLEWARE = [
    'onprem_web.web.SecurityHeadersMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'onprem_web.accounts.middleware.TridSessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
]

ROOT_URLCONF = 'onprem_web.urls'
WSGI_APPLICATION = None
TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': []},
}]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': _DATA_DIR / 'web.db',
        'OPTIONS': {'timeout': 5, 'init_command': 'PRAGMA journal_mode=WAL;'},
    }
}
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

PASSWORD_HASHERS = ['onprem_web.accounts.hashers.CompatibleScryptPasswordHasher']
AUTH_USER_MODEL = 'accounts.User'
AUTHENTICATION_BACKENDS = ['django.contrib.auth.backends.ModelBackend']  # rejects is_active=False
AUTH_PASSWORD_VALIDATORS = []  # rules live in trid.onprem.passwords.check_new_password

# Sessions and CSRF: the token lives in the session and travels in X-Trid-Token.
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_COOKIE_AGE = TRID_CONFIG.idle_seconds  # DB row lifetime; the middleware renews it on activity
SESSION_COOKIE_NAME = 'trid-dev' if TRID_DEV else '__Host-trid'
SESSION_COOKIE_SECURE = not TRID_DEV
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Strict'
SESSION_COOKIE_PATH = '/'
CSRF_USE_SESSIONS = True
CSRF_HEADER_NAME = 'HTTP_X_TRID_TOKEN'
CSRF_COOKIE_SECURE = not TRID_DEV
CSRF_FAILURE_VIEW = 'onprem_web.web.csrf_failure'

# Caddy terminates TLS in front of the app; no SECURE_* redirects here.
SECURE_SSL_REDIRECT = False

# Uploads stay in memory and are capped at the configured PDF size.
FILE_UPLOAD_HANDLERS = ['django.core.files.uploadhandler.MemoryFileUploadHandler']
DATA_UPLOAD_MAX_MEMORY_SIZE = TRID_CONFIG.pdf.max_bytes
FILE_UPLOAD_MAX_MEMORY_SIZE = TRID_CONFIG.pdf.max_bytes

# Content-free logging: one stderr handler, type/status only (see logging_policy). No mail_admins.
_LOG_LOGGER = {'handlers': ['stderr'], 'level': 'INFO', 'propagate': False}
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'type_only': {'()': 'onprem_web.logging_policy.TypeOnlyFormatter'}},
    'handlers': {'stderr': {'class': 'logging.StreamHandler', 'formatter': 'type_only'}},
    'loggers': {name: dict(_LOG_LOGGER)
                for name in ('django', 'django.request', 'django.server', 'django.security', 'waitress')},
    # Everything else (py.warnings via captureWarnings, library loggers) also goes through type_only.
    'root': {'handlers': ['stderr'], 'level': 'WARNING'},
}

APPEND_SLASH = False
USE_I18N = False
USE_TZ = True
TIME_ZONE = 'UTC'
