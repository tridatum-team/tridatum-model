from django.conf import settings
from django.urls import path

from onprem_web import web
from onprem_web.accounts import views as accounts

handler404 = 'onprem_web.web.not_found'
handler500 = 'onprem_web.web.server_error'

urlpatterns = [
    path('', accounts.home, name='home'),
    path('healthz', web.healthz),
    path('api/login', accounts.api_login),
    path('api/logout', accounts.api_logout),
    path('api/password', accounts.api_password),
    path('static/<path:name>', web.static_file),
    path('fonts/<str:name>', web.font_file),
]

if settings.TRID_TEST_ROUTES:
    urlpatterns.append(path('__boom', web.boom))
