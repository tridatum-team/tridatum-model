from django.conf import settings
from django.urls import path

from onprem_web import web
from onprem_web.accounts import views as accounts
from onprem_web.staffadmin import views as staffadmin
from onprem_web.workspace import api as workspace
from onprem_web.workspace import ui

handler404 = 'onprem_web.web.not_found'
handler500 = 'onprem_web.web.server_error'

urlpatterns = [
    path('', ui.home, name='home'),
    path('healthz', web.healthz),
    path('api/login', accounts.api_login),
    path('api/logout', accounts.api_logout),
    path('api/password', accounts.api_password),
    path('api/load', workspace.load),
    path('api/ask', workspace.ask),
    path('api/clear', workspace.clear),
    path('ui/ask', ui.ask),
    path('ui/document', ui.document),
    path('ui/clear', ui.clear),
    path('ui/password', ui.password),
    path('admin', staffadmin.admin_page),
    path('ui/admin/users', staffadmin.create_user),
    path('ui/admin/users/<str:username>', staffadmin.update_user),
    path('ui/admin/events', staffadmin.events),
    path('api/admin/events.csv', staffadmin.events_csv),
    path('static/<path:name>', web.static_file),
    path('fonts/<str:name>', web.font_file),
]

if settings.TRID_TEST_ROUTES:
    urlpatterns.append(path('__boom', web.boom))
