from django.urls import path

from onprem_web import web

handler404 = 'onprem_web.web.not_found'
handler500 = 'onprem_web.web.server_error'

urlpatterns = [
    path('healthz', web.healthz),
    path('static/<path:name>', web.static_file),
    path('fonts/<str:name>', web.font_file),
]
