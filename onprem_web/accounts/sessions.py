"""Ending logins: registry entries (in memory) and Django session rows (DB) together.

Usable without a web runtime (CLI): the registry part is then skipped.
"""

from __future__ import annotations

from django.contrib.sessions.models import Session
from django.utils import timezone

from onprem_web import runtime

from .models import User


def _registry():
    try:
        return runtime.get().registry
    except RuntimeError:
        return None


def delete_db_sessions(keys) -> None:
    keys = list(keys)
    if keys:
        Session.objects.filter(session_key__in=keys).delete()


def end_user_sessions(username: str, keep: str | None = None) -> None:
    """Drop every login of `username` except session `keep` (as Auth._drop_user_sessions did)."""
    registry = _registry()
    if registry is not None:
        registry.drop_user(username, keep=keep)
    pk = User.objects.filter(username=username).values_list('pk', flat=True).first()
    if pk is None:
        return
    want = str(pk)
    doomed = [row.session_key for row in Session.objects.filter(expire_date__gt=timezone.now())
              if row.session_key != keep and row.get_decoded().get('_auth_user_id') == want]
    delete_db_sessions(doomed)
