"""Audit-log service functions. Validation, ordering and CSV output follow trid.onprem.store.Store."""

from __future__ import annotations

import ipaddress
import json
import time

from django.db import transaction

from trid.onprem.store import (
    DETAIL_KEYS, EVENTS, Store, _DETAIL_STR_RE, _RESULT_RE, _fold, normalize_username)

from .models import Event, PurgeGate

_clock = time.time  # tests patch this for fixed timestamps


def log(event, username, ip, result, **detail) -> None:
    if event not in EVENTS:
        raise ValueError(f'unknown event: {event}')
    extra = set(detail) - DETAIL_KEYS
    if extra:
        raise ValueError(f'detail keys not allowed: {sorted(extra)}')
    for key, value in detail.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(f'detail value must be simple: {key}')
        if isinstance(value, str) and not _DETAIL_STR_RE.fullmatch(value):
            raise ValueError(f'detail string format invalid: {key}')
    name = normalize_username(username) if username is not None else None
    if ip is not None:
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            raise ValueError('ip format invalid') from None
    if not isinstance(result, str) or not _RESULT_RE.fullmatch(result):
        raise ValueError('result format invalid')
    Event.objects.create(
        ts=_clock(), username=name, ip=ip, event=event, result=result,
        detail=json.dumps(detail, ensure_ascii=False, sort_keys=True))


def query_events(since=None, until=None, username=None, event=None, limit=50, offset=0) -> list[dict]:
    """Same shape and order as Store.query_events: newest first, list of dicts with parsed detail."""
    qs = Event.objects.all()
    if since is not None:
        qs = qs.filter(ts__gte=since)
    if until is not None:
        qs = qs.filter(ts__lte=until)
    if username is not None:
        qs = qs.filter(username=_fold(username))
    if event is not None:
        qs = qs.filter(event=event)
    qs = qs.order_by('-ts', '-id')
    if limit is not None:
        qs = qs[int(offset):int(offset) + int(limit)]
    return [{'ts': e.ts, 'username': e.username, 'ip': e.ip, 'event': e.event,
             'result': e.result, 'detail': json.loads(e.detail)} for e in qs]


class _Rows:
    """Lets the old Store.export_csv format our rows, so the CSV rules live in one place."""

    @staticmethod
    def query_events(since, until, username, event, limit, offset):
        return query_events(since, until, username, event, limit, offset)


def export_csv(since=None, until=None, username=None, event=None) -> bytes:
    text = Store.export_csv(_Rows, since, until, username, event, None, 0)
    return b'\xef\xbb\xbf' + text.encode('utf-8')


def purge_before(ts: float) -> int:
    with transaction.atomic():
        PurgeGate.objects.filter(id=1).update(open=1)
        count, _ = Event.objects.filter(ts__lt=ts).delete()
        PurgeGate.objects.filter(id=1).update(open=0)
    return count
