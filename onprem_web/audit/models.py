"""Append-only usage log. The DB triggers (migration 0002) are the real guarantee; save/delete are a first guard."""

from django.db import models


class Event(models.Model):
    ts = models.FloatField(db_index=True)
    username = models.CharField(max_length=32, null=True)
    ip = models.CharField(max_length=45, null=True)
    event = models.CharField(max_length=32)
    result = models.CharField(max_length=32)
    detail = models.TextField()  # JSON with sorted keys; only validated, content-free values

    def save(self, *args, **kwargs):
        if self.pk is not None:
            raise PermissionError('append-only')
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise PermissionError('append-only')


class PurgeGate(models.Model):
    """Single row (id=1) seeded by migration 0002; open only inside services.purge_before."""

    open = models.IntegerField(default=0)
