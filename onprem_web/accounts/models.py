"""Staff accounts. Mirrors the users table of trid.onprem.store (disabled == not is_active)."""

from __future__ import annotations

import time

from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.db import models
from django.utils.crypto import salted_hmac


class UserManager(BaseUserManager):
    pass


class User(AbstractBaseUser):
    username = models.CharField(max_length=32, unique=True)  # already normalized by services
    role = models.CharField(max_length=8, choices=[('user', 'user'), ('admin', 'admin')])
    must_change = models.BooleanField(default=True)
    failed = models.IntegerField(default=0)
    locked_until = models.FloatField(default=0.0)
    is_active = models.BooleanField(default=True)
    password_version = models.IntegerField(default=0)
    created_at = models.FloatField(default=time.time, null=True)

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = []

    objects = UserManager()

    def __str__(self) -> str:
        return self.username

    def _get_session_auth_hash(self, secret=None):
        # Mixing in password_version means a bump ends every session, even if the hash is unchanged.
        key_salt = 'onprem_web.accounts.User.get_session_auth_hash'
        return salted_hmac(key_salt, f'{self.password}${self.password_version}',
                           secret=secret, algorithm='sha256').hexdigest()
