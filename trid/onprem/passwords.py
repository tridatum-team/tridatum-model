"""scrypt password hashing (standard library only)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_N = 16384
_R = 8
_P = 1
_DKLEN = 32
_SALT_BYTES = 16
_MIN_LENGTH = 10


def _scrypt(pw: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        pw.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=_DKLEN,
        maxmem=128 * n * r * 2 + 1024 * 1024,
    )


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _scrypt(pw, salt, _N, _R, _P)
    return "scrypt${}${}${}${}${}".format(
        _N, _R, _P,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(pw: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        actual = _scrypt(pw, salt, int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def check_new_password(pw: str) -> None:
    if len(pw) < _MIN_LENGTH:
        raise ValueError("비밀번호는 10자 이상이어야 합니다.")
