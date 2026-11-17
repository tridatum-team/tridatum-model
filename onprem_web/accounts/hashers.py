"""Accept native scrypt hashes unchanged; new passwords use Django's encoding."""

import base64
import binascii

from django.contrib.auth.hashers import ScryptPasswordHasher

from trid.onprem.passwords import verify_password


def is_legacy_hash(encoded: str) -> bool:
    """Only accept the native v1 format and its fixed, bounded scrypt cost."""
    if not isinstance(encoded, str):
        return False
    parts = encoded.split('$')
    if len(parts) != 6 or parts[:4] != ['scrypt', '16384', '8', '1']:
        return False
    try:
        salt = base64.b64decode(parts[4], validate=True)
        digest = base64.b64decode(parts[5], validate=True)
    except (ValueError, binascii.Error):
        return False
    return (len(salt) == 16 and len(digest) == 32
            and base64.b64encode(salt).decode('ascii') == parts[4]
            and base64.b64encode(digest).decode('ascii') == parts[5])


class CompatibleScryptPasswordHasher(ScryptPasswordHasher):
    def verify(self, password, encoded):
        if is_legacy_hash(encoded):
            return verify_password(password, encoded)
        return super().verify(password, encoded)

    def must_update(self, encoded):
        return is_legacy_hash(encoded) or super().must_update(encoded)
