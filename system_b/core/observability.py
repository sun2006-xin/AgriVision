"""Request observability helpers shared by System B HTTP handlers."""

import re
import uuid


_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def resolve_request_id(value):
    """Reuse a bounded safe request ID or generate a non-sensitive replacement."""
    if value and _SAFE_REQUEST_ID.fullmatch(value):
        return value
    return uuid.uuid4().hex
