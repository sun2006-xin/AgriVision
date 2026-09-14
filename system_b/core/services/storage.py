"""Validation helpers for files received from camera storage endpoints."""


ALLOWED_IMAGE_SUFFIXES = (".jpg", ".jpeg")
MAX_CAMERA_FILENAME_LENGTH = 255
MAX_CAMERA_FILE_LIST = 5000


def normalize_camera_filename(value):
    """Return a safe single-level image filename, or None for invalid input."""
    if not isinstance(value, str) or not value or len(value) > MAX_CAMERA_FILENAME_LENGTH:
        return None
    if value in {".", ".."} or any(separator in value for separator in ("/", "\\", "\x00")):
        return None
    if not value.lower().endswith(ALLOWED_IMAGE_SUFFIXES):
        return None
    return value


def filter_camera_filenames(values):
    """Bound and de-duplicate an untrusted camera file listing."""
    if not isinstance(values, list):
        return []
    result = []
    seen = set()
    for value in values[:MAX_CAMERA_FILE_LIST]:
        filename = normalize_camera_filename(value)
        if filename is not None and filename not in seen:
            seen.add(filename)
            result.append(filename)
    return result
