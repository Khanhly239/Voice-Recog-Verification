import os
import tempfile
from fastapi import UploadFile


async def save_upload_to_temp(upload: UploadFile, suffix: str = ".wav") -> str:
    """Save an uploaded file to a temporary location and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await upload.read()
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    except Exception:
        os.close(fd)
        raise
    return path


def cleanup_temp_files(*paths: str):
    """Remove temporary files, ignoring errors."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
