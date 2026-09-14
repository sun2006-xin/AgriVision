"""Load System B pages from editable UTF-8 template files."""

from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def load_page(name):
    """Read a named page from the repository-owned template directory."""
    if name not in {"main.html", "dashboard.html"}:
        raise ValueError("unsupported page template")
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")
