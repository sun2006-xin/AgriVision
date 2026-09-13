"""Fail when tracked source contains common credentials or private endpoints."""

from pathlib import Path
import re
import subprocess
import sys


PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(?:password|secret|api[_-]?key)\s*[:=]\s*[\"'](?!【)[^\"']{5,}[\"']"),
    re.compile(r"(?i)rtsp://[^\s/:]+:[^\s@]+@"),
    re.compile(r"https?://(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[0-1])\.)"),
)


def tracked_files():
    output = subprocess.check_output(["git", "ls-files", "-z"], text=False)
    return [Path(item) for item in output.decode("utf-8").split("\0") if item]


def find_violations(paths):
    violations = []
    for path in paths:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pt", ".onnx", ".pyc"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            if any(pattern.search(line) for pattern in PATTERNS):
                violations.append(f"{path}:{line_number}")
    return violations


def main():
    violations = find_violations(tracked_files())
    if violations:
        print("PUBLIC_REPO_SCAN_FAILED")
        print("\n".join(violations))
        return 1
    print("PUBLIC_REPO_SCAN_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
