"""Prevent private corpora, holdouts, weights, and credentials entering public Git."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from braid.public_boundary import (
    PRIVATE_PATHS,
    WEIGHT_SUFFIXES,
    detected_secret_kind,
    is_private_credential_filename,
)

ROOT = Path(__file__).resolve().parents[1]


def candidates() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    failures: list[str] = []
    for path in candidates():
        relative = path.relative_to(ROOT).as_posix()
        if any(relative.startswith(prefix) for prefix in PRIVATE_PATHS):
            failures.append(f"private path: {relative}")
            continue
        if is_private_credential_filename(path.name):
            failures.append(f"private credential filename: {relative}")
            continue
        is_reference_weight = relative.startswith("artifacts/reference/")
        if path.suffix.lower() in WEIGHT_SUFFIXES and not is_reference_weight:
            failures.append(f"private weight outside reference release: {relative}")
            continue
        if not path.is_file() or path.stat().st_size > 10_000_000:
            if path.is_file() and not relative.startswith("artifacts/reference/"):
                failures.append(f"oversized public file: {relative}")
            continue
        if relative in {"src/braid/public_boundary.py", "tools/check_public_boundary.py"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        secret_kind = detected_secret_kind(content)
        if secret_kind is not None:
            failures.append(f"credential-like content ({secret_kind}): {relative}")
    if failures:
        print("Public boundary check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Public boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
