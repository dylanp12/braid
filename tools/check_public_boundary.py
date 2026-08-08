"""Prevent private corpora, holdouts, weights, and credentials entering public Git."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATHS = (
    ".aws/",
    ".ssh/",
    "data/raw/",
    "data/sealed/",
    "holdouts/",
    "artifacts/private/",
    "credentials/",
    "secrets/",
)
PRIVATE_BASENAMES = {
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "service-account.json",
}
PUBLIC_ENV_SUFFIXES = (".example", ".sample", ".template")
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
SECRET_PATTERNS = (
    ("GitHub token", re.compile(r"(?:gh[opsur]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("AWS access key", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    (
        "assigned cloud secret",
        re.compile(
            r"(?i)(?:aws_secret_access_key|aws_session_token|client_secret|private_token)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9/+=._-]{24,}"
        ),
    ),
    (
        "OpenAI/Anthropic-style API key",
        re.compile(r"sk-(?:(?:proj|svcacct|ant)-)?[A-Za-z0-9_-]{20,}"),
    ),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("GitLab token", re.compile(r"glpat-[A-Za-z0-9_-]{20,}")),
    ("Hugging Face token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("npm token", re.compile(r"npm_[A-Za-z0-9]{36}")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("Stripe secret", re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{20,}")),
    (
        "bearer token",
        re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/-]{24,}"),
    ),
    (
        "JWT",
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    (
        "private key",
        re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"),
    ),
)


def detected_secret_kind(content: str) -> str | None:
    """Return the first recognized credential family without exposing the value."""

    for name, pattern in SECRET_PATTERNS:
        if pattern.search(content):
            return name
    return None


def is_private_credential_filename(name: str) -> bool:
    if name in PRIVATE_BASENAMES:
        return True
    return name.startswith(".env.") and not name.endswith(PUBLIC_ENV_SUFFIXES)


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
        if relative == "tools/check_public_boundary.py":
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
