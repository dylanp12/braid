"""Pure public-boundary rules shared by release tooling and tests."""

from __future__ import annotations

import re

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
