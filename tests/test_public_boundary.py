from braid.public_boundary import detected_secret_kind, is_private_credential_filename


def test_modern_secret_families_are_detected_without_literal_test_credentials() -> None:
    examples = {
        "GitHub token": "github" + "_pat_" + "A" * 28,
        "OpenAI/Anthropic-style API key": "sk-" + "proj-" + "A" * 28,
        "Google API key": "AI" + "za" + "A" * 35,
        "GitLab token": "gl" + "pat-" + "A" * 24,
        "Hugging Face token": "h" + "f_" + "A" * 32,
        "npm token": "np" + "m_" + "A" * 36,
        "Slack token": "xo" + "xb-" + "A" * 24,
        "Stripe secret": "sk" + "_live_" + "A" * 24,
        "assigned cloud secret": "client" + "_secret=" + "A" * 28,
        "bearer token": "Authorization: Bearer " + "A" * 28,
        "JWT": "ey" + "J" + "A" * 12 + "." + "B" * 12 + "." + "C" * 12,
        "private key": "-----BEGIN " + "PRIVATE KEY-----",
    }

    for expected_kind, value in examples.items():
        assert detected_secret_kind(value) == expected_kind


def test_public_examples_do_not_trigger_secret_detection() -> None:
    assert detected_secret_kind("sha256:" + "a" * 64) is None
    assert detected_secret_kind("https://github.com/dylanp12/braid") is None


def test_private_environment_filenames_fail_closed_but_templates_are_public() -> None:
    assert is_private_credential_filename(".env")
    assert is_private_credential_filename(".env.production")
    assert is_private_credential_filename(".npmrc")
    assert not is_private_credential_filename(".env.example")
    assert not is_private_credential_filename("settings.json")
