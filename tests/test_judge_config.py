def test_settings_reads_judge_config_from_env(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o")

    from app.config import Settings

    settings = Settings()

    assert settings.judge_provider == "openai"
    assert settings.judge_model == "gpt-4o"


def test_settings_has_judge_defaults(monkeypatch):
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    from app.config import Settings

    settings = Settings()

    # Mặc định: judge dùng model mạnh cố định, tách khỏi provider của session agent
    assert settings.judge_provider
    assert settings.judge_model
