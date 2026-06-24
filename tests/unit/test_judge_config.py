def test_judge_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o")

    from tests.eval.config import JudgeSettings

    settings = JudgeSettings()

    assert settings.judge_provider == "openai"
    assert settings.judge_model == "gpt-4o"


def test_judge_settings_has_strong_defaults(monkeypatch):
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    from tests.eval.config import JudgeSettings

    # The judge uses a fixed strong model, decoupled from the session agent's provider
    settings = JudgeSettings(_env_file=None)

    assert settings.judge_provider
    assert settings.judge_model


def test_judge_config_not_in_application_settings():
    from app.config import Settings

    # Judge config must live in .env.test, NOT in the application Settings (.env)
    assert not hasattr(Settings(), "judge_provider")
    assert not hasattr(Settings(), "judge_model")
