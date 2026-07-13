# Freeze the module-level `judge_settings` singleton at collection time, under a clean env.
# These tests monkeypatch JUDGE_* and then import tests.eval.config; without this top-level
# import the singleton would be created lazily during a monkeypatched test and leak a wrong
# provider into the whole session (scenario judge -> 401).
import tests.eval.config  # noqa: E402,F401


def test_judge_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "shared-key")
    monkeypatch.setenv("LLM_SECRET_KEY", "shared-secret")
    monkeypatch.setenv("LLM_REGION", "us-west-2")

    from tests.eval.config import JudgeSettings

    settings = JudgeSettings()

    assert settings.judge_provider == "openai"
    assert settings.judge_model == "gpt-4o"
    assert settings.judge_api_key == "shared-key"
    assert settings.judge_secret_key == "shared-secret"
    assert settings.judge_region == "us-west-2"


def test_judge_settings_has_strong_defaults(monkeypatch):
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    from tests.eval.config import JudgeSettings

    # Judge uses a separate strong model, decoupled from the analyst's weak model.
    settings = JudgeSettings(_env_file=None)

    assert settings.judge_provider
    assert settings.judge_model
    assert settings.llm_model_name


def test_judge_config_not_in_application_settings():
    from app.config import Settings

    # Judge config must live in .env.test, not in application Settings (.env).
    assert not hasattr(Settings(), "judge_provider")
    assert not hasattr(Settings(), "judge_model")


def test_bedrock_byok_config_not_in_application_settings():
    from app.config import Settings

    settings = Settings()

    assert not hasattr(settings, "aws_access_key_id")
    assert not hasattr(settings, "aws_secret_access_key")
    assert not hasattr(settings, "aws_region")
    assert not hasattr(settings, "bedrock_notation_model")


def test_application_settings_ignore_deployment_only_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/reqflow",
                "POSTGRES_DB=reqflow",
                "DOCKER_USERNAME=example",
                "WEB_CONCURRENCY=4",
            ]
        ),
        encoding="utf-8",
    )

    from app.config import Settings

    settings = Settings(_env_file=env_file)

    assert settings.database_url == "postgresql+asyncpg://postgres:password@localhost:5432/reqflow"
