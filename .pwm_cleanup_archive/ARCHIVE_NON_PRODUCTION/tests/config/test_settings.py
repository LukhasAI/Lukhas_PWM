from config import settings


def test_settings_loads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("REDIS_URL", "redis://test:6379")

    reloaded = settings.__class__()
    assert reloaded.OPENAI_API_KEY == "test-key"
    assert reloaded.DATABASE_URL == "sqlite:///test.db"
    assert reloaded.REDIS_URL == "redis://test:6379"
