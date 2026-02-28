"""Tests for config module."""


def test_settings_defaults():
    from scanner.config import Settings

    s = Settings()
    assert s.version == "5.0.0"
    assert s.env == "test"  # set by conftest
    assert s.rate_limit == 50
    assert s.detector_timeout == 30
    assert s.cache_ttl == 86400


def test_settings_is_production():
    from scanner.config import Settings

    s = Settings(env="production")
    assert s.is_production is True

    s2 = Settings(env="development")
    assert s2.is_production is False


def test_cors_origin_list():
    from scanner.config import Settings

    s = Settings(cors_origins="http://a.com, http://b.com")
    assert s.cors_origin_list == ["http://a.com", "http://b.com"]
