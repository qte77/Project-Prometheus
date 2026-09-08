"""TechnosignatureDB must read its Postgres credentials from the
environment rather than only ever using a hardcoded literal default.

Regression test for xenarch_mk3_script.py:327-332, which previously passed
password='postgres', user='postgres', host='localhost' straight into
psycopg2.connect(...) with no env-var override path at all.
"""
import xenarch_mk3_script


def test_technosignaturedb_reads_credentials_from_env(monkeypatch):
    monkeypatch.setenv("TECHNOSIG_DB_NAME", "sentinel_db")
    monkeypatch.setenv("TECHNOSIG_DB_USER", "sentinel_user")
    monkeypatch.setenv("TECHNOSIG_DB_PASSWORD", "sentinel_password")
    monkeypatch.setenv("TECHNOSIG_DB_HOST", "sentinel_host")
    monkeypatch.setenv("TECHNOSIG_DB_PORT", "6543")

    # xenarch_mk3_script.psycopg2 is stubbed (see tests/conftest.py) as a
    # MagicMock, so .connect(...) calls are recorded rather than opening a
    # real connection.
    xenarch_mk3_script.psycopg2.connect.reset_mock()

    xenarch_mk3_script.TechnosignatureDB()

    _, kwargs = xenarch_mk3_script.psycopg2.connect.call_args
    assert kwargs["dbname"] == "sentinel_db"
    assert kwargs["user"] == "sentinel_user"
    assert kwargs["password"] == "sentinel_password"
    assert kwargs["host"] == "sentinel_host"
    assert kwargs["port"] == 6543


def test_technosignaturedb_falls_back_to_generic_dev_default(monkeypatch):
    monkeypatch.delenv("TECHNOSIG_DB_NAME", raising=False)
    monkeypatch.delenv("TECHNOSIG_DB_USER", raising=False)
    monkeypatch.delenv("TECHNOSIG_DB_PASSWORD", raising=False)
    monkeypatch.delenv("TECHNOSIG_DB_HOST", raising=False)
    monkeypatch.delenv("TECHNOSIG_DB_PORT", raising=False)

    xenarch_mk3_script.psycopg2.connect.reset_mock()

    xenarch_mk3_script.TechnosignatureDB()

    _, kwargs = xenarch_mk3_script.psycopg2.connect.call_args
    # Falling back to the generic Postgres dev default when nothing is set
    # is fine (that's not the hardcoded-credential problem); the point is
    # that an env var, when present, always wins.
    assert kwargs["password"] == "postgres"
    assert kwargs["host"] == "localhost"
