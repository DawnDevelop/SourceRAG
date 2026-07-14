import datetime

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import auth
import main

ISSUER = "https://login.microsoftonline.com/test-tenant/v2.0"
AUDIENCE = "api://test-client-id"


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def client():
    # No lifespan on purpose: REST routes and the 401-before-session path of
    # the MCP mount don't need the session manager running.
    return TestClient(main.app, raise_server_exceptions=False)


def make_jwt(rsa_key, issuer=ISSUER, audience=AUDIENCE, expires_in=3600):
    now = datetime.datetime.now(datetime.timezone.utc)
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": "user-1",
        "azp": "client-1",
        "iat": now,
        "exp": now + datetime.timedelta(seconds=expires_in),
    }
    return jwt.encode(claims, rsa_key, algorithm="RS256")


def make_jwt_verifier(rsa_key, monkeypatch):
    verifier = auth.CombinedVerifier("", ISSUER, AUDIENCE, [])
    monkeypatch.setattr(verifier, "_signing_key", lambda token: rsa_key.public_key())
    return verifier


class TestCombinedVerifierApiKey:
    def test_valid_key_returns_agent_token(self):
        token = auth.verifier.verify("test-agent-key")
        assert token is not None
        assert token.client_id == "agent"

    def test_near_miss_key_rejected(self):
        assert auth.verifier.verify("test-agent-keyX") is None
        assert auth.verifier.verify("test-agent-ke") is None

    def test_key_is_not_a_valid_jwt_fallthrough(self):
        # No issuer configured in tests: anything but the key must fail.
        assert auth.verifier.verify("eyJhbGciOiJSUzI1NiJ9.e30.sig") is None


class TestCombinedVerifierJwt:
    def test_valid_jwt_accepted(self, rsa_key, monkeypatch):
        verifier = make_jwt_verifier(rsa_key, monkeypatch)
        token = verifier.verify(make_jwt(rsa_key))
        assert token is not None
        assert token.subject == "user-1"
        assert token.client_id == "client-1"

    def test_expired_jwt_rejected(self, rsa_key, monkeypatch):
        verifier = make_jwt_verifier(rsa_key, monkeypatch)
        assert verifier.verify(make_jwt(rsa_key, expires_in=-60)) is None

    def test_wrong_audience_rejected(self, rsa_key, monkeypatch):
        verifier = make_jwt_verifier(rsa_key, monkeypatch)
        assert verifier.verify(make_jwt(rsa_key, audience="api://someone-else")) is None

    def test_wrong_issuer_rejected(self, rsa_key, monkeypatch):
        verifier = make_jwt_verifier(rsa_key, monkeypatch)
        assert verifier.verify(make_jwt(rsa_key, issuer="https://evil.example/v2.0")) is None

    def test_wrong_signature_rejected(self, rsa_key, monkeypatch):
        verifier = make_jwt_verifier(rsa_key, monkeypatch)
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        assert verifier.verify(make_jwt(other_key)) is None

    def test_audience_list_accepts_either(self, rsa_key, monkeypatch):
        verifier = auth.CombinedVerifier("", ISSUER, "client-guid, api://test-client-id", [])
        monkeypatch.setattr(verifier, "_signing_key", lambda token: rsa_key.public_key())
        assert verifier.verify(make_jwt(rsa_key, audience="client-guid")) is not None
        assert verifier.verify(make_jwt(rsa_key, audience="api://test-client-id")) is not None


class TestRestAuth:
    def test_health_open(self, client):
        assert client.get("/health").status_code == 200

    def test_search_without_credentials_401(self, client):
        resp = client.get("/search", params={"q": "x"})
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"

    def test_search_with_garbage_token_401(self, client):
        resp = client.get("/search", params={"q": "x"}, headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401

    def test_search_with_basic_scheme_401(self, client):
        resp = client.get("/search", params={"q": "x"}, headers={"Authorization": "Basic dXNlcjpwdw=="})
        assert resp.status_code == 401

    def test_duplicates_with_key_gets_past_auth(self, client):
        # DB is a dead port in tests -- 503 IndexNotReady proves auth passed.
        resp = client.get("/duplicates", headers={"Authorization": "Bearer test-agent-key"})
        assert resp.status_code == 503

    def test_search_with_key_succeeds(self, client, monkeypatch):
        monkeypatch.setattr(main, "search_chunks", lambda *a, **k: [])
        resp = client.get("/search", params={"q": "x"}, headers={"Authorization": "Bearer test-agent-key"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_duplicates_with_jwt_gets_past_auth(self, client, rsa_key, monkeypatch):
        monkeypatch.setattr(auth.verifier, "issuer", ISSUER)
        monkeypatch.setattr(auth.verifier, "audiences", [AUDIENCE])
        monkeypatch.setattr(auth.verifier, "_signing_key", lambda token: rsa_key.public_key())
        resp = client.get("/duplicates", headers={"Authorization": f"Bearer {make_jwt(rsa_key)}"})
        assert resp.status_code == 503

    def test_duplicates_with_expired_jwt_401(self, client, rsa_key, monkeypatch):
        monkeypatch.setattr(auth.verifier, "issuer", ISSUER)
        monkeypatch.setattr(auth.verifier, "audiences", [AUDIENCE])
        monkeypatch.setattr(auth.verifier, "_signing_key", lambda token: rsa_key.public_key())
        resp = client.get("/duplicates",
                          headers={"Authorization": f"Bearer {make_jwt(rsa_key, expires_in=-60)}"})
        assert resp.status_code == 401


class TestAuthDisabled:
    # Auth is optional: with neither AGENT_API_KEY nor OAUTH_ISSUER configured,
    # REST serves openly. (The MCP mount's open mode is decided at import time
    # in main.py and can't be re-tested without reloading the module.)
    def test_rest_open_without_credentials(self, client, monkeypatch):
        monkeypatch.setattr(auth, "AUTH_ENABLED", False)
        monkeypatch.setattr(main, "search_chunks", lambda *a, **k: [])
        assert client.get("/search", params={"q": "x"}).status_code == 200

    def test_rest_enforced_again_when_reenabled(self, client, monkeypatch):
        monkeypatch.setattr(auth, "AUTH_ENABLED", True)
        assert client.get("/search", params={"q": "x"}).status_code == 401


class TestMcpAuth:
    def test_mcp_without_credentials_401_with_resource_metadata(self, client):
        resp = client.post("/mcp/", json={})
        assert resp.status_code == 401
        expected = 'resource_metadata="http://localhost:4141/.well-known/oauth-protected-resource/mcp"'
        assert expected in resp.headers["WWW-Authenticate"]

    def test_protected_resource_metadata_route(self, client):
        resp = client.get("/.well-known/oauth-protected-resource/mcp")
        assert resp.status_code == 200
        meta = resp.json()
        assert meta["resource"] == "http://localhost:4141/mcp"
        assert meta["authorization_servers"] == ["http://localhost:4141"]
        assert meta["bearer_methods_supported"] == ["header"]

    def test_protected_resource_metadata_fallback_route(self, client):
        assert client.get("/.well-known/oauth-protected-resource").status_code == 200
