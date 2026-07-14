"""Bearer auth for both entry points (REST routes and the mounted MCP app):
a single shared API key for agents, or a JWT from the configured OIDC issuer
for humans. The server is a pure OAuth2 resource server -- it validates
tokens, it never runs login flows itself.
"""
import hmac
import json
import logging
import os
import re
import urllib.request

import anyio
import jwt
from fastapi import HTTPException, Request
from mcp.server.auth.provider import AccessToken

AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")
OAUTH_ISSUER = os.environ.get("OAUTH_ISSUER", "")
OAUTH_AUDIENCE = os.environ.get("OAUTH_AUDIENCE", "")
OAUTH_SCOPES = os.environ.get("OAUTH_SCOPES", "").split()
# The externally visible base URL -- used in 401 WWW-Authenticate headers and
# the RFC 9728 metadata, so it must match what clients actually connect to.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:4141").rstrip("/")

# Auth is optional: with neither credential type configured the API serves
# openly (rely on the loopback-only port binding in that case).
AUTH_ENABLED = bool(AGENT_API_KEY or OAUTH_ISSUER)
if not AUTH_ENABLED:
    logging.getLogger(__name__).warning(
        "No auth configured (AGENT_API_KEY / OAUTH_ISSUER) -- /search, /duplicates and /mcp "
        "serve the indexed source code to anyone who can reach them."
    )
if OAUTH_ISSUER and not OAUTH_AUDIENCE:
    raise RuntimeError(
        "OAUTH_ISSUER is set but OAUTH_AUDIENCE is not -- JWTs can't be validated without an audience."
    )


class CombinedVerifier:
    """One verifier for both credential types: the shared agent API key, or a
    JWT from the OIDC issuer. Satisfies the MCP TokenVerifier protocol
    structurally via verify_token()."""

    def __init__(self, api_key, issuer, audience, scopes):
        self.api_key = api_key
        # Compare iss against the raw configured string, never a parsed URL --
        # URL normalization (trailing slash) would break the exact match.
        self.issuer = issuer.rstrip("/")
        # Entra issues aud as either the client ID GUID or api://<client-id>
        # depending on how the scope was requested -- accept a list.
        self.audiences = [a for a in re.split(r"[ ,]+", audience) if a]
        self.scopes = scopes
        self._jwks = None  # PyJWKClient, built lazily from OIDC discovery

    def verify(self, token):
        # Both paths return the *configured* scopes: IdPs name scopes
        # differently in tokens than in requests (Entra: client requests
        # api://<id>/mcp.read, the token's scp says mcp.read), and with a
        # single privilege level the signature/iss/aud/exp checks (or the
        # static key) are the real gates -- scope matching is deliberately
        # vacuous so RequireAuthMiddleware's required_scopes check passes.
        if self.api_key and hmac.compare_digest(token.encode(), self.api_key.encode()):
            return AccessToken(token=token, client_id="agent", scopes=self.scopes)
        if not self.issuer:
            return None
        try:
            claims = jwt.decode(token, self._signing_key(token), algorithms=["RS256", "ES256"],
                                issuer=self.issuer, audience=self.audiences)
        except Exception:
            # Covers bad signature/iss/aud/exp and IdP discovery failures
            # alike: never 500 on an unverifiable token.
            return None
        return AccessToken(token=token, client_id=claims.get("azp", ""), scopes=self.scopes,
                           expires_at=claims.get("exp"), subject=claims.get("sub"), claims=claims)

    def _signing_key(self, token):  # separate method so tests can monkeypatch it
        if self._jwks is None:
            with urllib.request.urlopen(f"{self.issuer}/.well-known/openid-configuration") as r:
                self._jwks = jwt.PyJWKClient(json.load(r)["jwks_uri"])
        return self._jwks.get_signing_key_from_jwt(token).key

    async def verify_token(self, token):
        # verify() blocks (JWKS fetch on cold cache) -- keep it off the event loop.
        return await anyio.to_thread.run_sync(self.verify, token)


verifier = CombinedVerifier(AGENT_API_KEY, OAUTH_ISSUER, OAUTH_AUDIENCE, OAUTH_SCOPES)


def require_auth(request: Request):
    # Sync on purpose: FastAPI runs sync dependencies in its threadpool, so
    # the blocking JWKS fetch inside verify() is safe here too.
    if not AUTH_ENABLED:
        return
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer ") or verifier.verify(header[7:]) is None:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
