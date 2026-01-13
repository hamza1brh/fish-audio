"""
Token server for LiveKit voice call.

Generates access tokens for clients to connect to LiveKit rooms.
Also provides a simple web UI to test the voice call.

Usage:
    python -m voice_call_neymar.server
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from livekit import api

load_dotenv()

app = FastAPI(title="Neymar Voice Call Server")

# LiveKit credentials
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")


@app.get("/")
async def index():
    """Serve the test UI."""
    ui_path = Path(__file__).parent / "test_ui.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return HTMLResponse("<h1>Voice Call Neymar</h1><p>test_ui.html not found</p>")


# #region server log - debug helper
import json as _json
_LOG_PATH = r"c:\Users\PC\Desktop\fish-speech\.cursor\debug.log"
def _dbg(hyp: str, loc: str, msg: str, data: dict = None):
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(_json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data or {}, "timestamp": int(time.time() * 1000)}) + "\n")
# #endregion


@app.get("/token")
async def get_token(room: str = "neymar-room", identity: str = None, delay: float = 0.8):
    """Generate a LiveKit access token with response delay metadata."""
    # #region server log
    _dbg("E", "server.py:get_token", "Token request received", {"room": room, "identity": identity, "delay": delay})
    # #endregion
    if identity is None:
        identity = f"user-{int(time.time())}"
    
    # Clamp delay to valid range
    delay = max(0.3, min(2.0, delay))
    
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity)
    token.with_name(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
    ))
    # Include delay in participant metadata so agent can read it
    token.with_metadata(f'{{"response_delay": {delay}}}')
    
    jwt = token.to_jwt()
    # #region server log
    _dbg("E", "server.py:get_token", "Token generated successfully", {"room": room, "identity": identity})
    # #endregion
    return {"token": jwt, "url": LIVEKIT_URL, "room": room, "identity": identity, "delay": delay}


@app.get("/join")
async def join_redirect(room: str = "neymar-room", identity: str = None):
    """Redirect to LiveKit playground with token pre-filled."""
    from fastapi.responses import RedirectResponse
    import urllib.parse
    
    if identity is None:
        identity = f"user-{int(time.time())}"
    
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity)
    token.with_name(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
    ))
    
    jwt = token.to_jwt()
    
    # Build playground URL with token
    playground_url = f"https://agents-playground.livekit.io/?tab=connection&url={urllib.parse.quote(LIVEKIT_URL)}&token={jwt}"
    return RedirectResponse(url=playground_url)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "livekit_url": LIVEKIT_URL}


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Neymar Voice Call - Token Server")
    print("=" * 60)
    print(f"LiveKit URL: {LIVEKIT_URL}")
    print(f"API Key: {LIVEKIT_API_KEY}")
    print()
    print("Endpoints:")
    print("  GET /        - Test UI")
    print("  GET /token   - Get access token")
    print("  GET /health  - Health check")
    print()
    print("Open http://localhost:8080 in your browser")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
