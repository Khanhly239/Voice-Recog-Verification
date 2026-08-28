"""Standalone ASGI app serving ONLY the public data-collection site.

api.main mounts the whole KYC system: registration, verification, the PoC flows --
and it refuses to start without Postgres, because its lifespan creates tables.
None of that belongs on an internet-facing host whose single job is to let
volunteers record clips:

  * attack surface -- /register and /verify accept biometric uploads and touch the
    speaker DB; a public collection page has no reason to expose them;
  * availability -- collection only writes files under data/public_collect, so it
    should keep working when the KYC database is down or absent;
  * deployment -- this app has no DB dependency at all, so it can run on a cheap
    separate host or container from the verification stack.

Run it instead of api.main when the host is public:

    uvicorn api.collect_app:app --host 0.0.0.0 --port 8080

api.main still serves /gop-du-lieu too, which is handy for local testing on a
machine that already has the full stack running.
"""

import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(
    title="Thu dữ liệu giọng nói & khẩu hình tiếng Việt",
    description="Trang thu thập dữ liệu công khai cho hệ thống xác thực giọng nói và khẩu hình.",
    version="1.0.0",
    docs_url=None,      # nothing to explore here; the page is the interface
    redoc_url=None,
    openapi_url=None,
)

# No CORS middleware on purpose: the page is served from the same origin as the
# API, so cross-origin access would only help someone else's site post clips here.

from api.routes import collect_public  # noqa: E402

app.include_router(collect_public.router, prefix="/api/v1", tags=["Public Data Collection"])


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/gop-du-lieu")


@app.get("/gop-du-lieu", response_class=HTMLResponse, include_in_schema=False)
async def page():
    return HTMLResponse((_TEMPLATE_DIR / "collect_public.html").read_text(encoding="utf-8"))


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok", "enabled": collect_public.ENABLED}
