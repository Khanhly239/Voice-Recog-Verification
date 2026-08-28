import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ASV_System.db_utils import create_tables_if_not_exists

    logging.info("Initializing database tables...")
    create_tables_if_not_exists()
    logging.info("Video KYC API ready.")
    yield
    logging.info("Shutting down Video KYC API.")


app = FastAPI(
    title="Video KYC Speaker Verification API",
    description=(
        "Multi-layer speaker verification system for Video KYC with "
        "anti-spoofing, replay detection, AI voice detection, face liveness, "
        "and lip-sync verification."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import register, challenge, verify, auth
from api.routes import ws_asr
from api.routes import asr_sherpa
from api.routes import collect, collect_public

app.include_router(register.router, prefix="/api/v1", tags=["Registration"])
app.include_router(auth.router, prefix="/api/v1", tags=["Auth"])
app.include_router(challenge.router, prefix="/api/v1", tags=["Challenge"])
app.include_router(verify.router, prefix="/api/v1", tags=["Verification"])
app.include_router(asr_sherpa.router, prefix="/api/v1", tags=["ASR (Sherpa VI)"])
app.include_router(collect.router, prefix="/api/v1", tags=["Data Collection"])
app.include_router(collect_public.router, prefix="/api/v1", tags=["Public Data Collection"])
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.include_router(ws_asr.router)


@app.get("/", tags=["System"], response_class=HTMLResponse)
async def root():
    html = (_TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/voice-kyc", tags=["System"], response_class=HTMLResponse)
async def voice_kyc_page():
    html = (_TEMPLATE_DIR / "voice_kyc.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/collect-digits", tags=["System"], response_class=HTMLResponse)
async def collect_digits_page():
    html = (_TEMPLATE_DIR / "collect_digits.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/gop-du-lieu", tags=["System"], response_class=HTMLResponse)
async def public_collect_page():
    """Public crowd-sourcing page for lip-shape + voice clips (Vietnamese)."""
    html = (_TEMPLATE_DIR / "collect_public.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/v1/health", tags=["System"])
async def health_check():
    from sonic_cipher.sherpa_vi_asr import is_sherpa_model_available

    return {
        "status": "ok",
        "version": "2.0.0",
        "sherpa_vi_asr": is_sherpa_model_available(),
    }
