"""Central configuration for the deck-processing pipeline."""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PAGES_DIR = DATA_DIR / "pages"

MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", 25 * 1024 * 1024))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 30))
MIN_VISIBLE_FONT_SIZE = float(os.getenv("MIN_VISIBLE_FONT_SIZE", "3"))
TEXTLESS_CHARACTER_LIMIT = int(os.getenv("TEXTLESS_CHARACTER_LIMIT", "30"))
RENDER_DPI = int(os.getenv("RENDER_DPI", "200"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_MODEL)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Minimum spacing between starts of requests to the same provider.
PROVIDER_RATE_LIMIT_SECONDS = {
    "gemini": float(os.getenv("GEMINI_RATE_LIMIT_SECONDS", "4")),
    "groq": float(os.getenv("GROQ_RATE_LIMIT_SECONDS", "1")),
}
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
BACKOFF_SECONDS = float(os.getenv("LLM_BACKOFF_SECONDS", "1"))
