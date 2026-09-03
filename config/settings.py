from pathlib import Path
import os

from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

CFBD_API_KEY = os.getenv("CFBD_API_KEY")

CFBD_BASE_URL = "https://api.collegefootballdata.com"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASE_PATH = PROJECT_ROOT / "data" / "cfb_prediction.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

CURRENT_SEASON = 2026

PROJECT_NAME = "CFB Prediction Centre"