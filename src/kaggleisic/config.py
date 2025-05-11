import os
from pathlib import Path

from dotenv import load_dotenv
from upath import UPath

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"
REPORTS_DIR = Path(os.getenv("DIR_REPORTS", PROJ_ROOT / "reports"))

DATA_DIR = UPath(os.getenv("DATA_DIR_FSSPEC_URI", f"""file://{PROJ_ROOT / "data"}"""))

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
SUBMISSION_DATA_DIR = DATA_DIR / "submission"
MODELS_DATA_DIR = DATA_DIR / "models"
