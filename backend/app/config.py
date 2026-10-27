from pathlib import Path
import os
from dotenv import load_dotenv
ROOT = Path(os.getenv("CAMPUSPROOF_ROOT", str(Path(__file__).resolve().parents[2])))
load_dotenv(ROOT / ".env")
MODEL_DIR = ROOT / "models"
DB_PATH = os.getenv("CAMPUSPROOF_DB", str(ROOT / "artifacts/reports/campusproof.sqlite3"))
MODEL_PATH = Path(os.getenv("CAMPUSPROOF_MODEL", str(MODEL_DIR / "active.joblib")))

