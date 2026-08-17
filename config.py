import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
CACHE_DIR = ASSETS_DIR / ".cache"
LOGS_DIR = ROOT_DIR / "logs"
SCENARIOS_PATH = ROOT_DIR / "scripts" / "scenarios.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RERENDER_PENALTY = 8.0
EPS_TIE = 0.05
STALE_THRESHOLD_HOURS = 1.0

WEIGHTS = {
    "dialogue": {"w_caption": 1.0, "w_audio": 0.4, "w_rerender": 1.0},
    "music":    {"w_caption": 0.3, "w_audio": 1.0, "w_rerender": 1.0},
    "mixed":    {"w_caption": 0.7, "w_audio": 0.7, "w_rerender": 1.0},
}

EPS_START = 0.15
EPS_END = 0.15
PAIR_TOLERANCE = 1.0

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "3"))

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "gpt-oss-20b")
MULTI_CONFLICT_GROQ_THRESHOLD = int(os.getenv("MULTI_CONFLICT_GROQ_THRESHOLD", "2"))
DECIDER_MAX_TOKENS = int(os.getenv("DECIDER_MAX_TOKENS", "1024"))

FFMPEG_BIN = os.getenv(
    "FFMPEG_BIN",
    r"C:\Users\Dell\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin",
)
FFMPEG_EXE = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
FFPROBE_EXE = os.path.join(FFMPEG_BIN, "ffprobe.exe")

