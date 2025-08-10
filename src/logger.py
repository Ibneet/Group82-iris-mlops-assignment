from loguru import logger
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

logger.remove()
logger.add(LOG_FILE, rotation="1 week", retention="4 weeks", enqueue=True, backtrace=True, diagnose=False)
logger.add(lambda msg: print(msg, end=""))

__all__ = ["logger"]
