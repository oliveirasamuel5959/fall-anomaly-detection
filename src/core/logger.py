import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger(name: str, log_file: bool = True) -> logging.Logger:
  logger = logging.getLogger(name)

  if logger.handlers:
    return logger  # avoid duplicate handlers on re-import

  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  # Console handler
  console = logging.StreamHandler(sys.stdout)
  console.setLevel(logging.INFO)
  console.setFormatter(formatter)
  logger.addHandler(console)

  # File handler (optional)
  if log_file:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

  return logger