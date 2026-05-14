import logging
import sys
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.status import Status

console = Console()


def get_logger(name: str, log_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Rich console handler (replaces plain StreamHandler)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)

    # File handler (plain text, no rich markup)
    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)

    return logger


@contextmanager
def log_task(description: str, success_msg: str):
    """Context manager that shows a spinner while a task runs."""
    with console.status(f"[bold cyan]{description}...", spinner="dots") as status:
        try:
            yield status
            console.log(f"[bold green]✔[/] {success_msg}")
        except Exception as e:
            console.log(f"[bold red]✘[/] Failed: {e}")
            raise