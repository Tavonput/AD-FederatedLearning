import logging
import logging.handlers
import os


LOG_FILE = "../log.log"
GLOBAL_LEVEL = logging.INFO

RESET = "\033[0m"
COLORS = {
    'DEBUG': "\033[36m",    # Cyan
    'INFO': "\033[32m",     # Green
    'WARNING': "\033[33m",  # Yellow
    'ERROR': "\033[31m",    # Red
    'CRITICAL': "\033[41m", # Red background
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, RESET)
        message = super().format(record) 
        return f"{log_color}{message}{RESET}" 


def init_logging() -> None:
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(GLOBAL_LEVEL)

    if not logger.handlers:
        format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

        coloered_formatter = ColoredFormatter(format)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(coloered_formatter)
        logger.addHandler(console_handler)

        normal_formatter = logging.Formatter(format)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(normal_formatter)
        logger.addHandler(file_handler)

    return logger
