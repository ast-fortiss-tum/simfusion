import logging as log
import logging.config
from importlib import reload

def setup_logging(log_to: str = None):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates or stale configs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to:
        file_handler = logging.FileHandler(log_to, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.info("Logging setup completed.")

def disable_pymoo_warnings():
    from pymoo.config import Config

    Config.warnings['not_compiled'] = False