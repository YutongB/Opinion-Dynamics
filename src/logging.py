import subprocess
import logging
from pathlib import Path
import multiprocessing

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def create_logger(directory: Path):
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.DEBUG)

    if needs_setup := not len(logger.handlers): 
        formatter = logging.Formatter('[%(asctime)s| %(levelname)s| %(processName)s] %(message)s')
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(directory / 'output.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)

        logger.info(f"Logging started")
        logger.info(f"githash={get_git_revision_hash()}")

    return logger