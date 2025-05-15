
from datetime import datetime


def timestamp():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


class color:
    # https://stackoverflow.com/questions/8924173/how-to-print-bold-text-in-python
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def printb(*arg1, **argv):
    print(color.BOLD + " ".join([str(x) for x in arg1]) + color.END, **argv)

from uuid import uuid4
from dataclasses import field
uuid_field = lambda: field(default_factory=lambda: str(uuid4()))
UUID = str


def make_out_dir(out_dir):
    import multiprocessing
    logger = multiprocessing.get_logger()

    if out_dir.exists():
        # prompt the user to overwrite
        warning = f"Output directory {out_dir} already exists, are you sure you want to overwrite? (y) "
        if input(warning).strip().lower() != "y":
            logger.info("Exiting")
            return
        out_dir.rmdir()
    out_dir.mkdir(parents=True, exist_ok=True)
