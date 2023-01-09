import logging
from pathlib import Path


def create_python_logger(name=None, level=logging.DEBUG):
    name = "base_logger" if name is None else name
    pylogger = logging.getLogger(name)
    pylogger.setLevel(level)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f"logs/{name}.log")
    (Path.cwd() / "logs").mkdir(
        parents=True, exist_ok=True
    )  # writes to "./logs" so make sure dir exists
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    pylogger.addHandler(console_handler)
    pylogger.addHandler(file_handler)
    return pylogger
