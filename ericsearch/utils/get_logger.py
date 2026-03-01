import logging
from logging import Logger


def es_get_logger() -> Logger:
    logger = logging.getLogger("ericsearch")
    handler = logging.StreamHandler()
    handler.addFilter(logging.Filter("ericsearch"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[handler],
    )
    return logger
