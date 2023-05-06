import logging


def get_pipeline_logger(component_name: str) -> logging.Logger:
    """Get a logger for a pipeline component.

    Constructs a logger that follows a uniform
    logging format for all pipeline components.


    Args:
        component_name (str): The name of the component.

    Returns:
        logging.Logger: The logger for the component.
    """

    logger = logging.getLogger(component_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger