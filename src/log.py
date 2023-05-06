import logging

import colorlog


def get_pipeline_logger(component_name: str, component_color: str) -> logging.Logger:
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
    # Remove all handlers associated with the logger    
    logger.handlers.clear()    
    logger.propagate = False
    formatter = colorlog.ColoredFormatter(
        f"%(asctime)s %(levelname)s\t%(log_color)s%(name)s%(reset)s\t%(message)s",
        log_colors={
            'DEBUG':    f'{component_color}',
            'INFO':     f'{component_color}',
            'WARNING':  f'{component_color},bold',
            'ERROR':    f'{component_color},bg_pink',
            'CRITICAL': f'{component_color},bg_red',
        }
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
