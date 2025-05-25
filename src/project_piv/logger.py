# logger.py
import logging
import os
from datetime import datetime


class Logger:
    def __init__(self, logger_name="ES_MINI_App", log_file_prefix="es_mini_data"):
        """
        Initialize the logger with a specific name and log file prefix.

        Args:
            logger_name (str): Name of the logger.
            log_file_prefix (str): Prefix for the log file name.
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Create a unique log file name with date
        current_date = datetime.now().strftime('%Y%m%d')
        log_file_path = os.path.join(
            logs_dir, f"{log_file_prefix}_{current_date}.log")

        # File handler for logging to a file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Console handler for logging to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and set it to the handlers
        formatter = logging.Formatter('[%(asctime)s | %(levelname)s] [%(name)s.%(module_name)s.%(function_name)s] %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Store extra context for the formatter
        self.extra = {'module_name': '', 'function_name': ''}

    def _log(self, level, class_name, function_name, message):
        """
        Internal method to log messages with the appropriate context.

        Args:
            level: Logging level (e.g., logging.INFO, logging.ERROR)
            class_name (str): Name of the class where the log is generated
            function_name (str): Name of the function where the log is generated
            message (str): Log message
        """
        self.extra['module_name'] = class_name
        self.extra['function_name'] = function_name
        self.logger.log(level, message, extra=self.extra)

    def debug(self, class_name, function_name, message):
        """Log a debug message."""
        self._log(logging.DEBUG, class_name, function_name, message)

    def info(self, class_name, function_name, message):
        """Log an info message."""
        self._log(logging.INFO, class_name, function_name, message)

    def warning(self, class_name, function_name, message):
        """Log a warning message."""
        self._log(logging.WARNING, class_name, function_name, message)

    def error(self, class_name, function_name, message):
        """Log an error message."""
        self._log(logging.ERROR, class_name, function_name, message)

    def critical(self, class_name, function_name, message):
        """Log a critical message."""
        self._log(logging.CRITICAL, class_name, function_name, message)
