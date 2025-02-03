import logging

# Configure logging settings
logging.basicConfig(
    filename="view_logs",  # Log will be written to 'view_logs' file
    level=logging.INFO,  # Log all INFO level and higher messages
    format='%(asctime)s - %(message)s'
)

def log_message(message):
    """Log an INFO message."""
    logging.info(message)

def log_error(message):
    """Log an ERROR message."""
    logging.error(message)
