# Import the logging module to create log messages
import logging
import os
from datetime import datetime


# Handlers list — stdout logging always happens (works everywhere, including Vercel)
handlers = [logging.StreamHandler()]

# Only attempt file-based logging when NOT running on Vercel
# (Vercel's filesystem is read-only except /tmp, and doesn't persist between requests anyway)
if not os.environ.get("VERCEL"):

    # Create a log file name based on the current date and time.
    # Example: 07_27_2026_21_35_1753632310.log
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

    # Create the path where the log folder will be created.
    # os.getcwd() returns the current working directory.
    #
    # Example:
    # Current Directory:
    # D:\Projects\ML_Project
    #
    # logs_path becomes:
    # D:\Projects\ML_Project\logs
    logs_path = os.path.join(os.getcwd(), "logs")

    # Create the directory if it doesn't already exist.
    # exist_ok=True means: if the directory already exists, don't raise an error.
    os.makedirs(logs_path, exist_ok=True)

    # Create the complete path of the log file.
    #
    # Example:
    # D:\Projects\ML_Project\logs\07_27_2026_21_35_1753632310.log
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

    handlers.append(logging.FileHandler(LOG_FILE_PATH))


# Configure the logging system
logging.basicConfig(

    # Format of each log message
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    # Log only INFO and above
    level=logging.INFO,

    # Where logs get sent: stdout always, plus a file when running locally
    handlers=handlers,
)

# Shared logger object for the whole project. Exposes the configured Logger as
# `.logging` (so `logger.logging.info(...)` works) and forwards standard calls
# like `logger.info(...)` to it.
class ProjectLogger:

    logging = logging.getLogger("superconductivity_stress")

    def __getattr__(self, name):
        return getattr(self.logging, name)


logger = ProjectLogger()