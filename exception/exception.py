# Import the sys module to get detailed information about exceptions
import sys

# Import the logger object from the logging module
from logger.logger import logger


# Create a custom exception class by inheriting from Python's built-in Exception class
class CustomException(Exception):

    # Constructor of the custom exception class
    # error_message -> Original exception message
    # error_details -> sys module (used to fetch traceback information)
    def __init__(self,error_message,error_details:sys):

        # Store the original error message
        self.error_message = error_message

        # exc_info() returns a tuple:
        # (Exception Type, Exception Object, Traceback Object)
        # We only need the traceback object, so the first two values are ignored using '_'
        _,_,exc_tb = error_details.exc_info()

        # Fall back gracefully if there is no active traceback (e.g. the
        # exception was constructed outside an except block).
        self.lineno = exc_tb.tb_lineno if exc_tb is not None else "?"
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb is not None else "?"

    # This method is automatically called whenever the exception object is printed
    def __str__(self):

        # Return a formatted error message containing
        # file name, line number, and the original error message
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))


# This block executes only when this file is run directly
if __name__=='__main__':

    # Start of the try block
    try:

        # Log a message indicating that execution has entered the try block
        logger.logging.info("Enter the try block")

        # Intentionally generate a ZeroDivisionError
        a=1/0

        # This line will never execute because the exception occurs above
        print("This will not be printed",a)

    # Catch any exception that occurs inside the try block
    except Exception as e:

        # Raise our custom exception and pass:
        # e   -> original exception
        # sys -> used to extract traceback details
        raise CustomException(e,sys)