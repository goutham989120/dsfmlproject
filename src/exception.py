import sys
import logging
#from src.logger import logging

def error_message_detail(error, error_detail:sys):
    # error_detail is expected to be the 'sys' module so we can call sys.exc_info().
    # However, sys.exc_info() may return (None, None, None) when there is no active exception.
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        # No traceback available. Try to get caller frame info as a best-effort fallback.
        try:
            import inspect
            # inspect.stack()[2] should point to the caller of the code that raised
            frame_info = inspect.stack()[2]
            file_name = frame_info.filename
            line_no = frame_info.lineno
        except Exception:
            file_name = "<unknown>"
            line_no = 0
        error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            file_name, line_no, str(error)
        )
    else:
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error)
        )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
# if __name__ == "__main__":
#         try:
#             a = 1/0
#         except Exception as e:
#             logging.info("Divide by zero error")
#             raise CustomException(e, sys)   