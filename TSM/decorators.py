import functools
import os
import time
import traceback

def log_exceptions(logger=None, raise_exc=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = f"{func.__name__} error: {e}\n{traceback.format_exc()}"
                if logger:
                    logger.error(msg)
                else:
                    print(msg)
                if raise_exc:
                    raise
        return wrapper
    return decorator

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} cost: {time.time() - start:.2f}s")
        return result
    return wrapper