import warnings
import functools
import time


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # Turn off warnings filter
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f'Call to deprecated function {func.__name__}.',
                      category=DeprecationWarning,
                      stacklevel=2)
        # Reset warning filter
        warnings.simplefilter('default', DeprecationWarning)
        return func(*args, **kwargs)

    return new_func


def timeit(func):
    """This is a decorator which can be used to measure function time spent."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = time.time()
        ret_val = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f'function [{func.__name__}] finished in {int(elapsed_time * 1000)} ms')
        return ret_val

    return new_func
