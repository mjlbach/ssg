import numpy as np

def retry(times):
    """
    Retry Decorator

    Retries the wrapped function/method `times` times if exception is thrown

    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except:
                    attempt += 1
                    print(
                        "Exception thrown when attempting to run %s, attempt "
                        "%d of %d" % (func, attempt, times)
                    )
            return func(*args, **kwargs)

        return newfn

    return decorator

def convert_to_onehot(array):
    arr = np.eye(len(array))
    return {key: arr[idx] for idx, key in enumerate(array)}
