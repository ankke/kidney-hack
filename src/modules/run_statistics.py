import time


def execution_time(func):

    def inner(*args, **kwargs):
        begin = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("exec time: ", func.__name__, end - begin)
        return res

    return inner
