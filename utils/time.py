import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    start_time = time.time()
    print("[{0}] is BEGIN in {1}.".format(name, time.asctime(time.localtime(start_time))))
    yield
    finish_time = time.time()
    print("[{0}] is DONE in {1}. Time to calculate: {2:.2f} sec".format(name,
                                                                        time.asctime(time.localtime(finish_time)),
                                                                        finish_time - start_time))
