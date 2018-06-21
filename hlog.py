from contextlib import contextmanager
import logging
import threading
import time

state = threading.local()
state.path = []

@contextmanager
def task(name, timer=True):
    state.path.append(name)
    begin = time.time()
    yield
    end = time.time()
    if timer:
        print('%s{%0.2fs}' % ('/'.join(state.path), end - begin))
    state.path.pop()

def group(name):
    return task(name, timer=False)

def log(value):
    if isinstance(value, float):
        value = "%0.4f" % value
    print('%s %s' % ('/'.join(state.path), value))

def value(name, value):
    with task(name, timer=False):
        log(value)

def loop(template, coll=None, counter=None, timer=True):
    assert not (coll is None and counter is None)
    if coll is None:
        seq = zip(counter, counter)
    elif counter is None:
        seq = enumerate(coll)
    else:
        assert len(counter) == len(coll)
        seq = zip(counter, coll)
    for i, item in seq:
        with task(template % i, timer):
            yield item

def fn(name, timer=True):
    def wrap(underlying):
        def wrapped(*args, **kwargs):
            with task(name, timer):
                result = underlying(*args, **kwargs)
            return result
        return wrapped
    return wrap
