"""Cooperative cancellation and immediate termination of worker subprocesses."""
import subprocess
import threading


class Cancelled(BaseException):
    """A user stop request, not an analysis error."""


class TaskControl:
    def __init__(self):
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.process = None

    def stop(self):
        self.event.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.kill()
                except OSError:
                    pass


_local = threading.local()


def bind(control):
    _local.control = control


def stopped():
    control = getattr(_local, 'control', None)
    return control is not None and control.event.is_set()


def check():
    if stopped():
        raise Cancelled()


def reap():
    control = getattr(_local, 'control', None)
    if control is not None and control.process is not None:
        if control.event.is_set() and control.process.poll() is None:
            control.process.kill()
        control.process.wait()


def popen(*args, **kwargs):
    control = getattr(_local, 'control', None)
    if control is None:
        return subprocess.Popen(*args, **kwargs)
    with control.lock:
        check()
        process = subprocess.Popen(*args, **kwargs)
        control.process = process
        return process


def run(args, *, check=False, timeout=None, capture_output=False, **kwargs):
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = popen(args, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        process.kill()
        process.wait()
        raise
    if stopped():
        raise Cancelled()
    result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
