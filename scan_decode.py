"""Probe FFmpeg decoding independently of PyTorch; keep streaming errors off pipes."""
from __future__ import annotations

import functools
import subprocess
import sys
import threading
import time

import task_control as tasks


@functools.lru_cache(maxsize=8)
def hardware_methods(ffmpeg):
    try:
        result = tasks.run([ffmpeg, '-hide_banner', '-hwaccels'],
                           capture_output=True, timeout=8)
        return set(result.stdout.decode('utf-8', 'replace').split()) if not result.returncode else set()
    except (OSError, subprocess.TimeoutExpired):
        return set()


def decoder_args(ffmpeg, source, threads, video_filter, frame_bytes, on_status=lambda text: None):
    """Only select hardware after it produced a real frame with our scan filter."""
    methods = hardware_methods(ffmpeg)
    candidates = ['cuda'] + (['d3d11va', 'dxva2'] if sys.platform == 'win32' else [])
    failures = []
    for method in candidates:
        if method not in methods:
            continue
        tasks.check()
        on_status('Проверка декодирования ' + method.upper())
        args = ['-hwaccel', method]
        command = [ffmpeg, '-v', 'error', '-nostdin', '-threads', str(threads),
                   *args, '-i', str(source), '-vf', video_filter,
                   '-frames:v', '1', '-an', '-sn', '-dn', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
        try:
            result = tasks.run(command, capture_output=True, timeout=15)
            if result.returncode == 0 and len(result.stdout) == frame_bytes:
                return args, method, failures
            failures.append(method + ': ' + result.stderr.decode('utf-8', 'replace')[-300:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(method + ': ' + str(exc))
    return [], 'cpu', failures


def error_tail(stream, limit=4096):
    stream.flush()
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell()-limit))
    return stream.read().decode('utf-8', 'replace')


class StreamWatchdog:
    """Terminate a decoder only if it produces no data, not for a slow whole file."""
    def __init__(self, process, timeout=90.):
        self.process, self.timeout = process, timeout
        self.last_data = time.monotonic()
        self.stalled = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        while not self.done.wait(min(1., self.timeout/4)):
            if self.process.poll() is not None:
                return
            if time.monotonic()-self.last_data >= self.timeout:
                self.stalled = True
                try:
                    self.process.kill()
                except OSError:
                    pass
                return

    def touch(self):
        self.last_data = time.monotonic()

    def check(self):
        if self.stalled:
            raise RuntimeError(f'FFmpeg не выдаёт данные {self.timeout:g} с. '
                               'Проверьте доступность исходника и попробуйте локальную копию MP4.')

    def close(self):
        self.done.set()
        self.thread.join(timeout=2.)
