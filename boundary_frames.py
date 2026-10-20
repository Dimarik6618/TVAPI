"""Small, accurately sought boundary images; no player or optional AI packages."""
from __future__ import annotations

import math
import queue
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


class PreviewCancelled(Exception):
    pass


@dataclass(frozen=True)
class BoundaryFrame:
    png: bytes
    timestamp: float


def decode_boundary_frame(ffmpeg: str, source: Path, position: float,
                          duration: float, *, end: bool = False,
                          cancelled: threading.Event | None = None,
                          timeout: float = 20.) -> BoundaryFrame:
    """Start: first frame at/after the cut. End: last frame before the cut.

    Decode at most a short local interval for the end image. Scale before
    reversing so a 4K/120fps source does not fill memory with original frames.
    showinfo retains the selected frame's actual source time, including VFR.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError('Не найден исходный MP4. Подключите диск или выберите файл заново.')
    if not all(math.isfinite(v) for v in (position, duration)) or duration <= 0:
        raise ValueError('Некорректный таймкод кадра.')
    if not 0 <= position <= duration or (end and position <= 0):
        raise ValueError('Граница должна находиться внутри исходного видео.')
    cancelled = cancelled or threading.Event()
    if cancelled.is_set():
        raise PreviewCancelled()
    seek = max(0., position-1.) if end else position
    interval = position-seek
    args = [ffmpeg, '-hide_banner', '-nostdin', '-loglevel', 'info',
            '-threads', '2', '-ss', f'{seek:.9f}']
    if end:
        args += ['-t', f'{interval:.9f}']
    args += ['-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn']
    filters = []
    if end:
        filters.append(f'trim=end={interval:.9f}')
    filters += ["scale=w='min(360,202*dar)':h='min(202,360/dar)'",
                'setsar=1', 'showinfo=checksum=0']
    if end:
        filters.append('reverse')
    args += ['-vf', ','.join(filters), '-frames:v', '1', '-vsync', '0',
             '-c:v', 'png', '-compression_level', '1', '-threads', '1',
             '-f', 'image2pipe', '-']
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=flags)
    deadline = time.monotonic()+timeout
    try:
        while True:
            if cancelled.is_set():
                raise PreviewCancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError('Кадр не загрузился за 20 секунд. Попробуйте другой таймкод.')
            try:
                png, error = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    if cancelled.is_set():
        raise PreviewCancelled()
    info = error.decode('utf-8', 'replace')
    if process.returncode or not png.startswith(b'\x89PNG\r\n\x1a\n'):
        if not png and not process.returncode:
            raise ValueError('В этой точке нет видеокадра. Сдвиньте границу немного раньше.')
        messages = [line for line in info.splitlines() if not line.startswith('[Parsed_showinfo')]
        raise RuntimeError('Не удалось прочитать кадр: '+ '\n'.join(messages[-3:])[-300:])
    times = [float(value) for value in re.findall(r'pts_time:\s*([-+\d.eE]+)', info)]
    if not times:
        raise RuntimeError('Не удалось определить таймкод видеокадра.')
    relative = max(times) if end else times[0]
    return BoundaryFrame(png, max(0., seek+relative))


class BoundaryPreviewWorker:
    """One decoder, two replaceable requests, bounded cache; never calls Tk."""
    def __init__(self, ffmpeg: str, source: Path, duration: float,
                 decoder=decode_boundary_frame):
        self.ffmpeg, self.source, self.duration = ffmpeg, Path(source), duration
        self.decoder = decoder
        self.results = queue.Queue()
        self._condition = threading.Condition()
        self._pending = OrderedDict()
        self._active = None
        self._closed = False
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='moto-boundary-preview')
        self.thread.start()

    def invalidate(self, side):
        with self._condition:
            self._pending.pop(side, None)
            if self._active and self._active[0] == side:
                self._active[1].set()

    def request(self, side: int, revision: int, position: float):
        with self._condition:
            if self._closed:
                return
            self._pending[side] = (revision, position)
            if self._active and self._active[0] == side:
                self._active[1].set()
            self._condition.notify()

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            if self._active:
                self._active[1].set()
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                side, (revision, position) = self._pending.popitem(last=False)
                cancelled = threading.Event()
                self._active = (side, cancelled)
            frame, error = None, None
            try:
                stat = self.source.stat()
                key = (stat.st_size, stat.st_mtime_ns, side, round(position, 9))
                frame = self._cache.get(key)
                if frame is not None:
                    self._cache.move_to_end(key)
                else:
                    frame = self.decoder(self.ffmpeg, self.source, position,
                                         self.duration, end=bool(side), cancelled=cancelled)
                    if not cancelled.is_set():
                        self._cache[key] = frame
                        self._cache_bytes += len(frame.png)
                        while len(self._cache) > 24 or self._cache_bytes > 5*1024*1024:
                            _, evicted = self._cache.popitem(last=False)
                            self._cache_bytes -= len(evicted.png)
            except PreviewCancelled:
                pass
            except Exception as exc:
                error = str(exc)
            with self._condition:
                self._active = None
                if not self._closed and not cancelled.is_set() and (frame or error):
                    self.results.put((side, revision, frame, error))
