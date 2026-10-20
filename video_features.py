"""Cached, local X-CLIP features shared by annotation training and selection."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import task_control as tasks
from hardware import Runtime
from project_store import digest, fingerprint
from local_privacy import configure_local_models

configure_local_models()

VIDEO_MODEL = 'microsoft/xclip-base-patch32'
VIDEO_REVISION = 'a2e27a78a2b5d802e894b8a1ef14f3a8ce490963'
FEATURE_SPEC = {
    'version': 1, 'model': VIDEO_MODEL, 'revision': VIDEO_REVISION,
    'frames': 8, 'window_seconds': 6., 'stride_seconds': 6.,
    'pixels': 'rgb_letterbox_224', 'pooling': 'mean_unit_video_features',
    'transformers': '4.56.2',
}


def window_ranges(start: float, end: float):
    """Bounded consecutive windows, with a full tail window, including short clips."""
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise ValueError('Некорректные границы фрагмента для обучения.')
    width = FEATURE_SPEC['window_seconds']
    if end - start <= width:
        return [(float(start), float(end))]
    starts = list(np.arange(start, end - width + .001, FEATURE_SPEC['stride_seconds']))
    if abs(starts[-1] - (end - width)) > .001:
        starts.append(end - width)
    return [(round(float(s), 4), round(float(s + width), 4)) for s in starts]


def unit_vector(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-8:
        raise ValueError('Модель вернула некорректные признаки видео.')
    return value / np.linalg.norm(value)


class VideoFeatureExtractor:
    """Lazy weights, reusable features, cancellation between bounded FFmpeg/model calls."""
    def __init__(self, ffmpeg, folder, runtime=None, on_status=lambda text: None, archive=None):
        self.ffmpeg = ffmpeg
        self.root = Path(folder) / 'moto_cache' / 'video_features'
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime or Runtime.detect()
        self.on_status = on_status
        self.model = self.processor = self.torch = None
        self.archive = archive
        self.archive_sources = {}

    def load(self):
        if self.model is not None:
            return
        tasks.check()
        try:
            import torch
            import transformers
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError('Для обучения запустите Установить_обучение.bat, затем откройте программу снова.') from exc
        if transformers.__version__ != FEATURE_SPEC['transformers']:
            raise RuntimeError('Обучение использует Transformers 4.56.2. Запустите Установить_обучение.bat.')
        self.torch = torch
        self.runtime.select_torch(torch)
        self.on_status('Загрузка X-CLIP: первый запуск скачает веса; видео остаётся на компьютере.')
        self.processor = AutoImageProcessor.from_pretrained(
            VIDEO_MODEL, revision=VIDEO_REVISION, use_fast=False)
        self.model = AutoModel.from_pretrained(
            VIDEO_MODEL, revision=VIDEO_REVISION, use_safetensors=True).eval()
        try:
            self.model.to(self.runtime.device)
        except RuntimeError:
            self.runtime.cpu_fallback(torch)
            self.model.to('cpu')
        tasks.check()
        self.on_status(self.runtime.summary())

    def _frames(self, source, start, end):
        size = 224
        duration = end - start
        command = [self.ffmpeg, '-v', 'error', '-threads', str(self.runtime.threads),
                   '-ss', f'{start:.4f}', '-i', str(source), '-t', f'{duration:.4f}',
                   '-vf', f'fps={8 / duration:.9f},scale={size}:{size}:force_original_aspect_ratio=decrease,'
                          f'pad={size}:{size}:(ow-iw)/2:(oh-ih)/2,format=rgb24',
                   '-frames:v', '8', '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
        result = tasks.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError('Кадры для обучения: ' + result.stderr.decode('utf-8', 'replace')[-300:])
        frame_bytes = size * size * 3
        frames = [np.frombuffer(result.stdout[i:i + frame_bytes], np.uint8).reshape(size, size, 3)
                  for i in range(0, len(result.stdout) - frame_bytes + 1, frame_bytes)]
        if not frames:
            raise ValueError('Нет кадров в размеченном отрезке. Проверьте исходник и таймкоды.')
        while len(frames) < 8:
            frames.append(frames[-1])
        return frames

    def _infer(self, frames):
        pixels = self.processor(frames, return_tensors='pt', do_resize=False,
                                do_center_crop=False)['pixel_values']
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(0)
        if tuple(pixels.shape[:2]) != (1, 8):
            raise RuntimeError('X-CLIP: неправильное число кадров.')
        try:
            with self.torch.inference_mode():
                value = self.model.get_video_features(pixel_values=pixels.to(self.runtime.device))
        except RuntimeError:
            if self.runtime.device == 'cpu':
                raise
            self.runtime.cpu_fallback(self.torch)
            self.model.to('cpu')
            with self.torch.inference_mode():
                value = self.model.get_video_features(pixel_values=pixels.to('cpu'))
        tasks.check()
        # This application pins the Tensor-returning 4.56.2 implementation.
        vector = unit_vector(value[0].detach().float().cpu().numpy())
        if vector.shape != (512,):
            raise RuntimeError(f'X-CLIP: ожидалось 512 признаков, получено {vector.shape}.')
        return vector

    def window(self, source, start, end, source_fingerprint=None):
        tasks.check()
        identity = source_fingerprint or fingerprint(Path(source))
        sid = None
        if self.archive:
            key_source = digest([identity['size'], identity['sample']])
            sid = self.archive_sources.get(key_source)
            if sid is None:
                sid = self.archive.register(source)
                self.archive_sources[key_source] = sid
            saved = self.archive.window(sid, start, end)
            if saved is not None:
                return saved
        key = digest([FEATURE_SPEC, identity, round(start, 4), round(end, 4)])
        path = self.root / (key + '.npz')
        try:
            with np.load(path, allow_pickle=False) as data:
                vector = data['vector']
                if vector.shape == (512,) and np.isfinite(vector).all() and np.linalg.norm(vector) > .99:
                    if sid:
                        self.archive.save_window(sid, start, end, vector)
                    return unit_vector(vector)
        except (OSError, ValueError, KeyError, EOFError):
            pass
        self.load()
        vector = self._infer(self._frames(source, start, end))
        tasks.check()
        fd, name = tempfile.mkstemp(suffix='.npz', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, vector=vector)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        if sid:
            self.archive.save_window(sid, start, end, vector)
        return vector

    def extract(self, source, start, end):
        identity = fingerprint(Path(source))
        features = [self.window(source, a, b, identity) for a, b in window_ranges(start, end)]
        return unit_vector(np.mean(features, axis=0))
