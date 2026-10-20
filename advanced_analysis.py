"""Optional local AI and telemetry signals for motorcycle video highlights.

Models are loaded only for features explicitly enabled by the user. No video is
sent to a cloud API; model weights may be downloaded on first use.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import task_control as tasks
from hardware import Runtime
from temporal_events import track_events
from local_privacy import configure_local_models

configure_local_models()

MODEL_OBJECTS = 'yolo11n.pt'
MODEL_SOUND = 'MIT/ast-finetuned-audioset-10-10-0.4593'
MODEL_VIDEO = 'microsoft/xclip-base-patch32'
VIDEO_REVISION = 'a2e27a78a2b5d802e894b8a1ef14f3a8ce490963'

PROMPTS = [
    'A first person motorcycle ride on a winding road',
    'A motorcyclist riding past another motorcycle or vehicle',
    'A close vehicle approaching during a motorcycle ride',
    'A scenic mountain or coastal road filmed from a motorcycle',
    'A group of motorcycles riding together',
    'An ordinary straight road filmed from a motorcycle',
    'A stationary motorcycle camera with little happening',
    'A blurry shaky video filmed from a motorcycle',
]
PROMPT_LABELS = ['Извилистая дорога', 'Обгон или встреча', 'Близкий транспорт',
                 'Красивый маршрут', 'Группа мотоциклов']
POSITIVE_PROMPTS = 5
ROAD_USERS = {'car', 'motorcycle', 'bus', 'truck', 'person', 'bicycle'}
SOUND_WORDS = ('horn', 'siren', 'speech', 'shout', 'yell', 'revving', 'accelerat',
               'screaming', 'skidding', 'tire squeal')


@dataclass(frozen=True)
class Options:
    objects: bool = False
    sound: bool = False
    meaning: bool = False
    gps: bool = False
    speech: bool = False

    def enabled(self) -> bool:
        return any((self.objects, self.sound, self.meaning, self.gps))


def ffmpeg_frames(ffmpeg: str, path: Path, start: float, duration: float,
                  count: int = 8, width: int = 448, height: int = 252) -> list[np.ndarray]:
    sample_rate = count / max(.2, duration)
    command = [ffmpeg, '-v', 'error', '-ss', str(start), '-i', str(path),
               '-t', str(max(.2, duration)), '-vf',
               f'fps={sample_rate:.6f},scale={width}:{height},format=rgb24',
               '-frames:v', str(count), '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    result = tasks.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError('Кадры для ИИ: ' + result.stderr.decode('utf-8', 'replace')[-400:])
    size = width * height * 3
    frames = [np.frombuffer(result.stdout[i:i+size], np.uint8).reshape(height, width, 3).copy()
              for i in range(0, len(result.stdout) - size + 1, size)]
    if not frames:
        raise RuntimeError('Для ИИ не удалось получить кадры.')
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return frames


def ffmpeg_audio(ffmpeg: str, path: Path, start: float, duration: float) -> np.ndarray:
    command = [ffmpeg, '-v', 'error', '-ss', str(start), '-i', str(path),
               '-t', str(min(10., duration)), '-vn', '-ac', '1', '-ar', '16000',
               '-f', 'f32le', '-']
    result = tasks.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError('Звук для ИИ: ' + result.stderr.decode('utf-8', 'replace')[-400:])
    return np.frombuffer(result.stdout, dtype='<f4').copy()


def _haversine(lat1, lon1, lat2, lon2) -> float:
    radius = 6_371_000.0
    a1, a2 = math.radians(lat1), math.radians(lat2)
    da, db = a2-a1, math.radians(lon2-lon1)
    h = math.sin(da/2)**2 + math.cos(a1)*math.cos(a2)*math.sin(db/2)**2
    return 2 * radius * math.asin(min(1., math.sqrt(h)))


def sample_seconds(value: str) -> float:
    """ExifTool sample time as seconds or HH:MM:SS.sss."""
    match = re.search(r'\d+(?::\d+){0,2}(?:\.\d+)?', value)
    if not match:
        raise ValueError('No sample time')
    parts = match.group().split(':')
    return sum(float(part) * 60**index for index, part in enumerate(reversed(parts)))


def read_gpx(path: Path) -> list[tuple[float, float, float]]:
    """Return elapsed seconds and coordinates; first GPX point is time zero."""
    import datetime as dt
    root = ET.parse(path).getroot()
    points = []
    start_time = None
    for node in root.iter():
        if node.tag.rsplit('}', 1)[-1] != 'trkpt':
            continue
        time_node = next((child for child in node if child.tag.rsplit('}', 1)[-1] == 'time'), None)
        if time_node is None or not time_node.text:
            continue
        when = dt.datetime.fromisoformat(time_node.text.strip().replace('Z', '+00:00'))
        if start_time is None:
            start_time = when
        try:
            points.append(((when-start_time).total_seconds(), float(node.attrib['lat']),
                           float(node.attrib['lon'])))
        except (KeyError, ValueError):
            continue
    return points


def read_embedded_gps(path: Path) -> list[tuple[float, float, float]]:
    """Use ExifTool's embedded sample extraction if the user installed ExifTool."""
    executable = shutil.which('exiftool')
    if not executable:
        return []
    command = [executable, '-api', 'LargeFileSupport=1', '-ee3', '-n', '-p',
               '${SampleTime}\t${GPSLatitude}\t${GPSLongitude}', str(path)]
    output = tasks.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace', check=False)
    points = []
    for line in output.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        try:
            seconds = sample_seconds(parts[0])
            lat, lon = float(parts[1]), float(parts[2])
            if abs(lat) <= 90 and abs(lon) <= 180:
                points.append((seconds, lat, lon))
        except ValueError:
            pass
    return points


def gps_points(path: Path) -> list[tuple[float, float, float]]:
    points = read_embedded_gps(path)
    if points:
        return sorted(points)
    for candidate in path.parent.iterdir():
        if candidate.stem.lower() == path.stem.lower() and candidate.suffix.lower() == '.gpx':
            return read_gpx(candidate)
    return []


def camera_motion(path: Path) -> list[tuple[float, float]]:
    """Read optional time-stamped accelerometer/gyro triples exported by ExifTool."""
    executable = shutil.which('exiftool')
    if not executable:
        return []
    output = tasks.run(
        [executable, '-api', 'LargeFileSupport=1', '-ee3', '-n', '-p',
         '${SampleTime}\t${Accelerometer}\t${Gyroscope}', str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding='utf-8', errors='replace', check=False)
    samples = []
    for line in output.stdout.splitlines():
        fields = line.split('\t')
        if len(fields) != 3:
            continue
        try:
            when = sample_seconds(fields[0])
        except ValueError:
            continue
        strengths = []
        for raw in fields[1:]:
            values = [float(number) for number in re.findall(r'-?\d+(?:\.\d+)?', raw)]
            if len(values) == 3:
                strengths.append(float(np.linalg.norm(values)))
        if strengths:
            samples.append((when, max(strengths)))
    return samples


def camera_score(samples: list[tuple[float, float]], start: float,
                 end: float) -> tuple[float, str]:
    if len(samples) < 20:
        return 0., ''
    values = np.asarray([value for _, value in samples], dtype=np.float64)
    center = float(np.median(values))
    deviations = np.abs(values - center)
    threshold = float(np.percentile(deviations, 95))
    if threshold < 1e-6:
        return 0., ''
    within = [abs(value-center) for when, value in samples if start <= when <= end]
    if len(within) < 3:
        return 0., ''
    score = min(1., float(np.percentile(within, 85)) / threshold)
    return score, 'Движение по датчикам камеры'


def gps_score(points: list[tuple[float, float, float]], start: float,
              end: float) -> tuple[float, str]:
    data = [p for p in points if start-2 <= p[0] <= end+2]
    if len(data) < 3:
        return 0., ''
    speeds = []
    headings = []
    for a, b in zip(data, data[1:]):
        delta = b[0] - a[0]
        if not .2 <= delta <= 30:
            continue
        distance = _haversine(a[1], a[2], b[1], b[2])
        if distance / delta > 70:  # Ignore implausible GPS jumps.
            continue
        speeds.append(distance / delta)
        dx = (b[2]-a[2]) * math.cos(math.radians((a[1]+b[1])/2))
        dy = b[1]-a[1]
        if distance > 2:
            headings.append(math.atan2(dx, dy))
    if len(speeds) < 2:
        return 0., ''
    changes = [abs(b-a) for a, b in zip(speeds, speeds[1:])]
    turns = [abs(math.atan2(math.sin(b-a), math.cos(b-a)))
             for a, b in zip(headings, headings[1:])]
    acceleration = min(1., max(changes, default=0.) / 7.)
    turn = min(1., max(turns, default=0.) / .8)
    score = .55 * acceleration + .45 * turn
    return score, (('Поворот по GPS' if turn > acceleration else 'Изменение темпа по GPS')
                   if score > .05 else '')


class AdvancedAnalyzer:
    def __init__(self, options: Options, ffmpeg: str,
                 on_status: Callable[[str], None] = lambda message: None, runtime=None):
        self.options = options
        self.runtime = runtime or Runtime.detect()
        self.ffmpeg = ffmpeg
        self.object_model = None
        self.audio_model = None
        self.video_model = None
        self.video_processor = None
        self.torch = None
        self.object_tracks = {}
        self.sound_events = []
        self.warnings: list[str] = []
        self.diagnostics: list[str] = []
        tasks.check()
        if options.objects or options.sound or options.meaning:
            try:
                import torch
                self.torch = torch
                self.runtime.select_torch(torch)
                on_status(self.runtime.summary())
            except ImportError as exc:
                raise RuntimeError('Для ИИ запустите Установить_ИИ.bat.') from exc
        if options.objects:
            on_status('Загрузка модели объектов (первый запуск скачает веса)')
            try:
                from ultralytics import YOLO, settings
                settings.update({'sync':False})
            except ImportError as exc:
                raise RuntimeError('Для объектов запустите Установить_ИИ.bat.') from exc
            self.object_model = YOLO(MODEL_OBJECTS)
        if options.sound or options.meaning:
            try:
                import torch
                from transformers import pipeline, AutoProcessor, AutoModel
            except ImportError as exc:
                raise RuntimeError('Для анализа звука и сюжета запустите Установить_ИИ.bat.') from exc
            self.torch = torch
            if options.sound:
                on_status('Загрузка модели звуковых событий')
                self.audio_model = pipeline('audio-classification', model=MODEL_SOUND,
                                            device=self.runtime.device)
            if options.meaning:
                on_status('Загрузка модели коротких видео')
                self.video_processor = AutoProcessor.from_pretrained(MODEL_VIDEO, revision=VIDEO_REVISION)
                self.video_model = AutoModel.from_pretrained(MODEL_VIDEO, revision=VIDEO_REVISION, use_safetensors=True).eval()
                try:
                    self.video_model.to(self.runtime.device)
                except RuntimeError:
                    self.cpu_fallback()

    def cpu_fallback(self):
        self.runtime.cpu_fallback(self.torch)
        if self.object_model is not None and hasattr(self.object_model,'to'):
            self.object_model.to('cpu')
        if self.video_model is not None:self.video_model.to('cpu')
        if self.audio_model is not None:
            self.audio_model.model.to('cpu')
            self.audio_model.device=self.torch.device('cpu')
        if self.torch and self.torch.cuda.is_available():self.torch.cuda.empty_cache()

    def objects(self, frames: list[np.ndarray], start=0., step=1., reset=True) -> tuple[float, str]:
        if reset:
            self.object_tracks = {}
            predictor = getattr(self.object_model, 'predictor', None)
            for tracker in getattr(predictor, 'trackers', []) if predictor else []:
                tracker.reset()
        largest = 0.
        for index, frame in enumerate(frames):
            tasks.check()
            try:
                results = self.object_model.track(frame[:, :, ::-1].copy(), persist=True,
                    tracker='botsort.yaml', conf=.25, verbose=False, device=self.runtime.device)
            except RuntimeError:
                if self.runtime.device!='cpu':
                    self.cpu_fallback()
                    results = self.object_model.track(frame[:, :, ::-1].copy(), persist=True,
                        tracker='botsort.yaml', conf=.25, verbose=False, device='cpu')
                else:
                    raise
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    name = result.names.get(int(box.cls.item()), '')
                    if name not in ROAD_USERS:
                        continue
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    cx, cy = (x1+x2)/2/frame.shape[1], (y1+y2)/2/frame.shape[0]
                    area = (x2-x1)*(y2-y1)/(frame.shape[0]*frame.shape[1])
                    largest = max(largest, area)
                    if box.id is None:
                        continue
                    tid = int(box.id.item())
                    key = (name, tid)
                    track = self.object_tracks.setdefault(key, {'id': tid, 'class': name, 'points': []})
                    track['points'].append({'time': round(start+index*step, 3), 'x': round(cx, 4),
                                            'y': round(cy, 4), 'area': round(area, 5)})
        events = track_events(self.object_tracks)
        score = max((e['score'] for e in events), default=min(.35, largest*3))
        return score, events[0]['label'] if events else ('Транспорт в кадре' if largest else '')

    def sound(self,path:Path,start:float,duration:float):
        best=(0.,'')
        self.sound_events = []
        offset=0.
        while offset<duration:
            tasks.check()
            signal=ffmpeg_audio(self.ffmpeg,path,start+offset,min(10.,duration-offset))
            if len(signal):
                try:
                    predictions=self.audio_model(signal,top_k=8,function_to_apply='sigmoid')
                except RuntimeError:
                    if self.runtime.device=='cpu':raise
                    self.cpu_fallback();predictions=self.audio_model(signal,top_k=8,function_to_apply='sigmoid')
                relevant=[(float(p['score']) * (.35 if any(w in p['label'].lower() for w in ('revving', 'accelerat')) else 1.),p['label']) for p in predictions
                          if any(word in p['label'].lower() for word in SOUND_WORDS)]
                self.sound_events.extend({'start': round(start+offset, 3), 'end': round(start+min(duration,offset+10), 3),
                                          'label': label, 'score': round(score, 4)} for score,label in relevant if score>=.1)
                if relevant:
                    score,name=max(relevant)
                    if score>best[0]:best=(min(1.,score),'Звук: '+name)
            if offset+10>=duration:break
            offset+=8
        return best

    def meaning_window(self, frames: list[np.ndarray]) -> tuple[float, str]:
        # Some Transformers releases route `videos=` through ProcessorMixin
        # differently and return only tokenizer fields. Use X-CLIP's own
        # image processor and tokenizer explicitly so both inputs are present.
        pixels = self.video_processor.image_processor(frames[:8], return_tensors='pt')
        pixel_values = pixels.get('pixel_values')
        if pixel_values is None:
            pixel_values = pixels.get('pixel_values_videos')
        if pixel_values is None:
            raise RuntimeError('X-CLIP: обработчик кадров не вернул pixel_values.')
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 5 or pixel_values.shape[0] != 1 or pixel_values.shape[1] != 8:
            raise RuntimeError(f'X-CLIP: неожиданная форма кадров {tuple(pixel_values.shape)}.')
        inputs = dict(self.video_processor.tokenizer(PROMPTS, return_tensors='pt', padding=True))
        inputs['pixel_values'] = pixel_values
        inputs={k:v.to(self.runtime.device) for k,v in inputs.items()}
        try:
            with self.torch.inference_mode():
                logits = self.video_model(**inputs).logits_per_video[0]
        except RuntimeError:
            if self.runtime.device=='cpu':raise
            self.cpu_fallback()
            with self.torch.inference_mode():
                logits=self.video_model(**{k:v.to('cpu') for k,v in inputs.items()}).logits_per_video[0]
        good = logits[:POSITIVE_PROMPTS]
        bad = logits[POSITIVE_PROMPTS:]
        margin = (good.max() - bad.max()).clamp(-12, 12)
        score = float(self.torch.sigmoid(margin).item())
        label = PROMPT_LABELS[int(good.argmax().item())]
        return score, 'Сюжет: ' + label

    def meaning(self,frames):
        starts=list(range(0,max(1,len(frames)-7),4))
        if len(frames)>8 and starts[-1]!=len(frames)-8:starts.append(len(frames)-8)
        best=(0.,'')
        for start in starts:
            tasks.check()
            group=list(frames[start:start+8])
            while len(group)<8:group.append(group[-1])
            score,reason=self.meaning_window(group)
            if score>best[0]:best=(score,reason)
        return best

    def enrich(self, path: Path, clips, on_status=lambda message, fraction: None,
               frame_source: Path | None = None):
        self.warnings = []
        self.diagnostics = []
        failed_features: set[str] = set()

        def record(feature: str, clip, exc: Exception):
            if feature in failed_features:
                return
            failed_features.add(feature)
            start = clip.start if clip is not None else 0.
            end = clip.end if clip is not None else 0.
            self.warnings.append(f'{feature}, {start:.1f} с: '
                                 f'{type(exc).__name__}: {exc}')
            self.diagnostics.append(f'{path.name} · {feature} · {start:.2f}–{end:.2f} с\n'
                                    + traceback.format_exc())

        points, motion = [], []
        if self.options.gps:
            try:
                points = gps_points(path)
                motion = camera_motion(path)
            except Exception as exc:
                record('Телеметрия', None, exc)
        for index, clip in enumerate(clips):
            tasks.check()
            on_status(f'ИИ {path.name}: {index+1}/{len(clips)}', index / len(clips))
            scores = {}
            descriptions = {}
            if self.options.objects or self.options.meaning:
                offset=0.
                while offset<clip.end-clip.start and 'Кадры для ИИ' not in failed_features:
                    tasks.check()
                    duration=min(24.,clip.end-clip.start-offset)
                    try:
                        frames=ffmpeg_frames(self.ffmpeg,frame_source or path,
                            clip.start+offset,duration,count=max(8,math.ceil(duration*(6 if self.options.objects else 1))))
                    except Exception as exc:
                        record('Кадры для ИИ',clip,exc);break
                    for enabled,feature,name,method in (
                        (self.options.objects,'Объекты','objects',self.objects),
                        (self.options.meaning,'Сюжет','meaning',self.meaning)):
                        if enabled and feature not in failed_features:
                            try:
                                if name == 'objects':
                                    step = duration / len(frames)
                                    skip = min(len(frames)-1, round(4/step)) if offset else 0
                                    score,reason = self.objects(frames[skip:], clip.start+offset+skip*step,
                                                               step, reset=offset==0)
                                else:
                                    score,reason=method(frames[::6] if self.options.objects else frames)
                                if name not in scores or score>scores[name]:
                                    scores[name],descriptions[name]=score,reason
                            except Exception as exc:record(feature,clip,exc)
                    if all((not self.options.objects or 'Объекты' in failed_features,
                            not self.options.meaning or 'Сюжет' in failed_features)):break
                    if offset+24>=clip.end-clip.start:break
                    offset+=20. # overlap windows so events on a boundary are not lost
                if self.options.objects and 'Объекты' not in failed_features:
                    clip.details['tracks'] = list(self.object_tracks.values())
                    clip.details['object_events'] = track_events(self.object_tracks)
            if self.options.sound and 'Звуки' not in failed_features:
                try:
                    scores['sound'],descriptions['sound']=self.sound(path,clip.start,clip.end-clip.start)
                    clip.details['sound_events'] = list(self.sound_events)
                except Exception as exc:record('Звуки',clip,exc)
            if self.options.gps and sum(clip.start-2 <= t <= clip.end+2
                                        for t, _, _ in points) >= 3:
                scores['gps'], descriptions['gps'] = gps_score(points, clip.start, clip.end)
            if self.options.gps and sum(clip.start <= t <= clip.end
                                        for t, _ in motion) >= 3:
                scores['camera'], descriptions['camera'] = camera_score(motion, clip.start,
                                                                         clip.end)
            tasks.check()
            clip.signals.update(scores)
            if scores:
                clip.score = round(.55 * clip.score + .45 * float(np.mean(list(scores.values()))), 3)
                key = max(scores, key=scores.get)
                if descriptions.get(key):
                    clip.reason = descriptions[key]
        on_status(f'ИИ {path.name}: готово', 1.)
        tasks.check()
        return clips
