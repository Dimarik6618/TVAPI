#!/usr/bin/env python3
"""Local motorcycle ride highlight finder and vertical short exporter."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from advanced_analysis import Options, AdvancedAnalyzer
from color_grade import ColorGrade
from preferences import PreferenceStore
import task_control as tasks
from neon_widgets import RoundedCard, RoundedButton, rounded_rect, blend, ui_font
from project_store import CacheStore, ProjectStore, fingerprint, digest, atomic_json
from hardware import Runtime
from learning import LearningStore, EventSelector, EXCLUDED_REASONS
from temporal_events import smart_bounds, diverse_order, cosine

FPS_SAMPLE = 2
WIDTH, HEIGHT = 160, 90


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        from shutil import which
        executable = which('ffmpeg')
        if executable:
            return executable
        raise RuntimeError('Установите зависимости: pip install -r requirements.txt')


def run_capture(args: list[str]) -> subprocess.CompletedProcess:
    return tasks.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def video_info(path: Path, ffmpeg: str) -> tuple[float, bool]:
    if not Path(path).is_file():
        raise FileNotFoundError('Не найден исходный MP4: ' + str(path) + '. Подключите диск или выберите файл заново.')
    probe = run_capture([ffmpeg, '-hide_banner', '-i', str(path)])
    info = probe.stderr.decode('utf-8', 'replace')
    match = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', info)
    if not match:
        raise ValueError('Не удалось определить продолжительность видео: ' + info[-500:])
    duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
    if not re.search(r'Stream #.*: Video:', info):
        raise ValueError('В файле нет видеодорожки.')
    return duration, bool(re.search(r'Stream #.*: Audio:', info))


def motion_samples(path: Path, ffmpeg: str, duration: float,
                   on_progress=lambda fraction: None, threads: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    args = [ffmpeg, '-v', 'error', '-threads', str(threads), '-i', str(path), '-vf',
            f'fps={FPS_SAMPLE},scale={WIDTH}:{HEIGHT},format=gray',
            '-an', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    proc = tasks.popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = WIDTH * HEIGHT
    previous = None
    scores = []
    quality = []
    signatures = []
    try:
        while True:
            tasks.check()
            frame = proc.stdout.read(frame_bytes)
            if len(frame) != frame_bytes:
                break
            current = np.frombuffer(frame, dtype=np.uint8).reshape(HEIGHT, WIDTH)
            # A blurred or nearly black frame carries less useful visual detail.
            sharpness = (np.abs(np.diff(current.astype(np.int16), axis=0)).mean()
                         + np.abs(np.diff(current.astype(np.int16), axis=1)).mean())
            brightness = min(1.0, float(current.mean()) / 45.0)
            quality.append(float(sharpness * brightness))
            signatures.append(current.reshape(9,10,16,10).mean(axis=(1,3)).ravel()/255.)
            # Take the median of local changes so one bright flash does not dominate.
            if previous is not None:
                changes = np.abs(current.astype(np.int16) - previous.astype(np.int16))
                blocks = changes.reshape(9, 10, 16, 10).mean(axis=(1, 3))
                scores.append(float(np.median(blocks)))
            else:
                scores.append(0.0)
            previous = current.copy()
            if len(scores) % 12 == 0:
                on_progress(min(1.0, len(scores) / max(1, duration * FPS_SAMPLE)))
        proc.stdout.close()
        stderr = proc.stderr.read().decode('utf-8', 'replace') if proc.stderr else ''
        return_code = proc.wait()
        tasks.check()
        if return_code != 0:
            raise RuntimeError('Ошибка чтения видеокадров: ' + stderr[-500:])
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout:proc.stdout.close()
        if proc.stderr:proc.stderr.close()
    if not scores:
        raise ValueError('Видео не содержит кадров.')
    on_progress(1.0)
    return (np.asarray(scores, dtype=np.float32),
            np.asarray(quality, dtype=np.float32),
            np.asarray(signatures,dtype=np.float32))


def audio_samples(path: Path, ffmpeg: str, seconds: int,
                  on_progress=lambda fraction: None) -> np.ndarray:
    args = [ffmpeg, '-v', 'error', '-i', str(path), '-vn', '-ac', '1',
            '-ar', '8000', '-f', 's16le', '-']
    proc = tasks.popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    result = []
    try:
        while True:
            tasks.check()
            chunk = proc.stdout.read(8000 * 2)
            if not chunk:
                break
            samples = np.frombuffer(chunk[:len(chunk) // 2 * 2], dtype='<i2').astype(np.float32)
            if len(samples):
                result.append(float(np.sqrt(np.mean(samples * samples))))
                if len(result) % 10 == 0:
                    on_progress(min(1.0, len(result) / max(1, seconds)))
        proc.stdout.close()
        stderr = proc.stderr.read().decode('utf-8', 'replace') if proc.stderr else ''
        return_code = proc.wait()
        tasks.check()
        if return_code != 0:
            raise RuntimeError('Ошибка чтения аудио: ' + stderr[-500:])
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout:proc.stdout.close()
        if proc.stderr:proc.stderr.close()
    on_progress(1.0)
    return np.asarray(result[:seconds], dtype=np.float32)


def normalize(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values
    low, high = np.percentile(values, [15, 90])
    if high - low < 1e-5:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def local_change(values: np.ndarray, radius: int = 12) -> np.ndarray:
    """How much a second differs from its nearby routine riding baseline."""
    if not len(values):
        return values
    baseline = np.empty(len(values), dtype=np.float32)
    # Chunked medians avoid one Python/NumPy call per second of a long ride.
    for i in range(0,len(values),4096):
        tasks.check()
        stop = min(len(values),i+4096)
        lo, hi = max(0,i-radius), min(len(values),stop+radius)
        chunk = values[lo:hi].astype(float)
        padded = np.pad(chunk,(max(0,radius-i),max(0,stop+radius-len(values))),
                        constant_values=np.nan).astype(float)
        windows = np.lib.stride_tricks.sliding_window_view(padded,2*radius+1)
        baseline[i:stop] = np.nanmedian(windows,axis=1)
    return np.maximum(0, values - baseline)


@dataclass
class Clip:
    start: float
    end: float
    score: float
    selected: bool = True
    reason: str = ''
    reviewed: bool = False
    signals: dict[str, float] = field(default_factory=dict)
    signature: list[float] = field(default_factory=list)
    edit_origin: list[float] | None = None
    embedding: list[float] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    decision_reason: str = ''


def event_clips(activity: np.ndarray, detail: np.ndarray,
                visual: np.ndarray, sound: np.ndarray,
                duration: float, count: int) -> list[Clip]:
    """Expand activity peaks to quiet boundaries, retaining setup and aftermath."""
    if duration <= 10:
        return [Clip(0., duration, float(np.mean(activity)), reason='Вся запись')]
    # Edge padding preserves events at the beginning and end of a recording.
    smooth = np.convolve(np.pad(activity, (1, 1), mode='edge'),
                         np.ones(3) / 3, mode='valid')
    peaks = [i for i in range(len(smooth))
             if (i == 0 or smooth[i] >= smooth[i-1])
             and (i == len(smooth)-1 or smooth[i] >= smooth[i+1])]
    chosen = []
    for peak in sorted(peaks, key=lambda i: smooth[i] * detail[i], reverse=True):
        if any(c.start-3 <= peak <= c.end+3 for c in chosen):
            continue
        threshold = max(.08, float(smooth[peak]) * .45)
        left = right = peak
        # Three quiet seconds terminate the event; isolated dips do not split it.
        for direction in (-1, 1):
            quiet = 0
            edge = peak
            for step in range(1, 43):
                index = peak + direction * step
                if not 0 <= index < len(smooth):
                    break
                quiet = quiet + 1 if smooth[index] < threshold else 0
                if quiet == 3:
                    break
                edge = index
            if direction < 0:
                left = edge
            else:
                right = edge
        start, end = max(0., left-3.), min(duration, right+4.)
        # Very brief events still need context. Limit broad plateaus to 90 s.
        if end-start < 10:
            start = max(0., peak-5.)
            end = min(duration, start+10.)
            start = max(0., end-10.)
        end = min(end, start+90.)
        if any(not (end+3 <= c.start or start >= c.end+3) for c in chosen):
            continue
        lo, hi = int(start), min(len(activity), math.ceil(end))
        segment = activity[lo:hi]
        rating = (.65 * float(np.mean(segment)) + .35 * float(np.max(segment)))
        rating *= float(np.mean(detail[lo:hi]))
        v, s = float(np.mean(visual[lo:hi])), float(np.mean(sound[lo:hi]))
        reason = 'Движение и звук' if v > .2 and s > .2 else (
            'Изменение звука' if s > v and s > .1 else 'Изменение в кадре')
        chosen.append(Clip(start, end, round(rating, 3), reason=reason))
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda c: c.start)


def choose_clips(motion: np.ndarray, audio: np.ndarray, duration: float,
                 clip_length: int, count: int,
                 quality: np.ndarray | None = None) -> list[Clip]:
    seconds = max(1, math.ceil(duration))
    motion_sec = np.asarray([np.mean(motion[i * FPS_SAMPLE:(i + 1) * FPS_SAMPLE])
                             if len(motion[i * FPS_SAMPLE:(i + 1) * FPS_SAMPLE]) else 0
                             for i in range(seconds)], dtype=np.float32)
    sound = np.zeros(seconds, dtype=np.float32)
    sound[:min(seconds, len(audio))] = audio[:seconds]
    detail = np.ones(seconds, dtype=np.float32)
    if quality is not None and len(quality):
        detail = np.asarray([np.mean(quality[i * FPS_SAMPLE:(i + 1) * FPS_SAMPLE])
                             if len(quality[i * FPS_SAMPLE:(i + 1) * FPS_SAMPLE]) else 0
                             for i in range(seconds)], dtype=np.float32)
        detail = 0.70 + 0.30 * normalize(detail)
    # Contrast against local baseline helps avoid choosing constant highway motion
    # or steady wind solely because both are loud/high-motion throughout a ride.
    visual_event = normalize(local_change(motion_sec))
    audio_event = normalize(local_change(sound)) if len(audio) else np.zeros(seconds)
    activity = (0.30 * normalize(motion_sec) + 0.50 * visual_event
                + (0.20 * audio_event if len(audio) else 0))
    if clip_length == 0:
        return event_clips(activity, detail, visual_event, audio_event, duration, count)
    length = min(float(clip_length), duration)
    if duration <= length:
        return [Clip(0, duration, float(np.mean(activity)), reason='Вся запись')]
    window = max(1, math.ceil(length))
    candidates = []
    for start in range(0, max(1, math.floor(duration - length)) + 1, 2):
        segment = activity[start:min(seconds, start + window)]
        # Favor sustained activity over a single abrupt cut.
        sustained = float(np.mean(segment))
        interesting = float(np.mean(np.sort(segment)[len(segment)//2:]))
        rating = (0.65 * sustained + 0.35 * interesting) * float(
            np.mean(detail[start:min(seconds, start + window)]))
        visual = float(np.mean(visual_event[start:min(seconds, start+window)]))
        sound_event = float(np.mean(audio_event[start:min(seconds, start+window)]))
        reason = 'Движение и звук' if visual > .2 and sound_event > .2 else (
            'Изменение звука' if sound_event > visual and sound_event > .1 else
            'Изменение в кадре')
        candidates.append((rating, float(start), reason))
    if not candidates or candidates[-1][1] < duration - length - 1:
        start = duration - length
        candidates.append((float(np.mean(activity[int(start):])), start, 'Конец записи'))
    chosen = []
    for rating, start, reason in sorted(candidates, reverse=True):
        end = min(duration, start + length)
        gap = min(5, length * .2)
        if all(end + gap <= item.start or start >= item.end + gap for item in chosen):
            chosen.append(Clip(round(start, 2), round(end, 2), round(rating, 3), reason=reason))
            if len(chosen) == count:
                break
    return sorted(chosen, key=lambda item: item.start)


def prepare_scan_scores(data):
    """Derive ride-wide scales once per scan, retaining the original score formula.

    Derived arrays live only in memory. A new scan (including a disk-cache hit)
    prepares fresh scales; persisted caches and event timelines keep raw signals.
    Treat returned scan arrays as immutable during an analysis pass.
    """
    size = max(1, math.ceil(len(data['motion']) / FPS_SAMPLE))
    motion = np.asarray([np.mean(data['motion'][i*FPS_SAMPLE:(i+1)*FPS_SAMPLE])
                         for i in range(size)])
    sound = np.zeros(size)
    sound[:min(size, len(data['audio']))] = data['audio'][:size]
    activity = .3*normalize(motion) + .5*normalize(local_change(motion))
    if len(data['audio']):
        activity += .2*normalize(local_change(sound))
    return {**data, '_score_timeline': {
        'activity': activity,
        'visual': normalize(data['motion']),
        'audio': normalize(data['audio']),
    }}


def scan_video(path, ffmpeg, on_progress, proxy=None, cache=None, runtime=None):
    key = cache.key(path,proxy) if cache else None
    cached = cache.load_scan(key) if cache else None
    if cached:
        on_progress(.95,'Сканирование из кэша')
        return cached[0],prepare_scan_scores(cached[1]),key
    duration, has_audio = video_info(path,ffmpeg)
    runtime = runtime or Runtime.detect()
    stage = 'Сканирование LRF' if proxy else 'Сканирование движения'
    motion,quality,signature = motion_samples(proxy or path,ffmpeg,duration,
        lambda f:on_progress(.02+f*.70,stage), runtime.threads)
    audio = (audio_samples(path,ffmpeg,math.ceil(duration),
        lambda f:on_progress(.72+f*.23,'Анализ звука')) if has_audio else np.array([]))
    tasks.check()
    data = dict(motion=motion,quality=quality,audio=audio,signature=signature)
    if cache:
        cache.save_scan(key,duration,**data)
    return duration,prepare_scan_scores(data),key


def describe_clip(clip, data):
    scales = data.get('_score_timeline')
    if scales is None:
        scales = prepare_scan_scores(data)['_score_timeline']
    lo,hi=int(clip.start*FPS_SAMPLE),max(1,math.ceil(clip.end*FPS_SAMPLE))
    audio=data['audio'][int(clip.start):math.ceil(clip.end)]
    quality=data['quality'][lo:hi]
    motion=data['motion'][lo:hi]
    sig=data['signature'][lo:hi]
    clip.signature=(np.mean(sig,axis=0).round(4).tolist() if len(sig) else [])
    clip.signals.update(baseline=clip.score,duration=min(2.,(clip.end-clip.start)/90.),
        quality=float(np.clip(np.mean(quality)/30.,0,1)) if len(quality) else 0.,
        visual=float(np.mean(scales['visual'][lo:hi])) if len(motion) else 0.,
        audio=float(np.mean(scales['audio'][int(clip.start):math.ceil(clip.end)])) if len(audio) else 0.)


def baseline_at(data, start, end):
    scales = data.get('_score_timeline')
    if scales is None:
        scales = prepare_scan_scores(data)['_score_timeline']
    activity = scales['activity']
    segment=activity[int(start):max(int(start)+1,math.ceil(end))]
    return round(.65*float(np.mean(segment))+.35*float(np.max(segment)),3)


def select_best(clips, count, threshold=.22, deduplicate=True, prior=None):
    selected=[]
    candidates = diverse_order(clips, prior or []) if deduplicate and any(c.embedding for c in clips) else sorted(clips,key=lambda c:c.score,reverse=True)
    for clip in candidates:
        if ('learned_event' in clip.signals and clip.signals['learned_event'] < .5
                and clip.signals.get('sound',0.) < .8 and clip.signals.get('speech',0.) < .65):
            continue
        if clip.score < threshold:
            continue
        # Candidates from the visual sweep and motion scan can describe the same
        # event. Keep one interval per moment even when visual dedup is disabled.
        if any(min(clip.end,old.end)>max(clip.start,old.start) for old in selected):
            continue
        repeated=False
        if deduplicate and clip.embedding:
            repeated = any((similarity := cosine(clip.embedding, old.embedding)) is not None and similarity > .97
                           for old in selected + (prior or []))
        if repeated:
            continue
        if deduplicate and clip.signature:
            a=np.asarray(clip.signature);a=a-a.mean()
            for old in selected + (prior or []):
                if len(old.signature)!=len(a) or old.reason.split(' · ')[0]!=clip.reason.split(' · ')[0]:
                    continue
                b=np.asarray(old.signature);b=b-b.mean()
                similarity=float(a@b/max(1e-9,np.linalg.norm(a)*np.linalg.norm(b)))
                ratio=(clip.end-clip.start)/max(.1,old.end-old.start)
                if similarity>.985 and .65<ratio<1.55:
                    repeated=True;break
        if not repeated:
            selected.append(clip)
            if len(selected)>=count:break
    return sorted(selected,key=lambda c:c.start)


def analyze(path: Path, clip_length: int, count: int,
            on_progress=lambda fraction, message: None,
            motion_path: Path | None = None, cache=None, runtime=None):
    duration,data,key=scan_video(path,ffmpeg_exe(),on_progress,motion_path,cache,runtime)
    clips=choose_clips(data['motion'],data['audio'],duration,clip_length,count,data['quality'])
    for clip in clips:describe_clip(clip,data)
    on_progress(1.,'Анализ завершён')
    return duration,clips


def mp4_files(path: Path) -> list[Path]:
    """Accept one MP4 or the direct MP4 children of a folder."""
    if path.is_file() and path.suffix.lower() == '.mp4':
        return [path]
    if path.is_dir():
        return sorted((p for p in path.iterdir()
                       if p.is_file() and p.suffix.lower() == '.mp4'),
                      key=lambda p: p.name.lower())
    return []


def paired_proxy(path: Path, ffmpeg: str) -> tuple[Path | None, str | None]:
    """Use an equally timed DJI LRF next to the source MP4, when available."""
    proxy = next((candidate for candidate in path.parent.iterdir()
                  if candidate.is_file() and candidate.stem.lower() == path.stem.lower()
                  and candidate.suffix.lower() == '.lrf'), None)
    if proxy is None:
        return None, None
    try:
        source_duration, _ = video_info(path, ffmpeg)
        proxy_duration, _ = video_info(proxy, ffmpeg)
        if abs(source_duration - proxy_duration) > max(.5, min(2., source_duration * .001)):
            return None, (f'LRF {proxy.name}: длительность отличается от MP4 '
                          f'({proxy_duration:.1f} / {source_duration:.1f} с); использован MP4')
    except Exception as exc:
        return None, f'LRF {proxy.name} недоступен ({exc}); использован MP4'
    return proxy, None


def analyze_batch(paths: list[Path], clip_length: int, count: int,
                  on_progress=lambda fraction, message: None,
                  options: Options | None = None,
                  preferences_folder: Path | None = None,
                  use_proxies: bool = True, threshold: float = .22,
                  deduplicate: bool = True, profile: str = 'auto',
                  resume: dict | None = None, on_complete=lambda path, clips: None,
                  model_pool: dict | None = None, learning_folder: Path | None = None,
                  source_loader=None, source_origin=None, after_complete=None, capture_memory=False):
    rows,errors,diagnostics=[],[],[]
    options=options or Options()
    preferences=PreferenceStore(preferences_folder) if preferences_folder else None
    model=preferences.fitted() if preferences else None
    cache=CacheStore(preferences_folder) if preferences_folder else None
    runtime=Runtime.detect(profile)
    pool=model_pool if model_pool is not None else {}
    selector=None
    selection_token='off'
    learning_folder=learning_folder or preferences_folder
    archive = None
    if capture_memory and learning_folder:
        from event_archive import EventArchive
        archive = EventArchive(learning_folder)
    if learning_folder:
        try:
            learning=LearningStore(learning_folder)
            if learning.enabled():
                selection_token='unavailable'
                token=learning.token()
                pool_key='trained-event:'+digest([token,profile])
                for key in list(pool):
                    if key.startswith('trained-event:') and key!=pool_key:pool.pop(key)
                selector=pool.get(pool_key)
                if selector is None:
                    selector=EventSelector(learning,ffmpeg_exe(),runtime,lambda text:on_progress(0.,text))
                    pool[pool_key]=selector
                selection_token=token
        except (OSError,ValueError,KeyError,TypeError) as exc:
            if paths:errors.append((paths[0],f'Обученная модель: {exc}; используется обычный отбор.'))
    on_progress(0.,runtime.summary())
    for index,path in enumerate(paths):
        label=f'{index+1}/{len(paths)} · {path.name}'
        try:
            tasks.check()
            entry=(resume or {}).get(str(path.resolve()))
            if entry and entry.get('event_model','off')==selection_token:
                valid=ProjectStore.valid_completed({'completed':{str(path.resolve()):entry}},archive)
                if valid and (not capture_memory or (archive and archive.can_release(path))):
                    clips=[Clip(**item) for item in entry['clips']]
                    rows.extend((path,clip) for clip in clips)
                    on_progress((index+1)/len(paths),label+' · восстановлено из проекта и памяти')
                    tasks.check()
                    if after_complete:after_complete(path,clips)
                    continue
            if source_loader:
                source_loader(path, lambda f:on_progress(index/len(paths),label+f' · загрузка {f:.0%}'))
            if not path.is_file():
                raise FileNotFoundError(f'Не найден MP4: {path}. Подключите диск или выберите файл заново.')
            entry=(resume or {}).get(str(path.resolve()))
            if (entry and entry.get('fingerprint')==fingerprint(path) and entry.get('event_model','off')==selection_token
                and (not capture_memory or (archive and archive.can_release(path)))):
                clips=[Clip(**item) for item in entry['clips']]
                rows.extend((path,clip) for clip in clips)
                on_progress((index+1)/len(paths),label+' · восстановлено из проекта')
                tasks.check()
                if after_complete:after_complete(path,clips)
                continue
            proxy,warning=paired_proxy(path,ffmpeg_exe()) if use_proxies else (None,None)
            if warning:errors.append((path,warning))
            duration,data,key=scan_video(path,ffmpeg_exe(),lambda f,stage:on_progress(
                (index+f*.45)/len(paths),f'{label} · {stage}'),proxy,cache,runtime)
            # A wider candidate pool lets thresholds/diversity filter rather than fill quotas.
            clips=choose_clips(data['motion'],data['audio'],duration,clip_length,
                               min(80,max(12,count*4)),data['quality'])
            speech_segments = []
            if options.speech and len(data['audio']):
                try:
                    from speech_analysis import SpeechAnalyzer, speech_candidates
                    speech = pool.get('speech')
                    if speech is None:
                        speech = SpeechAnalyzer(ffmpeg_exe(), learning_folder or preferences_folder or path.parent, runtime)
                        pool['speech'] = speech
                    speech_segments = speech.recognize(path, duration, lambda f:on_progress(
                        (index+.45+.13*f)/len(paths),label+' · распознавание речи'))
                    for item in speech_candidates(speech_segments, duration):
                        a,b=item['start'],item['end']
                        if clip_length:
                            a=max(0.,min((a+b-clip_length)/2,duration-clip_length))
                            b=min(duration,a+clip_length)
                        clips.append(Clip(a,b,baseline_at(data,a,b),reason='Разговор'))
                except Exception as exc:
                    errors.append((path,'Речь: '+str(exc)))
            if selector:
                try:
                    clips=selector.discover(path,duration,data,clips,clip_length,count,
                        lambda f:on_progress((index+.58+.15*f)/len(paths),label+' · поиск обученных событий'))
                except Exception as exc:
                    errors.append((path,f'Обученный сюжет: {exc}; используется обычный отбор.'))
                    selector=None
            for clip in clips:
                if preferences:
                    origin=[clip.start,clip.end]
                    clip.start,clip.end=preferences.adjust_bounds(clip.start,clip.end,duration)
                    if [clip.start,clip.end]!=origin:
                        clip.score=baseline_at(data,clip.start,clip.end)
                        clip.reason+=' · ваши границы'
                describe_clip(clip,data)
                utterances = [s for s in speech_segments if s['start']<clip.end and s['end']>clip.start]
                if utterances:
                    clip.details['speech'] = utterances
                    words = sum(len(s['text'].split()) for s in utterances)
                    clip.signals['speech'] = min(.8,.45+.02*words)
                    clip.score = round(.55*clip.score+.45*clip.signals['speech'],3)
            # The trained visual classifier replaces generic text prompts, avoiding
            # two copies of X-CLIP and two visual inference passes for the same role.
            effective=Options(options.objects,options.sound,options.meaning and selector is None,options.gps,options.speech)
            if effective.enabled() and clips:
                pending=[]
                telemetry = path.with_suffix('.gpx')
                feature_options={'options':asdict(effective),'algorithm':4,
                                 'gpx':fingerprint(telemetry) if options.gps and telemetry.exists() else None}
                for clip in clips:
                    item=cache.load_features(key,clip.start,clip.end,feature_options) if cache else None
                    if item:
                        clip.signals.update(item['signals']);clip.score=item['score'];clip.reason=item['reason']
                        clip.details.update(item.get('details',{}))
                    else:pending.append(clip)
                if pending:
                    pool_key=digest([asdict(effective),profile])
                    advanced=pool.get(pool_key)
                    if advanced is None:
                        for old_key in list(pool):
                            if not old_key.startswith(('trained-event:','memory-features:','speech')):pool.pop(old_key)
                        import gc
                        gc.collect()
                        advanced=AdvancedAnalyzer(effective,ffmpeg_exe(),lambda message:on_progress(
                            (index+.73)/len(paths),message),runtime=runtime)
                        pool[pool_key]=advanced
                    try:
                        advanced.enrich(path,pending,lambda stage,f:on_progress(
                            (index+.73+.14*f)/len(paths),stage),frame_source=proxy)
                        errors.extend((path,w) for w in advanced.warnings)
                        diagnostics.extend(advanced.diagnostics)
                        if cache and not advanced.warnings:
                            for clip in pending:cache.save_features(key,clip,feature_options)
                    except Exception as exc:
                        errors.append((path,f'ИИ: {exc}; использован базовый отбор'))
                        diagnostics.append(path.name+'\n'+traceback.format_exc())
            if clip_length == 0:
                for clip in clips:
                    bounds = smart_bounds(clip.start,clip.end,duration,speech_segments,clip.details.get('object_events',[]))
                    if bounds != (clip.start,clip.end):
                        clip.start,clip.end = bounds
                        clip.embedding = []
                        clip.score = baseline_at(data,*bounds)
                        extras = [clip.signals[k] for k in ('objects','sound','meaning','gps','camera','speech') if k in clip.signals]
                        if extras:clip.score=round(.55*clip.score+.45*float(np.mean(extras)),3)
                        describe_clip(clip,data)
                        clip.reason += ' · границы события'
            if selector and clips:
                original=[Clip(**asdict(clip)) for clip in clips]
                try:
                    selector.enrich(path,clips,lambda f:on_progress((index+.87+.05*f)/len(paths),
                                                                    label+' · оценка обученного сюжета'))
                except Exception as exc:
                    clips=original
                    errors.append((path,f'Обученный сюжет: {exc}; оценки восстановлены без этой модели.'))
                    selector=None
            if model:
                for clip in clips:
                    personal=preferences.probability(clip.signals,model)
                    clip.score=round(.7*clip.score+.3*personal,3)
                    clip.reason+=' · ваш выбор'
            if capture_memory and learning_folder:
                from event_memory import remember
                warning = remember(path,duration,data,clips,learning_folder,runtime,pool,selector,
                    source_origin(path) if source_origin else None,
                    lambda f:on_progress((index+.92+.079*f)/len(paths),label+' · память событий'))
                if warning:
                    errors.append((path,'Анализ сохранён, но признаки для обучения не готовы: '+warning))
            clips=select_best(clips,count,threshold,deduplicate,[c for _,c in rows])
            tasks.check()
            rows.extend((path,clip) for clip in clips)
            on_complete(path,clips)
            tasks.check()
            if after_complete:
                after_complete(path,clips)
            on_progress((index+1)/len(paths),label+f' · найдено {len(clips)}')
        except tasks.Cancelled:break
        except Exception as exc:errors.append((path,str(exc)))
    if diagnostics and preferences_folder:
        log_path=Path(preferences_folder)/'moto_analysis.log'
        try:
            log_path.parent.mkdir(parents=True,exist_ok=True)
            with log_path.open('a',encoding='utf-8') as log:
                log.write('\n'+datetime.datetime.now().isoformat(timespec='seconds')+'\n')
                log.write('\n'.join(diagnostics)+'\n')
            errors.append((paths[0],f'Подробности ИИ: {log_path}'))
        except OSError as exc:errors.append((paths[0],f'Журнал не сохранён: {exc}'))
    return rows,errors


def reanalyze_clip(source, clip, options, folder, use_proxies=True, profile='auto',
                   on_progress=lambda f,s:None, model_pool=None, learning_folder=None):
    cache=CacheStore(folder);runtime=Runtime.detect(profile)
    proxy,_=paired_proxy(source,ffmpeg_exe()) if use_proxies else (None,None)
    duration,data,key=scan_video(source,ffmpeg_exe(),on_progress,proxy,cache,runtime)
    clip.embedding=[]
    clip.details={}
    clip.signals={}
    # Score only the edited range, against the same ride-wide feature scales.
    clip.score=baseline_at(data,clip.start,clip.end)
    describe_clip(clip,data)
    warnings=[]
    selector=None
    pool=model_pool if model_pool is not None else {}
    try:
        learning=LearningStore(learning_folder or folder)
        if learning.enabled():
            selector=EventSelector(learning,ffmpeg_exe(),runtime,lambda s:on_progress(0.,s))
    except (OSError,ValueError,KeyError,TypeError) as exc:warnings.append('Обученная модель: '+str(exc))
    if options.speech and len(data['audio']):
        try:
            from speech_analysis import SpeechAnalyzer
            speech=pool.get('speech') or SpeechAnalyzer(ffmpeg_exe(),learning_folder or folder,runtime)
            pool['speech']=speech
            segments=speech.recognize(source,duration,lambda f:on_progress(f,'Проверка речи'))
            utterances=[s for s in segments if s['start']<clip.end and s['end']>clip.start]
            if utterances:
                clip.details['speech']=utterances
                clip.signals['speech']=min(.8,.45+.02*sum(len(s['text'].split()) for s in utterances))
                clip.score=round(.55*clip.score+.45*clip.signals['speech'],3)
        except Exception as exc:warnings.append('Речь: '+str(exc))
    effective=Options(options.objects,options.sound,options.meaning and selector is None,options.gps,options.speech)
    if effective.enabled():
        pool=model_pool if model_pool is not None else {}
        pool_key=digest([asdict(effective),profile]);advanced=pool.get(pool_key)
        if advanced is None:
            for key in list(pool):
                if not key.startswith(('trained-event:','memory-features:','speech')):pool.pop(key)
            import gc
            gc.collect()
            advanced=AdvancedAnalyzer(effective,ffmpeg_exe(),lambda s:on_progress(0.,s),runtime=runtime)
            pool[pool_key]=advanced
        advanced.enrich(source,[clip],lambda s,f:on_progress(f,s),frame_source=proxy)
        warnings.extend(advanced.warnings)
    if selector:
        original=Clip(**asdict(clip))
        try:selector.enrich(source,[clip],lambda f:on_progress(f,'Проверка обученного сюжета'))
        except Exception as exc:
            clip=original;warnings.append('Обученный сюжет: '+str(exc))
    clip.reason='Ваши границы · '+clip.reason
    if learning_folder:
        from event_memory import remember
        warning=remember(source,duration,data,[clip],learning_folder,runtime,pool,selector,
                         on_progress=lambda f:on_progress(f,'Сохранение признаков изменённого отрезка'))
        if warning:warnings.append('Память событий: '+warning)
    return clip,warnings


def encode_with_progress(cmd,length,on_fraction,cwd=None):
    with tempfile.TemporaryFile() as errors:
        proc=tasks.popen(cmd,stdout=subprocess.PIPE,stderr=errors,cwd=cwd,text=True,encoding='utf-8',errors='replace')
        try:
            for line in proc.stdout:
                tasks.check()
                key,sep,value=line.strip().partition('=')
                if sep and key in ('out_time_us','out_time_ms'):
                    try:on_fraction(min(.99,int(value)/1_000_000/max(.01,length)))
                    except ValueError:pass
            code=proc.wait();tasks.check();errors.seek(0)
            return code,errors.read().decode('utf-8','replace')[-900:]
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            proc.stdout.close()


def export(path: Path, destination: Path, clips: list[Clip], framing: str,
           on_progress=lambda message: None,
           on_fraction=lambda fraction, message: None,
           color: ColorGrade | None = None, runtime=None, on_done=lambda index,path:None) -> list[Path]:
    ffmpeg = ffmpeg_exe()
    runtime=runtime or Runtime.detect()
    color = color or ColorGrade()
    color_filters = color.filters()
    destination.mkdir(parents=True, exist_ok=True)
    _, has_audio = video_info(path, ffmpeg)
    result = []
    selected = [c for c in clips if c.selected]
    pending = None
    try:
        with tempfile.TemporaryDirectory(prefix='moto_grade_') as temp_folder:
            if color.mode == 'dlog':
                shutil.copyfile(color.lut_path(), Path(temp_folder) / 'grade.cube')
            for index, clip in enumerate(selected, 1):
                tasks.check()
                output = destination / f'{path.stem}_short_{index:02d}.mp4'
                pending = output.with_name(output.stem+'.part.mp4')
                beginning = ('[0:v]' + ','.join(color_filters) + '[graded];') if color_filters else ''
                input_label = '[graded]' if color_filters else '[0:v]'
                # Preserve the whole wide view with a softly blurred vertical background.
                if framing == 'full':
                    video_filter = (beginning + input_label + 'split=2[base][front];'
                                    '[base]scale=720:1280:force_original_aspect_ratio=increase,'
                                    'crop=720:1280,boxblur=20:2[bg];'
                                    '[front]scale=720:1280:force_original_aspect_ratio=decrease[fg];'
                                    '[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]')
                else:
                    video_filter = (beginning + input_label + r'crop=w=min(iw\,ih*9/16):h=min(ih\,iw*16/9),'
                                    'scale=720:1280,setsar=1,format=yuv420p[v]')
                cmd = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-progress', 'pipe:1',
                       '-ss', str(clip.start), '-i', str(path.resolve()), '-t', str(clip.end - clip.start),
                       '-filter_complex', video_filter, '-map', '[v]']
                if has_audio:
                    cmd += ['-map', '0:a:0', '-c:a', 'aac', '-b:a', '160k']
                cmd += runtime.codec_args() + ['-color_primaries', 'bt709', '-color_trc', 'bt709',
                        '-colorspace', 'bt709',
                        '-movflags', '+faststart', str(pending.resolve())]
                label = f'Экспорт {index}/{len(selected)}: {output.name}'
                on_progress(label)
                on_fraction((index - 1) / len(selected), label)
                progress=lambda f:on_fraction((index-1+f)/len(selected),label)
                return_code,error_text=encode_with_progress(cmd,clip.end-clip.start,progress,temp_folder)
                if return_code and runtime.encoder!='libx264':
                    on_progress('Ускоритель недоступен; повтор экспорта на CPU')
                    runtime.encoder='libx264'
                    lo,hi=cmd.index('-c:v'),cmd.index('-color_primaries')
                    cmd[lo:hi]=runtime.codec_args()
                    return_code,error_text=encode_with_progress(cmd,clip.end-clip.start,progress,temp_folder)
                if return_code:
                    pending.unlink(missing_ok=True)
                    raise RuntimeError(f'Не удалось создать {output.name}: '+error_text)
                os.replace(pending,output)
                result.append(output)
                pending = None
                on_done(index-1,output)
                on_fraction(index / len(selected), label)
    except tasks.Cancelled:
        tasks.reap()
        if pending is not None:
            pending.unlink(missing_ok=True)
    manifest = destination / f'{path.stem}_shorts.json'
    manifest.write_text(json.dumps({'source': str(path), 'framing': framing,
                                    'color': color.manifest(),
                                    'clips': [asdict(c) for c in clips],
                                    'exports': [str(p) for p in result]},
                                   ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def export_batch(rows: list[tuple[Path, Clip]], destination: Path, framing: str,
                 on_fraction=lambda fraction, message: None,
                 color: ColorGrade | None = None, profile='auto',
                 on_complete=lambda index,path:None, source_loader=None, after_source=None
                 ) -> tuple[list[tuple[int, Path]], list[tuple[Path, str]]]:
    runtime=Runtime.detect(profile)
    on_fraction(0.,'Проверка ускорителя экспорта')
    runtime.select_encoder(ffmpeg_exe())
    on_fraction(0.,runtime.summary())
    sources = list(dict.fromkeys(path for path, clip in rows if clip.selected))
    results: list[tuple[int, Path]] = []
    errors: list[tuple[Path, str]] = []
    for group_index, source in enumerate(sources):
        selected = [(i, clip) for i, (path, clip) in enumerate(rows)
                    if path == source and clip.selected]
        # A separate subfolder prevents collisions between identical filenames.
        folder = (destination / f'{group_index+1:02d}_{source.stem}'
                  if len(sources) > 1 else destination)
        try:
            tasks.check()
            if source_loader:
                source_loader(source,lambda f:on_fraction(group_index/len(sources),source.name+f' · загрузка {f:.0%}'))
            outputs = export(source, folder, [clip for _, clip in selected], framing,
                             on_fraction=lambda fraction, label: on_fraction(
                                 (group_index + fraction) / len(sources),
                                 f'{group_index+1}/{len(sources)} · {label}'),
                             color=color,runtime=runtime,
                             on_done=lambda i,p:on_complete(selected[i][0],p))
            results.extend((selected[i][0], path) for i, path in enumerate(outputs))
            if after_source and not tasks.stopped():after_source(source)
        except tasks.Cancelled:
            break
        except Exception as exc:
            errors.append((source, str(exc)))
        if tasks.stopped():
            break
        on_fraction((group_index + 1) / len(sources),
                    f'{group_index+1}/{len(sources)} · {source.name}')
    return results, errors


def timestamp(value: float) -> str:
    minutes, seconds = divmod(value, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:05.2f}'


def parse_timecode(text: str) -> float:
    parts = text.strip().replace(',', '.').split(':')
    if not 1 <= len(parts) <= 3:
        raise ValueError('Введите секунды или таймкод ЧЧ:ММ:СС.')
    numbers = [float(part) for part in parts]
    if any(not math.isfinite(n) or n < 0 for n in numbers):
        raise ValueError('Таймкод должен быть неотрицательным числом.')
    if len(numbers) > 1 and (any(n >= 60 for n in numbers[1:])
                            or any(n != int(n) for n in numbers[:-1])):
        raise ValueError('Минуты и секунды в таймкоде должны быть меньше 60.')
    return sum(n * 60**i for i, n in enumerate(reversed(numbers)))


def set_clip_bounds(clip: Clip, start: float, end: float, duration: float) -> bool:
    if not all(math.isfinite(v) for v in (start, end, duration)) or not 0 <= start < end <= duration:
        raise ValueError('Начало должно быть раньше конца, а границы — внутри исходного видео.')
    if abs(clip.start-start) < .001 and abs(clip.end-end) < .001:
        return False
    if clip.edit_origin is None:clip.edit_origin=[clip.start,clip.end]
    clip.start, clip.end = start, end
    clip.reviewed = False
    clip.signals.clear()
    clip.score = 0.
    clip.reason = 'Границы изменены вручную'
    return True


def table_sort_value(column: str, source: Path, clip: Clip):
    """Use source values, preserving precision hidden by formatted table cells."""
    if column == 'video':
        # Camera sequences such as DJI_2/DJI_10 follow their numeric order.
        return tuple(int(part) if part.isdigit() else part
                     for part in re.split(r'(\d+)', source.name.casefold()))
    if column == 'score':
        return clip.score if clip.signals and math.isfinite(clip.score) else None
    return {'use': clip.selected, 'reviewed': clip.reviewed,
            'from': clip.start, 'to': clip.end, 'duration': clip.end-clip.start,
            'reason': clip.reason.casefold()}[column]


def external_player_command(source: Path, start: float, end: float) -> list[str] | None:
    """Return a native player invocation with an exact timeline start if available."""
    import shutil
    for name in ('vlc', 'mpv', 'ffplay'):
        executable = shutil.which(name)
        if not executable and sys.platform == 'win32':
            roots = [os.environ.get('ProgramFiles', ''), os.environ.get('ProgramFiles(x86)', ''),
                     os.environ.get('LOCALAPPDATA', '')]
            paths = ([Path(root) / 'VideoLAN' / 'VLC' / 'vlc.exe' for root in roots if root]
                     if name == 'vlc' else
                     [Path(root) / 'mpv' / 'mpv.exe' for root in roots if root] if name == 'mpv'
                     else [])
            executable = next((str(candidate) for candidate in paths if candidate.is_file()), None)
        if not executable:
            continue
        video = str(source.resolve())
        if name == 'vlc':
            return [executable, '--no-one-instance', f'--start-time={start:.3f}',
                    f'--stop-time={end:.3f}', video]
        if name == 'mpv':
            return [executable, f'--start={start:.3f}', f'--end={end:.3f}', video]
        return [executable, '-autoexit', '-ss', f'{start:.3f}', '-t',
                f'{max(.1, end-start):.3f}', '-i', video]
    return None


class App:
    INK = '#06111F'
    CARD = '#0C1B2E'
    EDGE = '#2A455F'
    CYAN = '#27E5E8'
    PINK = '#ED36A4'
    PALE = '#F4F8FF'
    MUTED = '#ADC0DC'

    def __init__(self, root: tk.Tk):
        self.root = root
        self.font_family=ui_font(root)
        self.font_bold=ui_font(root,True)
        root.title('Мото Шортсы — NEON v23.1')
        root.geometry('980x700')
        root.minsize(900, 630)
        self.events: queue.Queue = queue.Queue()
        self.task_control = None
        self.current_project = None
        self.project_state = {}
        self.analysis_settings = {}
        self.model_pool = {}
        self.training_ui = None
        self.boundary_editor = None
        self.boundary_step = 1
        self.drive_ui = None
        self.archive_ui = None
        self.table_sort_column = None
        self.table_sort_reverse = False
        self.table_titles = {}
        self._displayed_rows = {}
        self.learning_folder = Path(__file__).resolve().parent / 'learning_data'
        from drive_source import DriveSource
        self.drive_store = DriveSource(self.learning_folder)
        self.path = tk.StringVar()
        self.folder = tk.StringVar()
        self.length = tk.IntVar(value=30)
        self.auto_length = tk.BooleanVar(value=True)
        self.count = tk.IntVar(value=5)
        self.use_proxies = tk.BooleanVar(value=True)
        self.threshold = tk.DoubleVar(value=.22)
        self.deduplicate = tk.BooleanVar(value=True)
        self.performance = tk.StringVar(value='Авто')
        self.framing = tk.StringVar(value='full')
        self.color_mode = tk.StringVar(value='Без коррекции')
        self.color_lut = tk.StringVar()
        self.color_brightness = tk.IntVar(value=0)
        self.color_contrast = tk.IntVar(value=0)
        self.color_saturation = tk.IntVar(value=0)
        self.ai_flags = {key: tk.BooleanVar(value=False)
                         for key in ('objects', 'sound', 'meaning', 'gps', 'speech')}
        self.files: list[Path] = []
        self.sources: list[Path] = []
        self.clips: list[Clip] = []
        self.exports: dict[int, Path] = {}
        self.busy = False
        self.status = tk.StringVar(value='Выберите MP4 или папку с поездками.')
        self.stage = tk.StringVar(value='ГОТОВ К РАБОТЕ')
        self.percent = tk.StringVar(value='0%')
        self.progress_value = 0.
        self.preview_dir = tempfile.TemporaryDirectory(prefix='moto_open_',
                                                        ignore_cleanup_errors=True)
        self.build()
        root.after(100, self.poll)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def button(self, parent, title, command, accent=None, filled=False):
        accent = accent or self.CYAN
        return RoundedButton(parent, text=title, command=command,
                             accent=accent, filled=filled)

    def label(self, parent, title, fg=None, size=9):
        return tk.Label(parent, text=title, bg=parent.cget('bg'),
                        fg=fg or self.MUTED, font=(self.font_family, size))

    def build(self):
        self.root.configure(bg=self.INK)
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('Neon.Treeview', background=self.CARD,
                        fieldbackground=self.CARD, foreground=self.PALE,
                        rowheight=38, borderwidth=0, font=(self.font_family, 10))
        style.configure('Neon.Treeview.Heading', background='#10263B',
                        foreground=self.MUTED, borderwidth=0, padding=9,
                        font=(self.font_bold, 9))
        style.map('Neon.Treeview', background=[('selected', '#14516C')],
                  foreground=[('selected', '#FFFFFF')])
        style.configure('Neon.TEntry', fieldbackground='#0B1A2C',
                        foreground=self.PALE, bordercolor=self.EDGE,
                        lightcolor=self.EDGE, darkcolor=self.EDGE, padding=7)
        style.map('Neon.TEntry', fieldbackground=[('readonly', '#0B1A2C')],
                  foreground=[('readonly', self.PALE)])
        style.configure('Neon.TCombobox', fieldbackground='#0B1A2C',
                        background='#173047', foreground=self.PALE,
                        bordercolor=self.EDGE, lightcolor=self.EDGE,
                        darkcolor=self.EDGE, padding=6)
        style.map('Neon.TCombobox', fieldbackground=[('readonly', '#0B1A2C')],
                  foreground=[('readonly', self.PALE)])
        style.configure('Neon.TSpinbox', fieldbackground='#0B1A2C',
                        foreground=self.PALE, arrowsize=12,
                        bordercolor=self.EDGE, lightcolor=self.EDGE,
                        darkcolor=self.EDGE, padding=5)
        style.configure('Neon.Vertical.TScrollbar', background='#36526A',
                        troughcolor='#0B1A2C', bordercolor=self.EDGE,
                        lightcolor='#36526A', darkcolor='#36526A',
                        arrowcolor=self.CYAN)

        shell = tk.Frame(self.root, bg=self.INK, padx=16, pady=12)
        shell.pack(fill='both', expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        head = tk.Canvas(shell, bg=self.INK, height=70, highlightthickness=0)
        head.grid(row=0, column=0, sticky='ew', pady=(0, 12))
        head.bind('<Configure>', self.draw_header)

        material_card = RoundedCard(shell, color=self.CARD, edge=self.EDGE)
        material_card.grid(row=1, column=0, sticky='ew')
        material = material_card.body
        material.columnconfigure(0, weight=1)
        self.label(material, 'МАТЕРИАЛ', self.PINK, 10).grid(
            row=0, column=0, sticky='w', pady=(0, 8))
        ttk.Entry(material, textvariable=self.path, style='Neon.TEntry',
                  state='readonly').grid(row=1, column=0, sticky='ew', padx=(0, 9))
        self.files_btn = self.button(material, 'ФАЙЛЫ', self.pick_files)
        self.files_btn.grid(row=1, column=1, padx=(0, 8))
        self.folder_btn = self.button(material, 'ПАПКА MP4', self.pick_input_folder)
        self.folder_btn.grid(row=1, column=2)
        self.project_btn=self.button(material,'ПРОЕКТ',self.project_popup)
        self.project_btn.grid(row=1,column=3,padx=(8,0))

        results_card = RoundedCard(shell, color=self.CARD, edge=self.EDGE)
        results_card.grid(row=2, column=0, sticky='nsew', pady=12)
        results = results_card.body
        results.columnconfigure(0, weight=1)
        results.rowconfigure(2, weight=1)
        self.label(results, 'НАЙДЕННЫЕ МОМЕНТЫ', self.PINK, 10).grid(
            row=0, column=0, sticky='w')
        self.label(results, 'Заголовок — сортировка · Двойной щелчок — открыть отрезок с его таймкода').grid(
            row=1, column=0, sticky='w', pady=(3, 8))
        area = tk.Frame(results, bg=self.CARD)
        area.grid(row=2, column=0, sticky='nsew')
        area.columnconfigure(0, weight=1)
        area.rowconfigure(0, weight=1)
        columns = ('use', 'reviewed', 'video', 'from', 'to', 'duration', 'score', 'reason')
        self.table = ttk.Treeview(area, columns=columns, show='headings',
                                  style='Neon.Treeview', height=7)
        specs = [('use', 'ВЫБРАН', 85), ('reviewed', 'ПРОВ.', 75),
                 ('video', 'ВИДЕО', 140), ('from', 'НАЧАЛО', 95),
                 ('to', 'КОНЕЦ', 95), ('duration', 'ДЛИНА', 75), ('score', 'БАЛЛ', 75),
                 ('reason', 'ПРИЧИНА', 180)]
        self.table_titles = {key: title for key, title, _ in specs}
        for key, title, width in specs:
            self.table.heading(key, text=title, command=lambda column=key: self.sort_table(column))
            self.table.column(key, width=width, anchor='center',
                              stretch=key in ('video', 'reason'))
        self.table.tag_configure('chosen', background='#103745', foreground=self.PALE)
        self.table.tag_configure('alternate', background='#11243A')
        self.table.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(area, orient='vertical', command=self.table.yview,
                               style='Neon.Vertical.TScrollbar')
        scroll.grid(row=0, column=1, sticky='ns')
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind('<Double-1>', self.open_selected)
        self.table.bind('<Return>', self.open_selected)
        self.table.bind('<space>', lambda _: self.mark_selected(None))
        self.table.bind('<F2>', lambda _: self.edit_bounds())
        actions = tk.Frame(results, bg=self.CARD)
        actions.grid(row=3, column=0, sticky='w', pady=(9, 0))
        self.button(actions, '✓ ПРИНЯТЬ', lambda: self.mark_selected(True)).pack(side='left')
        self.button(actions, '✕ ИСКЛЮЧИТЬ', lambda: self.mark_selected(False),
                    self.PINK).pack(side='left', padx=7)
        self.button(actions, '▶ ОТКРЫТЬ', self.open_selected).pack(side='left')
        self.button(actions, 'ГРАНИЦЫ', self.edit_bounds).pack(side='left', padx=(7, 0))
        self.button(actions, 'ГОТОВЫЙ РОЛИК', self.open_exported).pack(side='left', padx=7)
        self.button(actions, 'АНАЛИЗ', self.inspect_selected).pack(side='left')

        processing_card = RoundedCard(shell, color=self.CARD, edge=self.EDGE, pady=12)
        processing_card.grid(row=3, column=0, sticky='ew')
        processing = processing_card.body
        processing.columnconfigure(0, weight=1)
        bottom = tk.Frame(processing, bg=self.CARD)
        bottom.grid(row=0, column=0, sticky='ew')
        self.analyze_btn = self.button(bottom, '✦ НАЙТИ МОМЕНТЫ',
                                        self.analysis_popup, self.CYAN, True)
        self.analyze_btn.pack(side='left', fill='x', expand=True)
        self.export_btn = self.button(bottom, '✂ СОЗДАТЬ ШОРТСЫ',
                                       self.export_popup, self.PINK, True)
        self.export_btn.pack(side='left', padx=10, fill='x', expand=True)
        self.export_btn.config(state='disabled')
        self.stop_btn = self.button(bottom, '■ ОСТАНОВИТЬ', self.stop_work, self.PINK)
        self.stop_btn.pack(side='left', fill='x', expand=True)
        self.stop_btn.config(state='disabled')
        progress = tk.Frame(processing, bg=self.CARD)
        progress.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        progress.columnconfigure(0, weight=1)
        tk.Label(progress, textvariable=self.stage, bg=self.CARD, fg=self.CYAN,
                 font=(self.font_bold, 9)).grid(row=0, column=0, sticky='w')
        tk.Label(progress, textvariable=self.percent, bg=self.CARD, fg=self.PINK,
                 font=(self.font_bold, 9)).grid(row=0, column=1, sticky='e')
        self.bar = tk.Canvas(progress, height=12, bg=self.CARD, highlightthickness=0)
        self.bar.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(5, 5))
        self.bar.bind('<Configure>', lambda _: self.draw_progress())
        tk.Label(progress, textvariable=self.status, bg=self.CARD, fg=self.MUTED,
                 anchor='w', font=(self.font_family, 9)).grid(row=2, column=0,
                                                      columnspan=2, sticky='ew')
        self.draw_progress()

    @staticmethod
    def blend(first, second, amount):
        amount = max(0., min(1., amount))
        a = tuple(int(first[i:i+2], 16) for i in (1, 3, 5))
        b = tuple(int(second[i:i+2], 16) for i in (1, 3, 5))
        return '#{:02x}{:02x}{:02x}'.format(*(
            round(left * (1-amount) + right * amount) for left, right in zip(a, b)))

    def draw_gradient(self, canvas, width, height):
        """Soft cyan and pink glows built from color tiles; no raster asset."""
        import math
        canvas.delete('all')
        step = 3
        for y in range(0, height, step):
            radius = min(22., height/2)
            center = y+step/2
            dy = max(0., radius-center, center-(height-radius))
            inset = radius-math.sqrt(max(0., radius*radius-dy*dy))
            base = self.blend('#071529', '#102139', y / max(height, 1) * .55)
            for x in range(0, width, step):
                u = x / max(width, 1)
                v = y / max(height, 1)
                cyan = .42 * math.exp(-((u-.72)/.19)**2 - ((v-.30)/.85)**2)
                pink = .48 * math.exp(-((u-1.04)/.23)**2 - ((v-.75)/.85)**2)
                color = self.blend(self.blend(base, '#087A91', cyan), '#9B226D', pink)
                left, right = max(x, inset), min(x+step, width-inset)
                if right > left:
                    canvas.create_rectangle(left, y, right, min(y+step, height),
                                            fill=color, outline=color)

    def draw_header(self, event):
        canvas = event.widget
        width = max(400, event.width)
        self.draw_gradient(canvas, width, 70)
        canvas.create_text(20, 17, anchor='nw', text='МОТО ШОРТСЫ',
                           fill=self.PALE, font=(self.font_bold, 23))

    def draw_progress(self):
        if not hasattr(self, 'bar'):
            return
        self.bar.delete('all')
        width = self.bar.winfo_width()
        if width < 3:
            return
        rounded_rect(self.bar, 0, 1, width, 11, 5, fill='#213951', outline='')
        filled = width * self.progress_value
        if filled <= 0:
            return
        # Clip gradient strips to a capsule instead of a square-ended bar.
        radius = min(5., filled/2)
        for y in range(1, 11):
            dy = abs(y+.5-6)
            inset = radius-math.sqrt(max(0., radius*radius-min(radius,dy)**2))
            for x in range(0, math.ceil(filled), 3):
                left, right = max(float(x), inset), min(x+3., filled-inset)
                if right > left:
                    self.bar.create_rectangle(left, y, right, y+1, outline='',
                                              fill=blend(self.CYAN, self.PINK, x/max(1.,filled)))

    def update_progress(self, fraction, message):
        self.progress_value = min(1., max(0., float(fraction)))
        self.percent.set(f'{self.progress_value:.0%}')
        self.stage.set(message.upper())
        self.draw_progress()

    def pick_files(self):
        names = filedialog.askopenfilenames(filetypes=[('MP4 видео', '*.mp4'),
                                                    ('Все файлы', '*.*')])
        if names:
            self.set_files([Path(name) for name in names if Path(name).suffix.lower() == '.mp4'])

    def pick_input_folder(self):
        path = filedialog.askdirectory(title='Папка с MP4')
        if path:
            self.set_files(mp4_files(Path(path)))
            if not self.files:
                messagebox.showinfo('MP4', 'В выбранной папке нет MP4.')

    def set_files(self, files):
        self.current_project=None
        self.project_state={}
        self.files = list(dict.fromkeys(files))
        self.path.set(self.files[0].name if len(self.files) == 1 else
                      f'{len(self.files)} MP4 · ' + ', '.join(p.name for p in self.files[:3])
                      if self.files else '')
        if self.files:
            self.folder.set(str(self.files[0].parent / 'shorts'))
        self.clips = []
        self.sources = []
        self.exports = {}
        self.refresh()

    def selected_index(self):
        rows = self.table.selection()
        return int(rows[0]) if rows else None

    def mark_selected(self, accepted, reason=None):
        index = self.selected_index()
        if index is None or self.busy:
            return
        clip = self.clips[index]
        if accepted is False and reason is None:
            window=self.popup('Причина исключения')
            frame=self.popup_body(window,20,16)
            value=tk.StringVar(value='Неинтересно')
            ttk.Combobox(frame,textvariable=value,values=['Неинтересно','Тряска / плохое качество','Неудачные границы','Повтор','Другое'],state='readonly',style='Neon.TCombobox',width=28).pack(pady=8)
            def apply():
                chosen=value.get();window.destroy();self.mark_selected(False,chosen)
            self.button(frame,'ИСКЛЮЧИТЬ',apply,self.PINK).pack(fill='x')
            return
        clip.selected = not clip.selected if accepted is None else accepted
        clip.reviewed = True
        clip.decision_reason = reason or ('Принято' if clip.selected else 'Исключено')
        try:
            if hasattr(self,'learning_folder'):
                from event_archive import EventArchive
                archive = EventArchive(self.learning_folder)
                source = archive.source_for(self.sources[index])
                if source:archive.save_clip(source['id'],clip)
            if clip.signals:
                PreferenceStore(Path(self.folder.get())).add(self.sources[index], clip,
                                                               clip.selected, reason or '')
            if accepted is not None and (clip.selected or (reason or '') not in EXCLUDED_REASONS):
                LearningStore(self.learning_folder).add_user(self.sources[index],clip.start,clip.end,
                                                                clip.selected,clip.reason)
        except (OSError,ValueError) as exc:
            self.status.set(f'Не удалось сохранить решение: {exc}')
        self.refresh()
        self.table.selection_set(str(index))
        self.save_project()

    def edit_bounds(self):
        if self.ensure_selected_source(self.edit_bounds):return
        index = self.selected_index()
        if index is None or self.busy:
            return
        if self.boundary_editor is not None:
            self.boundary_editor.window.lift()
            return
        clip, source = self.clips[index], self.sources[index]
        try:
            ffmpeg = ffmpeg_exe()
            duration, _ = video_info(source, ffmpeg)
        except Exception as exc:
            messagebox.showerror('Границы', str(exc))
            return

        def save(start, end, open_video=False):
            try:
                changed = set_clip_bounds(clip, start, end, duration)
            except ValueError as exc:
                messagebox.showerror('Границы', str(exc), parent=self.boundary_editor.window)
                return False
            if changed:
                self.exports.pop(index,None);self.refresh();self.table.selection_set(str(index))
                self.save_project()
                cfg=self.analysis_settings or {'options':{},'use_proxies':True,'profile':'auto'}
                folder=Path(self.folder.get())
                draft=Clip(**asdict(clip))
                def work():
                    result,warnings=reanalyze_clip(source,draft,Options(**cfg.get('options',{})),folder,
                        cfg.get('use_proxies',True),cfg.get('profile','auto'),
                        lambda f,s:self.events.put(('progress',(f,s))),self.model_pool,self.learning_folder)
                    return ('edited',(index,result,warnings,open_video))
                self.background(work)
            else:
                if open_video:self.open_selected()
            return True

        from boundary_editor import BoundaryEditor
        self.boundary_editor = BoundaryEditor(self, source, duration, clip.start, clip.end,
                                             save, ffmpeg=ffmpeg,
                                             parse_timecode=parse_timecode, timestamp=timestamp)

    def open_selected(self, event=None):
        if event is not None and getattr(event, 'type', None) == tk.EventType.ButtonPress:
            row = self.table.identify_row(event.y)
            if not row:
                return
            self.table.selection_set(row)
        index = self.selected_index()
        if index is None or self.busy:
            return
        if self.ensure_selected_source(self.open_selected):return
        clip = self.clips[index]
        source = self.sources[index]
        command = external_player_command(source, clip.start, clip.end)
        if command:
            try:
                subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=(sys.platform != 'win32'))
                self.status.set(f'Открыт {source.name} с {timestamp(clip.start)}.')
            except OSError as exc:
                messagebox.showerror('Внешний плеер', str(exc))
            return
        self.status.set('Подготовка отрезка для системного плеера…')
        self.background(lambda: ('open_segment', self.make_segment(index)))

    def make_segment(self, index):
        clip = self.clips[index]
        source = self.sources[index]
        output = Path(self.preview_dir.name) / f'fragment_{index:03d}.mp4'
        command = [ffmpeg_exe(), '-y', '-v', 'error', '-ss', str(clip.start),
                   '-i', str(source.resolve()), '-t', str(clip.end-clip.start),
                   '-map', '0:v:0', '-map', '0:a?',
                   '-vf', 'scale=-2:720:force_original_aspect_ratio=decrease',
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
                   '-c:a', 'aac', '-b:a', '128k', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(output)]
        result = tasks.run(command, capture_output=True)
        if result.returncode or not output.is_file() or not output.stat().st_size:
            raise RuntimeError('Не удалось открыть фрагмент: ' +
                               result.stderr.decode('utf-8', 'replace')[-300:])
        return output

    def open_exported(self):
        index = self.selected_index()
        if index is None or index not in self.exports:
            messagebox.showinfo('Готовый ролик', 'Для этой строки ещё нет готового ролика.')
            return
        path = self.exports[index]
        command = external_player_command(path, 0., self.clips[index].end-self.clips[index].start)
        try:
            if command:
                subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=(sys.platform != 'win32'))
            else:
                self.system_open(path)
        except OSError as exc:
            messagebox.showerror('Готовый ролик', str(exc))

    def ensure_selected_source(self, action):
        index = self.selected_index()
        if index is None or self.sources[index].is_file():return False
        drive = getattr(self,'drive_store',None)
        if drive is None or not drive.entry(self.sources[index]):
            messagebox.showinfo('Исходник','MP4 удалён. Анализ доступен через АНАЛИЗ; для кадров и экспорта верните исходник.')
            return True
        source = self.sources[index]
        def ready(_):
            self.table.selection_set(str(index))
            action()
        def work():
            drive.ensure(source,lambda f:self.events.put(('progress',(f,'Повторная загрузка '+source.name))))
            return ('callback',(ready,None))
        self.background(work)
        return True

    def inspect_selected(self):
        index=self.selected_index()
        if index is None:return
        from archive_ui import show_evidence
        from event_archive import EventArchive
        archive=EventArchive(self.learning_folder)
        source=archive.source_for(self.sources[index])
        clip=self.clips[index]
        if source and not clip.embedding:
            saved=archive.feature(source['id'],clip.start,clip.end)
            if saved:clip.embedding=saved[0].tolist()
        show_evidence(self,clip,source)

    def drive_popup(self):
        if self.busy:return
        from drive_ui import DriveWindow
        if self.drive_ui and self.drive_ui.alive():
            self.drive_ui.window.lift();return
        self.drive_ui=DriveWindow(self)

    def archive_popup(self):
        if self.busy:return
        from archive_ui import ArchiveWindow
        if self.archive_ui and self.archive_ui.window.winfo_exists():
            self.archive_ui.window.lift();return
        self.archive_ui=ArchiveWindow(self)

    @staticmethod
    def system_open(path: Path):
        if sys.platform == 'win32':
            os.startfile(str(path.resolve()))
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(path.resolve())])
        else:
            subprocess.Popen(['xdg-open', str(path.resolve())])

    def popup(self, title):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.configure(bg=self.INK)
        window.resizable(False, False)
        window.transient(self.root)
        header = tk.Canvas(window, bg=self.INK, height=52, highlightthickness=0)
        header.pack(fill='x')
        header.bind('<Configure>', lambda event: self.draw_popup_header(event, title))
        window.grab_set()
        window.focus_set()
        return window

    def popup_body(self, window, padx, pady):
        card = RoundedCard(window, color=self.CARD, edge=self.EDGE, padx=padx, pady=pady)
        card.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        return card.body

    def draw_popup_header(self, event, title):
        canvas = event.widget
        self.draw_gradient(canvas, max(300, event.width), 52)
        canvas.create_text(17, 16, anchor='nw', text=title.upper(),
                           fill=self.PALE, font=(self.font_bold, 12))

    def analysis_popup(self):
        if not self.files or self.busy:
            messagebox.showerror('Видео', 'Сначала выберите MP4 или папку.')
            return
        window = self.popup('Параметры анализа')
        frame = self.popup_body(window, 22, 20)
        frame.pack()
        self.label(frame, '01 / ПОИСК МОМЕНТОВ', self.PINK, 11).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 15))
        self.label(frame, 'Длина фрагмента, сек').grid(row=1, column=0, sticky='w')
        length_box = tk.Frame(frame, bg=self.CARD)
        length_box.grid(row=1, column=1, sticky='w', pady=5)
        length_input = ttk.Spinbox(length_box, from_=10, to=120, width=7,
                                   textvariable=self.length, style='Neon.TSpinbox')
        length_input.pack(side='right')
        def toggle_length():
            length_input.config(state='disabled' if self.auto_length.get() else 'normal')
        tk.Checkbutton(length_box, text='Авто', variable=self.auto_length,
                       command=toggle_length, bg=self.CARD, fg=self.CYAN,
                       selectcolor='#173047', activebackground=self.CARD,
                       activeforeground=self.CYAN).pack(side='left', padx=(0, 12))
        toggle_length()
        self.label(frame, 'На одно видео').grid(row=2, column=0, sticky='w')
        ttk.Spinbox(frame, from_=1, to=20, width=7, textvariable=self.count,
                    style='Neon.TSpinbox').grid(
            row=2, column=1, sticky='w', pady=5)
        self.label(frame, 'ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ', self.PINK).grid(
            row=3, column=0, columnspan=2, sticky='w', pady=(18, 5))
        for row, (key, title) in enumerate((('objects', 'Объекты'),
                                             ('sound', 'Звуковые события'),
                                             ('meaning', 'Сюжет нескольких секунд'),
                                             ('gps', 'Телеметрия DJI / GPS'),
                                             ('speech', 'Речь и границы реплик')), 4):
            tk.Checkbutton(frame, text=title, variable=self.ai_flags[key],
                           bg=self.CARD, fg=self.PALE, selectcolor='#173047',
                           activebackground=self.CARD,
                           activeforeground=self.CYAN).grid(row=row, column=0,
                                                            columnspan=2, sticky='w')
        tk.Checkbutton(frame, text='Ускорять анализ через одноимённые .LRF',
                       variable=self.use_proxies, bg=self.CARD, fg=self.CYAN,
                       selectcolor='#173047', activebackground=self.CARD,
                       activeforeground=self.CYAN).grid(row=9, column=0,
                                                        columnspan=2, sticky='w', pady=(12, 0))
        self.label(frame,'Порог интересности (0–1)').grid(row=10,column=0,sticky='w',pady=(12,4))
        ttk.Spinbox(frame,from_=0,to=1,increment=.05,width=7,textvariable=self.threshold,style='Neon.TSpinbox').grid(row=10,column=1,sticky='w')
        tk.Checkbutton(frame,text='Убирать похожие фрагменты',variable=self.deduplicate,bg=self.CARD,fg=self.PALE,selectcolor='#173047').grid(row=11,column=0,columnspan=2,sticky='w')
        self.label(frame,'Производительность').grid(row=12,column=0,sticky='w',pady=8)
        ttk.Combobox(frame,textvariable=self.performance,values=['Авто','Экономный','Максимальная скорость'],state='readonly',width=22,style='Neon.TCombobox').grid(row=12,column=1,sticky='w')
        self.label(frame,'Число фрагментов — максимум; слабые кандидаты пропускаются.').grid(row=13,column=0,columnspan=2,sticky='w',pady=6)
        def apply():
            try:
                length = 0 if self.auto_length.get() else self.length.get()
                count = self.count.get()
                if not ((length == 0 or 10 <= length <= 120) and 1 <= count <= 20 and 0<=self.threshold.get()<=1):
                    raise ValueError()
            except (ValueError, tk.TclError):
                messagebox.showerror('Параметры', 'Длина: авто или 10–120 секунд; количество 1–20.',
                                     parent=window)
                return
            flags = Options(**{key: variable.get() for key, variable in self.ai_flags.items()})
            use_proxies = self.use_proxies.get()
            window.destroy()
            self.start_analyze(length, count, flags, use_proxies)
        self.button(frame, '✦ НАЧАТЬ АНАЛИЗ', apply, self.CYAN, True).grid(
            row=14, column=0, columnspan=2, sticky='ew', pady=(18, 0))

    def profile_key(self):
        return {'Авто':'auto','Экономный':'economy','Максимальная скорость':'fast'}.get(self.performance.get(),'auto')

    def snapshot(self):
        return {'files':[str(p.resolve()) for p in self.files], 'output':self.folder.get(),
                'analysis':self.analysis_settings,
                'export':{'framing':self.framing.get(),'mode':self.color_mode.get(),
                          'lut':self.color_lut.get(),'brightness':self.color_brightness.get(),
                          'contrast':self.color_contrast.get(),'saturation':self.color_saturation.get()},
                'rows':[{'source':str(p.resolve()),'clip':asdict(c)} for p,c in zip(self.sources,self.clips)],
                'exports':{str(i):str(p.resolve()) for i,p in self.exports.items()},
                'completed':self.project_state.get('completed',{})}

    def save_project(self, choose=False):
        if self.busy:return
        if not self.files:return
        if choose:
            name=filedialog.asksaveasfilename(title='Сохранить проект',defaultextension='.json',
                       initialfile='moto_project.json',filetypes=[('Проект Мото Шортсы','*.json')])
            if not name:return
            self.current_project=Path(name)
        self.current_project=self.current_project or Path(self.folder.get())/'moto_project.json'
        state=self.snapshot()
        # Keep edits and decisions in the completed-file checkpoint too.
        for source,entry in state['completed'].items():
            entry['clips']=[r['clip'] for r in state['rows'] if r['source']==source]
        try:
            ProjectStore(self.current_project).save(state)
            self.project_state=state
        except (OSError,ValueError) as exc:
            self.status.set('Проект не сохранён: '+str(exc))

    def open_project(self):
        if self.busy:return
        name=filedialog.askopenfilename(title='Открыть проект',filetypes=[('Проект Мото Шортсы','*.json')])
        if not name:return
        try:
            state=ProjectStore(Path(name)).load()
            self.files=[Path(p) for p in state['files']]
            self.current_project=Path(name);self.project_state=state
            self.folder.set(state['output'])
            self.analysis_settings=state.get('analysis',{})
            cfg=self.analysis_settings
            length=cfg.get('length',0);self.auto_length.set(length==0);self.length.set(length or 30)
            self.count.set(cfg.get('count',5));self.use_proxies.set(cfg.get('use_proxies',True))
            self.threshold.set(cfg.get('threshold',.22));self.deduplicate.set(cfg.get('deduplicate',True))
            self.performance.set({'auto':'Авто','economy':'Экономный','fast':'Максимальная скорость'}.get(cfg.get('profile'),'Авто'))
            for key,value in cfg.get('options',{}).items():self.ai_flags[key].set(value)
            self.sources=[Path(r['source']) for r in state.get('rows',[])]
            self.clips=[Clip(**r['clip']) for r in state.get('rows',[])]
            self.exports={int(i):Path(p) for i,p in state.get('exports',{}).items() if Path(p).is_file()}
            exp=state.get('export',{})
            for key,var in [('framing',self.framing),('mode',self.color_mode),('lut',self.color_lut),
                            ('brightness',self.color_brightness),('contrast',self.color_contrast),('saturation',self.color_saturation)]:
                if key in exp:var.set(exp[key])
            self.path.set(f'{len(self.files)} MP4 · проект')
            self.refresh()
            from event_archive import EventArchive
            valid=ProjectStore.valid_completed(state,EventArchive(self.learning_folder))
            self.status.set(f'Проект открыт: {len(valid)}/{len(self.files)} видео обработано, {len(self.clips)} фрагментов.')
            missing=sum(not p.is_file() and not self.drive_store.entry(p) for p in self.files)
            if missing:messagebox.showwarning('Проект',f'Не найдены исходники: {missing}. Верните их по прежним путям.')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            messagebox.showerror('Проект',str(exc))

    def project_popup(self):
        if self.busy:return
        window=self.popup('Проект и оборудование')
        frame=self.popup_body(window,22,18)
        self.label(frame,f'Видео в очереди: {len(self.files)} · фрагментов: {len(self.clips)}').pack(anchor='w',pady=(0,12))
        def action(fn):
            window.destroy();fn()
        for title,fn in [('ОТКРЫТЬ ПРОЕКТ',self.open_project),
                         ('СОХРАНИТЬ КАК…',lambda:self.save_project(True)),
                         ('ПРОДОЛЖИТЬ ОЧЕРЕДЬ',self.resume_project),
                         ('ПРОВЕРИТЬ ОБОРУДОВАНИЕ',self.hardware_report),
                         ('ОБУЧЕНИЕ ОТБОРА',self.training_popup),
                         ('ПАМЯТЬ СОБЫТИЙ',self.archive_popup),
                         ('GOOGLE DRIVE',self.drive_popup)]:
            self.button(frame,title,lambda f=fn:action(f)).pack(fill='x',pady=4)
        self.label(frame,'После каждого файла проект сохраняется автоматически.').pack(pady=(12,0))

    def training_popup(self):
        if self.busy:return
        from training_ui import TrainingWindow
        if self.training_ui and self.training_ui.alive():
            self.training_ui.window.lift();return
        try:
            self.training_ui=TrainingWindow(self)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            messagebox.showerror('Обучение',str(exc))

    def hardware_report(self):
        profile=self.profile_key()
        def work():
            runtime=Runtime.detect(profile)
            try:
                import torch
                runtime.select_torch(torch)
            except ImportError:runtime.notes.append('Пакеты ИИ не установлены.')
            runtime.select_encoder(ffmpeg_exe())
            return ('hardware',runtime.summary()+'\nGPU: '+runtime.gpu+'\n'+'\n'.join(runtime.notes))
        self.background(work)

    def resume_project(self):
        if not self.files or not self.analysis_settings:
            messagebox.showinfo('Проект','Откройте проект с очередью анализа.');return
        cfg=self.analysis_settings
        self.threshold.set(cfg.get('threshold',.22))
        self.deduplicate.set(cfg.get('deduplicate',True))
        self.performance.set({'auto':'Авто','economy':'Экономный','fast':'Максимальная скорость'}.get(cfg.get('profile'),'Авто'))
        self.start_analyze(cfg['length'],cfg['count'],Options(**cfg['options']),cfg['use_proxies'],resume=True)

    def start_analyze(self,length,count,flags,use_proxies,resume=False):
        paths=list(self.files)
        if not paths:return
        folder=Path(self.folder.get());profile=self.profile_key()
        self.analysis_settings={'length':length,'count':count,'options':asdict(flags),
            'use_proxies':use_proxies,'threshold':self.threshold.get(),
            'deduplicate':self.deduplicate.get(),'profile':profile}
        try:
            token=LearningStore(self.learning_folder).token()
        except (OSError,ValueError,KeyError,TypeError):token='unavailable'
        previous_token=self.project_state.get('analysis',{}).get('event_model','off')
        self.analysis_settings['event_model']=token
        from event_archive import EventArchive
        restored=ProjectStore.valid_completed(self.project_state,EventArchive(self.learning_folder)) if resume else {}
        if token!=previous_token:restored={}
        self.restored_exports={}
        if resume:
            for i,row in enumerate(self.project_state.get('rows',[])):
                output=self.project_state.get('exports',{}).get(str(i))
                if row['source'] in restored and output and Path(output).is_file():
                    saved=row['clip']
                    self.restored_exports[(row['source'],saved['start'],saved['end'])]=Path(output)
        old_rows=[r for r in self.project_state.get('rows',[]) if r['source'] in restored] if resume else []
        self.sources=[Path(r['source']) for r in old_rows];self.clips=[Clip(**r['clip']) for r in old_rows]
        self.exports={i:self.restored_exports[(str(p.resolve()),c.start,c.end)]
                      for i,(p,c) in enumerate(zip(self.sources,self.clips))
                      if (str(p.resolve()),c.start,c.end) in self.restored_exports}
        self.refresh()
        self.current_project=self.current_project or folder/'moto_project.json'
        state=self.snapshot();state['completed']=restored
        store=ProjectStore(self.current_project)
        store.save(state)
        self.project_state=state
        cfg=dict(self.analysis_settings)
        def complete(path,clips):
            source=str(path.resolve())
            state['rows']=[r for r in state['rows'] if r['source']!=source]+[
                {'source':source,'clip':asdict(c)} for c in clips]
            state['completed'][source]={'fingerprint':fingerprint(path),'clips':[asdict(c) for c in clips],
                                        'event_model':cfg['event_model']}
            store.save(state)
        self.update_progress(0.,'Подготовка анализа')
        def work():
            result=analyze_batch(paths,length,count,lambda f,s:self.events.put(('progress',(f,s))),
                flags,folder,use_proxies,cfg['threshold'],cfg['deduplicate'],profile,
                restored,complete,self.model_pool,self.learning_folder,
                source_loader=self.drive_store.ensure,
                source_origin=lambda p:self.drive_store.entry(p)['metadata'] if self.drive_store.entry(p) else None,
                after_complete=lambda p,c:self.cleanup_drive_source(p),capture_memory=True)
            self.project_state=state
            return ('analysis',result)
        self.background(work)

    def export_popup(self):
        if self.busy or not any(clip.selected for clip in self.clips):
            return
        window = self.popup('Экспорт и цвет')
        frame = self.popup_body(window, 22, 18)
        frame.pack(fill='both')
        self.label(frame, 'ЭКСПОРТ / DJI OSMO ACTION 5 PRO', self.PINK, 11).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 12))
        self.label(frame, 'Папка вывода').grid(row=1, column=0, sticky='w')
        ttk.Entry(frame, textvariable=self.folder, width=40, style='Neon.TEntry').grid(
            row=1, column=1, sticky='ew', padx=8, pady=6)
        self.button(frame, 'ОБЗОР', lambda: self.pick_output(window)).grid(
            row=1, column=2)
        self.label(frame, 'Кадр').grid(row=2, column=0, sticky='w')
        for column, (title, value) in enumerate((('Весь + фон', 'full'),
                                                  ('По центру', 'crop')), 1):
            tk.Radiobutton(frame, text=title, value=value, variable=self.framing,
                           bg=self.CARD, fg=self.PALE, selectcolor='#173047',
                           activebackground=self.CARD,
                           activeforeground=self.CYAN).grid(row=2, column=column, sticky='w')
        self.label(frame, 'Цвет').grid(row=3, column=0, sticky='w', pady=(12, 0))
        mode = ttk.Combobox(frame, textvariable=self.color_mode, state='readonly',
                            style='Neon.TCombobox', width=24,
                            values=('Без коррекции', 'Ручная настройка', 'D-Log M + LUT'))
        mode.grid(row=3, column=1, sticky='w', pady=(12, 0))
        lut_label = self.label(frame, '')
        lut_label.configure(wraplength=380, justify='left')
        lut_label.grid(row=5, column=0, columnspan=2, sticky='w', pady=6)
        def show_lut():
            name = ('Свой LUT: ' + Path(self.color_lut.get()).name if self.color_lut.get()
                    else 'Встроенный LUT DJI Osmo Action 5 Pro → Rec.709')
            lut_label.configure(text=name)
        def choose_lut():
            self.pick_lut(window)
            show_lut()
        def use_builtin_lut():
            self.color_lut.set('')
            self.color_mode.set('D-Log M + LUT')
            show_lut()
        show_lut()
        self.button(frame, 'ДРУГОЙ LUT', choose_lut).grid(
            row=3, column=2, pady=(12, 0))
        self.button(frame, 'ВСТРОЕННЫЙ DJI', use_builtin_lut).grid(
            row=5, column=2, pady=6)
        self.label(frame, 'LUT применяйте только к съёмке D-Log M.').grid(
            row=4, column=1, columnspan=2, sticky='w')
        for row, (title, variable) in enumerate((('Яркость', self.color_brightness),
                                                  ('Контраст', self.color_contrast),
                                                  ('Насыщенность', self.color_saturation)), 6):
            self.label(frame, title).grid(row=row, column=0, sticky='w')
            tk.Scale(frame, from_=-30, to=30, orient='horizontal', length=250,
                     variable=variable, bg=self.CARD, fg=self.CYAN,
                     troughcolor='#243C54', activebackground=self.CYAN,
                     highlightthickness=0, bd=0).grid(row=row, column=1, sticky='w')
        def apply():
            try:
                color = self.current_color()
                color.validate()
                if not self.folder.get().strip():
                    raise ValueError('Выберите папку вывода.')
            except ValueError as exc:
                messagebox.showerror('Экспорт', str(exc), parent=window)
                return
            window.destroy()
            self.start_export(color)
        self.button(frame, '✂ СОЗДАТЬ ШОРТСЫ', apply, self.PINK, True).grid(
            row=9, column=0, columnspan=3, sticky='ew', pady=(14, 0))

    def pick_output(self, parent):
        folder = filedialog.askdirectory(parent=parent)
        if folder:
            self.folder.set(folder)

    def pick_lut(self, parent):
        filename = filedialog.askopenfilename(parent=parent,
                                             filetypes=[('3D LUT', '*.cube')])
        if filename:
            self.color_lut.set(filename)
            self.color_mode.set('D-Log M + LUT')

    def current_color(self):
        key = {'Без коррекции': 'none', 'Ручная настройка': 'manual',
               'D-Log M + LUT': 'dlog'}[self.color_mode.get()]
        return ColorGrade(key, self.color_brightness.get(), self.color_contrast.get(),
                          self.color_saturation.get(), self.color_lut.get())

    def start_export(self,color):
        rows=[(source,Clip(**asdict(clip))) for source,clip in zip(self.sources,self.clips)]
        folder,framing=Path(self.folder.get()),self.framing.get()
        profile=self.profile_key()
        self.save_project()
        state=self.snapshot()
        store=ProjectStore(self.current_project) if self.current_project else None
        def complete(index,path):
            state['exports'][str(index)]=str(path.resolve())
            if store:store.save(state)
        self.update_progress(0.,'Подготовка экспорта')
        self.background(lambda:('exports',export_batch(rows,folder,framing,
            lambda f,s:self.events.put(('progress',(f,s))),color,profile,complete,
            source_loader=self.drive_store.ensure,after_source=self.cleanup_drive_source)))

    def cleanup_drive_source(self,path):
        if self.drive_store.auto_cleanup:
            from event_archive import EventArchive
            try:self.drive_store.release(path,EventArchive(self.learning_folder))
            except OSError:pass

    def sort_table(self, column):
        if column not in self.table_titles:
            return
        self.table_sort_reverse = (not self.table_sort_reverse
                                   if self.table_sort_column == column
                                   else column in ('score', 'use', 'reviewed'))
        self.table_sort_column = column
        self.apply_table_sort()
        self.table.yview_moveto(0)

    def apply_table_sort(self):
        rows = list(self.table.get_children())
        column = self.table_sort_column
        if column is not None:
            values = [(row, table_sort_value(column, self.sources[int(row)], self.clips[int(row)]))
                      for row in rows]
            # Fixed original-id order keeps ties stable in either direction.
            known = sorted((entry for entry in values if entry[1] is not None),
                           key=lambda entry: int(entry[0]))
            rows = [row for row, _ in sorted(known, key=lambda entry: entry[1],
                                             reverse=self.table_sort_reverse)]
            # Unchecked/invalidated scores stay last, including descending order.
            rows += [row for row, value in sorted(values, key=lambda entry: int(entry[0]))
                     if value is None]
            for position, row in enumerate(rows):
                self.table.move(row, '', position)
        for position, row in enumerate(rows):
            clip = self.clips[int(row)]
            tags = ('chosen',) if clip.selected else ('alternate',) if position % 2 else ()
            self.table.item(row, tags=tags)
        for key, title in self.table_titles.items():
            arrow = (' ▼' if self.table_sort_reverse else ' ▲') if key == column else ''
            self.table.heading(key, text=title+arrow)

    def refresh(self):
        selected = self.table.selection()
        focused = self.table.focus()
        previous = self._displayed_rows
        self._displayed_rows = {}
        self.table.delete(*self.table.get_children())
        for index, (source, clip) in enumerate(zip(self.sources, self.clips)):
            tags = ('chosen',) if clip.selected else ('alternate',) if index % 2 else ()
            self.table.insert('', 'end', iid=str(index), values=(
                '✓' if clip.selected else '—', '✓' if clip.reviewed else '…',
                source.name, timestamp(clip.start), timestamp(clip.end),
                f'{clip.end-clip.start:.0f} с',
                f'{clip.score:.2f}' if clip.signals else '—', clip.reason), tags=tags)
            self._displayed_rows[str(index)] = (source, clip)
        self.apply_table_sort()
        def same_clip(row):
            old, current = previous.get(row), self._displayed_rows.get(row)
            return old and current and old[0] == current[0] and old[1] is current[1]
        retained = [row for row in selected if same_clip(row)]
        if retained:
            self.table.selection_set(*retained)
        if focused and same_clip(focused):
            self.table.focus(focused)
            if retained:
                self.table.see(focused)
        self.export_btn.config(state=('normal' if not self.busy and
                                      any(clip.selected for clip in self.clips)
                                      else 'disabled'))

    def stop_work(self):
        if self.busy and self.task_control:
            self.task_control.stop()
            self.stop_btn.config(state='disabled')
            self.stage.set('ОСТАНОВКА…')
            self.status.set('Остановка обработки. Ожидаем завершения текущего вызова ИИ, если он выполняется…')

    def set_busy(self, value):
        self.busy = value
        self.stop_btn.config(state='normal' if value else 'disabled')
        self.analyze_btn.config(state='disabled' if value else 'normal')
        self.files_btn.config(state='disabled' if value else 'normal')
        self.folder_btn.config(state='disabled' if value else 'normal')
        self.project_btn.config(state='disabled' if value else 'normal')
        if self.training_ui:self.training_ui.set_busy(value)
        self.refresh()

    def background(self, work):
        self.task_control = tasks.TaskControl()
        control = self.task_control
        self.set_busy(True)
        def worker():
            tasks.bind(control)
            try:
                self.events.put(('result', work()))
            except tasks.Cancelled:
                self.events.put(('stopped', None))
            except Exception as exc:
                self.events.put(('error', str(exc)))
            finally:
                tasks.reap()
                tasks.bind(None)
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == 'progress':
                    if not self.task_control or not self.task_control.event.is_set():
                        self.update_progress(*payload)
                elif kind == 'stopped':
                    self.set_busy(False)
                    self.stage.set('ОСТАНОВЛЕНО')
                    self.status.set('Обработка остановлена.')
                elif kind == 'error':
                    self.set_busy(False)
                    self.stage.set('ОШИБКА')
                    self.status.set('Ошибка: ' + payload)
                    messagebox.showerror('Ошибка', payload)
                elif kind == 'result':
                    subtype, value = payload
                    was_stopped = self.task_control is not None and self.task_control.event.is_set()
                    self.set_busy(False)
                    if subtype=='callback':
                        callback,result=value
                        if not was_stopped:callback(result)
                        continue
                    if was_stopped and subtype == 'open_segment':
                        self.stage.set('ОСТАНОВЛЕНО')
                        self.status.set('Подготовка фрагмента остановлена.')
                        continue
                    if subtype=='hardware':
                        self.update_progress(1.,'Проверка завершена')
                        messagebox.showinfo('Оборудование',value)
                        continue
                    if subtype=='training':
                        if not was_stopped:
                            self.update_progress(1.,'Обучение: готово')
                            self.status.set(value.split('\n')[0])
                            messagebox.showinfo('Обучение',value)
                        continue
                    if subtype=='edited':
                        index,clip,warnings,open_video=value
                        if not was_stopped:
                            self.clips[index]=clip
                            self.refresh();self.table.selection_set(str(index));self.save_project()
                            self.update_progress(1.,'Границы проверены')
                            if warnings:messagebox.showwarning('Проверка границ','\n'.join(warnings))
                            if open_video:self.open_selected()
                        continue
                    if subtype == 'open_segment':
                        try:
                            self.system_open(value)
                            self.update_progress(1, 'Фрагмент открыт')
                            self.status.set(f'Открыт фрагмент: {value.name}')
                        except OSError as exc:
                            messagebox.showerror('Системный плеер', str(exc))
                        continue
                    if was_stopped:
                        self.stage.set('ОСТАНОВЛЕНО')
                    else:
                        self.update_progress(1, 'Готово')
                    if subtype == 'analysis':
                        rows, errors = value
                        self.sources = [path for path, _ in rows]
                        self.clips = [clip for _, clip in rows]
                        retained=getattr(self,'restored_exports',{})
                        self.exports = {i:retained[(str(p.resolve()),c.start,c.end)]
                                        for i,(p,c) in enumerate(rows)
                                        if (str(p.resolve()),c.start,c.end) in retained}
                        self.refresh()
                        self.save_project()
                        self.status.set(('Остановлено. ' if was_stopped else '') + f'Найдено {len(rows)} фрагментов из '
                                        f'{len(set(self.sources))} MP4; '
                                        f'замечаний: {len(errors)}.')
                        if errors:
                            messagebox.showwarning('Замечания анализа', '\n'.join(
                                f'{path.name}: {error[:140]}' for path, error in errors[:5]))
                    elif subtype == 'exports':
                        outputs, errors = value
                        self.exports.update(dict(outputs))
                        self.save_project()
                        self.status.set(('Остановлено. Сохранено: ' if was_stopped else 'Готово: ') + f'{len(outputs)} шортсов. '
                                        f'Папка: {self.folder.get()}')
                        if errors:
                            messagebox.showwarning('Замечания экспорта', '\n'.join(
                                f'{path.name}: {error[:140]}' for path, error in errors[:5]))
                        elif not was_stopped:
                            messagebox.showinfo('Готово',
                                                f'Создано {len(outputs)} шортсов.\n'
                                                f'Папка: {self.folder.get()}')
        except queue.Empty:
            pass
        try:
            self.root.after(100, self.poll)
        except tk.TclError:
            pass

    def close(self):
        if self.boundary_editor is not None:
            self.boundary_editor.close()
        if not self.busy:self.save_project()
        if self.task_control:
            self.task_control.stop()
        self.preview_dir.cleanup()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='Поиск ярких фрагментов поездки и экспорт вертикальных MP4')
    parser.add_argument('video', nargs='?', type=Path, help='MP4 или папка с MP4 (без аргумента — окно)')
    parser.add_argument('-o', '--output', type=Path, help='Папка назначения')
    parser.add_argument('--length', type=int, default=0,
                        help='0 — авто (по умолчанию), или длина 10–120 секунд')
    parser.add_argument('--count', type=int, default=5, help='Количество фрагментов 1–20')
    parser.add_argument('--framing', choices=['full', 'crop'], default='full')
    parser.add_argument('--color', choices=['none', 'manual', 'dlog'], default='none')
    parser.add_argument('--lut', type=Path,
                        help='Свой .cube для --color dlog; по умолчанию встроенный LUT DJI Action 5 Pro')
    parser.add_argument('--brightness', type=int, default=0, help='−30…30')
    parser.add_argument('--contrast', type=int, default=0, help='−30…30')
    parser.add_argument('--saturation', type=int, default=0, help='−30…30')
    parser.add_argument('--objects', action='store_true', help='Объекты и короткие траектории')
    parser.add_argument('--sound-events', action='store_true', help='Звуковые события')
    parser.add_argument('--meaning', action='store_true', help='Смысл восьми кадров клипа')
    parser.add_argument('--gps', action='store_true', help='GPX / ExifTool GPS')
    parser.add_argument('--threshold',type=float,default=.22,help='Порог интересности 0–1')
    parser.add_argument('--keep-similar',action='store_true',help='Оставлять похожие кандидаты')
    parser.add_argument('--profile',choices=['auto','economy','fast'],default='auto')
    parser.add_argument('--resume',action='store_true',help='Продолжить moto_project.json в папке вывода')
    parser.add_argument('--learning-folder',type=Path,default=Path(__file__).resolve().parent/'learning_data',
                        help='Папка общей модели обучения (по умолчанию learning_data рядом с программой)')
    parser.add_argument('--no-lrf', action='store_true',
                        help='Не использовать одноимённые DJI .LRF для анализа')
    args = parser.parse_args()
    if args.video is None:
        root = tk.Tk()
        App(root)
        root.mainloop()
        return
    paths = mp4_files(args.video)
    if not paths:
        parser.error('Укажите существующий MP4 или папку, содержащую MP4.')
    if not ((args.length == 0 or 10 <= args.length <= 120) and 1 <= args.count <= 20):
        parser.error('Длина: 0 (авто) или 10–120 секунд, количество 1–20.')
    color = ColorGrade(args.color, args.brightness, args.contrast, args.saturation,
                       str(args.lut) if args.lut else '')
    try:
        color.validate()
    except ValueError as exc:
        parser.error(str(exc))
    if not 0<=args.threshold<=1:parser.error('Порог интересности: 0–1.')
    destination=args.output or (args.video if args.video.is_dir() else args.video.parent)/'shorts'
    project=ProjectStore(destination/'moto_project.json')
    cfg=dict(length=args.length,count=args.count,options=asdict(Options(args.objects,args.sound_events,args.meaning,args.gps)),use_proxies=not args.no_lrf,threshold=args.threshold,deduplicate=not args.keep_similar,profile=args.profile)
    try:cfg['event_model']=LearningStore(args.learning_folder).token()
    except (OSError,ValueError,KeyError,TypeError):cfg['event_model']='unavailable'
    state=project.load() if args.resume and project.path.exists() else {'files':[str(p.resolve()) for p in paths],'output':str(destination),'analysis':cfg,'completed':{},'rows':[],'exports':{}}
    previous_cfg={**state['analysis'],'event_model':state['analysis'].get('event_model','off')}
    if args.resume and previous_cfg!=cfg:parser.error('Настройки или обученная модель изменились. Запустите новый поиск без --resume.')
    restored=ProjectStore.valid_completed(state) if args.resume else {}
    def complete(source,clips):
        name=str(source.resolve())
        state['completed'][name]={'fingerprint':fingerprint(source),'clips':[asdict(c) for c in clips],
                                 'event_model':cfg['event_model']}
        state['rows']=[r for r in state['rows'] if r['source']!=name]+[{'source':name,'clip':asdict(c)} for c in clips]
        project.save(state)
    project.save(state)
    rows, errors = analyze_batch(paths, args.length, args.count,
                                 lambda f, s: print(f'{f:.0%} {s}', end='\r', flush=True),
                                 Options(args.objects, args.sound_events,
                                         args.meaning, args.gps),
                                 args.output or (args.video if args.video.is_dir() else args.video.parent) / 'shorts',
                                 not args.no_lrf,args.threshold,not args.keep_similar,args.profile,restored,complete,
                                 learning_folder=args.learning_folder)
    print(f'\nВидео: {len(paths)}; найдено {len(rows)} фрагментов:')
    for path, clip in rows:
        print(f'  {path.name}: {timestamp(clip.start)} — {timestamp(clip.end)} '
              f'({clip.score:.2f}; {clip.reason})')
    if rows:
        destination = args.output or (args.video if args.video.is_dir() else args.video.parent) / 'shorts'
        def exported(index,path):
            state['exports'][str(index)]=str(path);project.save(state)
        outputs, export_errors = export_batch(rows, destination, args.framing,
                                              lambda f, s: print(f'{f:.0%} {s}', end='\r', flush=True),
                                              color,args.profile,
                                              exported)
        state['exports'].update({str(i):str(p) for i,p in outputs});project.save(state)
        print('\nГотово:', *(str(p) for _, p in outputs), sep='\n')
        errors.extend(export_errors)
    if errors:
        for path, error in errors:
            print(f'Ошибка {path.name}: {error}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
