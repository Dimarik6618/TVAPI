"""Optional local speech recognition. Only text/timestamps are persisted."""
from __future__ import annotations

import subprocess
import numpy as np

import task_control as tasks
from project_store import atomic_json, digest, fingerprint
from local_privacy import configure_local_models

configure_local_models()


SPEECH_SPEC = {'version': 1, 'model': 'base', 'vad': True, 'words': True}


class SpeechAnalyzer:
    def __init__(self, ffmpeg, folder, runtime=None):
        self.ffmpeg, self.folder, self.runtime = ffmpeg, folder, runtime
        self.model = None

    def load(self):
        if self.model is not None:
            return
        try:
            import onnxruntime
            onnxruntime.disable_telemetry_events()
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Для речи запустите Установить_речь.bat.') from exc
        # CPU int8 avoids requiring separate CUDA/cuDNN libraries for CTranslate2.
        self.model = WhisperModel('base', device='cpu', compute_type='int8',
                                  cpu_threads=self.runtime.threads if self.runtime else 4)

    def recognize(self, path, duration, on_progress=lambda f: None):
        from pathlib import Path
        import json
        root = Path(self.folder) / 'moto_cache' / 'speech'
        root.mkdir(parents=True, exist_ok=True)
        cache = root / (digest([fingerprint(Path(path)), SPEECH_SPEC]) + '.json')
        try:
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved.get('complete'):
                return saved['segments']
        except (OSError, ValueError, KeyError):
            pass
        self.load()
        chunks, offset = [], 0.
        while offset < duration:
            tasks.check()
            # Overlap prevents a word at a decoding boundary from being truncated.
            length = min(60., duration-offset)
            command = [self.ffmpeg, '-v', 'error', '-ss', str(offset), '-i', str(path),
                       '-t', str(length), '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', '-']
            result = tasks.run(command, capture_output=True)
            if result.returncode:
                raise RuntimeError('Не удалось прочитать звук для распознавания речи.')
            audio = np.frombuffer(result.stdout, dtype='<f4').copy()
            if len(audio):
                segments, _ = self.model.transcribe(audio, vad_filter=True, word_timestamps=True,
                    condition_on_previous_text=False, beam_size=3, no_speech_threshold=.6)
                for segment in segments:
                    tasks.check()
                    if segment.no_speech_prob > .6 or segment.avg_logprob < -1.2 or not segment.text.strip():
                        continue
                    words = [{'start': round(offset+w.start, 3), 'end': round(offset+w.end, 3),
                              'text': w.word, 'probability': round(w.probability, 3)} for w in (segment.words or [])]
                    item = {'start': round(offset+segment.start, 3), 'end': min(duration, round(offset+segment.end, 3)),
                            'text': segment.text.strip(), 'words': words}
                    if item['end'] > item['start'] and not any(
                        old['text'].casefold() == item['text'].casefold() and abs(old['start']-item['start']) < 3
                        for old in chunks[-6:]):
                        chunks.append(item)
            on_progress(min(1., (offset+length)/max(.1, duration)))
            if offset+length >= duration:
                break
            offset += 58.
        chunks.sort(key=lambda s: s['start'])
        tasks.check()
        atomic_json(cache, {'complete': True, 'spec': SPEECH_SPEC, 'segments': chunks})
        return chunks


def speech_candidates(segments, duration):
    groups = []
    for s in segments:
        if len(s['text'].split()) < 2:
            continue
        if groups and s['start']-groups[-1]['end'] <= 3 and s['end']-groups[-1]['start'] <= 85:
            groups[-1]['end'] = max(groups[-1]['end'], s['end'])
            groups[-1]['segments'].append(s)
        else:
            groups.append({'start': s['start'], 'end': s['end'], 'segments': [s]})
    return [{'start': max(0., g['start']-1.5), 'end': min(duration, g['end']+2.),
             'segments': g['segments']} for g in groups]
