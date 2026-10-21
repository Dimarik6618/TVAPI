"""Local opt-in-by-review preferences: only scores and decisions are retained."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

FEATURES = ('baseline', 'objects', 'sound', 'meaning', 'gps', 'camera',
            'duration', 'quality', 'visual', 'audio', 'speech')
FILENAME = 'moto_preferences.json'
# These decisions describe presentation or repetition, not a boring event.
# Shared with the visual classifier so both learning paths interpret them alike.
EXCLUDED_REASONS = {'Неудачные границы', 'Повтор', 'Тряска / плохое качество'}


def feature_vector(signals: dict[str, float]) -> np.ndarray:
    return np.array([float(signals.get(key, 0.)) for key in FEATURES], dtype=np.float64)


class PreferenceStore:
    def __init__(self, folder: Path):
        self.path = folder / FILENAME
        try:
            self.samples = json.loads(self.path.read_text(encoding='utf-8')).get('samples', {})
            if not isinstance(self.samples, dict):
                self.samples = {}
        except (OSError, ValueError, TypeError):
            self.samples = {}

    def add(self, source: Path, clip, accepted: bool, reason: str = ''):
        identity = f'{source.resolve()}|{clip.start:.3f}|{clip.end:.3f}'
        key = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
        self.samples[key] = {
            'signals': {name: round(float(clip.signals.get(name, 0.)), 4)
                        for name in FEATURES},
            'accepted': bool(accepted),
            'reason': reason,
        }
        origin = getattr(clip, 'edit_origin', None)
        if origin and origin[1] > origin[0]:
            length = origin[1]-origin[0]
            self.samples[key]['bounds'] = [(clip.start-origin[0])/length,
                                           (clip.end-origin[1])/length]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'version': 1, 'samples': self.samples},
                                        ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, self.path)

    def adjust_bounds(self, start, end, duration):
        """Learn conservative start/end offsets from explicitly accepted edits."""
        examples = [s['bounds'] for s in self.samples.values()
                    if s.get('accepted') and isinstance(s.get('bounds'), list)
                    and len(s['bounds']) == 2]
        if len(examples) < 3:
            return start, end
        offsets = np.clip(np.median(examples,axis=0), -.25, .25)
        length = end-start
        new_start = max(0., start+float(offsets[0])*length)
        new_end = min(duration, end+float(offsets[1])*length)
        return (round(new_start,2),round(new_end,2)) if new_end-new_start >= 3 else (start,end)

    def fitted(self):
        # Keep every decision in history, but learn interest only from story labels.
        samples = [sample for sample in self.samples.values()
                   if sample.get('reason') not in EXCLUDED_REASONS]
        positives = sum(bool(s.get('accepted')) for s in samples)
        negatives = len(samples) - positives
        if positives < 5 or negatives < 5:
            return None
        X = np.stack([feature_vector(s.get('signals', {})) for s in samples])
        y = np.array([float(s.get('accepted', False)) for s in samples])
        weights = np.zeros(len(FEATURES), dtype=np.float64)
        bias = 0.
        sample_weights = np.where(y > .5, len(y)/(2*positives), len(y)/(2*negatives))
        for _ in range(240):
            predictions = 1/(1+np.exp(-np.clip(X @ weights+bias, -20, 20)))
            difference = (predictions-y)*sample_weights
            weights -= .25*(X.T @ difference/len(y) + .12*weights)
            bias -= .25*float(np.mean(difference))
        return weights, bias

    def probability(self, signals: dict[str, float], model=None) -> float | None:
        fitted = model if model is not None else self.fitted()
        if fitted is None:
            return None
        weights, bias = fitted
        value = float(np.dot(feature_vector(signals), weights)+bias)
        return float(1/(1+np.exp(-np.clip(value, -20, 20))))
