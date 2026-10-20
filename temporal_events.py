"""Temporal object evidence and conservative, sentence-aware event boundaries."""
from __future__ import annotations

import math
import numpy as np


def track_events(tracks):
    events = []
    for track in tracks.values():
        points = track['points']
        if len(points) < 3 or points[-1]['time'] - points[0]['time'] < .5:
            continue
        areas = np.asarray([p['area'] for p in points])
        xs = np.asarray([p['x'] for p in points])
        growth = float(areas.max() - areas[:max(1, len(areas)//3)].mean())
        crossing = float(xs.max() - xs.min())
        name = track['class']
        if growth > .025 and areas.max() > .06:
            label, kind, strength = 'Приближение транспорта' if name != 'person' else 'Человек приближается', 'approach', min(1., .55+growth*4)
        elif crossing > .30 and areas.max() > .012:
            label, kind, strength = ('Мотоциклист проходит через кадр' if name == 'motorcycle' else
                                    'Человек пересекает кадр' if name == 'person' else 'Транспорт пересекает кадр'), 'crossing', min(1., .4+crossing)
        elif name in ('person', 'motorcycle') and areas.max() > .025:
            label, kind, strength = 'Человек рядом' if name == 'person' else 'Мотоциклист рядом', 'encounter', .45
        else:
            continue
        events.append({'type': kind, 'label': label, 'class': name, 'track_id': track['id'],
                       'start': points[0]['time'], 'end': points[-1]['time'], 'score': strength})
    return sorted(events, key=lambda e: e['score'], reverse=True)


def smart_bounds(start, end, duration, speech=(), events=()):
    """Retain event setup/aftermath and complete intersecting speech segments."""
    original = (start, end)
    relevant = [e for e in events if e.get('score', 0) >= .55 and start <= e['start'] < end]
    if relevant:
        start = max(0., min(e['start'] for e in relevant)-2.5)
        end = min(duration, max(e['end'] for e in relevant)+3.5)
    utterances = [s for s in speech if s['end'] > start and s['start'] < end]
    # Do not join distant conversations just because an extension touched another.
    if utterances:
        start = min(start, max(0., min(s['start'] for s in utterances)-.6))
        end = max(end, min(duration, max(s['end'] for s in utterances)+.9))
    if end-start < min(10., duration):
        end = min(duration, start+min(10., duration))
        start = max(0., end-min(10., duration))
    # A very long utterance needs manual editing; preserve the original candidate.
    if end-start > 120 or not 0 <= start < end <= duration:
        return original
    return round(start, 3), round(end, 3)


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return None
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / max(1e-9, np.linalg.norm(a)*np.linalg.norm(b)))


def category(clip):
    events = clip.details.get('object_events', [])
    if events:
        return events[0]['type']
    if clip.details.get('speech'):
        return 'conversation'
    if clip.signals.get('sound', 0) >= .6:
        return 'sound'
    return 'visual'


def diverse_order(clips, prior=()):
    """Prefer variety softly; never promote a candidate below the score threshold."""
    pending = sorted(clips, key=lambda c: c.score, reverse=True)
    chosen = []
    while pending:
        def utility(c):
            similarities = [v for old in chosen + list(prior) if (v := cosine(c.embedding, old.embedding)) is not None]
            similar = max(similarities, default=0.)
            same = sum(category(old) == category(c) for old in chosen)
            return c.score - .12*max(0., (similar-.75)/.25) - .025*min(3, same)
        best = max(range(len(pending)), key=lambda i: utility(pending[i]))
        chosen.append(pending.pop(best))
    return chosen
