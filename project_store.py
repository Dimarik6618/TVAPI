"""Versioned, atomic project checkpoints and analysis caches (never video copies)."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np

CACHE_VERSION = 3


def fingerprint(path: Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    with path.open('rb') as stream:
        beginning = stream.read(65536)
        stream.seek(max(0, stat.st_size-65536))
        ending = stream.read(65536)
    return {'path': str(path), 'size': stat.st_size, 'mtime': stat.st_mtime_ns,
            'sample': hashlib.sha256(beginning+ending).hexdigest()}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class CacheStore:
    def __init__(self, folder: Path):
        self.root = Path(folder) / 'moto_cache'
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, source: Path, proxy: Path | None = None):
        return digest({'version': CACHE_VERSION, 'source': fingerprint(source),
                       'proxy': fingerprint(proxy) if proxy else None})

    def load_scan(self, key):
        try:
            with np.load(self.root/(key+'.npz'), allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in ('motion','quality','audio','signature')}
                duration = float(data['duration'])
                if duration <= 0 or any(not np.isfinite(a).all() for a in arrays.values()):
                    return None
                return duration, arrays
        except (OSError, ValueError, KeyError, EOFError):
            return None

    def save_scan(self, key, duration, **arrays):
        fd, name = tempfile.mkstemp(suffix='.npz', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, duration=duration, **arrays)
            os.replace(name, self.root/(key+'.npz'))
        finally:
            Path(name).unlink(missing_ok=True)

    def load_features(self, key, start, end, options):
        name = digest([key, round(start,3), round(end,3), options])
        try:
            data = json.loads((self.root/(name+'.json')).read_text(encoding='utf-8'))
            if data.get('complete'):
                return data
        except (OSError, ValueError, TypeError):
            pass
        return None

    def save_features(self, key, clip, options):
        name = digest([key, round(clip.start,3), round(clip.end,3), options])
        atomic_json(self.root/(name+'.json'), {'complete': True, 'signals': clip.signals,
                                              'score': clip.score, 'reason': clip.reason,
                                              'details': clip.details})


class ProjectStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def save(self, state):
        atomic_json(self.path, {'version': 1, **state})

    def load(self):
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('version') != 1 or not isinstance(data.get('files'), list):
            raise ValueError('Неподдерживаемый формат проекта.')
        return data

    @staticmethod
    def valid_completed(state, archive=None):
        valid = {}
        for source, entry in state.get('completed', {}).items():
            try:
                if fingerprint(Path(source)) == entry['fingerprint']:
                    valid[source] = entry
            except (OSError, KeyError, TypeError):
                if archive:
                    saved=archive.source_for(source)
                    before=entry.get('fingerprint',{})
                    if (saved and archive.can_release(source)
                        and all(saved['fingerprint'].get(k)==before.get(k) for k in ('size','sample'))):
                        valid[source]=entry
        return valid
