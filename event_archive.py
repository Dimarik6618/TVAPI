"""Durable event memory: numbers, text and frozen features; never media bytes."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import zlib
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from project_store import digest, fingerprint
from video_features import FEATURE_SPEC, unit_vector


class EventArchive:
    def __init__(self, folder):
        self.root = Path(folder) / 'moto_events'
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'events.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS aliases (path TEXT PRIMARY KEY, source_id TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS timelines (source_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, '
                       'start REAL NOT NULL, end REAL NOT NULL, payload TEXT NOT NULL, vector BLOB, spec TEXT)')
            db.execute('CREATE INDEX IF NOT EXISTS event_source ON events(source_id,start,end)')
            db.execute('CREATE TABLE IF NOT EXISTS windows (source_id TEXT, start REAL, end REAL, '
                       'spec TEXT, vector BLOB NOT NULL, PRIMARY KEY(source_id,start,end,spec))')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def encode(vector):
        value = unit_vector(vector)
        if value.shape != (512,):
            raise ValueError('Для памяти событий нужны 512 визуальных признаков.')
        return zlib.compress(value.astype('<f4').tobytes())

    @staticmethod
    def decode(value):
        if value is None:
            return None
        return unit_vector(np.frombuffer(zlib.decompress(value), dtype='<f4').copy())

    def register(self, path, duration=None, origin=None, scan=None):
        path = Path(path).resolve()
        fp = fingerprint(path)
        sid = digest([fp['size'], fp['sample']])[:32]
        source = self.source(sid) or {'id': sid, 'file': path.name, 'aliases': []}
        source['aliases'] = sorted(set(source.get('aliases', []) + [str(path)]))
        source.update(path=str(path), fingerprint=fp, updated=dt.datetime.now(dt.timezone.utc).isoformat())
        if duration is not None:
            source['duration'] = float(duration)
        if origin:
            source['origin'] = origin
            source['file'] = origin.get('name', source['file'])
        if scan is not None:
            # Per-second numeric signals also survive loss of the MP4.
            timeline = {k: np.asarray(scan[k]).round(5).tolist() for k in ('motion', 'quality', 'audio')}
            source['complete'] = False
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO sources VALUES (?,?)', (sid, json.dumps(source, ensure_ascii=False)))
            db.execute('INSERT OR REPLACE INTO aliases VALUES (?,?)', (str(path), sid))
            if scan is not None:
                db.execute('INSERT OR REPLACE INTO timelines VALUES (?,?)', (sid, json.dumps(timeline)))
        return sid

    def source(self, sid):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM sources WHERE id=?', (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def source_for(self, path):
        path = str(Path(path).resolve())
        with self.connect() as db:
            row = db.execute('SELECT s.payload FROM sources s JOIN aliases a ON a.source_id=s.id '
                             'WHERE a.path=?', (path,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_window(self, sid, start, end, vector):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO windows VALUES (?,?,?,?,?)',
                       (sid, round(start, 4), round(end, 4), digest(FEATURE_SPEC), self.encode(vector)))

    def window(self, sid, start, end):
        with self.connect() as db:
            row = db.execute('SELECT vector FROM windows WHERE source_id=? AND start=? AND end=? AND spec=?',
                             (sid, round(start, 4), round(end, 4), digest(FEATURE_SPEC))).fetchone()
        return self.decode(row[0]) if row else None

    def save_clip(self, sid, clip):
        eid = digest([sid, round(clip.start, 4), round(clip.end, 4)])[:32]
        payload = asdict(clip)
        vector = payload.pop('embedding', [])
        payload['event_id'] = eid
        with self.connect() as db:
            old = db.execute('SELECT payload,vector,spec FROM events WHERE id=?', (eid,)).fetchone()
            if old and not clip.reviewed:
                previous = json.loads(old[0])
                for key in ('reviewed', 'selected', 'decision_reason'):
                    if key in previous:
                        payload[key] = previous[key]
                        setattr(clip,key,previous[key])
            blob = self.encode(vector) if len(vector) else (old[1] if old else None)
            spec = json.dumps(FEATURE_SPEC, sort_keys=True) if len(vector) else (old[2] if old else None)
            db.execute('INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?)',
                       (eid, sid, clip.start, clip.end, json.dumps(payload, ensure_ascii=False), blob, spec))
        return eid

    def feature(self, sid, start, end):
        with self.connect() as db:
            row = db.execute('SELECT vector,spec,payload FROM events WHERE source_id=? '
                             'AND ABS(start-?)<0.0001 AND ABS(end-?)<0.0001', (sid, start, end)).fetchone()
        if not row or not row[0] or json.loads(row[1]) != FEATURE_SPEC:
            return None
        return self.decode(row[0]), json.loads(row[2])

    def event_for(self, path, start, end):
        source = self.source_for(path)
        if not source:
            return None
        with self.connect() as db:
            row = db.execute('SELECT id,payload FROM events WHERE source_id=? '
                             'AND ABS(start-?)<0.0001 AND ABS(end-?)<0.0001', (source['id'], start, end)).fetchone()
        return (source, json.loads(row[1])) if row else None

    def review(self, sid, start, end, accepted, reason=''):
        with self.connect() as db:
            row=db.execute('SELECT id,payload FROM events WHERE source_id=? AND ABS(start-?)<0.0001 '
                           'AND ABS(end-?)<0.0001',(sid,start,end)).fetchone()
            if row:
                payload=json.loads(row[1])
                decision = reason or (payload.get('decision_reason') if payload.get('reviewed') and payload.get('selected')==bool(accepted) else '')
                payload.update(reviewed=True,selected=bool(accepted),decision_reason=decision or ('Принято' if accepted else 'Неинтересно'))
                db.execute('UPDATE events SET payload=? WHERE id=?',(json.dumps(payload,ensure_ascii=False),row[0]))

    def finish(self, sid, event_ids):
        source = self.source(sid)
        source['complete'] = True
        source['current_events'] = list(event_ids)
        source['feature_spec'] = FEATURE_SPEC
        with self.connect() as db:
            entries = [db.execute('SELECT vector,spec FROM events WHERE id=?', (eid,)).fetchone() for eid in event_ids]
            source['feature_ready'] = bool(entries) and all(row and row[0] and row[1] and json.loads(row[1])==FEATURE_SPEC for row in entries)
            db.execute('UPDATE sources SET payload=? WHERE id=?', (json.dumps(source, ensure_ascii=False), sid))

    def can_release(self, path):
        source = self.source_for(path)
        if not (source and source.get('complete') and source.get('feature_ready') and source.get('feature_spec')==FEATURE_SPEC):
            return False
        ids=source.get('current_events',[])
        if not ids or not all(isinstance(eid,str) for eid in ids):
            return False
        ids=set(ids)
        with self.connect() as db:
            marks=','.join('?' for _ in ids)
            rows=db.execute(f'SELECT source_id,vector,spec FROM events WHERE id IN ({marks})',tuple(ids)).fetchall()
        if len(rows)!=len(ids):return False
        try:
            return all(sid==source['id'] and blob and spec and json.loads(spec)==FEATURE_SPEC
                       and self.decode(blob).shape==(512,) for sid,blob,spec in rows)
        except (ValueError,TypeError,zlib.error):
            return False

    def list_events(self, limit=1000):
        with self.connect() as db:
            rows = db.execute('SELECT e.payload,s.payload,e.vector,e.spec FROM events e '
                              'JOIN sources s ON s.id=e.source_id ORDER BY e.rowid DESC LIMIT ?', (limit,)).fetchall()
        result = []
        for payload, source, vector, spec in rows:
            value = json.loads(payload)
            value['source'] = json.loads(source)
            value['feature_ready'] = bool(vector and json.loads(spec) == FEATURE_SPEC)
            result.append(value)
        return result
