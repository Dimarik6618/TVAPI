"""Explicit video annotations, grouped validation and a small regularized classifier.

Public references and user annotations never write to moto_preferences.json.
The feature encoder stays frozen; only the binary event classifier is trained.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import task_control as tasks
from project_store import atomic_json, digest, fingerprint
from video_features import FEATURE_SPEC, VideoFeatureExtractor, unit_vector

MIN_PER_CLASS = 5
MODEL_VERSION = 1
EXCLUDED_REASONS = {'Неудачные границы', 'Повтор', 'Тряска / плохое качество'}


def read_annotations(path):
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            candidates = [n for n in archive.namelist() if Path(n).name in
                          ('urban_dataset.json', 'urban_annotations.json', 'annotations.json')]
            if len(candidates) != 1:
                raise ValueError('В архиве должен быть один файл urban_dataset.json или annotations.json.')
            if archive.getinfo(candidates[0]).file_size > 10 * 1024 ** 2:
                raise ValueError('Слишком большой файл разметки.')
            data = json.loads(archive.read(candidates[0]))
    else:
        if path.stat().st_size > 10 * 1024 ** 2:
            raise ValueError('Слишком большой файл разметки.')
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    return normalize_annotations(data, path.parent)


def normalize_annotations(data, base=Path('.')):
    """Accept both the previous urban experiment and the new portable JSON format."""
    if not isinstance(data, dict) or not isinstance(data.get('sources'), list) or not isinstance(data.get('examples'), list):
        raise ValueError('Нужны списки sources и examples в JSON разметки.')
    sources = []
    ids = set()
    for raw in data['sources']:
        sid = str(raw.get('id', ''))
        group = str(raw.get('group') or raw.get('source_group') or '')
        filename = raw.get('file') or raw.get('analysis_filename')
        if not sid or sid in ids or not group or not isinstance(filename, str):
            raise ValueError('У каждого исходника должны быть уникальный id, file и group (исходная запись).')
        file = Path(filename)
        if not file.is_absolute():
            file = Path(base) / file
        legacy = 'analysis_filename' in raw
        sources.append({'id': sid, 'group': group, 'file': Path(filename).name,
                        'path': str(file.resolve()) if file.is_file() else '',
                        'clip_url': str(raw.get('clip_url', '')),
                        'sha256': str(raw.get('sha256', raw.get('sha256_downloaded_file', ''))),
                        'analysis_sha256': str(raw.get('analysis_sha256', raw.get('sha256_analysis_file', ''))),
                        'analysis_offset': float(raw.get('analysis_offset', raw.get('analysis_time_offset_seconds', 0.))),
                        'offset': 0., 'legacy': legacy,
                        'group_basis': str(raw.get('group_basis', raw.get('source_group_basis', 'user_declared'))),
                        'title': str(raw.get('title', sid))})
        ids.add(sid)
    examples = []
    for raw in data['examples']:
        sid = str(raw.get('source_id', ''))
        start = float(raw.get('start', raw.get('clip_time_start', math.nan)))
        end = float(raw.get('end', raw.get('clip_time_end', math.nan)))
        if sid not in ids or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError('Метка ссылается на неизвестный исходник или содержит неверные таймкоды.')
        if not isinstance(raw.get('accepted'), bool):
            raise ValueError('У метки accepted должно быть true или false.')
        examples.append({'id': str(raw.get('id') or digest([sid, start, end])[:24]),
                         'source_id': sid, 'start': start, 'end': end, 'accepted': raw['accepted'],
                         'category': str(raw.get('category', 'event' if raw['accepted'] else 'routine')),
                         'label': str(raw.get('label', '')),
                         'origin': str(raw.get('origin', raw.get('annotation_origin', 'imported'))),
                         'reviewed_by_user': bool(raw.get('reviewed_by_user', False)),
                         'confidence': str(raw.get('confidence', raw.get('annotation_confidence', 'unknown')))})
    if len({e['id'] for e in examples}) != len(examples):
        raise ValueError('Повторяющиеся id меток.')
    return {'version': 1, 'sources': sources, 'examples': examples}


def full_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            tasks.check()
            result.update(chunk)
    return result.hexdigest()


class LearningStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.root = self.folder / 'moto_learning'
        self.path = self.root / 'annotations.json'
        self.model_path = self.root / 'event_model.json'
        self.settings_path = self.root / 'settings.json'
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            if self.data.get('version') != 1:
                raise ValueError('Неподдерживаемая версия разметки обучения.')
        else:
            self.data = {'version': 1, 'sources': [], 'examples': []}

    def save(self):
        atomic_json(self.path, self.data)

    def signature(self):
        # Moving files does not alter the meaning of the annotations.
        return digest({'sources': sorted((s['id'], s['group']) for s in self.data['sources']),
                       'examples': sorted((e['id'], e['source_id'], e['start'], e['end'], e['accepted'])
                                          for e in self.data['examples'])})

    def import_file(self, path):
        incoming = read_annotations(path)
        sources = {s['id']: dict(s) for s in self.data['sources']}
        examples = {e['id']: dict(e) for e in self.data['examples']}
        for s in incoming['sources']:
            if s['id'] in sources:
                old = sources[s['id']]
                if (old['group'], old['file']) != (s['group'], s['file']):
                    raise ValueError(f"Другой исходник с тем же id: {s['id']}.")
                s['path'], s['offset'] = old.get('path', ''), old.get('offset', 0.)
                for key in ('archive_id','bound_fingerprint'):
                    if key in old:s[key]=old[key]
            sources[s['id']] = s
        for e in incoming['examples']:
            old = examples.get(e['id'])
            if old and old.get('reviewed_by_user'):
                continue
            examples[e['id']] = e
        self.data = {'version': 1, 'sources': list(sources.values()), 'examples': list(examples.values())}
        self.save()
        return len(incoming['examples'])

    def bind_folder(self, folder):
        files = {p.name.lower(): p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() == '.mp4'}
        matched, errors = 0, []
        for source in self.data['sources']:
            tasks.check()
            candidate = files.get(source['file'].lower())
            if candidate:
                try:
                    self.bind_source(source['id'], candidate, save=False)
                    matched += 1
                except ValueError as exc:
                    errors.append(str(exc))
        self.save()
        return matched, errors

    def bind_source(self, sid, path, save=True):
        source = next(s for s in self.data['sources'] if s['id'] == sid)
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f'Не найден MP4: {path.name}')
        expected = source.get('sha256')
        offset = 0.
        if expected:
            actual = full_sha256(path)
            if actual != expected:
                if actual == source.get('analysis_sha256'):
                    offset = source.get('analysis_offset', 0.)
                else:
                    raise ValueError(f"{source['file']}: MP4 не совпадает с размеченным исходником.")
        source.update(path=str(path), offset=offset, bound_fingerprint=fingerprint(path))
        if save:
            self.save()

    def add_user(self, path, start, end, accepted, label='', group=None):
        from event_archive import EventArchive
        path = Path(path).resolve()
        archive = EventArchive(self.folder)
        remembered = archive.event_for(path, start, end)
        if (not path.is_file() and not remembered) or not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError('Нужны MP4 или сохранённое событие и корректные таймкоды.')
        sources = {s['id']: s for s in self.data['sources']}
        legacy_sid = 'user_' + digest(str(path))[:20]
        sid = legacy_sid
        if remembered:
            saved_source, _ = remembered
            matching = next((s for s in sources.values() if s.get('archive_id')==saved_source['id']),None)
            legacy = sources.get(legacy_sid)
            if matching:
                sid=matching['id']
            elif legacy and not legacy.get('archive_id') and (not legacy.get('bound_fingerprint') or all(
                legacy['bound_fingerprint'].get(k)==saved_source['fingerprint'].get(k) for k in ('size','sample'))):
                sid=legacy_sid
            else:
                sid='user_'+saved_source['id'][:20]
        old = sources.get(sid, {})
        sources[sid] = {**old, 'id': sid, 'file': path.name, 'path': str(path),
                        'group': group or old.get('group') or str(path), 'group_basis': 'user_declared',
                        'title': path.name, 'offset': 0., 'sha256': '', 'clip_url': ''}
        if remembered:
            saved_source, _ = remembered
            sources[sid].update(archive_id=saved_source['id'], bound_fingerprint=saved_source['fingerprint'],
                               file=saved_source['file'], title=saved_source['file'])
        eid = digest([sid, round(start, 3), round(end, 3)])[:24]
        examples = {e['id']: e for e in self.data['examples']}
        examples[eid] = {'id': eid, 'source_id': sid, 'start': start, 'end': end,
                         'accepted': bool(accepted), 'label': label, 'category': 'user_event' if accepted else 'routine',
                         'origin': 'user', 'reviewed_by_user': True, 'confidence': 'user'}
        self.data = {'version': 1, 'sources': list(sources.values()), 'examples': list(examples.values())}
        self.save()
        if remembered:
            archive.review(saved_source['id'],start,end,accepted)
        return eid

    def model(self):
        if not self.model_path.exists():
            return None
        data = json.loads(self.model_path.read_text(encoding='utf-8'))
        if data.get('version') != MODEL_VERSION or data.get('feature_spec') != FEATURE_SPEC:
            raise ValueError('Модель обучения несовместима с этой версией программы. Переобучите её.')
        model = LinearEventModel.from_dict(data['classifier'])
        return data, model

    def enabled(self):
        try:
            return bool(json.loads(self.settings_path.read_text(encoding='utf-8')).get('enabled'))
        except FileNotFoundError:
            return False

    def set_enabled(self, enabled):
        if enabled and self.model() is None:
            raise ValueError('Сначала обучите и проверьте модель.')
        atomic_json(self.settings_path, {'enabled': bool(enabled)})

    def install_reference_model(self, path):
        if self.model_path.exists():
            return False
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != MODEL_VERSION or data.get('feature_spec') != FEATURE_SPEC:
            raise ValueError('Пилотная модель несовместима с программой.')
        LinearEventModel.from_dict(data['classifier'])
        if not data.get('reference_model') or not isinstance(data.get('evaluation'), dict):
            raise ValueError('Некорректная пилотная модель.')
        self.set_enabled(False)
        atomic_json(self.model_path, data)
        atomic_json(self.root / 'training_report.json', data['evaluation'])
        return True

    def token(self):
        if not self.enabled():
            return 'off'
        value = self.model()
        return digest(value[0]) if value else 'off'

    def summary(self):
        from event_archive import EventArchive
        sources = self.data['sources']
        positives = sum(e['accepted'] for e in self.data['examples'])
        present = sum(bool(s.get('path')) and Path(s['path']).is_file() for s in sources)
        archive = EventArchive(self.folder)
        ready = sum(bool(s.get('archive_id') and archive.feature(s['archive_id'], e['start'] - s.get('offset', 0.),
                        e['end'] - s.get('offset', 0.))) for e in self.data['examples']
                    for s in sources if s['id'] == e['source_id'])
        return f"{positives} событий · {len(self.data['examples']) - positives} обычных · MP4 {present}/{len(sources)} · память {ready}/{len(self.data['examples'])}"


def download_references(store, on_progress=lambda f, text: None):
    """Download only the annotated public Coub video track, without added music."""
    sources = [s for s in store.data['sources'] if re.fullmatch(r'https://coub\.com/view/[a-zA-Z0-9]+', s.get('clip_url', ''))]
    if not sources:
        raise ValueError('В разметке нет публичных городских примеров для загрузки.')
    folder = store.root / 'reference_videos'
    folder.mkdir(parents=True, exist_ok=True)
    downloaded, errors = 0, []
    for i, source in enumerate(sources):
        tasks.check()
        on_progress(i / len(sources), f"Пример {i+1}/{len(sources)} · {source['title']}")
        if source.get('path') and Path(source['path']).is_file():
            downloaded += 1
            continue
        code = source['clip_url'].rsplit('/', 1)[-1]
        output = folder / (code + '.mp4')
        temporary = output.with_suffix('.part')
        try:
            with urllib.request.urlopen('https://coub.com/api/v2/coubs/' + code, timeout=15) as response:
                info = json.loads(response.read(2 * 1024 ** 2))
            versions = info['file_versions']['html5']['video']
            link = (versions.get('high') or versions.get('med') or versions.get('low'))['url']
            if not link.startswith('https://'):
                raise ValueError('У примера нет доступной HTTPS-видеодорожки.')
            with urllib.request.urlopen(link, timeout=15) as response, temporary.open('wb') as file:
                total = 0
                while chunk := response.read(256 * 1024):
                    tasks.check()
                    total += len(chunk)
                    if total > 100 * 1024 ** 2:
                        raise ValueError('Размер примера больше 100 МБ.')
                    file.write(chunk)
            if total < 1024:
                raise ValueError('Видеодорожка недоступна.')
            store.bind_source(source['id'], temporary, save=False)
            os.replace(temporary, output)
            source['path'] = str(output.resolve())
            source['bound_fingerprint'] = fingerprint(output)
            store.save()
            downloaded += 1
        except Exception as exc:
            errors.append(f"{source['id']}: {exc}")
        finally:
            temporary.unlink(missing_ok=True)
    on_progress(1., f'Доступно {downloaded}/{len(sources)} примеров.')
    return downloaded, errors


class LinearEventModel:
    """Balanced logistic classifier; all centering and fitting occur inside each fold."""
    SCALE = 10.
    L2 = .2

    def __init__(self, center, weights, bias):
        self.center = np.asarray(center, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        if self.center.ndim != 1 or self.weights.shape != self.center.shape or not np.isfinite(
                np.r_[self.center, self.weights, self.bias]).all():
            raise ValueError('Некорректные коэффициенты обученной модели.')

    @classmethod
    def fit(cls, x, y, groups):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if len(x) != len(y) or not len(x) or len(np.unique(y)) != 2:
            raise ValueError('Для обучения нужны события и обычные отрезки.')
        center = np.mean(x, axis=0)
        x = (x - center) * cls.SCALE
        # A long labelled source should not outweigh every other source.
        sample_weight = np.array([1. / sum(g == group for g in groups) for group in groups])
        for label in (0., 1.):
            mask = y == label
            sample_weight[mask] *= .5 / sample_weight[mask].sum()
        weights, bias = np.zeros(x.shape[1]), 0.
        norm = float(np.linalg.norm(x * np.sqrt(sample_weight[:, None]), ord=2))
        step = 1. / (.25 * (norm ** 2 + 1.) + cls.L2)
        for i in range(400):
            if i % 20 == 0:
                tasks.check()
            p = 1. / (1. + np.exp(-np.clip(x @ weights + bias, -30, 30)))
            error = (p - y) * sample_weight
            weights -= step * (x.T @ error + cls.L2 * weights)
            bias -= step * error.sum()
        return cls(center, weights, bias)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != len(self.center) or not np.isfinite(x).all():
            raise ValueError('Признаки не подходят обученной модели.')
        z = (x - self.center) * self.SCALE @ self.weights + self.bias
        return 1. / (1. + np.exp(-np.clip(z, -30, 30)))

    def to_dict(self):
        return {'center': self.center.tolist(), 'weights': self.weights.tolist(), 'bias': self.bias,
                'scale': self.SCALE, 'l2': self.L2}

    @classmethod
    def from_dict(cls, data):
        if data.get('scale') != cls.SCALE or data.get('l2') != cls.L2:
            raise ValueError('Неподдерживаемая версия классификатора.')
        model = cls(data['center'], data['weights'], data['bias'])
        if len(model.center) != 512:
            raise ValueError('У обученной модели должно быть 512 признаков.')
        return model


def metrics(labels, probabilities, threshold=.5):
    y, p = np.asarray(labels, dtype=bool), np.asarray(probabilities)
    predicted = p >= threshold
    tp, fn = int(np.sum(y & predicted)), int(np.sum(y & ~predicted))
    fp, tn = int(np.sum(~y & predicted)), int(np.sum(~y & ~predicted))
    recall, specificity = tp / max(1, tp+fn), tn / max(1, tn+fp)
    pairs = [(a > b) + .5*(a == b) for a in p[y] for b in p[~y]]
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'precision': tp/max(1, tp+fp),
            'recall': recall, 'specificity': specificity, 'balanced_accuracy': (recall+specificity)/2,
            'auc': float(np.mean(pairs)) if pairs else None, 'threshold': threshold}


def grouped_evaluate(x, labels, groups, baseline):
    x, y, groups = np.asarray(x), np.asarray(labels, dtype=int), np.asarray(groups)
    predictions, folds = np.zeros(len(y)), []
    if len(set(groups)) < 3 or min(np.sum(y == 0), np.sum(y == 1)) < MIN_PER_CLASS:
        raise ValueError('Нужно минимум 5 событий, 5 обычных отрезков и 3 независимые исходные записи.')
    for group in sorted(set(groups)):
        tasks.check()
        test = groups == group
        train = ~test
        if min(np.sum(y[train] == 0), np.sum(y[train] == 1)) < 2:
            raise ValueError('Слишком мало независимых событий или обычных отрезков вне одной записи.')
        model = LinearEventModel.fit(x[train], y[train], groups[train])
        predictions[test] = model.predict(x[test])
        folds.append({'held_out_group': str(group), 'train_groups': sorted(set(groups[train])),
                      'train_examples': int(train.sum()), 'test_indices': np.where(test)[0].tolist()})
    blended = .45 * np.asarray(baseline) + .55 * predictions
    selected_blend = np.where(predictions >= .5, blended, 0.)
    return {'protocol': 'leave_one_source_group_out', 'hyperparameters': 'fixed_before_validation',
            'assessment': 'development_pilot_not_final_holdout',
            'selection_rule': 'semantic_score >= 0.5 AND blended_score >= user_threshold',
            'folds': folds, 'predictions': predictions.tolist(), 'blended_predictions': blended.tolist(),
            'baseline': metrics(y, baseline, .22), 'semantic': metrics(y, predictions, .5),
            'blend': metrics(y, selected_blend, .22)}


def resolved_groups(sources):
    """Union declared parent groups and identical source copies before any split."""
    parent = list(range(len(sources)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, s in enumerate(sources):
        fp = s.get('bound_fingerprint')
        if not fp:
            if not s.get('path') or not Path(s['path']).is_file():
                raise ValueError('Нет идентификатора исходной записи для проверки обучения.')
            fp = fingerprint(Path(s['path']))
        for key in (('group', s['group']), ('copy', fp['size'], fp['sample'])):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    # Stable group names make the validation protocol inspectable.
    return {s['id']: min(sources[j]['group'] for j in range(len(sources)) if find(j) == find(i))
            for i, s in enumerate(sources)}


def train(store, ffmpeg, runtime=None, on_progress=lambda f, text: None, extractor=None):
    # Import here to keep the annotation/model module usable without GUI initialization.
    from moto_shorts import Clip, baseline_at, scan_video
    from event_archive import EventArchive
    examples = store.data['examples']
    annotation_signature = store.signature()
    if not examples:
        raise ValueError('Сначала добавьте разметку.')
    used_ids = {e['source_id'] for e in examples}
    sources = [s for s in store.data['sources'] if s['id'] in used_ids]
    archive = EventArchive(store.folder)
    by_id = {s['id']: s for s in sources}
    frozen = {}
    for e in examples:
        s = by_id[e['source_id']]
        if not s.get('archive_id') and s.get('path'):
            saved = archive.source_for(s['path'])
            if saved:
                s.update(archive_id=saved['id'], bound_fingerprint=saved['fingerprint'])
        if s.get('archive_id'):
            item = archive.feature(s['archive_id'], e['start'] - s.get('offset', 0.), e['end'] - s.get('offset', 0.))
            if item is not None:
                frozen[e['id']] = item
    missing = [by_id[e['source_id']]['file'] for e in examples if e['id'] not in frozen
               and (not by_id[e['source_id']].get('path') or not Path(by_id[e['source_id']]['path']).is_file())]
    if missing:
        raise ValueError(f"Нет сохранённых признаков для {len(missing)} меток: " + ', '.join(missing[:4]) +
                         '. Верните исходники один раз и запустите обучение, чтобы сохранить признаки.')
    groups = resolved_groups(sources)
    if len(set(groups.values())) < 3 or min(sum(e['accepted'] for e in examples), sum(not e['accepted'] for e in examples)) < MIN_PER_CLASS:
        raise ValueError('Нужно минимум 5 событий, 5 обычных отрезков и 3 независимые исходные записи.')
    from project_store import CacheStore
    from hardware import Runtime
    runtime = runtime or Runtime.detect()
    extractor = extractor or VideoFeatureExtractor(ffmpeg, store.folder, runtime,
                                                   lambda text: on_progress(0., text), archive=archive)
    scans, features, baselines = {}, [], []
    cache = CacheStore(store.folder)
    signatures = {}
    for i, example in enumerate(examples):
        tasks.check()
        source = by_id[example['source_id']]
        if example['id'] in frozen:
            vector, saved_clip = frozen[example['id']]
            features.append(vector)
            baselines.append(saved_clip.get('signals', {}).get('baseline', saved_clip['score']))
            on_progress((i+1) / len(examples) * .8, f'Из памяти: {i+1}/{len(examples)}')
            continue
        path = Path(source['path'])
        if source['id'] not in scans:
            if source.get('sha256'):
                current = fingerprint(path)
                if source.get('bound_fingerprint') != current:
                    store.bind_source(source['id'], path)
            scans[source['id']] = scan_video(path, ffmpeg, lambda f, text: on_progress(
                (i + f*.2) / len(examples) * .8, text), cache=cache, runtime=runtime)[:2]
            signatures[source['id']] = fingerprint(path)
        duration, data = scans[source['id']]
        start, end = example['start'] - source.get('offset', 0.), example['end'] - source.get('offset', 0.)
        if not 0 <= start < end <= duration + .06:
            raise ValueError(f"{source['file']}: таймкод метки {example['id']} вне записи ({duration:.2f} с).")
        end = min(end, duration)
        on_progress(i / len(examples) * .8, f"Признаки {i+1}/{len(examples)} · {source['file']}")
        vector = extractor.extract(path, start, end)
        if fingerprint(path) != signatures[source['id']]:
            raise ValueError('Один из MP4 изменился во время обучения. Повторите обработку.')
        base = baseline_at(data, start, end)
        features.append(vector)
        baselines.append(base)
        sid = archive.register(path, duration)
        source.update(archive_id=sid, bound_fingerprint=fingerprint(path))
        archive.save_clip(sid, Clip(start, end, base, selected=example['accepted'],
                          reviewed=example.get('reviewed_by_user', False), reason=example.get('label', ''),
                          signals={'baseline': base}, embedding=vector.tolist()))
    store.save()
    tasks.check()
    on_progress(.82, 'Проверка на отдельных исходных записях')
    x = np.asarray(features)
    labels = [int(e['accepted']) for e in examples]
    group_ids = [groups[e['source_id']] for e in examples]
    evaluation = grouped_evaluate(x, labels, group_ids, baselines)
    evaluation['examples'] = [{'id': e['id'], 'group': group_ids[i], 'accepted': e['accepted'],
                               'origin': e.get('origin'), 'baseline_score': baselines[i]}
                              for i, e in enumerate(examples)]
    evaluation['groups'] = len(set(group_ids))
    evaluation['examples_count'] = len(examples)
    evaluation['user_reviewed_count'] = sum(e.get('reviewed_by_user', False) for e in examples)
    evaluation['from_memory_count'] = len(frozen)
    evaluation['pilot'] = len(examples) < 30 or len(set(group_ids)) < 5
    evaluation['improved'] = (evaluation['blend']['balanced_accuracy'] >=
                             evaluation['baseline']['balanced_accuracy'] + .03 and
                             evaluation['semantic']['recall'] >= .5 and evaluation['semantic']['specificity'] >= .5)
    evaluation['limitations'] = ['Оценки событий не измеряют популярность рилса.',
                                  'Визуально изменённые копии нужно объединять вручную по исходной записи.',
                                  'Короткие опубликованные нарезки не проверяют поиск границ на полной поездке.',
                                  'Сигнал классификатора не является калиброванной вероятностью.']
    tasks.check()
    on_progress(.96, 'Сохранение модели и результата проверки')
    model = LinearEventModel.fit(x, labels, group_ids)
    for sid, before in signatures.items():
        if fingerprint(Path(by_id[sid]['path'])) != before:
            raise ValueError('Один из MP4 изменился во время обучения. Повторите обработку.')
    if LearningStore(store.folder).signature() != annotation_signature:
        raise ValueError('Разметка изменилась во время обучения. Запустите обучение снова.')
    output = {'version': MODEL_VERSION, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'feature_spec': FEATURE_SPEC, 'dataset_signature': store.signature(),
              'source_fingerprints': {s['id']: s.get('bound_fingerprint', signatures.get(s['id'])) for s in sources},
              'classifier': model.to_dict(), 'evaluation': evaluation,
              'encoder_retrained': False, 'selection_blend': {'baseline': .45, 'semantic': .55}}
    tasks.check()
    # Disable first: even a filesystem failure cannot silently activate new weights.
    store.set_enabled(False)
    atomic_json(store.root / 'training_report.json', evaluation)
    atomic_json(store.model_path, output)
    on_progress(1., 'Модель обучена. Проверьте результат и включите её при необходимости.')
    return output


class EventSelector:
    """Find quiet visual events as well as rescore the motion-generated candidates."""
    def __init__(self, store, ffmpeg, runtime=None, on_status=lambda text: None):
        from event_archive import EventArchive
        value = store.model()
        if value is None:
            raise ValueError('Нет обученной модели отбора событий.')
        self.output, self.model = value
        self.extractor = VideoFeatureExtractor(ffmpeg, store.folder, runtime, on_status, archive=EventArchive(store.folder))

    def discover(self, source, duration, data, existing, length, count, on_progress=lambda f: None):
        from moto_shorts import Clip, baseline_at
        from video_features import window_ranges
        spans = window_ranges(0., duration)
        identity = fingerprint(source)
        events = []
        for i, (start, end) in enumerate(spans):
            tasks.check()
            vector = self.extractor.window(source, start, end, identity)
            score = float(self.model.predict(vector))
            events.append((start, end, score))
            on_progress((i+1) / len(spans))
        # Join consecutive positive windows, retaining context and an automatic length.
        groups = []
        for a, b, score in events:
            if score < .5:
                continue
            if groups and a <= groups[-1][1] + .01 and b-groups[-1][0] <= 83:
                groups[-1][1] = b
                groups[-1][2] = max(groups[-1][2], score)
            else:
                groups.append([a, b, score])
        result = list(existing)
        added = []
        for a, b, score in sorted(groups, key=lambda x: x[2], reverse=True):
            if length:
                start = max(0., min((a+b-length)/2, duration-length))
                end = min(duration, start+length)
            else:
                start, end = max(0., a-3), min(duration, b+4)
            if any(min(end, c.end)-max(start, c.start) > .85 * max(end-start, c.end-c.start)
                   for c in result + added):
                continue
            added.append(Clip(round(start, 2), round(end, 2), baseline_at(data, start, end),
                              reason='Кандидат по обученному сюжету'))
            if len(added) >= max(12, count*4):
                break
        return result + added

    def enrich(self, source, clips, on_progress=lambda f: None):
        for i, clip in enumerate(clips):
            tasks.check()
            feature = self.extractor.extract(source, clip.start, clip.end)
            clip.embedding = feature.tolist()
            score = float(self.model.predict(feature))
            clip.signals['learned_event'] = score
            clip.signals['meaning'] = score
            clip.score = round(.45*clip.score + .55*score, 3)
            clip.reason += ' · обученный сюжет'
            on_progress((i+1) / max(1, len(clips)))


def report_text(output):
    e = output['evaluation']
    lines = [f"Примеров: {e['examples_count']} · исходных записей: {e['groups']}",
             f"Ваших проверенных меток: {e['user_reviewed_count']}",
             'Разработческий пилот: каждый исходник по очереди полностью исключается из обучения.', '']
    for key, title in [('baseline', 'Движение / звук'), ('semantic', 'Обученный сюжет'), ('blend', 'Совместный отбор')]:
        m = e[key]
        lines.append(f"{title}: события {m['tp']}/{m['tp']+m['fn']}; ошибочно выбрано обычных {m['fp']}/{m['fp']+m['tn']}; "
                     f"сбалансированная точность {m['balanced_accuracy']:.0%}.")
    lines += ['', 'Пилот: мало примеров; нужны ваши полные городские поездки.' if e['pilot'] else 'Результат относится только к этой выборке.',
              'Совместный отбор улучшился на проверке.' if e['improved'] else 'Проверка не показала уверенного улучшения совместного отбора.',
              'После обучения модель выключена до её включения в окне обучения.']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Локальное обучение отбора городских событий')
    parser.add_argument('--folder', type=Path, required=True, help='Та же папка вывода, что в основной программе')
    parser.add_argument('--import', dest='annotations', type=Path)
    parser.add_argument('--videos', type=Path)
    parser.add_argument('--download-references', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    store = LearningStore(args.folder)
    progress = lambda f, text: print(f'{f:.0%} {text}', flush=True)
    if args.annotations:
        store.import_file(args.annotations)
    if args.videos:
        print(store.bind_folder(args.videos))
    if args.download_references:
        print(download_references(store, progress))
    if args.train:
        from moto_shorts import ffmpeg_exe
        print(report_text(train(store, ffmpeg_exe(), on_progress=progress)))
    else:
        print(store.summary())
