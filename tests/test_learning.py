import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import learning as l
import moto_shorts as m
import task_control as tasks
from project_store import atomic_json
from video_features import FEATURE_SPEC, VideoFeatureExtractor, window_ranges


def feature(event):
    value = np.zeros(512, dtype=np.float32)
    value[0 if event else 1] = 1.
    return value


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        tasks.bind(None)
        self.temporary.cleanup()

    def dataset(self):
        store = l.LearningStore(self.folder)
        for i, color in enumerate(('red', 'blue', 'green')):
            source = self.folder / f'ride_{i}.mp4'
            tasks.run([m.ffmpeg_exe(), '-y', '-v', 'error', '-f', 'lavfi', '-i',
                       f'color=c={color}:s=160x90:r=24', '-t', '4', '-c:v', 'libx264', str(source)], check=True)
            for start in (0., 1., 2., 3.):
                store.add_user(source, start, start+1, start < 2, group=f'ride_{i}')
        return store

    def old_model(self, store):
        value = {'version': 1, 'feature_spec': FEATURE_SPEC, 'dataset_signature': store.signature(),
                 'classifier': l.LinearEventModel(np.zeros(512), np.zeros(512), 0.).to_dict(),
                 'evaluation': {'sentinel': 'previous model'}}
        atomic_json(store.model_path, value)
        store.set_enabled(True)
        return store.model_path.read_bytes()

    def test_legacy_import_uses_original_timeline_and_keeps_annotation_origin(self):
        asset = Path(__file__).resolve().parents[1] / 'training_examples' / 'urban_annotations.json'
        store = l.LearningStore(self.folder)
        store.import_file(asset)
        self.assertEqual(len(store.data['examples']), 18)
        example = next(e for e in store.data['examples'] if e['source_id'] == '1y740o')
        self.assertEqual(example['start'], 5.5)
        self.assertFalse(example['reviewed_by_user'])
        self.assertEqual(example['origin'], 'assistant_visual_review')
        example.update(accepted=True, reviewed_by_user=True, origin='user')
        store.save()
        store.import_file(asset)
        self.assertTrue(next(e for e in store.data['examples'] if e['id'] == example['id'])['accepted'])
        self.assertFalse((self.folder / 'moto_preferences.json').exists())

    def test_binding_normalized_copy_adjusts_offset_and_rejects_other_bytes(self):
        source = self.folder / 'normalized.mp4'
        source.write_bytes(b'normalized recording')
        store = l.LearningStore(self.folder)
        store.data['sources'] = [{'id': 'a', 'file': 'a.mp4', 'group': 'a',
                                  'sha256': 'raw_hash', 'analysis_sha256': l.full_sha256(source),
                                  'analysis_offset': 5.5}]
        store.bind_source('a', source)
        self.assertEqual(store.data['sources'][0]['offset'], 5.5)
        source.write_bytes(b'other recording')
        with self.assertRaisesRegex(ValueError, 'не совпадает'):
            store.bind_source('a', source)

    def test_identical_renamed_copies_are_grouped_before_validation(self):
        a, b, c = [self.folder / name for name in ('first.mp4', 'renamed.mp4', 'other.mp4')]
        a.write_bytes(b'one original recording')
        shutil.copyfile(a, b)
        c.write_bytes(b'another recording')
        groups = l.resolved_groups([{'id': 'a', 'group': 'ride_A', 'path': str(a)},
                                    {'id': 'b', 'group': 'ride_B', 'path': str(b)},
                                    {'id': 'c', 'group': 'ride_C', 'path': str(c)}])
        self.assertEqual(groups['a'], groups['b'])
        self.assertNotEqual(groups['a'], groups['c'])

    def test_source_group_never_appears_on_both_sides_of_a_fold(self):
        labels = [i % 2 for i in range(12)]
        groups = [f'ride_{i//4}' for i in range(12)]
        x = np.array([feature(y) for y in labels])
        result = l.grouped_evaluate(x, labels, groups, [.3]*12)
        for fold in result['folds']:
            self.assertNotIn(fold['held_out_group'], fold['train_groups'])
            self.assertTrue(all(groups[i] == fold['held_out_group'] for i in fold['test_indices']))
        self.assertEqual(result['semantic']['tp'], 6)
        self.assertEqual(result['semantic']['fp'], 0)
        with self.assertRaisesRegex(ValueError, 'независимые'):
            l.grouped_evaluate(x, labels, ['one']*12, [.3]*12)

    def test_gate_prevents_high_motion_from_overriding_a_routine_prediction(self):
        routine = m.Clip(0, 5, .9, signals={'learned_event': .49})
        event = m.Clip(8, 13, .7, signals={'learned_event': .8})
        self.assertEqual(m.select_best([routine, event], 2, threshold=0), [event])
        horn = m.Clip(15, 20, .6, signals={'learned_event': .2, 'sound': .95})
        self.assertEqual(m.select_best([horn], 1), [horn])

    def test_motion_and_semantic_proposals_do_not_duplicate_one_moment(self):
        first = m.Clip(0, 20, .8, signals={'learned_event': .9})
        same_event = m.Clip(7, 20, .7, signals={'learned_event': .85})
        other = m.Clip(24, 35, .6, signals={'learned_event': .8})
        self.assertEqual(m.select_best([first, same_event, other], 3, deduplicate=False), [first, other])

    def test_feature_cache_skips_weights_and_invalidates_after_media_changes(self):
        source = self.folder / 'ride.mp4'
        source.write_bytes(b'first media')
        extractor = VideoFeatureExtractor('ffmpeg', self.folder)
        with patch.object(extractor, 'load'), patch.object(extractor, '_frames', return_value=[]), \
                patch.object(extractor, '_infer', return_value=feature(True)) as infer:
            extractor.window(source, 0, 6)
            self.assertEqual(infer.call_count, 1)
            with patch.object(extractor, 'load', side_effect=AssertionError('weights requested for a cache hit')):
                cached = extractor.window(source, 0, 6)
            self.assertEqual(cached.shape, (512,))
            source.write_bytes(b'changed media')
            extractor.window(source, 0, 6)
            self.assertEqual(infer.call_count, 2)
        self.assertEqual(window_ranges(2, 4), [(2., 4.)])
        spans = window_ranges(0, 17)
        self.assertEqual(spans[-1][1], 17)
        self.assertTrue(all(b-a <= 6 for a, b in spans))

    def test_training_saves_checked_inactive_classifier_without_touching_preferences(self):
        store = self.dataset()
        preferences = self.folder / 'moto_preferences.json'
        preferences.write_text('{"sentinel":"personal decisions"}', encoding='utf-8')
        before = preferences.read_bytes()
        class Extractor:
            def extract(self, path, start, end):
                return feature(start < 2)
        output = l.train(store, m.ffmpeg_exe(), extractor=Extractor())
        self.assertFalse(store.enabled())
        self.assertFalse(output['encoder_retrained'])
        self.assertEqual(output['evaluation']['groups'], 3)
        self.assertEqual(preferences.read_bytes(), before)
        self.assertTrue(store.model())

    def test_stop_preserves_previous_model_and_activation(self):
        store = self.dataset()
        before = self.old_model(store)
        control = tasks.TaskControl()
        tasks.bind(control)
        class Extractor:
            count = 0
            def extract(self, path, start, end):
                self.count += 1
                if self.count == 2:
                    control.stop()
                    tasks.check()
                return feature(start < 2)
        with self.assertRaises(tasks.Cancelled):
            l.train(store, m.ffmpeg_exe(), extractor=Extractor())
        self.assertEqual(store.model_path.read_bytes(), before)
        self.assertTrue(store.enabled())

    def test_changed_source_prevents_replacing_a_verified_model(self):
        store = self.dataset()
        before = self.old_model(store)
        class Extractor:
            count = 0
            def extract(self, path, start, end):
                self.count += 1
                if self.count == 12:
                    with path.open('ab') as file:
                        file.write(b'changed while training')
                return feature(start < 2)
        with self.assertRaisesRegex(ValueError, 'изменился'):
            l.train(store, m.ffmpeg_exe(), extractor=Extractor())
        self.assertEqual(store.model_path.read_bytes(), before)

    def test_partial_model_failure_restores_original_candidate_scores(self):
        source = self.folder / 'ride.mp4'
        source.write_bytes(b'media')
        store = l.LearningStore(self.folder)
        self.old_model(store)
        data = {'motion': np.arange(12, dtype=np.float32), 'quality': np.ones(12),
                'audio': np.zeros(6), 'signature': np.ones((12, 144))}
        def enrich(path, clips, callback):
            clips[0].score = .99
            clips[0].signals['learned_event'] = .01
            raise RuntimeError('device failure')
        with patch.object(m, 'scan_video', return_value=(6., data, None)), \
                patch.object(m, 'choose_clips', return_value=[m.Clip(0, 6, .7)]), \
                patch.object(l.EventSelector, 'discover', side_effect=lambda path, duration, data, clips, *args: clips), \
                patch.object(l.EventSelector, 'enrich', side_effect=enrich):
            rows, errors = m.analyze_batch([source], 0, 1, use_proxies=False,
                                           preferences_folder=self.folder, threshold=.22)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1].score, .7)
        self.assertNotIn('learned_event', rows[0][1].signals)
        self.assertTrue(any('device failure' in error for _, error in errors))

    def test_quiet_event_candidates_do_not_require_a_motion_peak(self):
        store = l.LearningStore(self.folder)
        self.old_model(store)
        selector = l.EventSelector(store, 'ffmpeg')
        source = self.folder / 'quiet.mp4'
        source.write_bytes(b'quiet media')
        data = {'motion': np.zeros(48), 'audio': np.zeros(24)}
        with patch.object(selector.extractor, 'window', return_value=feature(True)), \
                patch.object(selector.model, 'predict', return_value=.9):
            candidates = selector.discover(source, 24, data, [], 0, 1)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].score, 0.)
        self.assertTrue(0 <= candidates[0].start < candidates[0].end <= 24)


if __name__ == '__main__':
    unittest.main()
