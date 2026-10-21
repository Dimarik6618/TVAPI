import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import event_memory
import moto_shorts as m
from event_archive import EventArchive
from hardware import Runtime
from project_store import fingerprint


def embedding():
    value = np.zeros(512, np.float32)
    value[0] = 1.
    return value


class Encoder:
    def __init__(self):
        self.calls = 0

    def extract(self, *args):
        self.calls += 1
        return embedding()


class MemoryPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='moto-memory-speed-')
        self.root = Path(self.temp.name)
        self.source = self.root/'ride.mp4'
        m.tasks.run([m.ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                     'testsrc2=s=160x90:r=24', '-t', '4', '-c:v', 'libx264',
                     str(self.source)], check=True)
        self.archive = EventArchive(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def analyze(self, **kwargs):
        return m.analyze_batch([self.source], 0, 2, threshold=0,
                               use_proxies=False, preferences_folder=self.root,
                               learning_folder=self.root, capture_memory=True, **kwargs)

    def test_disabling_frozen_features_never_constructs_encoder_and_retains_source(self):
        with patch.object(event_memory, 'VideoFeatureExtractor', side_effect=AssertionError('hidden encoder')):
            rows, errors = self.analyze(memory_features=False)
        self.assertFalse(errors)
        self.assertTrue(rows)
        events = self.archive.list_events()
        self.assertTrue(events)
        self.assertTrue(all(e['details']['memory'] == 'analysis_only' for e in events))
        self.assertFalse(any(e['feature_ready'] for e in events))
        self.assertFalse(self.archive.can_release(self.source))
        self.assertTrue(self.source.is_file())
        report=json.loads((self.root/'moto_performance.json').read_text(encoding='utf-8'))
        self.assertFalse(report['memory_features'])
        entry=report['files'][0]
        self.assertEqual(entry['status'],'complete')
        self.assertEqual(entry['duration_seconds'],4.)
        self.assertIn('scan',entry['stage_seconds'])
        self.assertIn('event_memory',entry['stage_seconds'])
        self.assertEqual(entry['candidates_returned'],len(rows))

    def test_failed_stage_is_in_timing_report_and_source_is_preserved(self):
        with patch.object(m,'scan_video',side_effect=RuntimeError('controlled scan failure')):
            rows,errors=self.analyze(memory_features=False)
        self.assertFalse(rows)
        self.assertTrue(errors)
        entry=json.loads((self.root/'moto_performance.json').read_text(encoding='utf-8'))['files'][0]
        self.assertEqual(entry['status'],'error')
        self.assertIn('scan',entry['stage_seconds'])
        self.assertTrue(self.source.is_file())

    def test_resume_without_frozen_features_uses_checkpoint_without_decoding(self):
        completed = {}
        def checkpoint(path, clips):
            completed[str(path.resolve())] = {'fingerprint': fingerprint(path),
                'clips': [m.asdict(c) for c in clips], 'event_model': 'off'}
        first, errors = self.analyze(memory_features=False, on_complete=checkpoint)
        self.assertFalse(errors)
        with patch.object(m, 'scan_video', side_effect=AssertionError('must reuse completed scan')):
            resumed, errors = self.analyze(memory_features=False, resume=completed)
        self.assertFalse(errors)
        self.assertEqual([m.asdict(c) for _, c in first], [m.asdict(c) for _, c in resumed])

    def test_enabling_features_after_fast_scan_prepares_them_before_cleanup(self):
        completed = {}
        def checkpoint(path, clips):
            completed[str(path.resolve())] = {'fingerprint': fingerprint(path),
                'clips': [m.asdict(c) for c in clips], 'event_model': 'off'}
        first, errors = self.analyze(memory_features=False, on_complete=checkpoint)
        self.assertFalse(errors)
        encoder = Encoder()
        second, errors = self.analyze(memory_features=True, resume=completed,
            model_pool={'memory-features:auto': encoder})
        self.assertFalse(errors)
        self.assertGreater(encoder.calls, 0)
        self.assertTrue(self.archive.can_release(self.source))
        self.assertEqual([(c.start,c.end,c.score) for _, c in first],
                         [(c.start,c.end,c.score) for _, c in second])

    def test_reanalysis_reuses_exact_clip_embedding_without_model_or_frame_windows(self):
        encoder = Encoder()
        first, errors = self.analyze(model_pool={'memory-features:auto': encoder})
        self.assertFalse(errors)
        self.assertGreater(encoder.calls, 0)
        with patch.object(event_memory, 'VideoFeatureExtractor', side_effect=AssertionError('must reuse event memory')):
            second, errors = self.analyze()
        self.assertFalse(errors)
        self.assertEqual([m.asdict(c) for _, c in first], [m.asdict(c) for _, c in second])
        self.assertTrue(self.archive.can_release(self.source))

    def test_corrupt_or_incompatible_saved_embedding_is_recomputed(self):
        duration, data, _ = m.scan_video(self.source,m.ffmpeg_exe(),lambda *a:None)
        for invalid in ('corrupt', 'incompatible'):
            with self.subTest(invalid=invalid):
                encoder = Encoder()
                original = m.Clip(0,4,.7,embedding=embedding().tolist())
                sid = self.archive.register(self.source,duration)
                eid = self.archive.save_clip(sid,original)
                with self.archive.connect() as db:
                    if invalid == 'corrupt':
                        db.execute('UPDATE events SET vector=? WHERE id=?',(b'bad-compressed-vector',eid))
                    else:
                        db.execute('UPDATE events SET spec=? WHERE id=?',(json.dumps({'version':999}),eid))
                clips = [m.Clip(0,4,.7)]
                failure = event_memory.remember(self.source,duration,data,clips,self.root,Runtime(),
                                               {'memory-features:auto':encoder})
                self.assertFalse(failure)
                self.assertEqual(encoder.calls,1)
                np.testing.assert_array_equal(clips[0].embedding,embedding())
                self.assertTrue(self.archive.can_release(self.source))

    def test_reused_encoder_keeps_its_original_status_callback(self):
        encoder=Encoder()
        original_status=lambda text:None
        encoder.on_status=original_status
        rows,errors=self.analyze(model_pool={'memory-features:auto':encoder})
        self.assertFalse(errors)
        self.assertTrue(rows)
        self.assertIs(encoder.on_status,original_status)


if __name__ == '__main__':
    unittest.main()
