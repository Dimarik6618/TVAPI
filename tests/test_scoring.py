import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import moto_shorts as m
from project_store import CacheStore


class ScoringTests(unittest.TestCase):
    def scan_data(self):
        motion = np.asarray([0,2,1,3,2,5,12,18,15,8,4,3,2,1,3,7,13,11,5,2,1,3,2,4,1], np.float32)
        return {
            'motion': motion,
            'audio': np.asarray([10,10,12,80,90,15,12,11,50,60,12,10,9], np.float32),
            'quality': np.linspace(1,30,len(motion),dtype=np.float32),
            'signature': np.arange(len(motion)*4,dtype=np.float32).reshape(-1,4)/100,
        }

    def test_scores_match_v23_1_at_fractional_bounds_and_partial_tail(self):
        # Golden values recorded from the supplied, unmodified v23.1 archive.
        expected = [
            ((0.,1.), 0., .0534722209, .0431034483, 0., [.02,.03,.04,.05]),
            ((1.25,6.75), .579, .3354166746, .3893678161, .3560606061, [.30,.31,.32,.33]),
            ((7.25,12.5), .471, .7986110449, .3181818182, .2348484848, [.76,.77,.78,.79]),
            ((11.,12.5), .031, .9597222209, .1149425287, 0., [.92,.93,.94,.95]),
        ]
        raw = self.scan_data()
        for mode, data in [('raw', raw), ('prepared', m.prepare_scan_scores(raw))]:
            for (start, end), score, quality, visual, audio, signature in expected:
                with self.subTest(mode=mode, bounds=(start,end)):
                    clip = m.Clip(start,end,m.baseline_at(data,start,end))
                    m.describe_clip(clip,data)
                    self.assertEqual(clip.score, score)
                    self.assertEqual(clip.signals['baseline'], score)
                    self.assertAlmostEqual(clip.signals['duration'], (end-start)/90.)
                    np.testing.assert_allclose(
                        [clip.signals[k] for k in ('quality','visual','audio')],
                        [quality,visual,audio], atol=1e-6)
                    np.testing.assert_allclose(clip.signature, signature, atol=1e-6)

    def test_silent_constant_ride_keeps_zero_activity(self):
        data = self.scan_data()
        data['motion'] = np.full_like(data['motion'], 5.)
        data['audio'] = np.asarray([], np.float32)
        prepared = m.prepare_scan_scores(data)
        clip = m.Clip(0.,12.5,m.baseline_at(prepared,0.,12.5))
        m.describe_clip(clip,prepared)
        self.assertEqual(clip.score, 0.)
        self.assertEqual(clip.signals['visual'], 0.)
        self.assertEqual(clip.signals['audio'], 0.)

    def test_cache_hit_prepares_once_and_does_not_persist_derived_scales(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root/'ride.mp4'
            source.write_bytes(b'cache identity; decoder must not be called')
            cache = CacheStore(root)
            key = cache.key(source)
            cache.save_scan(key,12.5,**self.scan_data())
            with patch.object(m,'motion_samples',side_effect=AssertionError('must use scan cache')):
                duration, data, _ = m.scan_video(source,'unused',lambda *a:None,cache=cache)
            self.assertEqual(duration, 12.5)
            with patch.object(m,'normalize',side_effect=AssertionError('repeated normalization')), \
                 patch.object(m,'local_change',side_effect=AssertionError('repeated ride baseline')):
                for start in range(5):
                    clip = m.Clip(start,start+3,m.baseline_at(data,start,start+3))
                    m.describe_clip(clip,data)
            self.assertEqual(set(cache.load_scan(key)[1]), {'motion','audio','quality','signature'})
            with np.load(cache.root/(key+'.npz'),allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {'duration','motion','audio','quality','signature'})


if __name__ == '__main__':
    unittest.main()
