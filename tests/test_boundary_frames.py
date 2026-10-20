import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from boundary_frames import (BoundaryFrame, BoundaryPreviewWorker,
                             PreviewCancelled, decode_boundary_frame)
from moto_shorts import ffmpeg_exe


class BoundaryFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.ffmpeg = ffmpeg_exe()
        cls.source = cls.folder/'городская поездка.mp4'
        cls.portrait = cls.folder/'portrait.mp4'
        cls.vfr = cls.folder/'vfr.mp4'
        for source, size, extra in [(cls.source, '640x360', []),
                                    (cls.portrait, '180x320', []),
                                    (cls.vfr, '320x180', ['-vf', 'select=eq(mod(n\\,7)\\,0)+eq(mod(n\\,7)\\,2)', '-vsync', 'vfr'])]:
            subprocess.run([cls.ffmpeg, '-y', '-v', 'error', '-f', 'lavfi', '-i',
                            f'testsrc2=size={size}:rate=30', '-t', '4', *extra,
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def reference_frame(self, index):
        # Independent full decode selects a known frame number, without seeking.
        command = [self.ffmpeg, '-v', 'error', '-i', str(self.source), '-vf',
                   f"select=eq(n\\,{index}),scale=w='min(360,202*dar)':h='min(202,360/dar)',setsar=1",
                   '-frames:v', '1', '-vsync', '0', '-c:v', 'png',
                   '-compression_level', '1', '-threads', '1', '-f', 'image2pipe', '-']
        return subprocess.run(command, check=True, capture_output=True).stdout

    def test_fractional_start_and_exclusive_end_match_real_frames(self):
        first = decode_boundary_frame(self.ffmpeg, self.source, 1.015, 4)
        last = decode_boundary_frame(self.ffmpeg, self.source, 1.015, 4, end=True)
        self.assertAlmostEqual(first.timestamp, 31/30, places=3)
        self.assertAlmostEqual(last.timestamp, 1., places=3)
        self.assertEqual(first.png, self.reference_frame(31))
        self.assertEqual(last.png, self.reference_frame(30))

    def test_file_edges_and_portrait_keep_visible_content(self):
        first = decode_boundary_frame(self.ffmpeg, self.source, 0, 4)
        last = decode_boundary_frame(self.ffmpeg, self.source, 4, 4, end=True)
        self.assertAlmostEqual(first.timestamp, 0)
        self.assertAlmostEqual(last.timestamp, 119/30, places=3)
        self.assertEqual(last.png, self.reference_frame(119))
        portrait = decode_boundary_frame(self.ffmpeg, self.portrait, .5, 4)
        width, height = struct.unpack('>II', portrait.png[16:24])
        self.assertLess(width, height)
        self.assertLessEqual(width, 360)
        self.assertLessEqual(height, 202)
        self.assertAlmostEqual(width/height, 180/320, delta=.01)

    def test_variable_rate_uses_real_frame_times(self):
        first = decode_boundary_frame(self.ffmpeg, self.vfr, .49, 4)
        last = decode_boundary_frame(self.ffmpeg, self.vfr, .49, 4, end=True)
        self.assertAlmostEqual(first.timestamp, 16/30, places=3)
        self.assertAlmostEqual(last.timestamp, 14/30, places=3)

    def test_invalid_missing_and_cancelled_requests_are_clear(self):
        with self.assertRaises(FileNotFoundError):
            decode_boundary_frame(self.ffmpeg, self.folder/'missing.mp4', 1, 4)
        with self.assertRaises(ValueError):
            decode_boundary_frame(self.ffmpeg, self.source, 0, 4, end=True)
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(PreviewCancelled):
            decode_boundary_frame(self.ffmpeg, self.source, 1, 4, cancelled=cancel)

    def test_repeated_boundary_uses_cache_and_changed_source_does_not(self):
        calls = []
        def decoder(ffmpeg, source, position, duration, **kwargs):
            calls.append(position)
            return BoundaryFrame(b'png', position)
        worker = BoundaryPreviewWorker(self.ffmpeg, self.source, 4, decoder)
        try:
            worker.request(0, 1, 1.); worker.results.get(timeout=3)
            worker.request(0, 2, 1.); worker.results.get(timeout=3)
            self.assertEqual(calls, [1.])
            with self.source.open('ab') as stream:
                stream.write(b'changed')
            worker.request(0, 3, 1.); worker.results.get(timeout=3)
            self.assertEqual(calls, [1., 1.])
        finally:
            worker.close(); worker.thread.join(timeout=3)
        self.assertFalse(worker.thread.is_alive())

    def test_quick_updates_cancel_old_request_and_close_stops_decoder(self):
        entered = threading.Event()
        closed_decode = threading.Event()
        calls = []
        def decoder(ffmpeg, source, position, duration, *, cancelled, **kwargs):
            calls.append(position)
            if position in (1., 3.):
                entered.set()
                if not cancelled.wait(3):
                    raise AssertionError('obsolete decoder was not cancelled')
                if position == 3.:
                    closed_decode.set()
                raise PreviewCancelled()
            return BoundaryFrame(b'png', position)
        worker = BoundaryPreviewWorker(self.ffmpeg, self.source, 4, decoder)
        try:
            worker.request(0, 1, 1.)
            self.assertTrue(entered.wait(3))
            worker.invalidate(0)
            worker.request(0, 2, 2.)
            side, revision, frame, error = worker.results.get(timeout=3)
            self.assertEqual((side, revision, frame.timestamp, error), (0, 2, 2., None))
            entered.clear(); worker.request(1, 3, 3.)
            self.assertTrue(entered.wait(3))
            worker.close(); worker.thread.join(timeout=3)
            self.assertTrue(closed_decode.is_set())
            self.assertTrue(worker.results.empty())
        finally:
            worker.close(); worker.thread.join(timeout=3)
        self.assertFalse(worker.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
