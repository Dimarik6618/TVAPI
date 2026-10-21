import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import moto_shorts as m
import scan_decode as decode
import task_control as tasks
from hardware import Runtime
from project_store import CacheStore


class ScanDecodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.source = self.folder/'поездка.mp4'
        subprocess.run([m.ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=size=320x180:rate=30', '-f', 'lavfi', '-i',
                        'sine=frequency=150:sample_rate=8000', '-t', '2',
                        '-c:v', 'libx264', '-threads', '2', '-c:a', 'aac', str(self.source)], check=True)

    def tearDown(self):
        tasks.bind(None)
        decode.hardware_methods.cache_clear()
        self.tmp.cleanup()

    def test_large_ffmpeg_error_output_cannot_block_motion_or_audio(self):
        # Each fake decoder fills stderr BEFORE writing stdout. The old PIPE
        # arrangement deadlocks here; the timer makes a regression fail quickly.
        control = tasks.TaskControl()
        tasks.bind(control)
        actual_popen = tasks.popen
        frame_bytes = m.WIDTH*m.HEIGHT
        for name, reader, data_bytes in [
            ('motion', lambda: m.motion_samples(self.source, 'unused', 1), frame_bytes*2),
            ('audio', lambda: m.audio_samples(self.source, 'unused', 1), 16000),
        ]:
            def noisy_decoder(args, **kwargs):
                code = ("import sys; sys.stderr.buffer.write(b'error '*100000); sys.stderr.flush(); "
                        f"sys.stdout.buffer.write(bytes({data_bytes})); sys.stdout.flush()")
                return actual_popen([sys.executable, '-c', code], **kwargs)
            timer = threading.Timer(4, control.stop)
            timer.start()
            try:
                with self.subTest(stream=name), patch.object(tasks, 'popen', side_effect=noisy_decoder):
                    result = reader()
                    self.assertFalse(control.event.is_set())
                    self.assertEqual(len(result[0]) if name=='motion' else len(result), 2 if name=='motion' else 1)
            finally:
                timer.cancel()
                timer.join()

    def test_error_tail_remains_available_after_large_stream_failure(self):
        actual_popen = tasks.popen
        def failing_decoder(args, **kwargs):
            code = "import sys; sys.stderr.write('x'*200000+' decoder failed at end'); sys.exit(1)"
            return actual_popen([sys.executable, '-c', code], **kwargs)
        with patch.object(tasks, 'popen', side_effect=failing_decoder):
            with self.assertRaisesRegex(RuntimeError, 'decoder failed at end'):
                m.motion_samples(self.source, 'unused', 1)

    def test_cuda_selected_only_after_real_filtered_frame(self):
        frame = bytes(m.WIDTH*m.HEIGHT)
        with patch.object(decode, 'hardware_methods', return_value={'cuda'}), \
                patch.object(tasks, 'run', return_value=subprocess.CompletedProcess([], 0, frame, b'')) as run:
            args, selected, errors = decode.decoder_args('ffmpeg', self.source, 2, 'fps=2', len(frame))
        self.assertEqual((args, selected, errors), (['-hwaccel', 'cuda'], 'cuda', []))
        self.assertIn('-frames:v', run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs['timeout'], 15)

    def test_failed_or_incomplete_hardware_probe_keeps_cpu_path(self):
        for result in [subprocess.CompletedProcess([], 1, b'', b'CUDA unavailable'),
                       subprocess.CompletedProcess([], 0, b'partial', b'')]:
            with patch.object(decode, 'hardware_methods', return_value={'cuda'}), \
                    patch.object(tasks, 'run', return_value=result):
                args, selected, errors = decode.decoder_args('ffmpeg', self.source, 2, 'fps=2', 14400)
            self.assertEqual((args, selected), ([], 'cpu'))
            self.assertEqual(len(errors), 1)
        with patch.object(decode, 'hardware_methods', return_value={'cuda'}), \
                patch.object(tasks, 'run', side_effect=subprocess.TimeoutExpired('probe', 15)):
            self.assertEqual(decode.decoder_args('ffmpeg', self.source, 2, 'fps=2', 14400)[1], 'cpu')

    def test_windows_uses_d3d11_after_cuda_failure(self):
        outcomes = [subprocess.CompletedProcess([], 1, b'', b'CUDA failed'),
                    subprocess.CompletedProcess([], 0, bytes(14400), b'')]
        with patch.object(decode.sys, 'platform', 'win32'), \
                patch.object(decode, 'hardware_methods', return_value={'cuda', 'd3d11va'}), \
                patch.object(tasks, 'run', side_effect=outcomes):
            args, selected, errors = decode.decoder_args('ffmpeg', self.source, 2, 'fps=2', 14400)
        self.assertEqual(args, ['-hwaccel', 'd3d11va'])
        self.assertEqual(selected, 'd3d11va')
        self.assertEqual(len(errors), 1)

    def test_metadata_probe_timeout_is_bounded_and_explained(self):
        with patch.object(tasks, 'run', side_effect=subprocess.TimeoutExpired('metadata', 30)) as run:
            with self.assertRaisesRegex(RuntimeError, 'параметры видео за 30 с'):
                m.video_info(self.source, 'unused')
        self.assertEqual(run.call_args.kwargs['timeout'], 30)

    def test_probe_absent_needs_no_extra_media_decode_and_scan_cache_is_preserved(self):
        runtime = Runtime.detect()
        cache = CacheStore(self.folder)
        with patch.object(decode, 'hardware_methods', return_value=set()):
            duration, arrays, key = m.scan_video(self.source, m.ffmpeg_exe(), lambda *args: None,
                                                cache=cache, runtime=runtime)
        self.assertEqual(runtime.scan['decoder'], 'cpu')
        self.assertEqual(runtime.scan['stage'], 'complete')
        self.assertGreater(runtime.scan['motion_seconds'], 0)
        self.assertGreater(runtime.scan['audio_seconds'], 0)
        self.assertEqual(runtime.scan['frame_count'], 4)
        with patch.object(m, 'decoder_args', side_effect=AssertionError('cached media probed')):
            cached_duration, cached, cached_key = m.scan_video(self.source, m.ffmpeg_exe(),
                lambda *args: None, cache=cache, runtime=runtime)
        self.assertEqual((cached_duration, cached_key), (duration, key))
        self.assertEqual(runtime.scan['decoder'], 'cache')
        for name in ('motion', 'quality', 'signature', 'audio'):
            np.testing.assert_array_equal(arrays[name], cached[name])

    def test_hardware_failure_during_scan_retries_cpu_without_partial_arrays(self):
        runtime = Runtime.detect()
        actual_motion = m.motion_samples
        calls = []
        def motion(path, ffmpeg, duration, progress, threads, args=()):
            calls.append(args)
            if args:
                raise RuntimeError('controlled hardware decode failure')
            return actual_motion(path, ffmpeg, duration, progress, threads)
        with patch.object(m, 'decoder_args', return_value=(['-hwaccel', 'cuda'], 'cuda', [])), \
                patch.object(m, 'motion_samples', side_effect=motion):
            _, data, _ = m.scan_video(self.source, m.ffmpeg_exe(), lambda *args: None, runtime=runtime)
        self.assertEqual(calls, [['-hwaccel', 'cuda'], ()])
        self.assertEqual(len(data['motion']), 4)
        self.assertEqual(runtime.scan['decoder'], 'cpu')
        self.assertIn('controlled hardware', runtime.scan['decoder_fallback'])

    def test_stopped_scan_report_keeps_duration_and_precise_substage(self):
        def stop(*args):
            raise tasks.Cancelled()
        with patch.object(m, 'decoder_args', return_value=([], 'cpu', [])), \
                patch.object(m, 'motion_samples', side_effect=stop):
            _, errors = m.analyze_batch([self.source], 0, 1, preferences_folder=self.folder,
                                        use_proxies=False, memory_features=False)
        self.assertFalse(errors)
        import json
        report = json.loads((self.folder/'moto_performance.json').read_text(encoding='utf-8'))['files'][0]
        self.assertEqual(report['status'], 'stopped')
        self.assertAlmostEqual(report['duration_seconds'], 2., delta=.1)
        self.assertEqual(report['scan_details']['stage'], 'motion')
        self.assertIn('motion_seconds', report['scan_details'])

    def test_watchdog_kills_only_a_stream_without_data(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        guard = decode.StreamWatchdog(process, timeout=.2)
        try:
            process.communicate(timeout=3)
            with self.assertRaisesRegex(RuntimeError, 'FFmpeg не выдаёт данные'):
                guard.check()
        finally:
            guard.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        self.assertFalse(guard.thread.is_alive())

    def test_watchdog_allows_longer_work_while_data_keeps_arriving(self):
        code = "import sys,time\nfor i in range(8):\n sys.stdout.write('x'); sys.stdout.flush(); time.sleep(.04)"
        process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
        guard = decode.StreamWatchdog(process, timeout=.15)
        try:
            result = []
            while chunk := process.stdout.read(1):
                guard.touch()
                result.append(chunk)
            process.wait(timeout=3)
            guard.check()
            self.assertEqual(len(result), 8)
        finally:
            guard.close()
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == '__main__':
    unittest.main()
