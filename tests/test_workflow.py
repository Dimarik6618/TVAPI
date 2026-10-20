import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import advanced_analysis as ai
import hardware
import moto_shorts as m
import task_control as tasks
from preferences import PreferenceStore
from project_store import CacheStore, ProjectStore, fingerprint, atomic_json

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.source = self.folder/'ride.mp4'
        sample=ROOT/'sample_ride.mp4'
        if sample.exists():
            shutil.copyfile(sample,self.source)
        else:
            tasks.run([m.ffmpeg_exe(),'-y','-v','error','-f','lavfi','-i',
                'testsrc2=size=160x90:rate=24','-f','lavfi','-i','sine=frequency=440',
                '-t','4','-c:v','libx264','-c:a','aac','-pix_fmt','yuv420p',str(self.source)],check=True)

    def tearDown(self):
        tasks.bind(None)
        self.temp.cleanup()

    def test_cache_reuses_scan_and_invalidates_changed_source(self):
        rows, errors = m.analyze_batch([self.source], 0, 3, preferences_folder=self.folder, threshold=0)
        self.assertFalse(errors)
        self.assertTrue(rows)
        with patch.object(m, 'motion_samples', side_effect=AssertionError('decoded twice')):
            _, errors = m.analyze_batch([self.source], 10, 1, preferences_folder=self.folder, threshold=0)
        self.assertFalse(errors)
        cache = CacheStore(self.folder)
        old = cache.key(self.source)
        with self.source.open('ab') as stream: stream.write(b'changed')
        self.assertNotEqual(old, cache.key(self.source))
        self.assertIsNone(cache.load_scan(cache.key(self.source)))

    def test_resume_after_stop_retains_empty_completed_files(self):
        other = self.folder/'second.mp4'
        shutil.copyfile(self.source, other)
        control = tasks.TaskControl(); tasks.bind(control)
        completed = {}
        def checkpoint(path, clips):
            completed[str(path.resolve())] = {'fingerprint': fingerprint(path), 'clips': [m.asdict(c) for c in clips]}
            control.stop()
        rows, errors = m.analyze_batch([self.source, other], 0, 3, preferences_folder=self.folder,
                                        threshold=1, on_complete=checkpoint)
        self.assertFalse(rows)
        self.assertFalse(errors)
        self.assertEqual(len(completed), 1)
        tasks.bind(None)
        decoded = []
        actual = m.motion_samples
        def scan(path, *args):
            decoded.append(path); return actual(path, *args)
        with patch.object(m, 'motion_samples', side_effect=scan):
            m.analyze_batch([self.source, other], 0, 3, preferences_folder=self.folder,
                             threshold=1, resume=completed)
        self.assertEqual(decoded, [other])

    def test_project_roundtrip_and_changed_file(self):
        clip = m.Clip(0, 3, .5, signals={'baseline': .5})
        m.set_clip_bounds(clip, .5, 2.5, 4)
        entry = {'fingerprint': fingerprint(self.source), 'clips': [m.asdict(clip)]}
        store = ProjectStore(self.folder/'project.json')
        store.save({'files': [str(self.source)], 'completed': {str(self.source): entry}})
        restored = store.load()
        actual = m.Clip(**restored['completed'][str(self.source)]['clips'][0])
        self.assertEqual(actual.edit_origin, [0, 3])
        self.assertEqual(actual.start, .5)
        self.assertTrue(ProjectStore.valid_completed(restored))
        self.source.unlink()
        self.assertFalse(ProjectStore.valid_completed(restored))

    def test_atomic_failed_write_keeps_old_project(self):
        path = self.folder/'state.json'
        atomic_json(path, {'old': 1})
        with patch('project_store.os.replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError): atomic_json(path, {'new': 1})
        self.assertEqual(json.loads(path.read_text()), {'old': 1})

    def test_threshold_and_diversity(self):
        signature = list(np.arange(12)/12)
        weak = m.Clip(0, 10, .1)
        strong = m.Clip(20, 30, .8, reason='Обгон', signature=signature)
        repeat = m.Clip(40, 50, .7, reason='Обгон', signature=signature)
        different = m.Clip(60, 70, .6, reason='Красивый маршрут', signature=signature)
        self.assertEqual(m.select_best([weak,strong,repeat,different],4), [strong,different])
        self.assertEqual(m.select_best([repeat],4,prior=[strong]), [])
        self.assertEqual(len(m.select_best([strong,repeat],4,deduplicate=False)), 2)

    def test_late_sound_event_is_examined(self):
        analyzer = ai.AdvancedAnalyzer.__new__(ai.AdvancedAnalyzer)
        analyzer.ffmpeg = 'ffmpeg'
        analyzer.runtime = hardware.Runtime()
        analyzer.audio_model = lambda signal, **kw: [{'label': 'Vehicle horn', 'score': float(signal[0])}]
        windows = []
        def waveform(ffmpeg,path,start,duration):
            windows.append((start,duration));return np.array([.9 if start>=16 else .05],np.float32)
        with patch.object(ai,'ffmpeg_audio',side_effect=waveform):
            score,reason = analyzer.sound(self.source,0,35)
        self.assertGreater(score,.8)
        self.assertGreaterEqual(windows[-1][0]+windows[-1][1],35)
        self.assertIn('horn',reason)

    def test_video_windows_include_tail_and_survive_object_failure(self):
        analyzer = ai.AdvancedAnalyzer.__new__(ai.AdvancedAnalyzer)
        analyzer.options = ai.Options(objects=True,meaning=True)
        analyzer.ffmpeg = 'ffmpeg'
        analyzer.objects = lambda frames: (_ for _ in ()).throw(RuntimeError('object error'))
        offsets = []
        analyzer.meaning = lambda frames: (float(frames[0][0,0,0])/100, 'Сюжет')
        def frames(ffmpeg,path,start,duration,count):
            offsets.append((start,duration,count))
            return [np.full((2,2,3),start,dtype=np.uint8)]*count
        with patch.object(ai,'ffmpeg_frames',side_effect=frames):
            clips = analyzer.enrich(self.source,[m.Clip(0,65,.2)])
        self.assertGreater(offsets[0][2],8)
        self.assertGreaterEqual(offsets[-1][0]+offsets[-1][1],65)
        self.assertGreater(clips[0].signals['meaning'],.3)
        self.assertEqual(len(analyzer.warnings),1)

    def test_meaning_visits_last_eight_frames(self):
        analyzer = ai.AdvancedAnalyzer.__new__(ai.AdvancedAnalyzer)
        groups = []
        def infer(frames):
            groups.append(frames);return (frames[-1]/100.,'Сюжет')
        analyzer.meaning_window = infer
        score,_ = analyzer.meaning(list(range(24)))
        self.assertEqual(groups[-1],list(range(16,24)))
        self.assertEqual(score,.23)

    def test_manual_boundaries_reanalyzed_then_learned(self):
        clip = m.Clip(0,3,.8,signals={'baseline':.8})
        m.set_clip_bounds(clip,.5,2.5,4)
        result, warnings = m.reanalyze_clip(self.source,clip,ai.Options(),self.folder,use_proxies=False)
        self.assertIn('duration',result.signals)
        self.assertFalse(warnings)
        store = PreferenceStore(self.folder)
        for i in range(3):
            example = m.Clip(10+i*20+2, 10+i*20+18, .8, signals={'baseline':.8}, edit_origin=[10+i*20,30+i*20])
            store.add(self.source,example,True)
        self.assertEqual(store.adjust_bounds(100,120,200),(102.,118.))

    def test_encoder_checks_real_run_not_only_listing(self):
        runtime = hardware.Runtime()
        commands = []
        def run(command,**kwargs):
            commands.append(command)
            if '-encoders' in command: return SimpleNamespace(stdout=b'h264_nvenc h264_qsv')
            return SimpleNamespace(returncode=0 if 'h264_qsv' in command else 1)
        with patch('hardware.tasks.run',side_effect=run),patch('hardware.sys.platform','win32'):
            runtime.select_encoder('ffmpeg')
        self.assertEqual(runtime.encoder,'h264_qsv')
        self.assertTrue(any('h264_nvenc' in c for c in commands))

    def test_tracking_gpu_oom_falls_back_without_losing_frames(self):
        analyzer=ai.AdvancedAnalyzer.__new__(ai.AdvancedAnalyzer)
        analyzer.runtime=hardware.Runtime(device='cuda:0',batch=4)
        analyzer.video_model=analyzer.audio_model=None
        analyzer.torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None,is_available=lambda:True))
        attempts=[]
        def track(frame,**kwargs):
            attempts.append(kwargs['device'])
            if kwargs['device']!='cpu':raise RuntimeError('CUDA out of memory')
            return [SimpleNamespace(boxes=None)]
        analyzer.object_model=SimpleNamespace(track=track)
        analyzer.objects([np.zeros((4,4,3),dtype=np.uint8)]*4)
        self.assertEqual(attempts,['cuda:0','cpu','cpu','cpu','cpu'])
        self.assertEqual(analyzer.runtime.device,'cpu')

    def test_learning_uses_new_features_and_excludes_bad_cut_labels(self):
        store=PreferenceStore(self.folder)
        for index in range(6):
            store.add(self.source,m.Clip(index*20,index*20+10,.8,signals={'baseline':.8,'quality':.9}),True)
            store.add(self.source,m.Clip(index*20+10,index*20+20,.1,signals={'baseline':.1,'quality':.2}),False,'Неинтересно')
        model=store.fitted()
        self.assertIsNotNone(model)
        self.assertGreater(store.probability({'baseline':.8,'quality':.9},model),
                           store.probability({'baseline':.1,'quality':.2},model))
        other=PreferenceStore(self.folder/'boundaries_only')
        for index in range(6):
            other.add(self.source,m.Clip(index*20,index*20+10,.8,signals={'baseline':.8}),True)
            other.add(self.source,m.Clip(index*20+10,index*20+20,.8,signals={'baseline':.8}),False,'Неудачные границы')
        self.assertIsNone(other.fitted())

    def test_lut_export_reads_cube_from_temporary_directory(self):
        lut=self.folder/'identity.cube'
        lut.write_text('LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n'
                       '0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n',encoding='utf-8')
        result=m.export(self.source,self.folder/'graded',[m.Clip(0,1,.8)],'crop',
                        color=m.ColorGrade(mode='dlog',lut=str(lut)),runtime=hardware.Runtime.detect())
        self.assertEqual(len(result),1)
        duration,audio=m.video_info(result[0],m.ffmpeg_exe())
        self.assertAlmostEqual(duration,1,delta=.2)
        self.assertTrue(audio)

    def test_stop_keeps_finished_exports_and_old_unfinished_target(self):
        destination=self.folder/'shorts'
        control=tasks.TaskControl();tasks.bind(control)
        def stop(fraction,message):
            if fraction==.5:control.stop()
        with patch.object(hardware.Runtime,'select_encoder'):
            outputs,errors=m.export_batch([(self.source,m.Clip(0,1,.5)),(self.source,m.Clip(1,2,.5))],
                                           destination,'crop',stop)
        self.assertEqual(len(outputs),1)
        self.assertFalse(errors)
        old=outputs[0][1].read_bytes()
        control=tasks.TaskControl();tasks.bind(control)
        def during(fraction,message):
            if 0<fraction<1:control.stop()
        with patch.object(hardware.Runtime,'select_encoder'):
            outputs,errors=m.export_batch([(self.source,m.Clip(0,2,.5))],destination,'crop',during)
        self.assertFalse(outputs)
        self.assertFalse(errors)
        self.assertEqual((destination/'ride_short_01.mp4').read_bytes(),old)
        self.assertFalse(list(destination.glob('*.part.mp4')))


if __name__ == '__main__':
    unittest.main()
