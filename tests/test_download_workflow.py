"""Integrated download/FFmpeg/archive/checkpoint/cleanup tests, without OAuth.

The Google downloader and MP4 decoding are real. Only the HTTP account transport
and large visual encoder are controlled fixtures, so tests never use an account.
"""
import hashlib
import re
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import moto_shorts as m
import task_control as tasks
from drive_source import DriveSource
from event_archive import EventArchive
from project_store import ProjectStore, fingerprint

try:
    import httplib2
    from googleapiclient.http import HttpRequest
    HAVE_GOOGLE=True
except ImportError:
    HAVE_GOOGLE=False


class Response:
    def __init__(self,value):self.value=value
    def execute(self,**kwargs):return self.value


class FrozenEncoder:
    def extract(self,*args):
        value=np.zeros(512,np.float32);value[0]=1
        return value


@unittest.skipUnless(HAVE_GOOGLE,'Google SDK is required for integration tests')
class DownloadWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media=tempfile.TemporaryDirectory(prefix='moto-fixture-')
        source=Path(cls.media.name)/'real.mp4'
        tasks.run([m.ffmpeg_exe(),'-v','error','-f','lavfi','-i','testsrc2=s=160x90:r=24',
            '-f','lavfi','-i','sine=frequency=440','-t','4','-c:v','libx264','-c:a','aac',str(source)],check=True)
        cls.video=source.read_bytes()

    @classmethod
    def tearDownClass(cls):cls.media.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='moto-download-cycle-')
        self.root=Path(self.temp.name)
        self.folder=self.root/'learning_data'
        self.archive=EventArchive(self.folder)
        self.metadata={'id':'ride-id','name':'DJI_7.mp4','size':str(len(self.video)),
            'md5Checksum':hashlib.md5(self.video).hexdigest(),'version':'1','capabilities':{'canDownload':True}}
        self.payload=self.video
        self.requests=[]
        test=self
        class HTTP:
            def request(self,uri,method='GET',headers=None,**kwargs):
                test.requests.append((method,(headers or {}).get('range','')))
                match=re.search(r'bytes=(\d+)-(\d+)',(headers or {}).get('range',''))
                if match:
                    a,b=int(match[1]),min(int(match[2]),len(test.payload)-1)
                    return httplib2.Response({'status':'206','content-range':f'bytes {a}-{b}/{len(test.payload)}'}),test.payload[a:b+1]
                return httplib2.Response({'status':'200','content-length':str(len(test.payload))}),test.payload
        class Files:
            def get(self,**kwargs):return Response(test.metadata.copy())
            def get_media(self,**kwargs):return HttpRequest(HTTP(),lambda *a:None,'https://example.invalid/media')
        self.drive=DriveSource(self.folder,SimpleNamespace(files=lambda:Files()))
        self.path=self.drive.add(self.metadata.copy())
        self.project=ProjectStore(self.root/'project.json')
        self.state={'files':[str(self.path)],'completed':{}}

    def tearDown(self):
        tasks.bind(None)
        self.temp.cleanup()

    def checkpoint(self,path,clips):
        self.state['completed'][str(path)]={'fingerprint':fingerprint(path),
            'clips':[m.asdict(c) for c in clips],'event_model':'off'}
        self.project.save(self.state)

    def analyze(self,encoder=None,checkpoint=None,progress=lambda *a:None):
        return m.analyze_batch([self.path],0,2,on_progress=progress,threshold=0,use_proxies=False,
            preferences_folder=self.root,learning_folder=self.folder,capture_memory=True,
            source_loader=self.drive.ensure,source_origin=lambda p:self.drive.entry(p)['metadata'],
            model_pool={'memory-features:auto':encoder or FrozenEncoder()},
            on_complete=checkpoint or self.checkpoint,
            after_complete=lambda p,c:self.drive.release(p,self.archive))

    def test_download_scan_checkpoint_delete_reload_and_resume(self):
        rows,errors=self.analyze()
        self.assertFalse(errors)
        self.assertTrue(rows)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix('.mp4.part').exists())
        events=self.archive.list_events()
        self.assertTrue(events and all(e['feature_ready'] for e in events))
        self.assertTrue(all(e['source']['origin']['id']=='ride-id' for e in events))
        self.assertTrue(self.project.path.exists())
        before=len(self.requests)
        restored=self.project.load()
        with patch.object(m,'scan_video',side_effect=AssertionError('must not decode completed source')):
            resumed,errors=m.analyze_batch([self.path],0,2,learning_folder=self.folder,capture_memory=True,
                resume=restored['completed'],source_loader=self.drive.ensure)
        self.assertFalse(errors)
        self.assertEqual(len(resumed),len(rows))
        self.assertEqual(len(self.requests),before)
        self.drive.ensure(self.path)
        self.assertEqual(self.path.read_bytes(),self.video)
        self.assertEqual(len(self.requests),before+1)
        self.assertTrue(self.drive.release(self.path,self.archive))
        self.assertEqual(len(self.archive.list_events()),len(events))
        self.assertFalse(any(p.suffix in ('.mp4','.jpg','.png','.wav') for p in self.archive.root.rglob('*')))

    def test_encoder_failure_keeps_mp4_and_partial_analysis(self):
        class Broken:
            def extract(self,*a):raise RuntimeError('encoder unavailable')
        rows,errors=self.analyze(Broken())
        self.assertTrue(rows)
        self.assertTrue(any('encoder unavailable' in message for _,message in errors))
        self.assertTrue(self.path.exists())
        self.assertFalse(self.archive.can_release(self.path))
        self.assertTrue(self.archive.list_events())
        self.assertTrue(all(not e['feature_ready'] for e in self.archive.list_events()))

    def test_checkpoint_failure_keeps_source_even_with_saved_features(self):
        def broken(*args):raise OSError('project disk full')
        _,errors=self.analyze(checkpoint=broken)
        self.assertTrue(any('project disk full' in message for _,message in errors))
        self.assertTrue(self.path.exists())
        self.assertTrue(self.archive.can_release(self.path))
        self.assertFalse(self.project.path.exists())

    def test_stop_while_freezing_keeps_source_and_unfinished_archive(self):
        control=tasks.TaskControl();tasks.bind(control)
        class Stopping(FrozenEncoder):
            def extract(self,*args):
                value=super().extract(*args);control.stop();return value
        _,errors=self.analyze(Stopping())
        self.assertFalse(errors)
        self.assertTrue(self.path.exists())
        self.assertFalse(self.archive.can_release(self.path))
        self.assertFalse(self.project.path.exists())

    def test_stop_after_checkpoint_keeps_mp4_until_resumed_cleanup(self):
        control=tasks.TaskControl();tasks.bind(control)
        def completed(path,clips):self.checkpoint(path,clips);control.stop()
        _,errors=self.analyze(checkpoint=completed)
        self.assertFalse(errors)
        self.assertTrue(self.project.path.exists())
        self.assertTrue(self.path.exists())
        tasks.bind(None)
        state=self.project.load()
        before=len(self.requests)
        with patch.object(m,'scan_video',side_effect=AssertionError('must reuse checkpoint')):
            _,errors=m.analyze_batch([self.path],0,2,learning_folder=self.folder,capture_memory=True,
                resume=state['completed'],source_loader=self.drive.ensure,
                after_complete=lambda p,c:self.drive.release(p,self.archive))
        self.assertFalse(errors)
        self.assertFalse(self.path.exists())
        self.assertEqual(len(self.requests),before)

    def test_wrong_download_checksum_never_reaches_analysis(self):
        self.payload=bytes([self.video[0]^1])+self.video[1:]
        with patch.object(m,'scan_video',side_effect=AssertionError('invalid media must not be analyzed')) as scan:
            _,errors=self.analyze()
        self.assertTrue(errors)
        scan.assert_not_called()
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix('.mp4.part').exists())
        self.assertFalse(self.archive.list_events())

    def test_undecodable_download_keeps_source_for_recovery(self):
        self.payload=b'not a decodable MP4 recording'
        self.metadata.update(size=str(len(self.payload)),md5Checksum=hashlib.md5(self.payload).hexdigest())
        self.path=self.drive.add(self.metadata.copy())
        rows,errors=self.analyze()
        self.assertFalse(rows)
        self.assertTrue(errors)
        self.assertEqual(self.path.read_bytes(),self.payload)
        self.assertFalse(self.project.path.exists())
        self.assertFalse(self.archive.can_release(self.path))

    def test_stop_after_first_chunk_removes_incomplete_download(self):
        self.payload=self.video*max(2,6*1024*1024//len(self.video)+1)
        self.metadata.update(size=str(len(self.payload)),md5Checksum=hashlib.md5(self.payload).hexdigest())
        self.path=self.drive.add(self.metadata.copy())
        control=tasks.TaskControl();tasks.bind(control)
        with self.assertRaises(tasks.Cancelled):
            self.drive.ensure(self.path,on_progress=lambda f:control.stop())
        self.assertEqual(len(self.requests),1)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix('.mp4.part').exists())

    def test_missing_archived_vector_blocks_cleanup(self):
        self.analyze()
        self.drive.ensure(self.path)
        for damaged in (None,b'corrupt storage',zlib.compress(np.ones(4,dtype='<f4').tobytes())):
            with self.subTest(damaged=damaged):
                with self.archive.connect() as db:db.execute('UPDATE events SET vector=?',(damaged,))
                self.assertFalse(self.drive.release(self.path,self.archive))
                self.assertTrue(self.path.exists())

    def test_batch_downloads_next_video_only_after_previous_cleanup(self):
        other=self.drive.add({**self.metadata,'id':'other-ride'})
        paths=[self.path,other]
        self.state['files']=[str(p) for p in paths]
        loaded=[]
        def load(path,progress):
            self.assertFalse(any(p.suffix=='.mp4' for p in self.drive.downloads.rglob('*')))
            loaded.append(path)
            return self.drive.ensure(path,progress)
        _,errors=m.analyze_batch(paths,0,2,threshold=0,use_proxies=False,preferences_folder=self.root,
            learning_folder=self.folder,capture_memory=True,source_loader=load,
            source_origin=lambda p:self.drive.entry(p)['metadata'],model_pool={'memory-features:auto':FrozenEncoder()},
            on_complete=self.checkpoint,after_complete=lambda p,c:self.drive.release(p,self.archive))
        self.assertFalse(errors)
        self.assertEqual(loaded,paths)
        self.assertEqual(len(self.project.load()['completed']),2)
        self.assertFalse(any(p.exists() for p in paths))
        self.assertTrue(all(method=='GET' for method,_ in self.requests))


if __name__=='__main__':unittest.main()
