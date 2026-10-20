import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import moto_shorts as m
import learning as l
from event_archive import EventArchive
from drive_source import DriveSource
from project_store import fingerprint, atomic_json, ProjectStore
from video_features import FEATURE_SPEC


def vector(event):
    v=np.zeros(512,np.float32)
    v[0 if event else 1]=1
    return v


class EventMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.archive=EventArchive(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self):
        store=l.LearningStore(self.root)
        paths=[]
        for i in range(3):
            path=self.root/f'ride{i}.mp4'
            path.write_bytes(f'original recording {i}'.encode())
            paths.append(path)
            sid=self.archive.register(path,4)
            for start in range(4):
                clip=m.Clip(start,start+1,.4,signals={'baseline':.4},embedding=vector(start<2).tolist(),
                            details={'speech':[{'start':start,'end':start+1,'text':'Встреча с другом'}]})
                self.archive.save_clip(sid,clip)
                store.add_user(path,start,start+1,start<2,group=f'ride{i}')
        return store,paths

    def test_all_originals_can_be_deleted_and_training_never_opens_media(self):
        store,paths=self.dataset()
        first=l.train(store,'not-a-real-ffmpeg')
        for p in paths:p.unlink()
        with patch.object(m,'scan_video',side_effect=AssertionError('media decoding forbidden')), \
             patch.object(l.VideoFeatureExtractor,'load',side_effect=AssertionError('weights forbidden')):
            second=l.train(l.LearningStore(self.root),'not-a-real-ffmpeg')
        self.assertEqual(second['evaluation']['from_memory_count'],12)
        self.assertEqual(second['evaluation']['semantic']['tp'],6)
        self.assertEqual(second['evaluation']['semantic']['fp'],0)
        self.assertEqual(first['classifier'],second['classifier'])
        events=self.archive.list_events()
        self.assertEqual(len(events),12)
        self.assertEqual(events[0]['details']['speech'][0]['text'],'Встреча с другом')
        self.assertFalse(any(p.suffix in ('.mp4','.png','.wav') for p in self.archive.root.rglob('*')))

    def test_decision_after_deletion_can_be_added_to_training(self):
        store,paths=self.dataset()
        source=self.archive.source_for(paths[0])
        paths[0].unlink()
        clip=m.Clip(0,1,.4,reviewed=True,selected=False,embedding=vector(True).tolist(),decision_reason='Неинтересно')
        self.archive.save_clip(source['id'],clip)
        store.add_user(paths[0],0,1,False)
        row=next(e for e in store.data['examples'] if e['source_id']==store.data['sources'][0]['id'] and e['start']==0)
        self.assertFalse(row['accepted'])
        self.assertEqual(self.archive.event_for(paths[0],0,1)[1]['decision_reason'],'Неинтересно')

    def test_replacing_file_at_same_path_keeps_distinct_training_sources(self):
        store=l.LearningStore(self.root)
        path=self.root/'ride.mp4'
        identities=[]
        for raw,accepted in ((b'first recording',True),(b'other recording',False)):
            path.write_bytes(raw)
            sid=self.archive.register(path,4)
            identities.append(sid)
            self.archive.save_clip(sid,m.Clip(0,4,.4,embedding=vector(accepted).tolist()))
            store.add_user(path,0,4,accepted)
        path.unlink()
        self.assertEqual(len(store.data['sources']),2)
        self.assertEqual({s['archive_id'] for s in store.data['sources']},set(identities))
        self.assertEqual(len(store.data['examples']),2)
        self.assertEqual({e['accepted'] for e in store.data['examples']},{False,True})
        for source in store.data['sources']:
            self.assertIsNotNone(self.archive.feature(source['archive_id'],0,4))

    def test_resume_from_archive_neither_downloads_nor_decodes_deleted_video(self):
        path=self.root/'managed.mp4';path.write_bytes(b'completed original')
        sid=self.archive.register(path,4)
        clip=m.Clip(0,4,.7,embedding=vector(True).tolist())
        eid=self.archive.save_clip(sid,clip)
        self.archive.finish(sid,[eid])
        entry={'fingerprint':fingerprint(path),'clips':[m.asdict(clip)],'event_model':'off'}
        completed={str(path.resolve()):entry}
        path.unlink()
        self.assertEqual(ProjectStore.valid_completed({'completed':completed},self.archive),completed)
        with patch.object(m,'scan_video',side_effect=AssertionError('decoding forbidden')):
            rows,errors=m.analyze_batch([path],0,3,learning_folder=self.root,resume=completed,
                source_loader=lambda *a:(_ for _ in ()).throw(AssertionError('download forbidden')),
                capture_memory=True)
        self.assertFalse(errors)
        self.assertEqual(len(rows),1)
        self.assertEqual((rows[0][1].start,rows[0][1].end),(0,4))

    def test_legacy_completed_project_freezes_features_before_cleanup(self):
        path=self.root/'old.mp4';path.write_bytes(b'legacy original')
        clip=m.Clip(0,4,.7)
        entry={'fingerprint':fingerprint(path),'clips':[m.asdict(clip)]}
        timeline={'motion':np.zeros(8),'audio':np.zeros(4),'quality':np.ones(8),'signature':np.zeros((8,12))}
        class Extractor:
            def extract(self,*args):return vector(True)
        with patch.object(m,'scan_video',return_value=(4,timeline,'legacy-key')) as scan:
            rows,errors=m.analyze_batch([path],0,1,threshold=0,use_proxies=False,
                learning_folder=self.root,resume={str(path.resolve()):entry},capture_memory=True,
                model_pool={'memory-features:auto':Extractor()})
        self.assertFalse(errors)
        scan.assert_called_once()
        self.assertTrue(self.archive.can_release(path))
        self.assertTrue(rows)

    def test_missing_features_and_other_boundaries_preserve_previous_model(self):
        store,paths=self.dataset()
        l.train(store,'not-a-real-ffmpeg')
        before=store.model_path.read_bytes()
        e=store.data['examples'][0]
        e['end']=.75
        store.save()
        paths[0].unlink()
        with self.assertRaisesRegex(ValueError,'сохранённых признаков'):
            l.train(store,'not-a-real-ffmpeg')
        self.assertEqual(store.model_path.read_bytes(),before)

    def test_feature_version_is_checked_without_media(self):
        store,paths=self.dataset()
        with self.archive.connect() as db:
            wrong={**FEATURE_SPEC,'version':999}
            db.execute('UPDATE events SET spec=?',(json.dumps(wrong),))
        for p in paths:p.unlink()
        with self.assertRaisesRegex(ValueError,'сохранённых признаков'):
            l.train(store,'not-a-real-ffmpeg')

    def test_identical_copies_stay_in_same_validation_group_after_deletion(self):
        a=self.root/'a.mp4';b=self.root/'b.mp4'
        a.write_bytes(b'one ride');b.write_bytes(a.read_bytes())
        sources=[{'id':str(i),'group':f'group{i}','path':str(p),'bound_fingerprint':fingerprint(p)} for i,p in enumerate((a,b))]
        a.unlink();b.unlink()
        self.assertEqual(len(set(l.resolved_groups(sources).values())),1)

    def test_only_complete_managed_downloads_with_features_are_removed(self):
        drive=DriveSource(self.root)
        path=drive.add({'id':'abc123','name':'DJI_1.mp4'})
        path.parent.mkdir(parents=True)
        path.write_bytes(b'managed source')
        drive.entry(path)['download_fingerprint']=fingerprint(path)
        sid=self.archive.register(path,8)
        a=m.Clip(0,4,.5,embedding=vector(True).tolist())
        b=m.Clip(4,8,.3)
        ids=[self.archive.save_clip(sid,c) for c in (a,b)]
        self.archive.finish(sid,ids)
        self.assertFalse(drive.release(path,self.archive))
        self.assertTrue(path.exists())
        b.embedding=vector(False).tolist()
        self.archive.save_clip(sid,b)
        self.archive.finish(sid,ids)
        local=self.root/'my_original.mp4';local.write_bytes(b'my original')
        self.assertFalse(drive.release(local,self.archive))
        self.assertTrue(local.exists())
        self.assertTrue(drive.release(path,self.archive))
        self.assertFalse(path.exists())
        self.assertEqual(len(self.archive.list_events()),2)

    def test_replaced_download_is_not_deleted(self):
        drive=DriveSource(self.root)
        path=drive.add({'id':'abc123','name':'DJI_1.mp4'})
        path.parent.mkdir(parents=True)
        path.write_bytes(b'first version')
        drive.entry(path)['download_fingerprint']=fingerprint(path)
        sid=self.archive.register(path,4)
        eid=self.archive.save_clip(sid,m.Clip(0,4,.5,embedding=vector(True).tolist()))
        self.archive.finish(sid,[eid])
        path.write_bytes(b'new replacement')
        self.assertFalse(drive.release(path,self.archive))
        self.assertTrue(path.exists())

    def test_moving_source_reuses_feature_identity(self):
        from video_features import VideoFeatureExtractor
        a=self.root/'a.mp4';a.write_bytes(b'original video bytes')
        sid=self.archive.register(a)
        self.archive.save_window(sid,0,6,vector(True))
        b=self.root/'moved.mp4';a.rename(b)
        extractor=VideoFeatureExtractor('none',self.root,archive=self.archive)
        with patch.object(extractor,'load',side_effect=AssertionError('must reuse frozen feature')):
            self.assertTrue(np.array_equal(extractor.window(b,0,6),vector(True)))

    def test_scan_and_event_memory_survive_removal_of_real_mp4(self):
        from event_memory import remember
        from hardware import Runtime
        p=self.root/'real.mp4'
        m.tasks.run([m.ffmpeg_exe(),'-v','error','-f','lavfi','-i','testsrc2=s=160x90:r=24',
                     '-t','4','-c:v','libx264',str(p)],check=True)
        duration,data,_=m.scan_video(p,m.ffmpeg_exe(),lambda *a:None)
        clips=m.choose_clips(data['motion'],data['audio'],duration,0,2,data['quality'])
        class Extractor:
            def extract(self,*args):return vector(True)
        remember(p,duration,data,clips,self.root,Runtime(),{'memory-features:auto':Extractor()})
        p.unlink()
        self.assertTrue(self.archive.can_release(p))
        self.assertTrue(all(e['feature_ready'] for e in self.archive.list_events()))


if __name__=='__main__':unittest.main()
