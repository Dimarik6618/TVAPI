import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import task_control as tasks
from drive_source import DriveSource,folder_id,SCOPES


try:
    import httplib2
    from googleapiclient.http import HttpRequest
    HAVE_GOOGLE=True
except ImportError:
    HAVE_GOOGLE=False


class Response:
    def __init__(self,value):self.value=value
    def execute(self,**kwargs):return self.value


@unittest.skipUnless(HAVE_GOOGLE,'optional Google SDK is required for actual downloader tests')
class DriveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.blob=b'controlled binary recording'*100
        self.metadata={'id':'test-file','name':'DJI_7.mp4','size':str(len(self.blob)),
                       'md5Checksum':hashlib.md5(self.blob).hexdigest(),'version':'1',
                       'capabilities':{'canDownload':True}}
        self.calls=[]
        test=self
        class HTTP:
            def request(self,uri,method='GET',headers=None,**kwargs):
                test.calls.append(method)
                return httplib2.Response({'status':'200','content-length':str(len(test.blob))}),test.blob
        class Files:
            def get(self,**kwargs):return Response(test.metadata)
            def get_media(self,**kwargs):return HttpRequest(HTTP(),lambda *a:None,'https://example.invalid/media')
            def list(self,**kwargs):
                if "'nested' in parents" in kwargs['q']:
                    return Response({'files':[test.metadata]})
                return Response({'files':[{'id':'nested','name':'trip','mimeType':'application/vnd.google-apps.folder'},
                                           {'id':'other','name':'photo.jpg','mimeType':'image/jpeg'}]})
        self.drive=DriveSource(self.root,SimpleNamespace(files=lambda:Files()))
        self.path=self.drive.add(self.metadata.copy())

    def tearDown(self):
        tasks.bind(None)
        self.tmp.cleanup()

    def test_actual_media_downloader_and_reuse_are_read_only(self):
        self.drive.ensure(self.path)
        self.assertEqual(self.path.read_bytes(),self.blob)
        self.drive.ensure(self.path)
        self.assertEqual(self.calls,['GET'])
        self.assertEqual(SCOPES,['https://www.googleapis.com/auth/drive.readonly'])

    def test_remote_rename_and_version_change_allow_same_content(self):
        self.metadata.update(name='renamed.mp4',version='2')
        self.drive.ensure(self.path)
        self.assertEqual(self.path.read_bytes(),self.blob)
        self.assertEqual(self.calls,['GET'])

    def test_readding_changed_cloud_file_keeps_old_source_identity(self):
        old_metadata=self.drive.entry(self.path)['metadata'].copy()
        new_metadata={**old_metadata,'md5Checksum':'new-contents','version':'2'}
        updated=self.drive.add(new_metadata)
        self.assertNotEqual(updated,self.path)
        self.assertEqual(self.drive.entry(self.path)['metadata'],old_metadata)
        self.assertEqual(self.drive.entry(updated)['metadata'],new_metadata)
        self.assertEqual(self.drive.add(new_metadata),updated)

    def test_local_replacement_is_neither_overwritten_nor_downloaded(self):
        self.drive.ensure(self.path)
        self.path.write_bytes(b'manually replaced recording')
        with self.assertRaisesRegex(ValueError,'изменён'):
            self.drive.ensure(self.path)
        self.assertEqual(self.path.read_bytes(),b'manually replaced recording')
        self.assertEqual(self.calls,['GET'])

    def test_cleanup_choice_survives_restart(self):
        self.drive.set_auto_cleanup(False)
        restarted=DriveSource(self.root)
        self.assertFalse(restarted.auto_cleanup)
        self.assertIsNotNone(restarted.entry(self.path))
        restarted.set_auto_cleanup(True)
        self.assertTrue(DriveSource(self.root).auto_cleanup)

    def test_changed_remote_is_not_replaced_and_partial_download_is_removed(self):
        checksum=self.metadata['md5Checksum']
        self.metadata['md5Checksum']='different-content'
        with self.assertRaisesRegex(ValueError,'изменилась'):self.drive.ensure(self.path)
        self.assertFalse(self.path.exists())
        self.metadata['md5Checksum']=checksum
        self.blob=b'corrupted bytes'
        with self.assertRaises(ValueError):self.drive.ensure(self.path)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix('.mp4.part').exists())

    def test_download_restriction_and_stop_preserve_no_partial_mp4(self):
        self.metadata['capabilities']['canDownload']=False
        with self.assertRaisesRegex(ValueError,'запретил'):self.drive.ensure(self.path)
        self.metadata['capabilities']['canDownload']=True
        control=tasks.TaskControl();tasks.bind(control);control.stop()
        with self.assertRaises(tasks.Cancelled):self.drive.ensure(self.path)
        self.assertFalse(self.path.exists())

    def test_nested_folder_and_safe_filename(self):
        result=self.drive.list_videos('root-folder',recursive=True)
        self.assertEqual([x['id'] for x in result],['test-file'])
        unsafe=self.drive.add({'id':'safe-id','name':'../../CON.mp4'})
        self.assertTrue(unsafe.is_relative_to(self.drive.downloads.resolve()))
        self.assertNotIn('/',unsafe.name)
        self.assertEqual(folder_id('https://drive.google.com/drive/folders/abc_123?usp=sharing'),'abc_123')
        with self.assertRaises(ValueError):folder_id('https://example.com/folders/abc')
        with self.assertRaises(ValueError):self.drive.add({'id':'../../escape','name':'x.mp4'})


if __name__=='__main__':unittest.main()
