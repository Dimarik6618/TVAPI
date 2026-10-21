import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from speech_analysis import SpeechAnalyzer
from moto_shorts import ffmpeg_exe


class SpeechTests(unittest.TestCase):
    def test_windows_word_times_vad_and_cache_use_no_audio_files(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source.mp4';source.write_bytes(b'source identity')
            recognizer=SpeechAnalyzer(ffmpeg_exe(),root)
            calls=[]
            class Model:
                def transcribe(self,audio,**kwargs):
                    calls.append(kwargs)
                    return iter([SimpleNamespace(start=1,end=3,text='Привет как дела',avg_logprob=-.3,no_speech_prob=.02,
                        words=[SimpleNamespace(start=1,end=2,word='Привет',probability=.95)])]),None
            recognizer.model=Model()
            audio=np.zeros(16000*3,dtype='<f4').tobytes()
            commands=[]
            def run(command,**kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=0,stdout=audio)
            with patch('speech_analysis.tasks.run',side_effect=run):
                segments=recognizer.recognize(source,121)
            self.assertEqual(len(calls),3)
            self.assertTrue(all(x['vad_filter'] and x['word_timestamps'] for x in calls))
            self.assertEqual([s['words'][0]['start'] for s in segments],[1,59,117])
            with patch.object(recognizer,'load',side_effect=AssertionError('must use text cache')):
                self.assertEqual(recognizer.recognize(source,121),segments)
            self.assertFalse(any(p.suffix in ('.wav','.mp3') for p in root.rglob('*')))


if __name__=='__main__':unittest.main()
