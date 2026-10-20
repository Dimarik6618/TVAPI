import unittest
import numpy as np

import moto_shorts as m
from temporal_events import smart_bounds,track_events
from speech_analysis import speech_candidates


class TemporalTests(unittest.TestCase):
    def test_growth_is_an_event_but_stationary_vehicle_is_not(self):
        track={'id':7,'class':'car','points':[{'time':i/6,'x':.5,'y':.5,'area':a} for i,a in enumerate([.01,.02,.035,.06,.10,.15])]}
        events=track_events({7:track})
        self.assertEqual(events[0]['type'],'approach')
        static={**track,'points':[{**p,'area':.04} for p in track['points']]}
        self.assertEqual(track_events({7:static}),[])

    def test_boundary_contains_full_utterance_and_action_reaction(self):
        speech=[{'start':4.,'end':18.,'text':'Завершённая реплика'}]
        start,end=smart_bounds(7.,15.,30.,speech)
        self.assertLess(start,4.)
        self.assertGreater(end,18.)
        start,end=smart_bounds(0,30,40,events=[{'start':8,'end':14,'score':.9}])
        self.assertLess(start,8)
        self.assertGreater(end,14)
        self.assertLess(end-start,30)

    def test_edge_events_and_long_speech_keep_valid_boundaries(self):
        for duration in (4.,12.,90.):
            start,end=smart_bounds(0,duration,duration,events=[{'start':0,'end':duration,'score':1}])
            self.assertTrue(0<=start<end<=duration)
        self.assertEqual(smart_bounds(10,30,600,[{'start':0,'end':300,'text':'Long'}]),(10,30))

    def test_speech_finds_quiet_candidates_and_preserves_tail(self):
        segments=[{'start':2.,'end':5.,'text':'Привет как дела'},
                  {'start':6.,'end':9.,'text':'Всё очень хорошо'},
                  {'start':95.,'end':99.,'text':'До встречи завтра'}]
        groups=speech_candidates(segments,100.)
        self.assertEqual(len(groups),2)
        self.assertGreaterEqual(groups[-1]['end'],99.)
        quiet=m.Clip(groups[0]['start'],groups[0]['end'],.4,signals={'learned_event':.1,'speech':.8})
        self.assertEqual(m.select_best([quiet],1),[quiet])

    def test_semantic_duplicates_are_filtered_across_reason_texts(self):
        a=np.zeros(512);a[0]=1
        b=np.zeros(512);b[1]=1
        first=m.Clip(0,10,.9,reason='Звук',embedding=a.tolist())
        same=m.Clip(20,30,.8,reason='Сюжет',embedding=a.tolist())
        other=m.Clip(40,50,.7,reason='Разговор',embedding=b.tolist())
        self.assertEqual(m.select_best([first,same,other],3,threshold=0),[first,other])
        self.assertEqual(m.select_best([same,other],2,threshold=0,prior=[first]),[other])
        self.assertEqual(len(m.select_best([first,same,other],3,threshold=0,deduplicate=False)),3)


if __name__=='__main__':unittest.main()
