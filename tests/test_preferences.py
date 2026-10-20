import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from preferences import PreferenceStore


class PreferenceReasonTests(unittest.TestCase):
    def add_examples(self, store, accepted, reason, offset=0):
        for i in range(5):
            clip = SimpleNamespace(start=offset+i*20, end=offset+i*20+10,
                                   signals={'baseline': .8 if accepted else .1})
            store.add(Path('ride.mp4'), clip, accepted, reason)

    def test_presentation_rejections_remain_saved_but_do_not_train_interest(self):
        for reason in ('Тряска / плохое качество', 'Неудачные границы', 'Повтор'):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as name:
                folder = Path(name)
                store = PreferenceStore(folder)
                self.add_examples(store, True, 'Принято')
                self.add_examples(store, False, reason, offset=200)
                restored = PreferenceStore(folder)
                self.assertEqual(len(restored.samples), 10)
                self.assertEqual(sum(s['reason'] == reason for s in restored.samples.values()), 5)
                self.assertIsNone(restored.fitted())

    def test_existing_interest_model_ignores_new_quality_rejections(self):
        with tempfile.TemporaryDirectory() as name:
            store = PreferenceStore(Path(name))
            self.add_examples(store, True, 'Принято')
            self.add_examples(store, False, 'Неинтересно', offset=200)
            before = store.fitted()
            self.assertIsNotNone(before)
            self.add_examples(store, False, 'Тряска / плохое качество', offset=400)
            after = PreferenceStore(Path(name)).fitted()
            np.testing.assert_array_equal(after[0], before[0])
            self.assertEqual(after[1], before[1])


if __name__ == '__main__':
    unittest.main()
