import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import moto_shorts as m


class MemoryTable:
    """A small Treeview contract, with no sorting logic of its own."""
    def __init__(self):
        self.order, self.rows, self.headers = [], {}, {}
        self.selected, self.focused = (), ''

    def get_children(self):
        return tuple(self.order)

    def insert(self, parent, position, *, iid, **values):
        self.rows[iid] = values; self.order.append(iid)

    def delete(self, *rows):
        for row in rows:
            self.order.remove(row); self.rows.pop(row)
        self.selected = tuple(row for row in self.selected if row in self.rows)
        if self.focused not in self.rows: self.focused = ''

    def move(self, row, parent, index):
        self.order.remove(row); self.order.insert(index, row)

    def item(self, row, **values):
        self.rows[row].update(values)
        return self.rows[row]

    def heading(self, key, **values):
        self.headers.setdefault(key, {}).update(values)

    def selection(self):
        return self.selected

    def selection_set(self, *rows):
        self.selected = tuple(rows)

    def focus(self, row=None):
        if row is not None: self.focused = row
        return self.focused

    def see(self, row):
        pass

    def yview_moveto(self, position):
        pass


class TableSortingTests(unittest.TestCase):
    def setUp(self):
        self.app = m.App.__new__(m.App)
        self.app.table = MemoryTable()
        self.app.table_sort_column = None
        self.app.table_sort_reverse = False
        self.app.table_titles = {key: key.upper() for key in
                                ('use', 'reviewed', 'video', 'from', 'to', 'duration', 'score', 'reason')}
        self.app._displayed_rows = {}
        self.app.busy = False
        self.app.export_btn = Mock()
        self.app.sources = [Path('DJI_10.mp4'), Path('dji_2.mp4'), Path('DJI_1.mp4'), Path('DJI_3.mp4')]
        self.app.clips = [m.Clip(100,109,.49,False,signals={'baseline':.49},reason='Б'),
                          m.Clip(2,102,.9,True,signals={'baseline':.9},reason='а'),
                          m.Clip(11,13,.491,False,signals={'baseline':.491},reason='В'),
                          m.Clip(7,17,0.,False,signals={},reason='Г')]
        self.app.exports = {0:Path('short_10.mp4'), 1:Path('short_2.mp4')}
        self.app.refresh()

    def test_camera_names_toggle_naturally_and_clear_old_arrow(self):
        self.app.sort_table('video')
        self.assertEqual(self.app.table.order, ['2','1','3','0'])
        self.assertEqual(self.app.table.headers['video']['text'], 'VIDEO ▲')
        self.app.sort_table('video')
        self.assertEqual(self.app.table.order, ['0','3','1','2'])
        self.assertEqual(self.app.table.headers['video']['text'], 'VIDEO ▼')
        self.app.sort_table('reason')
        self.assertEqual(self.app.table.order, ['1','0','2','3'])
        self.assertEqual(self.app.table.headers['video']['text'], 'VIDEO')

    def test_scores_use_full_precision_and_unchecked_scores_stay_last(self):
        self.app.sort_table('score')
        self.assertEqual(self.app.table.order, ['1','2','0','3'])
        self.assertEqual(self.app.table.headers['score']['text'], 'SCORE ▼')
        self.app.sort_table('score')
        self.assertEqual(self.app.table.order, ['0','2','1','3'])
        self.app.sort_table('duration')
        self.assertEqual(self.app.table.order, ['2','0','3','1'])
        self.app.sort_table('from')
        self.assertEqual(self.app.table.order, ['1','3','2','0'])
        self.app.sort_table('to')
        self.assertEqual(self.app.table.order, ['2','3','1','0'])
        self.app.sources.append(Path('DJI_0.mp4'))
        self.app.clips.append(m.Clip(0,1,0.,signals={}))
        self.app.refresh()
        self.app.sort_table('video')
        self.app.sort_table('score')
        self.assertEqual(self.app.table.order[-2:], ['3','4'])

    def test_selection_still_opens_and_marks_its_own_source_and_export(self):
        original_sources = list(self.app.sources)
        original_clips = list(self.app.clips)
        original_exports = dict(self.app.exports)
        self.app.table.selection_set('0'); self.app.table.focus('0')
        self.app.sort_table('score')
        self.assertEqual(self.app.selected_index(), 0)
        self.assertEqual(self.app.table.focus(), '0')
        self.app.status = Mock()
        with patch.object(m,'external_player_command',return_value=['player','video']) as player, \
                patch.object(m.subprocess,'Popen'), patch.object(Path,'is_file',return_value=True):
            self.app.open_selected()
            player.assert_called_once_with(original_sources[0],100,109)
            player.reset_mock()
            self.app.open_exported()
            player.assert_called_once_with(original_exports[0],0.,9)
        self.app.folder = SimpleNamespace(get=lambda:'.')
        self.app.save_project = Mock()
        with patch.object(m,'PreferenceStore') as store:
            self.app.mark_selected(None)
            store.return_value.add.assert_called_once_with(original_sources[0],original_clips[0],True,'')
        self.assertTrue(original_clips[0].selected)
        self.assertEqual(self.app.sources,original_sources)
        self.assertTrue(all(a is b for a,b in zip(self.app.clips,original_clips)))
        self.assertEqual(self.app.exports,original_exports)
        self.assertEqual(self.app.selected_index(),0)

    def test_refresh_reapplies_sort_and_preserves_only_the_same_clip(self):
        self.app.sort_table('score')
        self.app.table.selection_set('1');self.app.table.focus('1')
        self.app.clips[1].score=.1
        self.app.refresh()
        self.assertEqual(self.app.table.order,['2','0','1','3'])
        self.assertEqual(self.app.selected_index(),1)
        self.assertEqual(self.app.table.focus(),'1')
        self.app.clips[1]=m.Clip(20,30,.99,signals={'baseline':.99})
        self.app.refresh()
        self.assertEqual(self.app.table.order,['1','2','0','3'])
        self.assertIsNone(self.app.selected_index())
        # Identical scores keep their original fragment order in both directions.
        for clip in self.app.clips: clip.score=.5;clip.signals={'baseline':.5}
        self.app.refresh()
        self.assertEqual(self.app.table.order,['0','1','2','3'])
        self.app.sort_table('score')
        self.assertEqual(self.app.table.order,['0','1','2','3'])


if __name__ == '__main__':
    unittest.main()
