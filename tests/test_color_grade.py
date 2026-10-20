import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

import color_grade
from color_grade import ColorGrade
import hardware
import moto_shorts as m
import task_control as tasks


class ColorGradeTests(unittest.TestCase):
    def test_bundled_lut_matches_user_supplied_file(self):
        self.assertEqual(
            hashlib.sha256(color_grade.BUILTIN_LUT.read_bytes()).hexdigest(),
            'b18162854ab47702068410c33afa98a8cb6eef159fc5a04ce0e65fad0fd8947e')

    def test_dlog_defaults_to_bundled_lut_and_keeps_project_setting_portable(self):
        color = ColorGrade(mode='dlog')
        color.validate()
        self.assertEqual(color.lut_path(), color_grade.BUILTIN_LUT)
        self.assertIn('lut3d=file=grade.cube:interp=tetrahedral', color.filters())
        self.assertEqual(asdict(color)['lut'], '')
        self.assertEqual(color.manifest()['lut_source'], 'builtin')
        self.assertEqual(color.manifest()['lut_name'], color_grade.BUILTIN_LUT.name)

    def test_missing_custom_lut_is_reported_without_silent_substitution(self):
        with tempfile.TemporaryDirectory() as name:
            color = ColorGrade(mode='dlog', lut=str(Path(name)/'missing.cube'))
            with self.assertRaisesRegex(ValueError, 'выбранный'):
                color.validate()

    def test_missing_bundle_reports_incomplete_installation(self):
        with tempfile.TemporaryDirectory() as name, \
             patch.object(color_grade, 'BUILTIN_LUT', Path(name)/'missing.cube'):
            with self.assertRaisesRegex(ValueError, 'assets'):
                ColorGrade(mode='dlog').validate()

    def test_other_color_modes_do_not_require_or_apply_lut(self):
        with tempfile.TemporaryDirectory() as name, \
             patch.object(color_grade, 'BUILTIN_LUT', Path(name)/'missing.cube'):
            self.assertEqual(ColorGrade().filters(), [])
            self.assertEqual(ColorGrade(mode='manual').filters(), [])
            manual = ColorGrade(mode='manual', brightness=10)
            self.assertEqual(len(manual.filters()), 1)
            self.assertTrue(manual.filters()[0].startswith('eq='))
            self.assertEqual(ColorGrade().manifest(), asdict(ColorGrade()))

    def test_real_export_applies_bundle_and_custom_override_from_unicode_paths(self):
        with tempfile.TemporaryDirectory(prefix='moto-lut-') as name:
            root = Path(name)
            installed = root/'Программа для мото'/'assets'/'luts'/'DJI LUT.cube'
            installed.parent.mkdir(parents=True)
            shutil.copyfile(color_grade.BUILTIN_LUT, installed)
            source = root/'Городская поездка.mp4'
            ffmpeg = m.ffmpeg_exe()
            tasks.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                       'color=c=0x404040:s=160x90:r=10', '-t', '1',
                       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

            def mean_rgb(path):
                decoded = tasks.run([ffmpeg, '-v', 'error', '-i', str(path),
                                     '-frames:v', '1', '-vf', 'scale=16:16',
                                     '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                    capture_output=True, check=True)
                pixels = np.frombuffer(decoded.stdout, np.uint8)
                self.assertEqual(pixels.size, 16*16*3)
                return float(pixels.mean())

            with patch.object(color_grade, 'BUILTIN_LUT', installed):
                builtin_folder = root/'Рилсы со встроенным LUT'
                outputs = m.export(source, builtin_folder, [m.Clip(0,1,.8)], 'crop',
                                   color=ColorGrade(mode='dlog'), runtime=hardware.Runtime())
            self.assertEqual(len(outputs), 1)
            builtin_rgb = mean_rgb(outputs[0])
            source_rgb = mean_rgb(source)
            self.assertGreater(source_rgb-builtin_rgb, 5.)
            duration, _ = m.video_info(outputs[0], ffmpeg)
            self.assertAlmostEqual(duration, 1., delta=.1)
            manifest = json.loads((builtin_folder/'Городская поездка_shorts.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['color']['lut_source'], 'builtin')
            self.assertEqual(manifest['color']['lut_name'], installed.name)

            custom = root/'Свой нейтральный LUT.cube'
            custom.write_text('LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n'
                              '0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n', encoding='ascii')
            color = ColorGrade(mode='dlog', lut=str(custom))
            custom_outputs = m.export(source, root/'Рилсы со своим LUT', [m.Clip(0,1,.8)], 'crop',
                                      color=color, runtime=hardware.Runtime())
            self.assertEqual(len(custom_outputs), 1)
            self.assertAlmostEqual(mean_rgb(custom_outputs[0]), source_rgb, delta=3.)
            self.assertEqual(color.manifest()['lut_source'], 'external')
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)


if __name__ == '__main__':
    unittest.main()
