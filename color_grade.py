"""Color treatment with a bundled Action 5 Pro LUT and optional custom LUTs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


BUILTIN_LUT = (Path(__file__).resolve().parent / 'assets' / 'luts'
               / 'dji_osmo_action_5_pro_dlog_m_to_rec709_v1.cube')


@dataclass(frozen=True)
class ColorGrade:
    mode: str = 'none'  # none, manual, dlog
    brightness: int = 0  # -30..30
    contrast: int = 0    # -30..30
    saturation: int = 0  # -30..30
    lut: str = ''

    def lut_path(self) -> Path:
        """An empty override selects the bundled LUT, independent of working directory."""
        return Path(self.lut) if self.lut else BUILTIN_LUT

    def validate(self) -> None:
        if self.mode not in ('none', 'manual', 'dlog'):
            raise ValueError('Неизвестный режим цветокоррекции.')
        if any(not -30 <= value <= 30 for value in
               (self.brightness, self.contrast, self.saturation)):
            raise ValueError('Ползунки цвета должны быть в пределах −30…30.')
        if self.mode == 'dlog':
            path = self.lut_path()
            if path.suffix.lower() != '.cube' or not path.is_file():
                if self.lut:
                    raise ValueError('Не найден выбранный .cube LUT. Выберите файл или встроенный LUT DJI.')
                raise ValueError('Не найден встроенный LUT DJI. Распакуйте папку программы целиком, включая assets.')

    def filters(self) -> list[str]:
        self.validate()
        if self.mode == 'none':
            return []
        result = ['lut3d=file=grade.cube:interp=tetrahedral'] if self.mode == 'dlog' else []
        if any((self.brightness, self.contrast, self.saturation)):
            result.append('eq=brightness={:.3f}:contrast={:.3f}:saturation={:.3f}'.format(
                self.brightness / 300, 1 + self.contrast / 100,
                1 + self.saturation / 100))
        return result

    def manifest(self) -> dict:
        result = asdict(self)
        if self.mode == 'dlog':
            result.update(lut_source='external' if self.lut else 'builtin',
                          lut_name=self.lut_path().name)
        return result
