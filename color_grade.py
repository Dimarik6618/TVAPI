"""Color treatment for exported shorts, with an optional camera-specific 3D LUT."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ColorGrade:
    mode: str = 'none'  # none, manual, dlog
    brightness: int = 0  # -30..30
    contrast: int = 0    # -30..30
    saturation: int = 0  # -30..30
    lut: str = ''

    def validate(self) -> None:
        if self.mode not in ('none', 'manual', 'dlog'):
            raise ValueError('Неизвестный режим цветокоррекции.')
        if any(not -30 <= value <= 30 for value in
               (self.brightness, self.contrast, self.saturation)):
            raise ValueError('Ползунки цвета должны быть в пределах −30…30.')
        if self.mode == 'dlog':
            path = Path(self.lut)
            if path.suffix.lower() != '.cube' or not path.is_file():
                raise ValueError('Для D-Log M выберите .cube LUT для Osmo Action 5 Pro.')

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
        return asdict(self)
