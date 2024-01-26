from collections.abc import Generator
from pathlib import Path


class ImagesDir(type(Path())):
    IMAGES_EXTS = [".png", ".jpg", ".jpeg", ".gif"]

    @classmethod
    def _is_image(cls, path: Path):
        return path.suffix in cls.IMAGES_EXTS

    def glob(self, pattern: str) -> Generator[Path, None, None]:
        return (path for path in super().glob(pattern) if self._is_image(path))

    def rglob(self, pattern: str) -> Generator[Path, None, None]:
        return (path for path in super().rglob(pattern) if self._is_image(path))

    def iterdir(self) -> Generator[Path, None, None]:
        return (path for path in super().iterdir() if self._is_image(path))
