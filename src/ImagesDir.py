from collections.abc import Generator
from pathlib import Path


class ImagesDir(Path):
    IMAGES_EXTS = [".png", ".jpg", ".jpeg", ".gif"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def _is_image(cls, path: Path):
        return path.suffix in cls.IMAGES_EXTS

    def glob(self, pattern: str) -> Generator[Path, None, None]:
        return (path for path in super().glob(pattern) if self._is_image(path))

    def rglob(self, pattern: str) -> Generator[Path, None, None]:
        return (path for path in super().rglob(pattern) if self._is_image(path))

    def iterdir(self) -> Generator[Path, None, None]:
        return (path for path in super().iterdir() if self._is_image(path))
