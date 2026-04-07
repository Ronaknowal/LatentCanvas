import numpy as np
from PIL import Image

_THUMB_SIZE = 64
_PIXEL_TOL = 5


class SimilarityFilter:
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold
        self._prev: np.ndarray | None = None

    def _to_array(self, image: Image.Image) -> np.ndarray:
        return np.array(
            image.resize((_THUMB_SIZE, _THUMB_SIZE)).convert("RGB"),
            dtype=np.int32,
        )

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of pixels whose max-channel difference is within tolerance."""
        diff = np.abs(a - b).max(axis=2)
        matches = (diff <= _PIXEL_TOL).sum()
        return float(matches) / (_THUMB_SIZE * _THUMB_SIZE)

    def should_generate(self, image: Image.Image) -> bool:
        arr = self._to_array(image)

        if self._prev is None:
            self._prev = arr
            return True

        similarity = self._similarity(arr, self._prev)

        if similarity > self.threshold:
            return False

        self._prev = arr
        return True
