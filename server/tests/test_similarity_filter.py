import numpy as np
from PIL import Image
import pytest

from similarity_filter import SimilarityFilter


def _white_image(size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), (255, 255, 255))


def _black_image(size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), (0, 0, 0))


def _noisy_image(size: int = 64, seed: int = 42) -> Image.Image:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestSimilarityFilter:
    def test_first_image_always_passes(self):
        f = SimilarityFilter(threshold=0.98)
        assert f.should_generate(_white_image()) is True

    def test_identical_image_is_skipped(self):
        f = SimilarityFilter(threshold=0.98)
        f.should_generate(_white_image())
        assert f.should_generate(_white_image()) is False

    def test_very_different_image_passes(self):
        f = SimilarityFilter(threshold=0.98)
        f.should_generate(_white_image())
        assert f.should_generate(_black_image()) is True

    def test_threshold_controls_sensitivity(self):
        f = SimilarityFilter(threshold=0.99999)
        img1 = _noisy_image(seed=1)
        img2 = _noisy_image(seed=2)
        f.should_generate(img1)
        assert f.should_generate(img2) is True

    def test_low_threshold_skips_more(self):
        f = SimilarityFilter(threshold=0.5)
        f.should_generate(_noisy_image(seed=1))
        assert f.should_generate(_noisy_image(seed=2)) is True

    def test_previous_image_updates_on_pass(self):
        f = SimilarityFilter(threshold=0.98)
        f.should_generate(_white_image())
        f.should_generate(_black_image())  # passes, updates prev
        assert f.should_generate(_white_image()) is True

    def test_previous_image_does_not_update_on_skip(self):
        f = SimilarityFilter(threshold=0.98)
        f.should_generate(_white_image())
        f.should_generate(_white_image())  # skipped, prev stays white
        assert f.should_generate(_black_image()) is True

    def test_accepts_different_image_sizes(self):
        f = SimilarityFilter(threshold=0.98)
        big = Image.new("RGB", (1024, 1024), (255, 255, 255))
        assert f.should_generate(big) is True
        small = Image.new("RGB", (256, 256), (255, 255, 255))
        assert f.should_generate(small) is False  # still white
