import cv2
import numpy as np

from ai_server.services.patch_registration import refine_patch_translation


def test_patch_recovers_fractional_translation_despite_brightness_change():
    yy, xx = np.indices((100, 100))
    previous = (150 - 90 * np.exp(-((xx-50)**2+(yy-50)**2)/12)
                - 50 * np.exp(-((xx-55)**2+(yy-48)**2)/6)).astype(np.float32)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 2.3], [0, 1, -1.4]]), (100, 100)) + 10
    center = refine_patch_translation(previous, current, (50, 50), (53, 49), (25, 25))
    assert center is not None
    assert np.linalg.norm(np.array(center) - [52.3, 48.6]) < 0.25


def test_patch_rejects_flat_and_boundary_regions():
    image = np.full((100, 100), 100, dtype=np.uint8)
    assert refine_patch_translation(image, image, (50, 50), (50, 50), (25, 25)) is None
    assert refine_patch_translation(image, image, (1, 1), (1, 1), (25, 25)) is None
