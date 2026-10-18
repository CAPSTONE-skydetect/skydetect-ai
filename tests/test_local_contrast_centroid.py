import numpy as np

from ai_server.services.manual_roi_tracker import _refine_center_by_contrast


def test_centroid_ignores_strong_distant_background():
    yy, xx = np.indices((81, 81))
    image = np.full((81, 81), 100.0)
    image -= 50 * np.exp(-((xx - 40.3)**2 + (yy - 39.7)**2) / 8)
    image[15, 15] = 255
    center = _refine_center_by_contrast(
        image.astype(np.uint8), (40, 40), radius=32, centroid_sigma=4,
    )
    assert center is not None
    assert np.linalg.norm(np.array(center) - [40.3, 39.7]) < 0.5


def test_centroid_does_not_invent_target_in_flat_region():
    image = np.full((81, 81), 100, dtype=np.uint8)
    image[15, 15] = 255
    assert _refine_center_by_contrast(
        image, (40, 40), radius=32, centroid_sigma=4,
    ) is None
