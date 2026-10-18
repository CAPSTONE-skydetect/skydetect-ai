"""Local image registration around a KLT proposal; no trajectory smoothing."""

import cv2
import numpy as np

from ai_server.services.online_appearance import _subpixel_peak


def refine_patch_translation(
    previous: np.ndarray,
    current: np.ndarray,
    previous_center: tuple[float, float],
    proposed_center: tuple[float, float],
    target_size: tuple[float, float],
    *,
    search_radius: int = 3,
) -> tuple[float, float] | None:
    width, height = (max(11, int(round(size)) | 1) for size in target_size)
    radius = max(1, int(search_radius))
    for image, center, margin in (
        (previous, previous_center, 0), (current, proposed_center, radius),
    ):
        x, y = center
        if (x - width / 2 - margin < 0 or y - height / 2 - margin < 0
                or x + width / 2 + margin >= image.shape[1]
                or y + height / 2 + margin >= image.shape[0]):
            return None
    template = cv2.getRectSubPix(previous, (width, height), tuple(map(float, previous_center)))
    search = cv2.getRectSubPix(
        current, (width + 2 * radius, height + 2 * radius),
        tuple(map(float, proposed_center)),
    )
    # Remove slow background intensity variation before comparing image structure.
    template = template.astype(np.float32)
    search = search.astype(np.float32)
    template -= cv2.GaussianBlur(template, (0, 0), 2.0)
    search -= cv2.GaussianBlur(search, (0, 0), 2.0)
    if float(template.std()) < 2.0:
        return None
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(response)
    if not np.isfinite(score) or score < 0.7:
        return None
    if x in (0, 2 * radius) or y in (0, 2 * radius):
        return None
    others = response.copy()
    others[max(0, y-1):y+2, max(0, x-1):x+2] = -1
    if score - float(others.max()) < 0.02:
        return None
    dx, dy, _ = _subpixel_peak(response, x, y)
    return (float(proposed_center[0] + x - radius + dx),
            float(proposed_center[1] + y - radius + dy))
