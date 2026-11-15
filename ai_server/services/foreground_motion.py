from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ForegroundMask:
    mask: np.ndarray
    coverage: float
    segmented: bool


@dataclass(frozen=True, slots=True)
class MotionMatch:
    center: tuple[float, float]
    score: float
    area: int


class ConstantVelocityKalman:
    """Constant-velocity prediction in camera-compensated coordinates."""

    def __init__(self, center: tuple[float, float]) -> None:
        self._filter = cv2.KalmanFilter(4, 2)
        self._filter.transitionMatrix = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._filter.measurementMatrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self._filter.processNoiseCov = np.diag([0.05, 0.05, 0.35, 0.35]).astype(np.float32)
        self._filter.measurementNoiseCov = np.diag([0.8, 0.8]).astype(np.float32)
        self._filter.errorCovPost = np.diag([1.0, 1.0, 80.0, 80.0]).astype(np.float32)
        self._filter.statePost = np.array(
            [[float(center[0])], [float(center[1])], [0.0], [0.0]],
            dtype=np.float32,
        )

    def predict(self) -> tuple[float, float]:
        state = self._filter.predict()
        return float(state[0, 0]), float(state[1, 0])

    def correct(self, center: tuple[float, float]) -> tuple[float, float]:
        state = self._filter.correct(
            np.array([[float(center[0])], [float(center[1])]], dtype=np.float32)
        )
        return float(state[0, 0]), float(state[1, 0])


def build_foreground_mask(
    frame: np.ndarray,
    center: tuple[float, float],
    target_size: tuple[float, float],
) -> ForegroundMask:
    width = max(8, int(round(target_size[0])))
    height = max(8, int(round(target_size[1])))
    patch = cv2.getRectSubPix(frame, (width, height), center)
    if patch.ndim != 3 or patch.shape[2] != 3:
        return _full_mask(width, height)

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = _border_mask(height, width)
    background_color = np.median(lab[border], axis=0)

    center_radius_x = max(1, int(round(width * 0.12)))
    center_radius_y = max(1, int(round(height * 0.12)))
    center_x = width // 2
    center_y = height // 2
    center_seed = lab[
        max(0, center_y - center_radius_y) : min(height, center_y + center_radius_y + 1),
        max(0, center_x - center_radius_x) : min(width, center_x + center_radius_x + 1),
    ]
    target_color = np.median(center_seed.reshape(-1, 3), axis=0)
    target_to_background = float(np.linalg.norm(target_color - background_color))
    if target_to_background < 7.0:
        return _full_mask(width, height)

    background_distance = np.linalg.norm(lab - background_color, axis=2)
    target_distance = np.linalg.norm(lab - target_color, axis=2)
    threshold = max(5.0, min(30.0, target_to_background * 0.28))
    candidate = (
        (background_distance >= threshold)
        & (target_distance <= background_distance * 1.15 + 3.0)
    ).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.dilate(candidate, kernel, iterations=1)
    selected = _component_near_center(candidate)
    coverage = float(np.count_nonzero(selected)) / max(float(selected.size), 1.0)
    if coverage < 0.025 or coverage > 0.88:
        return _full_mask(width, height)
    return ForegroundMask(mask=(selected * 255).astype(np.uint8), coverage=coverage, segmented=True)


def foreground_contrast_score(
    gray: np.ndarray,
    center: tuple[float, float],
    target_size: tuple[float, float],
    foreground_mask: np.ndarray,
) -> float:
    width = max(8, int(round(target_size[0])))
    height = max(8, int(round(target_size[1])))
    patch = cv2.getRectSubPix(gray, (width, height), center).astype(np.float32)
    mask = _resize_mask(foreground_mask, width=width, height=height)
    foreground = mask > 0
    background = ~foreground
    if int(background.sum()) < 4:
        background = _border_mask(height, width)
    if int(foreground.sum()) < 4 or int(background.sum()) < 4:
        return 0.5

    mean_delta = abs(float(np.mean(patch[foreground])) - float(np.mean(patch[background])))
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    boundary = cv2.dilate(foreground.astype(np.uint8), np.ones((3, 3), np.uint8)) != cv2.erode(
        foreground.astype(np.uint8),
        np.ones((3, 3), np.uint8),
    )
    edge_strength = float(np.mean(gradient[boundary])) if boundary.any() else 0.0
    intensity_score = min(1.0, mean_delta / 58.0)
    edge_score = min(1.0, edge_strength / 160.0)
    return float(max(0.0, min(1.0, 0.62 * intensity_score + 0.38 * edge_score)))


def find_residual_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_to_current: np.ndarray,
    predicted_center: tuple[float, float],
    *,
    target_size: tuple[float, float],
    foreground_mask: np.ndarray,
    search_radius: int,
) -> MotionMatch | None:
    height, width = curr_gray.shape[:2]
    aligned_previous = cv2.warpAffine(
        prev_gray,
        prev_to_current,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    difference = cv2.absdiff(curr_gray, aligned_previous)
    difference = cv2.GaussianBlur(difference, (3, 3), 0)

    half_width = target_size[0] / 2.0
    half_height = target_size[1] / 2.0
    cx, cy = predicted_center
    x1 = max(0, int(round(cx - search_radius - half_width)))
    y1 = max(0, int(round(cy - search_radius - half_height)))
    x2 = min(width, int(round(cx + search_radius + half_width + 1)))
    y2 = min(height, int(round(cy + search_radius + half_height + 1)))
    crop = difference[y1:y2, x1:x2]
    if crop.size < 9:
        return None

    nonzero = crop[crop > 0]
    adaptive_threshold = float(np.percentile(nonzero, 70)) if nonzero.size else 0.0
    threshold = max(4.0, min(28.0, adaptive_threshold))
    binary = (crop.astype(np.float32) >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    expected_area = max(
        3.0,
        float(target_size[0] * target_size[1])
        * max(0.08, float(np.count_nonzero(foreground_mask)) / max(foreground_mask.size, 1))
        * 0.45,
    )
    max_area = max(12.0, float(target_size[0] * target_size[1]) * 2.5)
    best: MotionMatch | None = None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 2 or area > max_area:
            continue
        local_x, local_y = centroids[label]
        center = (float(x1 + local_x), float(y1 + local_y))
        distance = hypot(center[0] - cx, center[1] - cy)
        if distance > search_radius + max(target_size):
            continue

        pixels = crop[labels == label]
        strength = min(1.0, float(np.mean(pixels)) / 34.0)
        distance_score = exp(-distance / max(6.0, max(target_size) * 0.85))
        area_score = min(area, expected_area) / max(area, expected_area)
        score = 0.48 * strength + 0.37 * distance_score + 0.15 * area_score
        match = MotionMatch(center=center, score=float(score), area=area)
        if best is None or match.score > best.score:
            best = match
    return best


def place_foreground_mask(
    image_shape: tuple[int, ...],
    bbox: tuple[float, float, float, float],
    foreground_mask: np.ndarray,
) -> np.ndarray:
    height, width = image_shape[:2]
    x, y, bbox_width, bbox_height = bbox
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(width, int(np.ceil(x + bbox_width)))
    y2 = min(height, int(np.ceil(y + bbox_height)))
    canvas = np.zeros((height, width), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return canvas
    canvas[y1:y2, x1:x2] = _resize_mask(
        foreground_mask,
        width=x2 - x1,
        height=y2 - y1,
    )
    return canvas


def _component_near_center(candidate: np.ndarray) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        return candidate
    height, width = candidate.shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    center_label = int(labels[min(height - 1, height // 2), min(width - 1, width // 2)])
    if center_label > 0:
        return (labels == center_label).astype(np.uint8)

    best_label = min(
        range(1, count),
        key=lambda label: (
            float(np.linalg.norm(centroids[label] - center)),
            -int(stats[label, cv2.CC_STAT_AREA]),
        ),
    )
    max_distance = hypot(width, height) * 0.38
    if float(np.linalg.norm(centroids[best_label] - center)) > max_distance:
        return np.zeros_like(candidate)
    return (labels == best_label).astype(np.uint8)


def _border_mask(height: int, width: int) -> np.ndarray:
    thickness = max(1, int(round(min(height, width) * 0.14)))
    border = np.zeros((height, width), dtype=bool)
    border[:thickness, :] = True
    border[-thickness:, :] = True
    border[:, :thickness] = True
    border[:, -thickness:] = True
    return border


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 255, 0).astype(np.uint8)


def _full_mask(width: int, height: int) -> ForegroundMask:
    mask = np.full((height, width), 255, dtype=np.uint8)
    return ForegroundMask(mask=mask, coverage=1.0, segmented=False)
