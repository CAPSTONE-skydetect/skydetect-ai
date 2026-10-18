from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class AppearanceMatch:
    center: tuple[float, float]
    score: float
    anchor_score: float
    peak_score: float = 0.0
    second_peak_margin: float = 0.0
    peak_to_sidelobe_ratio: float = 0.0
    subpixel_offset: tuple[float, float] = (0.0, 0.0)
    refinement_status: str = "disabled"


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    center: tuple[float, float]
    peak_score: float
    second_peak_margin: float
    peak_to_sidelobe_ratio: float
    subpixel_offset: tuple[float, float]
    refinement_status: str


class OnlineAppearanceModel:
    """Small per-video classifier learned from the selected ROI and nearby background."""

    def __init__(
        self,
        gray: np.ndarray,
        center: tuple[float, float],
        target_size: tuple[float, float],
        foreground_mask: np.ndarray | None = None,
        local_refine: bool = False,
        subpixel: bool = False,
    ) -> None:
        self.target_size = (
            max(8.0, float(target_size[0])),
            max(8.0, float(target_size[1])),
        )
        self.local_refine = bool(local_refine)
        self.subpixel = bool(subpixel)
        self.anchor_template = _extract_patch(gray, center, self.target_size)
        self.short_template = self.anchor_template.copy()
        self.foreground_mask = _normalize_mask(
            foreground_mask,
            width=self.anchor_template.shape[1],
            height=self.anchor_template.shape[0],
        )
        self._initial_positives = [
            self._descriptor(gray, (center[0] + dx, center[1] + dy))
            for dx, dy in _positive_offsets()
        ]
        self._initial_negatives = [
            self._descriptor(gray, negative_center)
            for negative_center in self._negative_centers(center)
        ]
        self._recent_positives: list[np.ndarray] = []
        self._recent_negatives: list[np.ndarray] = []
        self._weights = np.zeros(len(self._initial_positives[0]) + 1, dtype=np.float32)
        self._midpoint = 0.0
        self._scale = 1.0
        self.update_count = 0
        self._fit()

    def score(self, gray: np.ndarray, center: tuple[float, float]) -> float:
        feature = self._descriptor(gray, center)
        raw_score = self._raw_score(feature)
        normalized = (raw_score - self._midpoint) / max(self._scale, 0.10)
        classifier_score = 1.0 / (1.0 + exp(-max(-12.0, min(12.0, normalized * 2.0))))
        anchor_score = _template_similarity(
            gray,
            self.anchor_template,
            center,
            self.target_size,
            self.foreground_mask,
        )
        short_score = _template_similarity(
            gray,
            self.short_template,
            center,
            self.target_size,
            self.foreground_mask,
        )
        return float(
            max(
                0.0,
                min(1.0, 0.58 * classifier_score + 0.42 * max(anchor_score, short_score)),
            )
        )

    def search(
        self,
        gray: np.ndarray,
        predicted_center: tuple[float, float],
        *,
        search_radius: int,
    ) -> AppearanceMatch | None:
        radius = max(6, int(search_radius))
        candidates = {self._clamp_center(gray, predicted_center)}
        template_matches: list[TemplateMatch] = []

        for template in (self.anchor_template, self.short_template):
            match = _match_template(
                gray,
                template,
                predicted_center,
                search_radius=radius,
                subpixel=self.subpixel,
            )
            if match is not None:
                template_matches.append(match)
                candidates.add(self._clamp_center(gray, match.center))

        step = max(3, int(round(max(radius / 8.0, min(self.target_size) / 5.0))))
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                candidates.add(
                    self._clamp_center(
                        gray,
                        (predicted_center[0] + dx, predicted_center[1] + dy),
                    )
                )

        best: AppearanceMatch | None = None
        for center in candidates:
            appearance_score = self.score(gray, center)
            anchor_score = _template_similarity(
                gray,
                self.anchor_template,
                center,
                self.target_size,
                self.foreground_mask,
            )
            distance = hypot(center[0] - predicted_center[0], center[1] - predicted_center[1])
            ranked_score = appearance_score - 0.08 * min(1.0, distance / max(radius, 1))
            if best is None or ranked_score > best.score:
                best = AppearanceMatch(
                    center=center,
                    score=float(ranked_score),
                    anchor_score=float(anchor_score),
                )
        if best is not None and self.local_refine:
            refine_radius = max(1, min(3, int(np.ceil(step / 2.0))))
            local_centers = {
                self._clamp_center(gray, (best.center[0] + dx, best.center[1] + dy))
                for dy in range(-refine_radius, refine_radius + 1)
                for dx in range(-refine_radius, refine_radius + 1)
            }
            for center in local_centers:
                appearance_score = self.score(gray, center)
                anchor_score = _template_similarity(
                    gray, self.anchor_template, center, self.target_size,
                    self.foreground_mask,
                )
                distance = hypot(
                    center[0] - predicted_center[0], center[1] - predicted_center[1],
                )
                ranked_score = appearance_score - 0.08 * min(
                    1.0, distance / max(radius, 1),
                )
                if ranked_score > best.score:
                    best = AppearanceMatch(
                        center=center, score=float(ranked_score),
                        anchor_score=float(anchor_score),
                        refinement_status="integer_local_refine",
                    )
        if best is not None and template_matches:
            nearest = min(
                template_matches,
                key=lambda item: hypot(
                    item.center[0] - best.center[0], item.center[1] - best.center[1],
                ),
            )
            if hypot(nearest.center[0] - best.center[0], nearest.center[1] - best.center[1]) <= max(2.0, step):
                best = AppearanceMatch(
                    center=best.center, score=best.score,
                    anchor_score=best.anchor_score,
                    peak_score=nearest.peak_score,
                    second_peak_margin=nearest.second_peak_margin,
                    peak_to_sidelobe_ratio=nearest.peak_to_sidelobe_ratio,
                    subpixel_offset=nearest.subpixel_offset,
                    refinement_status=(
                        nearest.refinement_status
                        if nearest.refinement_status != "disabled"
                        else best.refinement_status
                    ),
                )
        return best

    def update(self, gray: np.ndarray, center: tuple[float, float]) -> bool:
        self._recent_positives.append(self._descriptor(gray, center))
        self._recent_positives = self._recent_positives[-16:]

        hard_negatives = []
        for negative_center in self._negative_centers(center):
            feature = self._descriptor(gray, negative_center)
            hard_negatives.append((self._raw_score(feature), feature))
        hard_negatives.sort(key=lambda item: item[0], reverse=True)
        self._recent_negatives.extend(feature for _, feature in hard_negatives[:4])
        self._recent_negatives = self._recent_negatives[-32:]

        candidate_template = _extract_patch(gray, center, self.target_size)
        if candidate_template.shape == self.short_template.shape:
            blended = cv2.addWeighted(
                self.short_template,
                0.96,
                candidate_template,
                0.04,
                0.0,
            )
            selected = self.foreground_mask > 0
            self.short_template[selected] = blended[selected]

        self.update_count += 1
        if self.update_count % 4 == 0:
            self._fit()
        return True

    def _fit(self) -> None:
        positives = [*self._initial_positives, *self._recent_positives]
        negatives = [*self._initial_negatives, *self._recent_negatives]
        negatives = negatives[-max(len(positives), len(self._initial_negatives)) :]
        positive_features = np.stack(positives).astype(np.float32)
        negative_features = np.stack(negatives).astype(np.float32)
        positive_center = np.mean(positive_features, axis=0)
        negative_center = np.mean(negative_features, axis=0)
        direction = positive_center - negative_center
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-6:
            direction /= direction_norm
        bias = -0.5 * float(np.sum((positive_center + negative_center) * direction))
        self._weights = np.concatenate(
            [direction.astype(np.float32), np.array([bias], dtype=np.float32)]
        )

        positive_scores = np.sum(positive_features * direction, axis=1) + bias
        negative_scores = np.sum(negative_features * direction, axis=1) + bias
        positive_mean = float(np.mean(positive_scores))
        negative_mean = float(np.mean(negative_scores))
        self._midpoint = (positive_mean + negative_mean) / 2.0
        self._scale = max(0.10, (positive_mean - negative_mean) / 2.0)

    def _raw_score(self, feature: np.ndarray) -> float:
        vector = np.concatenate([feature, np.ones(1, dtype=np.float32)])
        return float(np.sum(vector * self._weights))

    def _descriptor(self, gray: np.ndarray, center: tuple[float, float]) -> np.ndarray:
        center = self._clamp_center(gray, center)
        patch = _extract_patch(gray, center, self.target_size)
        patch = cv2.resize(patch, (24, 24), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(self.foreground_mask, (24, 24), interpolation=cv2.INTER_NEAREST)
        weights = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
        selected = weights > 0
        values = patch.astype(np.float32) / 255.0
        mean = float(np.mean(values[selected])) if selected.any() else float(np.mean(values))
        std = float(np.std(values[selected])) if selected.any() else float(np.std(values))
        values = (values - mean) / max(std, 0.08)
        gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
        values *= weights
        gx *= weights
        gy *= weights
        pooled = [
            cv2.resize(channel, (12, 12), interpolation=cv2.INTER_AREA).reshape(-1)
            for channel in (values, gx, gy)
        ]
        feature = np.nan_to_num(
            np.concatenate(pooled),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm > 1e-6:
            feature /= norm
        return feature

    def _negative_centers(
        self,
        center: tuple[float, float],
    ) -> list[tuple[float, float]]:
        dx = max(12.0, self.target_size[0] * 1.15)
        dy = max(12.0, self.target_size[1] * 1.15)
        return [
            (center[0] - dx, center[1]),
            (center[0] + dx, center[1]),
            (center[0], center[1] - dy),
            (center[0], center[1] + dy),
            (center[0] - dx, center[1] - dy),
            (center[0] + dx, center[1] - dy),
            (center[0] - dx, center[1] + dy),
            (center[0] + dx, center[1] + dy),
        ]

    @staticmethod
    def _clamp_center(
        gray: np.ndarray,
        center: tuple[float, float],
    ) -> tuple[float, float]:
        height, width = gray.shape[:2]
        return (
            float(max(0.0, min(center[0], max(width - 1, 0)))),
            float(max(0.0, min(center[1], max(height - 1, 0)))),
        )


def _positive_offsets() -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 0.0),
        (-1.5, 0.0),
        (1.5, 0.0),
        (0.0, -1.5),
        (0.0, 1.5),
        (-1.5, -1.5),
        (1.5, -1.5),
        (-1.5, 1.5),
        (1.5, 1.5),
    )


def _extract_patch(
    gray: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
) -> np.ndarray:
    width = max(8, int(round(size[0])))
    height = max(8, int(round(size[1])))
    return cv2.getRectSubPix(
        gray,
        (width, height),
        (float(center[0]), float(center[1])),
    )


def _template_similarity(
    gray: np.ndarray,
    template: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    foreground_mask: np.ndarray,
) -> float:
    patch = _extract_patch(gray, center, size)
    if patch.shape != template.shape:
        return 0.0
    mask = _normalize_mask(
        foreground_mask,
        width=template.shape[1],
        height=template.shape[0],
    )
    selected = mask > 0
    if int(selected.sum()) < 4:
        selected = np.ones(template.shape[:2], dtype=bool)
    template_mean = float(np.mean(template[selected]))
    patch_mean = float(np.mean(patch[selected]))
    template_values = np.where(selected, template.astype(np.float32) - template_mean, 0.0)
    patch_values = np.where(selected, patch.astype(np.float32) - patch_mean, 0.0)
    denominator = float(np.linalg.norm(template_values) * np.linalg.norm(patch_values))
    if denominator <= 1e-6:
        difference = float(
            np.mean(
                np.abs(template.astype(np.float32)[selected] - patch.astype(np.float32)[selected])
            )
        )
        return float(max(0.0, 1.0 - difference / 255.0))
    correlation = float(np.sum(template_values * patch_values) / denominator)
    return float(max(0.0, min(1.0, (correlation + 1.0) / 2.0)))


def _match_template(
    gray: np.ndarray,
    template: np.ndarray,
    predicted_center: tuple[float, float],
    *,
    search_radius: int,
    subpixel: bool = False,
) -> TemplateMatch | None:
    height, width = gray.shape[:2]
    template_height, template_width = template.shape[:2]
    cx, cy = predicted_center
    x1 = max(0, int(round(cx - template_width / 2.0 - search_radius)))
    y1 = max(0, int(round(cy - template_height / 2.0 - search_radius)))
    x2 = min(width, int(round(cx + template_width / 2.0 + search_radius)))
    y2 = min(height, int(round(cy + template_height / 2.0 + search_radius)))
    search = gray[y1:y2, x1:x2]
    if search.shape[0] < template_height or search.shape[1] < template_width:
        return None
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, max_location = cv2.minMaxLoc(response)
    px, py = max_location
    offset_x = offset_y = 0.0
    status = "disabled"
    if subpixel:
        offset_x, offset_y, status = _subpixel_peak(response, px, py)
    exclusion = response.copy()
    ey1, ey2 = max(0, py - 1), min(response.shape[0], py + 2)
    ex1, ex2 = max(0, px - 1), min(response.shape[1], px + 2)
    exclusion[ey1:ey2, ex1:ex2] = -np.inf
    finite_sidelobes = exclusion[np.isfinite(exclusion)]
    second = float(np.max(finite_sidelobes)) if finite_sidelobes.size else float(peak)
    finite_response = response[np.isfinite(response)]
    mean = float(np.mean(finite_response)) if finite_response.size else 0.0
    std = float(np.std(finite_response)) if finite_response.size else 0.0
    psr = (float(peak) - mean) / max(std, 1e-6)
    return TemplateMatch(
        center=(
            float(x1 + px + offset_x + template_width / 2.0),
            float(y1 + py + offset_y + template_height / 2.0),
        ),
        peak_score=float(peak),
        second_peak_margin=float(peak) - second,
        peak_to_sidelobe_ratio=float(psr),
        subpixel_offset=(float(offset_x), float(offset_y)),
        refinement_status=status,
    )


def _subpixel_peak(
    response: np.ndarray, x: int, y: int,
) -> tuple[float, float, str]:
    if x <= 0 or y <= 0 or x >= response.shape[1] - 1 or y >= response.shape[0] - 1:
        return 0.0, 0.0, "fallback_boundary"

    def axis_offset(left: float, center: float, right: float) -> float | None:
        curvature = left - 2.0 * center + right
        if not np.isfinite(curvature) or curvature >= -1e-6:
            return None
        value = 0.5 * (left - right) / curvature
        if not np.isfinite(value) or abs(value) > 0.5:
            return None
        return float(value)

    center = float(response[y, x])
    dx = axis_offset(float(response[y, x - 1]), center, float(response[y, x + 1]))
    dy = axis_offset(float(response[y - 1, x]), center, float(response[y + 1, x]))
    if dx is None or dy is None:
        return 0.0, 0.0, "fallback_flat_or_unstable"
    return dx, dy, "subpixel_parabolic"


def _normalize_mask(
    foreground_mask: np.ndarray | None,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if foreground_mask is None or foreground_mask.size == 0:
        return np.full((height, width), 255, dtype=np.uint8)
    resized = cv2.resize(
        foreground_mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.where(resized > 0, 255, 0).astype(np.uint8)
