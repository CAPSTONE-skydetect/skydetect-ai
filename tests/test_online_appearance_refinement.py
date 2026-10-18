import cv2
import numpy as np

from ai_server.services.online_appearance import _match_template, _subpixel_peak


def test_parabolic_subpixel_peak_has_bounded_offset():
    response = np.array([
        [0.0, 0.1, 0.0],
        [0.2, 1.0, 0.6],
        [0.0, 0.4, 0.0],
    ], dtype=np.float32)
    dx, dy, status = _subpixel_peak(response, 1, 1)
    assert status == "subpixel_parabolic"
    assert -0.5 <= dx <= 0.5
    assert -0.5 <= dy <= 0.5


def test_subpixel_peak_falls_back_at_response_boundary():
    response = np.ones((3, 3), dtype=np.float32)
    assert _subpixel_peak(response, 0, 1) == (0.0, 0.0, "fallback_boundary")


def test_template_match_reports_peak_diagnostics():
    base = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(base, (30, 30), 4, 255, -1, cv2.LINE_AA)
    template = cv2.getRectSubPix(base, (16, 16), (30.0, 30.0))
    transform = np.array([[1.0, 0.0, 0.35], [0.0, 1.0, 0.25]], dtype=np.float32)
    shifted = cv2.warpAffine(base, transform, (64, 64), flags=cv2.INTER_CUBIC)
    match = _match_template(
        shifted, template, (30.0, 30.0), search_radius=8, subpixel=True,
    )
    assert match is not None
    assert np.isfinite(match.peak_to_sidelobe_ratio)
    assert match.second_peak_margin >= 0.0
    assert all(abs(value) <= 0.5 for value in match.subpixel_offset)
