from research.cmc_gt_diagnostics import _correlation, _percentile


def test_percentile_returns_none_for_empty_values() -> None:
    assert _percentile([], 0.95) is None


def test_correlation_detects_linear_relationship() -> None:
    assert _correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0


def test_correlation_rejects_constant_values() -> None:
    assert _correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None
