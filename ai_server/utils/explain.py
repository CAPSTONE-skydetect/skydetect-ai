"""Helpers for summarizing model explanations."""


def summarize_top_features(
    names: list[str],
    scores: list[float],
    top_k: int = 3,
) -> dict[str, float]:
    """Return the top-k explanation scores as a simple mapping."""

    pairs = sorted(zip(names, scores, strict=False), key=lambda item: item[1], reverse=True)
    return {name: float(score) for name, score in pairs[:top_k]}
