from ai_server.schemas import StabilizationInfo, StabilizationMethod


def build_stabilization_info(method: StabilizationMethod) -> StabilizationInfo:
    return StabilizationInfo(
        applied=method != "none",
        method=method,
    )
