"""Standard-library-only offload feasibility analysis contracts."""

from .cost import analyze_feasibility
from .replay import replay_trace, select_resident
from .schema import (
    DemandTrace,
    ReplayConfig,
    ReplayResult,
    TimingPoint,
    TraceEvent,
    TransferSample,
    diagnostic_artifact,
    parse_diagnostic_artifact,
    validate_transfer_contract,
)

__all__ = [
    "DemandTrace",
    "ReplayConfig",
    "ReplayResult",
    "TimingPoint",
    "TraceEvent",
    "TransferSample",
    "analyze_feasibility",
    "diagnostic_artifact",
    "parse_diagnostic_artifact",
    "replay_trace",
    "select_resident",
    "validate_transfer_contract",
]
