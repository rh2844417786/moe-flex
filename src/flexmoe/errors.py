"""Typed fail-closed exceptions used across moe-flex."""


class FluxMoEError(RuntimeError):
    """Base exception for expected moe-flex failures."""


class ConfigurationError(FluxMoEError):
    """Raised when a declared configuration is inconsistent."""


class PreflightError(FluxMoEError):
    """Raised when execution prerequisites are not satisfied."""


class IntegrityError(FluxMoEError):
    """Raised when bytes or model behavior fail integrity checks."""


class UnsupportedModeError(FluxMoEError):
    """Raised instead of silently falling back from FluxMoE."""
