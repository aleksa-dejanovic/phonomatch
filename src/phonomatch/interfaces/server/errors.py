"""Errors raised while processing HTTP requests."""


class RequestTimeoutError(ValueError):
    """Raised when a client does not finish an HTTP request in time."""


class ServerBusyError(ValueError):
    """Raised when accepting work would exceed server capacity."""
