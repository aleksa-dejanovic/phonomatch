"""A small local HTTP API for recognizing uploaded WAV audio."""

from ...application.phonomatch import PhonoMatch
from .errors import RequestTimeoutError, ServerBusyError
from .handlers import MAX_AUDIO_BYTES, UPLOAD_CHUNK_BYTES, create_recognition_handler
from .http_server import (
    MAX_CONCURRENT_REQUESTS,
    MAX_INFLIGHT_UPLOAD_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    BoundedThreadingHTTPServer,
)
from .lifecycle import ModelLifecycleGate
from .payloads import result_payload

__all__ = [
    "MAX_AUDIO_BYTES",
    "MAX_CONCURRENT_REQUESTS",
    "MAX_INFLIGHT_UPLOAD_BYTES",
    "REQUEST_TIMEOUT_SECONDS",
    "UPLOAD_CHUNK_BYTES",
    "BoundedThreadingHTTPServer",
    "ModelLifecycleGate",
    "RequestTimeoutError",
    "ServerBusyError",
    "create_server",
    "result_payload",
    "serve",
]


def create_server(
    host: str,
    port: int,
    analyzer: PhonoMatch,
    *,
    enable_model_lifecycle: bool = False,
) -> BoundedThreadingHTTPServer:
    """Create a server without starting it, primarily for embedding and tests."""
    return BoundedThreadingHTTPServer(
        (host, port),
        create_recognition_handler(
            analyzer,
            ModelLifecycleGate(),
            enable_model_lifecycle=enable_model_lifecycle,
        ),
    )


def serve(
    host: str,
    port: int,
    analyzer: PhonoMatch,
    *,
    enable_model_lifecycle: bool = False,
) -> None:
    """Run a recognition service until interrupted."""
    with create_server(
        host, port, analyzer, enable_model_lifecycle=enable_model_lifecycle
    ) as server:
        server.serve_forever()
