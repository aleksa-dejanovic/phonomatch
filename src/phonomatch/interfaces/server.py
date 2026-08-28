"""A small local HTTP API for recognizing uploaded WAV audio."""

from __future__ import annotations

import json
import logging
import socket
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock, RLock
from typing import Any, cast
from urllib.parse import urlparse

from ..application.phonomatch import PhonoMatch
from ..domain.models import AnalysisResult, PhraseAnalysisResult
from ..exceptions import PhonoMatchError

MAX_AUDIO_BYTES = 32 * 1024 * 1024
MAX_INFLIGHT_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_CONCURRENT_REQUESTS = 4
REQUEST_TIMEOUT_SECONDS = 15.0
UPLOAD_CHUNK_BYTES = 64 * 1024
LOGGER = logging.getLogger(__name__)


class RequestTimeoutError(ValueError):
    """Raised when a client does not finish an HTTP request in time."""


class ServerBusyError(ValueError):
    """Raised when accepting an upload would exceed server capacity."""


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with bounded active work and upload storage."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
        max_inflight_upload_bytes: int = MAX_INFLIGHT_UPLOAD_BYTES,
        request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        if max_inflight_upload_bytes < 1:
            raise ValueError("max_inflight_upload_bytes must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        super().__init__(server_address, request_handler_class)
        self.max_concurrent_requests = max_concurrent_requests
        self.max_inflight_upload_bytes = max_inflight_upload_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self._request_slots = BoundedSemaphore(max_concurrent_requests)
        self._upload_lock = Lock()
        self._inflight_upload_bytes = 0

    def process_request(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        if not self._request_slots.acquire(blocking=False):
            self._send_overloaded_response(request)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()

    def reserve_upload(self, size: int) -> bool:
        with self._upload_lock:
            if self._inflight_upload_bytes + size > self.max_inflight_upload_bytes:
                return False
            self._inflight_upload_bytes += size
            return True

    def release_upload(self, size: int) -> None:
        with self._upload_lock:
            self._inflight_upload_bytes -= size

    @staticmethod
    def _send_overloaded_response(
        request: socket.socket | tuple[bytes, socket.socket],
    ) -> None:
        body = b'{"error":"server is busy; try again later"}'
        response = (
            b"HTTP/1.1 503 Service Unavailable\r\n"
            b"Content-Type: application/json; charset=utf-8\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        with suppress(OSError):
            cast(socket.socket, request).sendall(response)


def result_payload(result: AnalysisResult | PhraseAnalysisResult) -> dict[str, Any]:
    """Convert a public result model into the JSON representation of the API."""
    payload = asdict(result)
    payload["accepted"] = (
        result.match.accepted if isinstance(result, AnalysisResult) else result.accepted
    )
    return payload


def create_server(
    host: str,
    port: int,
    analyzer: PhonoMatch,
    *,
    enable_model_lifecycle: bool = False,
) -> BoundedThreadingHTTPServer:
    """Create a server without starting it, primarily for embedding and tests."""

    # Recognition requests and lifecycle operations share a process-wide model
    # cache.  Do not clear it while an ONNX session is serving a request.
    model_lock = RLock()

    class RecognitionHandler(BaseHTTPRequestHandler):
        server_version = "PhonoMatch/0.1"

        def setup(self) -> None:
            super().setup()
            server = cast(BoundedThreadingHTTPServer, self.server)
            self.connection.settimeout(server.request_timeout_seconds)

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/health":
                self._send_error(HTTPStatus.NOT_FOUND, "endpoint not found")
                return
            self._send_json(HTTPStatus.OK, {"status": "ok"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if enable_model_lifecycle and path == "/v1/model/load":
                self._load_model()
                return
            if enable_model_lifecycle and path == "/v1/model/unload":
                self._unload_model()
                return
            if path not in {"/v1/recognize", "/v1/recognize/phrase"}:
                self._send_error(HTTPStatus.NOT_FOUND, "endpoint not found")
                return
            try:
                with self._temporary_audio() as audio_path:
                    result = self._analyze(audio_path, phrase=path.endswith("/phrase"))
            except RequestTimeoutError as exc:
                self._send_error(HTTPStatus.REQUEST_TIMEOUT, str(exc))
            except ServerBusyError as exc:
                self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except PhonoMatchError as exc:
                LOGGER.warning("recognition request failed: %s", exc)
                self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            else:
                self._send_json(HTTPStatus.OK, result_payload(result))

        def _load_model(self) -> None:
            try:
                with model_lock:
                    analyzer.load_model()
            except PhonoMatchError as exc:
                LOGGER.warning("model load failed: %s", exc)
                self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            else:
                self._send_json(HTTPStatus.OK, {"status": "loaded"})

        def _unload_model(self) -> None:
            try:
                with model_lock:
                    analyzer.unload_model()
            except PhonoMatchError as exc:
                LOGGER.warning("model unload failed: %s", exc)
                self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            else:
                self._send_json(HTTPStatus.OK, {"status": "unloaded"})

        @contextmanager
        def _temporary_audio(self) -> Iterator[Path]:
            """Stream a bounded WAV request into a temporary file."""
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
            if content_type not in {"audio/wav", "audio/x-wav"}:
                raise ValueError("Content-Type must be audio/wav")
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise ValueError("Content-Length is required")
            try:
                size = int(content_length)
            except ValueError as exc:
                raise ValueError("Content-Length must be an integer") from exc
            if not 0 < size <= MAX_AUDIO_BYTES:
                raise ValueError(f"audio must be between 1 and {MAX_AUDIO_BYTES} bytes")
            server = cast(BoundedThreadingHTTPServer, self.server)
            if not server.reserve_upload(size):
                raise ServerBusyError(
                    "server upload capacity is exhausted; try again later"
                )
            path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temporary:
                    path = Path(temporary.name)
                    remaining = size
                    while remaining:
                        try:
                            chunk = self.rfile.read(min(UPLOAD_CHUNK_BYTES, remaining))
                        except TimeoutError as exc:
                            raise RequestTimeoutError("request body timed out") from exc
                        if not chunk:
                            raise ValueError("request body ended before Content-Length")
                        temporary.write(chunk)
                        remaining -= len(chunk)
                yield path
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)
                server.release_upload(size)

        def _analyze(
            self, path: Path, *, phrase: bool
        ) -> AnalysisResult | PhraseAnalysisResult:
            with model_lock:
                if phrase:
                    return analyzer.analyze_phrase_file(path)
                return analyzer.analyze_file(path)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json(status, {"error": message})

        def log_message(self, format: str, *args: object) -> None:
            return

    return BoundedThreadingHTTPServer((host, port), RecognitionHandler)


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
