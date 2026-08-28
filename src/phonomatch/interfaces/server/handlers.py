"""HTTP request handlers for the recognition API."""

import json
import logging
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from ...application.phonomatch import PhonoMatch
from ...domain.models import AnalysisResult, PhraseAnalysisResult
from ...exceptions import PhonoMatchError
from .errors import RequestTimeoutError, ServerBusyError
from .http_server import BoundedThreadingHTTPServer
from .lifecycle import ModelLifecycleGate
from .payloads import result_payload

MAX_AUDIO_BYTES = 32 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024
LOGGER = logging.getLogger(__name__)


def create_recognition_handler(
    analyzer: PhonoMatch,
    model_gate: ModelLifecycleGate,
    *,
    enable_model_lifecycle: bool,
) -> type[BaseHTTPRequestHandler]:
    """Create a handler class bound to one analyzer and lifecycle gate."""

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
                with model_gate.lifecycle():
                    analyzer.load_model()
            except ServerBusyError as exc:
                self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            except PhonoMatchError as exc:
                LOGGER.warning("model load failed: %s", exc)
                self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            else:
                self._send_json(HTTPStatus.OK, {"status": "loaded"})

        def _unload_model(self) -> None:
            try:
                with model_gate.lifecycle():
                    analyzer.unload_model()
            except ServerBusyError as exc:
                self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            except PhonoMatchError as exc:
                LOGGER.warning("model unload failed: %s", exc)
                self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            else:
                self._send_json(HTTPStatus.OK, {"status": "unloaded"})

        @contextmanager
        def _temporary_audio(self) -> Generator[Path, None, None]:
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
            with model_gate.recognition():
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

    return RecognitionHandler
