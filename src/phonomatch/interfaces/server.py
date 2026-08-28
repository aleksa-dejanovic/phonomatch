"""A small local HTTP API for recognizing uploaded WAV audio."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlparse

from ..application.phonomatch import PhonoMatch
from ..domain.models import AnalysisResult, PhraseAnalysisResult
from ..exceptions import PhonoMatchError

MAX_AUDIO_BYTES = 32 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


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
) -> ThreadingHTTPServer:
    """Create a server without starting it, primarily for embedding and tests."""

    # Recognition requests and lifecycle operations share a process-wide model
    # cache.  Do not clear it while an ONNX session is serving a request.
    model_lock = RLock()

    class RecognitionHandler(BaseHTTPRequestHandler):
        server_version = "PhonoMatch/0.1"

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
                audio = self._read_audio()
                result = self._analyze(audio, phrase=path.endswith("/phrase"))
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

        def _read_audio(self) -> bytes:
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
            audio = self.rfile.read(size)
            if len(audio) != size:
                raise ValueError("request body ended before Content-Length")
            return audio

        def _analyze(
            self, audio: bytes, *, phrase: bool
        ) -> AnalysisResult | PhraseAnalysisResult:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                temporary.write(audio)
                path = Path(temporary.name)
            try:
                with model_lock:
                    if phrase:
                        return analyzer.analyze_phrase_file(path)
                    return analyzer.analyze_file(path)
            finally:
                path.unlink(missing_ok=True)

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

    return ThreadingHTTPServer((host, port), RecognitionHandler)


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
