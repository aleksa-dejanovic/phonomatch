import io
import tempfile
import unittest
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from phonomatch.application.phonomatch import PhonoMatch
from phonomatch.domain.models import (
    AnalysisResult,
    CandidateScore,
    MatchResult,
)
from phonomatch.exceptions import RecognitionError
from phonomatch.interfaces.server import (
    MAX_AUDIO_BYTES,
    BoundedThreadingHTTPServer,
    ModelLifecycleGate,
    ServerBusyError,
    create_server,
    result_payload,
    serve,
)


class ServerTests(unittest.TestCase):
    def test_model_gate_allows_recognition_requests_to_overlap(self) -> None:
        gate = ModelLifecycleGate()
        both_running = Event()
        release = Event()
        running = 0

        def recognize() -> None:
            nonlocal running
            with gate.recognition():
                running += 1
                if running == 2:
                    both_running.set()
                release.wait(timeout=1)
                running -= 1

        workers = [Thread(target=recognize), Thread(target=recognize)]
        for worker in workers:
            worker.start()
        self.assertTrue(both_running.wait(timeout=1))
        release.set()
        for worker in workers:
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())

    def test_model_gate_drains_existing_work_and_rejects_new_recognition(self) -> None:
        gate = ModelLifecycleGate()
        recognition_started = Event()
        release_recognition = Event()
        lifecycle_started = Event()
        release_lifecycle = Event()

        def recognize() -> None:
            with gate.recognition():
                recognition_started.set()
                release_recognition.wait(timeout=1)

        def unload() -> None:
            with gate.lifecycle():
                lifecycle_started.set()
                release_lifecycle.wait(timeout=1)

        recognition = Thread(target=recognize)
        recognition.start()
        self.assertTrue(recognition_started.wait(timeout=1))
        lifecycle = Thread(target=unload)
        lifecycle.start()

        with self.assertRaises(ServerBusyError), gate.recognition():
            pass
        self.assertFalse(lifecycle_started.is_set())

        release_recognition.set()
        self.assertTrue(lifecycle_started.wait(timeout=1))
        release_lifecycle.set()
        for worker in (recognition, lifecycle):
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())

    def test_model_gate_rejects_a_second_lifecycle_operation(self) -> None:
        gate = ModelLifecycleGate()

        def second_lifecycle() -> None:
            with gate.lifecycle():
                pass

        with gate.lifecycle(), self.assertRaises(ServerBusyError):
            second_lifecycle()

    def test_result_payload_is_json_compatible_and_includes_acceptance(self) -> None:
        result = AnalysisResult(
            recognized_ipa="ɡrun",
            match=MatchResult(
                decision="likely_match",
                best_candidate=CandidateScore("grun", "ɡrun", 0, 0, 1),
                second_candidate=None,
                relative_margin=1,
            ),
            recognition_seconds=0.1,
        )

        self.assertEqual(
            result_payload(result),
            {
                "recognized_ipa": "ɡrun",
                "match": {
                    "decision": "likely_match",
                    "best_candidate": {
                        "word": "grun",
                        "ipa": "ɡrun",
                        "raw_distance": 0,
                        "distance_ratio": 0,
                        "confidence": 1,
                    },
                    "second_candidate": None,
                    "relative_margin": 1,
                },
                "recognition_seconds": 0.1,
                "accepted": True,
            },
        )

    def test_lifecycle_endpoints_are_only_available_when_enabled(self) -> None:
        analyzer = SimpleNamespace(load_model=Mock(), unload_model=Mock())
        handler = self._handler_for(analyzer)
        handler.path = "/v1/model/unload"
        handler.do_POST()

        handler._send_error.assert_called_once()
        self.assertEqual(handler._send_error.call_args.args[0], 404)
        analyzer.unload_model.assert_not_called()

    def test_health_endpoint_reports_status_and_rejects_unknown_paths(self) -> None:
        handler = self._handler_for(SimpleNamespace())

        handler.path = "/health?ready=true"
        handler.do_GET()
        handler._send_json.assert_called_once_with(200, {"status": "ok"})

        handler._send_json.reset_mock()
        handler.path = "/missing"
        handler.do_GET()
        handler._send_error.assert_called_once_with(404, "endpoint not found")

    def test_enabled_lifecycle_endpoints_load_and_unload_the_model(self) -> None:
        analyzer = SimpleNamespace(load_model=Mock(), unload_model=Mock())
        handler = self._handler_for(analyzer, enable_model_lifecycle=True)

        for endpoint, method, expected in (
            ("/v1/model/load", analyzer.load_model, {"status": "loaded"}),
            ("/v1/model/unload", analyzer.unload_model, {"status": "unloaded"}),
        ):
            handler.path = endpoint
            handler._send_json.reset_mock()
            handler.do_POST()
            handler._send_json.assert_called_once_with(200, expected)
            method.assert_called_once_with()

    def test_recognition_routes_audio_and_returns_result_payload(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        result = self._analysis_result()
        handler.path = "/v1/recognize/phrase"

        @contextmanager
        def temporary_audio() -> Generator[Path, None, None]:
            yield Path("upload.wav")

        handler._temporary_audio = temporary_audio
        handler._analyze = Mock(return_value=result)

        handler.do_POST()

        handler._analyze.assert_called_once_with(Path("upload.wav"), phrase=True)
        handler._send_json.assert_called_once_with(200, result_payload(result))

    def test_recognition_maps_bad_audio_and_recognition_errors(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        handler.path = "/v1/recognize"
        handler._temporary_audio = Mock(side_effect=ValueError("bad audio"))

        handler.do_POST()
        handler._send_error.assert_called_once_with(400, "bad audio")

        handler._send_error.reset_mock()

        @contextmanager
        def temporary_audio() -> Generator[Path, None, None]:
            yield Path("upload.wav")

        handler._temporary_audio = temporary_audio
        handler._analyze = Mock(side_effect=RecognitionError("model failed"))
        handler.do_POST()
        handler._send_error.assert_called_once_with(422, "model failed")

    def test_recognition_rejects_unknown_endpoint(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        handler.path = "/v1/unknown"

        handler.do_POST()

        handler._send_error.assert_called_once_with(404, "endpoint not found")

    def test_load_endpoint_maps_model_errors(self) -> None:
        analyzer = SimpleNamespace(
            load_model=Mock(side_effect=RecognitionError("offline"))
        )
        handler = self._handler_for(analyzer, enable_model_lifecycle=True)
        handler.path = "/v1/model/load"

        handler.do_POST()

        handler._send_error.assert_called_once_with(422, "offline")

    def test_temporary_audio_validates_headers_and_body(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        cases: tuple[tuple[dict[str, str], bytes, str], ...] = (
            ({}, b"", "Content-Type"),
            ({"Content-Type": "audio/wav"}, b"", "Content-Length is required"),
            (
                {"Content-Type": "audio/wav", "Content-Length": "wat"},
                b"",
                "Content-Length must be an integer",
            ),
            (
                {"Content-Type": "audio/x-wav", "Content-Length": "0"},
                b"",
                "audio must be between",
            ),
            (
                {
                    "Content-Type": "audio/wav",
                    "Content-Length": str(MAX_AUDIO_BYTES + 1),
                },
                b"",
                "audio must be between",
            ),
            (
                {"Content-Type": "audio/wav", "Content-Length": "4"},
                b"abc",
                "request body ended",
            ),
        )
        for headers, body, message in cases:
            with self.subTest(headers=headers):
                handler.headers = headers
                handler.rfile = io.BytesIO(body)
                with (
                    self.assertRaisesRegex(ValueError, message),
                    handler._temporary_audio(),
                ):
                    pass

    def test_temporary_audio_streams_wav_and_removes_it(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        handler.headers = {
            "Content-Type": "audio/wav; charset=binary",
            "Content-Length": "3",
        }
        handler.rfile = io.BytesIO(b"wav")

        with handler._temporary_audio() as path:
            self.assertEqual(path.read_bytes(), b"wav")
            self.assertTrue(path.exists())
        self.assertFalse(path.exists())

    def test_recognition_returns_503_when_upload_capacity_is_exhausted(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        handler.server.max_inflight_upload_bytes = 2
        handler.path = "/v1/recognize"
        handler.headers = {"Content-Type": "audio/wav", "Content-Length": "3"}
        handler.rfile = io.BytesIO(b"wav")

        handler.do_POST()

        handler._send_error.assert_called_once_with(503, unittest.mock.ANY)

    def test_recognition_returns_408_when_upload_times_out(self) -> None:
        class TimedOutStream:
            def read(self, _size: int) -> bytes:
                raise TimeoutError()

        handler = self._handler_for(SimpleNamespace())
        handler.path = "/v1/recognize"
        handler.headers = {"Content-Type": "audio/wav", "Content-Length": "3"}
        handler.rfile = TimedOutStream()

        handler.do_POST()

        handler._send_error.assert_called_once_with(408, "request body timed out")

    def test_server_rejects_connections_when_all_worker_slots_are_busy(self) -> None:
        server = self._server_for(SimpleNamespace())
        request = Mock()
        try:
            for _ in range(server.max_concurrent_requests):
                self.assertTrue(server._request_slots.acquire(blocking=False))

            with patch.object(server, "shutdown_request") as shutdown_request:
                server.process_request(request, ("127.0.0.1", 1))

            request.sendall.assert_called_once()
            shutdown_request.assert_called_once_with(request)
        finally:
            for _ in range(server.max_concurrent_requests):
                server._request_slots.release()
            server.server_close()

    def test_analyze_uses_the_temporary_wav(self) -> None:
        observed_paths: list[Path] = []

        def analyze(path: Path) -> AnalysisResult:
            observed_paths.append(path)
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), b"wav")
            return self._analysis_result()

        handler = self._handler_for(SimpleNamespace(analyze_file=analyze))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            temporary.write(b"wav")
            path = Path(temporary.name)
        result = handler._analyze(path, phrase=False)

        self.assertEqual(result, self._analysis_result())
        self.assertTrue(observed_paths[0].exists())
        path.unlink()

    def test_analyze_routes_phrase_requests(self) -> None:
        analyze_phrase = Mock(return_value=self._analysis_result())
        handler = self._handler_for(SimpleNamespace(analyze_phrase_file=analyze_phrase))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            temporary.write(b"phrase")
            path = Path(temporary.name)
        result = handler._analyze(path, phrase=True)

        self.assertEqual(result, self._analysis_result())
        analyzed_path = analyze_phrase.call_args.args[0]
        self.assertEqual(analyzed_path, path)
        path.unlink()

    def test_send_json_writes_a_json_response(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        handler._send_json = type(handler)._send_json.__get__(handler, type(handler))
        handler.wfile = io.BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()

        handler._send_json(200, {"ipa": "ɡrun"})

        self.assertEqual(handler.wfile.getvalue(), '{"ipa": "ɡrun"}'.encode())
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_any_call(
            "Content-Type", "application/json; charset=utf-8"
        )
        handler.end_headers.assert_called_once_with()

    def test_send_error_serializes_the_message_and_suppresses_request_logs(
        self,
    ) -> None:
        handler = self._handler_for(SimpleNamespace())
        handler._send_error = type(handler)._send_error.__get__(handler, type(handler))
        handler._send_json = Mock()

        self.assertIsNone(handler.log_message("%s", "quiet"))
        handler._send_error(400, "invalid input")

        handler._send_json.assert_called_once_with(400, {"error": "invalid input"})

    def test_serve_creates_a_lifecycle_enabled_server(self) -> None:
        server = Mock()
        server.__enter__ = Mock(return_value=server)
        server.__exit__ = Mock(return_value=False)
        analyzer = cast(PhonoMatch, SimpleNamespace())
        with patch(
            "phonomatch.interfaces.server.create_server", return_value=server
        ) as create:
            serve("127.0.0.1", 9999, analyzer, enable_model_lifecycle=True)

        create.assert_called_once_with(
            "127.0.0.1", 9999, analyzer, enable_model_lifecycle=True
        )
        server.serve_forever.assert_called_once_with()

    @staticmethod
    def _analysis_result() -> AnalysisResult:
        return AnalysisResult(
            recognized_ipa="ɡrun",
            match=MatchResult(
                decision="likely_match",
                best_candidate=CandidateScore("grun", "ɡrun", 0, 0, 1),
                second_candidate=None,
                relative_margin=1,
            ),
            recognition_seconds=0.1,
        )

    def _handler_for(
        self, analyzer: SimpleNamespace, *, enable_model_lifecycle: bool = False
    ) -> Any:
        server = self._server_for(
            analyzer, enable_model_lifecycle=enable_model_lifecycle
        )
        handler = cast(
            Any,
            object.__new__(cast(type[object], server.RequestHandlerClass)),
        )
        handler.server = server
        server.server_close()
        handler._send_error = Mock()
        handler._send_json = Mock()
        return handler

    @staticmethod
    def _server_for(
        analyzer: SimpleNamespace, *, enable_model_lifecycle: bool = False
    ) -> BoundedThreadingHTTPServer:
        return create_server(
            "127.0.0.1",
            0,
            cast(PhonoMatch, analyzer),
            enable_model_lifecycle=enable_model_lifecycle,
        )
