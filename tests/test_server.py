import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from phonomatch.application.analyzer import PhonoMatch
from phonomatch.domain.models import (
    AnalysisResult,
    CandidateScore,
    MatchResult,
)
from phonomatch.exceptions import RecognitionError
from phonomatch.interfaces.server import (
    MAX_AUDIO_BYTES,
    create_server,
    result_payload,
    serve,
)


class ServerTests(unittest.TestCase):
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
        handler._read_audio = Mock(return_value=b"wav")
        handler._analyze = Mock(return_value=result)

        handler.do_POST()

        handler._analyze.assert_called_once_with(b"wav", phrase=True)
        handler._send_json.assert_called_once_with(200, result_payload(result))

    def test_recognition_maps_bad_audio_and_recognition_errors(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        handler.path = "/v1/recognize"
        handler._read_audio = Mock(side_effect=ValueError("bad audio"))

        handler.do_POST()
        handler._send_error.assert_called_once_with(400, "bad audio")

        handler._send_error.reset_mock()
        handler._read_audio = Mock(return_value=b"wav")
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

    def test_read_audio_validates_headers_and_body(self) -> None:
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
                with self.assertRaisesRegex(ValueError, message):
                    handler._read_audio()

    def test_read_audio_accepts_wav_content_type_parameters(self) -> None:
        handler = self._handler_for(SimpleNamespace())
        handler.headers = {
            "Content-Type": "audio/wav; charset=binary",
            "Content-Length": "3",
        }
        handler.rfile = io.BytesIO(b"wav")

        self.assertEqual(handler._read_audio(), b"wav")

    def test_analyze_writes_then_removes_the_temporary_wav(self) -> None:
        observed_paths: list[Path] = []

        def analyze(path: Path) -> AnalysisResult:
            observed_paths.append(path)
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), b"wav")
            return self._analysis_result()

        handler = self._handler_for(SimpleNamespace(analyze_file=analyze))
        result = handler._analyze(b"wav", phrase=False)

        self.assertEqual(result, self._analysis_result())
        self.assertFalse(observed_paths[0].exists())

    def test_analyze_routes_phrase_requests(self) -> None:
        analyze_phrase = Mock(return_value=self._analysis_result())
        handler = self._handler_for(SimpleNamespace(analyze_phrase_file=analyze_phrase))

        result = handler._analyze(b"phrase", phrase=True)

        self.assertEqual(result, self._analysis_result())
        analyzed_path = analyze_phrase.call_args.args[0]
        self.assertFalse(analyzed_path.exists())

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
        captured: dict[str, Any] = {}

        class CapturingServer:
            def __init__(self, _address: object, handler: type[object]) -> None:
                captured["handler"] = handler

        with patch("phonomatch.interfaces.server.ThreadingHTTPServer", CapturingServer):
            create_server(
                "127.0.0.1",
                0,
                cast(PhonoMatch, analyzer),
                enable_model_lifecycle=enable_model_lifecycle,
            )
        handler = object.__new__(captured["handler"])
        handler._send_error = Mock()
        handler._send_json = Mock()
        return handler
