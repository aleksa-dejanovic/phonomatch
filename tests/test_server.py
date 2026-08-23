import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from phonomatch.application.analyzer import PhonoMatch
from phonomatch.domain.models import (
    AnalysisResult,
    CandidateScore,
    MatchResult,
)
from phonomatch.interfaces.server import create_server, result_payload


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
