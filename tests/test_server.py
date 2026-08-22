import unittest

from phonomatch.domain.models import (
    AnalysisResult,
    CandidateScore,
    MatchResult,
)
from phonomatch.interfaces.server import result_payload


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
