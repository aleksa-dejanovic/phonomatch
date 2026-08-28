"""JSON representations of server responses."""

from dataclasses import asdict
from typing import Any

from ...domain.models import AnalysisResult, PhraseAnalysisResult


def result_payload(result: AnalysisResult | PhraseAnalysisResult) -> dict[str, Any]:
    """Convert a public result model into the JSON representation of the API."""
    payload = asdict(result)
    payload["accepted"] = (
        result.match.accepted if isinstance(result, AnalysisResult) else result.accepted
    )
    return payload
