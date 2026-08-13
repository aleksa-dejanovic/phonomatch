# Sound Analyzer

Sound Analyzer records speech, transcribes it to IPA with Allosaurus, and
matches the transcription against a configurable vocabulary using PanPhon.

## Run with uv

```console
uv sync
uv run sound-analyzer
```

The console reports when recording stops and recognition begins. The final
report includes the elapsed recognition and matching time.

## Development

```console
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -m unittest discover -s tests -v
```
