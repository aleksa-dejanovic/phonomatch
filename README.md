# Sound Analyzer

Sound Analyzer records a short utterance, uses Allosaurus to transcribe it to IPA,
and compares it with a small configurable vocabulary using PanPhon's weighted
feature edit distance. Weak and ambiguous matches are rejected.

Distances are percentages of PanPhon's theoretical maximum edit cost for the
two phone sequences. A match must be no more than 5% of that maximum, distinct
from the runner-up, and sufficiently confident.

## Requirements

- Python 3.9 or newer
- A working microphone and PortAudio installation
- [`uv`](https://docs.astral.sh/uv/) for the commands below

## Run

```console
uv sync
uv run sound-analyzer
```

Use `--seconds 3` to change the recording duration. Exit status `0` indicates an
accepted match, `2` an unknown or ambiguous utterance, and `1` an audio or model
error. The first run may take longer while Allosaurus prepares its model.

The vocabulary is defined by `DEFAULT_WORDS` in
`src/sound_analyzer/analyzer.py`. Each key is the displayed word and each value
is its expected IPA transcription.

## Test

The matching tests do not require a microphone or model download:

```console
uv run python -m unittest discover -s tests -v
```
