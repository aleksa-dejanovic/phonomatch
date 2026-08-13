# Sound Analyzer

Sound Analyzer records a short utterance, recognizes its phones with Meta's
Wav2Vec2Phoneme model, and compares the resulting IPA transcription with a
configurable vocabulary using PanPhon's weighted feature edit distance.

The application runs locally. It restricts CTC decoding to the phones in the
configured vocabulary and rejects results that are too distant, insufficiently
separated from the runner-up, or not confident enough.

## Requirements

- Python 3.9 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A microphone and PortAudio for live recording
- Approximately 1.3 GB of disk space for the default model

Install PortAudio on Debian or Ubuntu:

```console
sudo apt install libportaudio2
```

On macOS with Homebrew:

```console
brew install portaudio
```

## Install and run

From the repository root:

```console
uv sync
uv run sound-analyzer
```

Use `uv sync --locked` in production and CI so dependency resolution cannot
change after review. Linux and Windows installations use the CPU-only PyTorch
index; this avoids bundling unused CUDA libraries in the recognition service.

The first run downloads `facebook/wav2vec2-lv-60-espeak-cv-ft` at the pinned
revision `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`. Later runs reuse the local
Hugging Face cache, so model contents and licensing cannot drift silently.

The loader explicitly uses the model's official PyTorch weights and disables
Transformers' automatic download of an unmerged SafeTensors conversion. A clean
model cache therefore occupies approximately 1.2 GiB instead of 2.4 GiB.

The console announces when recording ends and recognition starts. The final
report includes the total recognition and matching time.

Useful CLI options:

```console
uv run sound-analyzer --seconds 3
uv run sound-analyzer --max-distance 0.08
uv run sound-analyzer --color never
uv run sound-analyzer --help
```

Exit codes are `0` for an accepted match, `2` for an ambiguous or unknown
utterance, and `1` for an operational error.

## Python API

```python
from sound_analyzer import MatchSettings, SoundAnalyzer

words = {
    "grun": "ɡrun",
    "shaki": "ʃaki",
}

analyzer = SoundAnalyzer(
    words,
    MatchSettings(max_distance_ratio=0.10),
)

# Record, recognize, and match.
result = analyzer.listen(seconds=2)

# Recognize an existing 16-bit PCM WAV file and match it.
result = analyzer.analyze_file("recording.wav")

# Match an existing IPA transcription without loading the speech model.
result = analyzer.match_ipa("gɾun")

print(result.recognized_ipa)
print(result.match.decision)
print(result.match.best_candidate.word)
```

You can use a compatible local or Hugging Face model identifier:

```python
analyzer = SoundAnalyzer(
    words,
    model_id="/path/to/local/model",
    model_revision=None,
)
```

The model must be compatible with `AutoModelForCTC` and its tokenizer must use
phone tokens matching the vocabulary's normalized IPA phones.

Expected operational exceptions inherit from `SoundAnalyzerError`:

- `AudioRecordingError`
- `RecognitionError`
- `UnsupportedPhoneError`

## Matching

The default acceptance rules are:

| Check | Default | Meaning |
| --- | ---: | --- |
| Maximum distance | 10% | Best match may consume at most 10% of its theoretical maximum edit cost |
| Relative separation | 20% | Best candidate must be sufficiently favored over the runner-up given their mutual distance |
| Confidence | 80% | Best candidate must dominate the complete vocabulary |

All checks must pass. Confidence is relative to the configured vocabulary; it
does not replace the absolute distance check.

The vocabulary is `DEFAULT_WORDS` in `src/sound_analyzer/config.py`. The decoder
inventory is derived automatically from it. Unsupported model phones produce a
clear error instead of silently degrading recognition.

## Development

```console
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -m unittest discover -s tests -v
uv build
```

Apply automatic formatting and safe lint fixes with:

```console
uv run ruff format .
uv run ruff check . --fix
```

Unit tests do not download the Wav2Vec2 model. A real-model smoke test requires
network access on its first run and can be performed with the installed CLI.

## Licensing

This branch does not depend on GPL-licensed Allosaurus. The default Meta model
is published under Apache License 2.0, as are Hugging Face Transformers. PyTorch
uses a BSD-style license. Exact runtime licenses are collected under
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES), with the generated inventory in
[`DEPENDENCY_LICENSE_REPORT.md`](DEPENDENCY_LICENSE_REPORT.md). Follow the
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) and verify all distribution
obligations with qualified counsel before shipping a commercial product.
