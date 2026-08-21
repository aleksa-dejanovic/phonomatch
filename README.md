# PhoneMatch

PhoneMatch records a short utterance, recognizes its phones with Meta's
Wav2Vec2Phoneme model, and compares the resulting IPA transcription with a
configurable vocabulary using PanPhon's weighted feature edit distance.

It can recognize either one word or a phrase containing an arbitrary sequence
of configured words. Phrase mode uses lexicon-constrained CTC beam search to
infer word boundaries without requiring pauses, then aligns and evaluates each
word independently.

The application runs locally. It restricts CTC decoding to the phones in the
configured vocabulary and rejects results that are too distant, insufficiently
separated from the runner-up, or not confident enough.

## Requirements

- Python 3.10 or newer
- A microphone and PortAudio for live recording
- Approximately 1.3 GB of disk space for the default model

## Platform support

PhoneMatch supports Linux, macOS, and Windows. Microphone access must be
permitted for the terminal or Python environment that runs the command.

On Linux, install PortAudio with your system package manager before installing
PhoneMatch. For Debian or Ubuntu:

```console
sudo apt install libportaudio2
```

When installed with `pip`, the `sounddevice` dependency supplies PortAudio on
macOS and Windows. If macOS cannot find the library, install it with Homebrew:

```console
brew install portaudio
```

If Windows cannot detect an input device, first check its microphone privacy
settings and confirm that the selected input device works in another
application.

## Install

Install from PyPI:

```console
# Linux and Windows: prefer PyTorch's CPU-only wheels. This avoids installing
# the NVIDIA CUDA runtime packages pulled in by the default PyPI Linux wheels.
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu phonematch
```

On macOS, `python -m pip install phonematch` installs the platform-native
PyTorch wheel; it does not include NVIDIA CUDA libraries. The application uses
CPU inference, so a GPU build is not needed on any platform.

Then confirm the command is available:

```console
phonematch --help
```

For a development checkout, use [`uv`](https://docs.astral.sh/uv/):

```console
uv sync
uv run phonematch
```

Use `uv sync --locked` in production and CI so dependency resolution cannot
change after review.

The first run downloads `facebook/wav2vec2-lv-60-espeak-cv-ft` at the pinned
revision `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`. Later runs reuse the local
Hugging Face cache, so model contents and licensing cannot drift silently.

The loader explicitly uses the model's official PyTorch weights and disables
Transformers' automatic download of an unmerged SafeTensors conversion. A clean
model cache therefore occupies approximately 1.2 GiB instead of 2.4 GiB.

The console loads the model before recording begins, announces when recording
ends and recognition starts, and reports the recognition and matching time.
Keeping the loaded analyzer instance alive also keeps the model resident for
subsequent requests.

## Privacy and network access

PhoneMatch records audio from the selected microphone and processes it locally.
It does not upload recorded audio to a PhoneMatch service or any other remote
service. The first use of the default model requires network access to download
the pinned model files from Hugging Face; later uses read them from the local
cache. You can instead configure a compatible local model path to avoid that
download.

Useful CLI options:

```console
phonematch --seconds 3
phonematch --phrase --seconds 5
phonematch --phrase --beam-size 96 --max-words 8
phonematch --max-distance 0.08
phonematch --color never
phonematch --help
```

Exit codes are `0` for an accepted match, `2` for an ambiguous or unknown
utterance, and `1` for an operational error.

## Python API

```python
from phonematch import MatchSettings, PhoneMatch

words = {
    "grun": "ɡrun",
    "shaki": "ʃaki",
}

analyzer = PhoneMatch(
    words,
    MatchSettings(max_distance_ratio=0.10),
)

# Record, recognize, and match.
result = analyzer.listen(seconds=2)

# Recognize an existing 16-bit PCM WAV file and match it.
result = analyzer.analyze_file("recording.wav")

# Decode a sequence of vocabulary words and match each word independently.
phrase = analyzer.analyze_phrase_file(
    "phrase.wav",
    beam_size=64,
    max_words=12,
)

for word in phrase.words:
    print(
        word.match.best_candidate.word,
        word.start_seconds,
        word.end_seconds,
        word.match.accepted,
    )

# Record and recognize a phrase. The default recording duration is four seconds.
phrase = analyzer.listen_for_phrase(seconds=4)

# Match an existing IPA transcription without loading the speech model.
result = analyzer.match_ipa("gɾun")

print(result.recognized_ipa)
print(result.match.decision)
print(result.match.best_candidate.word)
```

The original `listen`, `analyze_file`, `match_ipa`, and `listen_and_match`
single-word APIs are unchanged. Phrase acceptance requires both an unambiguous
word sequence and successful independent matching of every aligned word.

### Phrase decoding

Phrase mode operates directly on the model's frame-level CTC probabilities.
Every configured IPA pronunciation is converted to the model's phone tokens
and inserted into a trie. Beam search can finish a word at a terminal trie node
and immediately begin any vocabulary word, so boundaries do not depend on
silence in the recording.

The winning token sequence is Viterbi-aligned back to the model frames. Each
word receives an approximate start and end time, a recognized IPA segment, and
the same distance, relative-separation, and confidence checks used by
single-word recognition. The phrase is rejected when its runner-up sequence is
too competitive or any individual word fails.

`--beam-size` trades decoding speed and memory for a wider hypothesis search.
`--max-words` bounds phrase length and prevents unreasonably long segmentations.
These options only affect phrase mode.

You can use a compatible local or Hugging Face model identifier:

```python
analyzer = PhoneMatch(
    words,
    model_id="/path/to/local/model",
    model_revision=None,
)
```

The model must be compatible with `AutoModelForCTC` and its tokenizer must use
phone tokens matching the vocabulary's normalized IPA phones.

Expected operational exceptions inherit from `PhoneMatchError`:

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

The vocabulary is `DEFAULT_WORDS` in `src/phonematch/domain/config.py`.
The decoder inventory is derived automatically from it. Unsupported model
phones produce a clear error instead of silently degrading recognition.

## Development

```console
uv sync --group dev
./scripts/check.sh
```

The script verifies patch whitespace, lockfile consistency, formatting, lint,
strict typing, unit tests, and both distribution formats. Build artifacts are
created in a temporary directory and removed automatically.

Apply automatic formatting and safe lint fixes with:

```console
uv run ruff format .
uv run ruff check . --fix
```

Unit tests do not download the Wav2Vec2 model. A real-model smoke test requires
network access on its first run and can be performed with the installed CLI.
