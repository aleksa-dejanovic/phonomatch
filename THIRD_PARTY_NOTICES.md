# Third-party notices

This project uses third-party software and model weights. This file is a
convenience summary, not a substitute for the license texts shipped by those
projects or for legal advice.

## Default phoneme-recognition model

- Name: `facebook/wav2vec2-lv-60-espeak-cv-ft`
- Revision: `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`
- Publisher: Meta / Facebook
- License declared by the model publisher: Apache License 2.0
- Source: <https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft>

## Major runtime components

- Hugging Face Transformers — Apache License 2.0
- PyTorch — BSD-style license
- SciPy — BSD 3-Clause license
- NumPy — BSD 3-Clause license
- sounddevice — MIT license
- PanPhon — MIT license
- tqdm — MPL-2.0 and MIT licenses
- certifi — MPL-2.0 license

When redistributing dependencies or model files, include the license and notice
materials required by their respective versions. NumPy and SciPy wheels also
carry notices for bundled native libraries, including runtime-exception and
LGPL components; retain their complete bundled texts. The exact audited
inventory is in `DEPENDENCY_LICENSE_REPORT.md`, and full texts are under
`THIRD_PARTY_LICENSES/`.
