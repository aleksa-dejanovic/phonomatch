# Release checklist

Run this checklist separately on every operating system and architecture that
will be shipped. Python wheels and their bundled native libraries vary by
platform.

1. Create the exact environment and refuse lock-file changes:

   ```console
   uv sync --locked --group dev
   ```

2. Run every automated check, including regeneration and comparison of the
   runtime license inventory and bundled texts in a temporary directory:

   ```console
   ./scripts/check.sh --release
   ```

3. Confirm the default model ID and immutable revision in
   `sound_analyzer.infrastructure.recognition`. If deploying a local model
   directory, record the model files' origin, revision, checksums, and license
   separately.

4. Confirm the check script completed successfully. It validates formatting,
   lint, strict typing, unit tests, lockfile consistency, package builds, patch
   whitespace, and generated licensing artifacts.

5. Test recognition on the target machine with networking disabled after the
   model has been provisioned. This verifies that the pinned files are present
   and that production does not depend on a mutable remote revision.

6. Ship `THIRD_PARTY_NOTICES.md`, `DEPENDENCY_LICENSE_REPORT.md`, and the entire
   `THIRD_PARTY_LICENSES/` directory beside the service. Preserve the model's
   Apache-2.0 license with redistributed model weights.

7. Review the generated report and texts for the exact artifact with qualified
   counsel. The audit script detects missing texts and declared GPL/AGPL
   packages, but it is not a legal opinion and does not replace review of weak
   copyleft, native-library, trademark, patent, or model-data obligations.
