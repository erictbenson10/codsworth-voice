# Codsworth expanded adapter · step 300

This is a trained VoxCPM2 LoRA adapter from the September 7, 2026 experiment:
431 training clips, 41.59 minutes,
and 300 completed optimizer updates. Effective batch size was four. It
adapts the language model and diffusion transformer with rank-32, alpha-32 LoRA;
the AudioVAE and non-LoRA model parameters were frozen.

The weights are distributed as a separate release asset. The Git directory
contains the portable configuration, checksums, and provenance. No game audio,
dialogue transcripts, frozen base model, or optimizer state is included.

## Load

Use `load_adapter.py` from the parent voice directory after extracting the model
release asset. It downloads the exact public base revision and requires your own
reference recording. See [the voice guide](../../README.md) for the command.
The exported weights are byte-identical to the archived checkpoint; only the
configuration's private base path was replaced with the public model ID.

## Provenance and limits

Base: `openbmb/VoxCPM2@32279effe8c19989596f05d353d1447f51d9e915`. Upstream code revision:
`f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`. All 300 optimizer updates completed. This records an
actual adaptation run, but does not claim convergence or better voice quality.
Use matched prompts, reference audio, and seeds to compare the base, pilot, and
expanded adapters. The two experiments have different development sets.

The [base model card](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/README.md)
declares Apache-2.0. No separate adapter license has been declared in this
project. This documentation does not relicense the game dialogue, character,
or training corpus. The training dataset's Hugging Face release is pending.

Verify the extracted files with `shasum -a 256 -c SHA256SUMS`.
