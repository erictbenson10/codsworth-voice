# codsworth-expanded-v1: split metadata

This is the exact split inventory used for the 300-update expanded adapter.
The audio and dialogue transcripts are **not included**. The future dataset
location is recorded in [dataset.json](../../dataset.json), which currently has
no published Hugging Face URL.

The train split contains 431 clips / 2495.15542
seconds. The development split contains 6 clips /
23.08075 seconds. Paths are relative to the eventual
dataset root. `*.metadata.jsonl` contains stable recording IDs, split assignments,
group IDs, duration, frame count, and file/PCM SHA-256 checksums. These inventory
files are not runnable trainer manifests because they omit text.

Assignments are taken from the actual trainer inputs, not from the historical
source-pool label in the private provenance file. No audio, original split, or
holdout was modified for this export. The training and development IDs and
leakage groups are disjoint. Historical RVC models had holdout contamination;
those old evaluation labels do not establish an untouched test set.

`summary.json` records historical verification and selection limitations.
Speaker similarity and contextual review support curation; neither certifies
the identity of every recording. Game assets retain their original ownership.
