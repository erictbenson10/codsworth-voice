# Attribution and licensing status

| Component | Source | Recorded status |
| --- | --- | --- |
| VoxCPM2 base model | `openbmb/VoxCPM2`, revision `32279effe8c19989596f05d353d1447f51d9e915` | Model card declares Apache-2.0 |
| VoxCPM implementation | `OpenBMB/VoxCPM`, revision `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69` | Loaded separately from upstream; upstream notices apply |
| Codsworth LoRA weights | This project's September 7, 2026 runs | No separate adapter license declared |
| Source recordings and dialogue | `AquaV/fallout-4-voices`; Fallout 4 / Far Harbor game assets | Original rights retained; not included in this release |
| Project code and documentation | Eric Benson's Codsworth project | No additional license grant declared here |

The base model license is recorded at the
[exact pinned model card](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/README.md).
It is not represented as a license for the training recordings or character.
This release is not presented as an official Bethesda or Fallout product.

Audio and transcripts are excluded from Git. Hugging Face publication is a later,
separate task; [dataset.json](dataset.json) is the authoritative publication-status
pointer. Its repository and download URLs remain null until a dataset is actually
published.
