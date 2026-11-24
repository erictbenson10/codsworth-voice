# Portable VoxCPM2 LoRA training recipe

`pilot.json` and `expanded.json` preserve the historical hyperparameters while
replacing machine-specific paths with paths relative to the voice repository.
No training is started during release preparation.

## Inputs

1. Clone `https://github.com/OpenBMB/VoxCPM.git` outside this repository and check
   out revision `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`.
2. Install that checkout in an isolated environment and then install
   `training/requirements.txt`. The original runtime used Torch 2.7.1+cu128.
3. Download `openbmb/VoxCPM2` revision
   `32279effe8c19989596f05d353d1447f51d9e915` into
   `base-model/32279effe8c19989596f05d353d1447f51d9e915/`, or set
   `pretrained_path` to the corresponding cached snapshot directory.
4. Supply a local, authorized dataset and full trainer manifests under
   `data/codsworth-expanded-v1/` or `data/codsworth-pilot-v1/`. The future dataset
   location is [dataset.json](../dataset.json); it is not published yet.

Each trainer JSONL row must contain `audio` (a path resolvable from the working
directory), `text`, `duration` (seconds), and `dataset_id` (0 for this recipe).
Keep development audio out of training. The published `*.metadata.jsonl` files
are inventories, not substitutes for the full text manifests.

From the repository root, select a GPU explicitly and use a fresh output path:

```sh
export CUDA_VISIBLE_DEVICES=GPU-YOUR-SELECTED-UUID
python training/run.py \
  --source /path/to/pinned/VoxCPM \
  --config training/expanded.json \
  --expected-gpu-uuid "$CUDA_VISIBLE_DEVICES"
```

The wrapper checks the upstream revision and base snapshot name, verifies the
selected Torch GPU UUID, refuses implicit resume into an existing `latest`
checkpoint, freezes the AudioVAE, rejects non-LoRA trainable parameters, and
blocks nonfinite losses/gradients before optimizer updates. It records actual
optimizer progress. Validation measures loss only; separate paired auditions
are required to assess voice quality.

The expanded recipe uses 431 clips, effective batch four, and 300 updates. The
pilot uses 61 clips and 100 updates. Both use rank-32/alpha-32 language-model and
diffusion-transformer LoRA. Neither the base weights nor AudioVAE is trained.

This portable wrapper intentionally contains no service-stop commands, remote
hosts, or shared-GPU borrowing logic. It operates only on the GPU and local paths
you explicitly select.
