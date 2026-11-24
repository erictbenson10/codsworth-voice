# Codsworth voice experiments

Trained VoxCPM2 LoRA adapters and the reproducible recipe behind Eric Benson’s
Codsworth voice experiment. The UI is a separate project:
[codsworth-ui](https://github.com/erictbenson10/codsworth-ui).

The September 7, 2026 experiments produced two actual trained adapters:

| Adapter | Training data | Optimizer updates | Purpose |
| --- | --- | --- | --- |
| [Expanded step 300](models/expanded-step300/README.md) | 431 clips / 41.59 minutes | 300 | Larger adaptation candidate |
| [Pilot step 100](models/pilot-step100/README.md) | 61 clips / 7.50 minutes | 100 | Earlier comparison checkpoint |

Both use rank-32, alpha-32 LoRA on the language model and diffusion transformer.
Only LoRA parameters were trained; the AudioVAE stayed frozen. A completed run
does not establish convergence or improved voice quality.

## What is available

- Git contains model cards, configurations, checksums, split inventories, a
  portable training wrapper, and an inference example.
- [Model release assets](https://github.com/erictbenson10/codsworth-voice/releases)
  contain the adapter weights. Each adapter is approximately 69 MiB before
  compression, with 384 tensors and 18,087,936 trained parameters.
- **Training audio and transcripts are not published here.** The future Hugging
  Face dataset URL is tracked in [dataset.json](dataset.json). It is deliberately
  unset until that separate publication happens.
- Private deployment configuration, hostnames, GPU identifiers, service control
  scripts, raw game recordings, and historical experiment logs are excluded.

## Try an adapter

Use an isolated Python environment with a compatible CUDA/PyTorch installation.
The training environment used Torch 2.7.1 with CUDA 12.8. Install the pinned
VoxCPM source used by the experiment:

```sh
python -m venv .venv
. .venv/bin/activate
pip install 'git+https://github.com/OpenBMB/VoxCPM.git@f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69'
```

Download and extract one model release asset. Verify it without using a GPU:

```sh
tar -xzf codsworth-expanded-step300.tar.gz
python verify.py codsworth-expanded-step300
```

Supply a reference recording you are entitled to use:

```sh
python load_adapter.py \
  --checkpoint codsworth-expanded-step300 \
  --reference /path/to/reference.wav \
  --text 'Hello. This is a synthetic voice demonstration.' \
  --output outputs/demo.wav
```

The loader downloads the exact public base revision, verifies every adapter
tensor, and rejects skipped weights. It writes a sidecar identifying the output
as synthetic audio. Base weights and the AudioVAE are not duplicated in the
release. The publication checks do not run new GPU inference or training.

## Reproduce the training

See [training/README.md](training/README.md). The recipes use local dataset paths,
not private services. Actual audio and full text manifests are required; the
public `*.metadata.jsonl` inventories intentionally omit both.

The public inventories preserve the original train/development assignments and
group IDs. Older RVC experiments exposed nominal holdout recordings during
pretraining, so those historical evaluation labels must not be treated as clean
test results. Neither the pilot nor expanded adaptation reused the older model.

## Attribution and licensing status

The pinned [openbmb/VoxCPM2 base model card](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/README.md)
declares Apache-2.0. No separate license has been declared for these adaptation
artifacts. This project does not assign a new license to the game dialogue,
character, or training corpus. See [THIRD_PARTY.md](THIRD_PARTY.md).

The dataset originated from `AquaV/fallout-4-voices`, containing Fallout 4 and
Far Harbor game voice assets. `RobotMrHandy` is a voice type, not proof that
every source clip is the Codsworth character. Selection limitations are recorded
with each [dataset inventory](datasets/).
