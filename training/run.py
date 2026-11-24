"""Guarded entry point for pinned upstream VoxCPM2 LoRA adaptation."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SOURCE_REVISION = 'f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69'
MODEL_REVISION = '32279effe8c19989596f05d353d1447f51d9e915'


def check_finite(values):
    if not all(math.isfinite(float(v)) for v in values):
        raise FloatingPointError('Nonfinite training value; refusing optimizer update')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-gpu-uuid', required=True, help='Explicit GPU UUID selected for this run')
    args = parser.parse_args()
    expected_gpu = 'GPU-' + args.expected_gpu_uuid.removeprefix('GPU-')
    import yaml
    config = yaml.safe_load(args.config.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_gpu:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must match --expected-gpu-uuid')
    actual_revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_revision != SOURCE_REVISION:
        raise RuntimeError('Unexpected upstream revision')
    if Path(config['pretrained_path']).name != MODEL_REVISION:
        raise RuntimeError('Unexpected pretrained model revision')
    output = Path(config['save_path'])
    if (output / 'latest').exists():
        raise RuntimeError('Use a fresh output directory; implicit upstream resume is forbidden')
    output.mkdir(parents=True, exist_ok=True)
    import torch
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if actual_uuid != expected_gpu.removeprefix('GPU-'):
        raise RuntimeError('Actual Torch GPU UUID mismatch')
    torch.set_num_threads(6)
    sys.path.insert(0, str(args.source / 'src'))
    spec = importlib.util.spec_from_file_location('upstream_train', args.source / 'scripts/train_voxcpm_finetune.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    original_loader = upstream.VoxCPM2Model.from_local.__func__
    def frozen_loader(cls, *a, **kw):
        model = original_loader(cls, *a, **kw)
        model.audio_vae.requires_grad_(False)
        model.audio_vae.eval()
        trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in name for name, _ in trainable):
            raise RuntimeError('Only LoRA parameters may be trained')
        (output / 'trainable-parameters.json').write_text(json.dumps(dict(parameters=trainable, total=sum(n for _, n in trainable)), indent=2))
        return model
    upstream.VoxCPM2Model.from_local = classmethod(frozen_loader)
    original_backward = upstream.Accelerator.backward
    def checked_backward(self, loss):
        check_finite([loss.detach().item()])
        return original_backward(self, loss)
    upstream.Accelerator.backward = checked_backward
    original_step = upstream.Accelerator.step
    def checked_step(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite gradient; refusing optimizer update')
        result = original_step(self, optimizer)
        provenance['optimizer_steps_completed'] += 1
        provenance.update(training_performed=True, status='training')
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        return result
    upstream.Accelerator.step = checked_step
    # Validation is loss-only here. Upstream's optional generation mutates model
    # structure and is unbounded; fixed paired auditions run separately.
    upstream.generate_sample_audio = lambda *a, **kw: None
    manifest_hashes = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                       for key in ('train_manifest', 'val_manifest') if config.get(key)}
    provenance = dict(config=config, model_revision=MODEL_REVISION,
        source_revision=actual_revision, gpu_uuid=actual_uuid, torch=torch.__version__,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), manifest_sha256=manifest_hashes,
        training_performed=False, status='initializing', optimizer_steps_completed=0, audio_vae_trained=False)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(event='training_start', output=str(output), gpu_uuid=actual_uuid, steps=config['num_iters'])), flush=True)
    try:
        upstream.train(**config)
        provenance['status'] = 'complete'
    except BaseException:
        provenance['status'] = 'failed'
        raise
    finally:
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(event='training_complete', output=str(output))), flush=True)

if __name__ == '__main__':
    main()
