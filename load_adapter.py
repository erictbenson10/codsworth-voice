"""Generate explicitly synthetic audio with a verified Codsworth adapter."""
import argparse
import json
from pathlib import Path

from verify import verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='A recording you are entitled to use')
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    verify_checkpoint(args.checkpoint)
    if not args.reference.is_file():
        parser.error('--reference must be an existing audio file')
    config = json.loads((args.checkpoint / 'lora_config.json').read_text())
    from huggingface_hub import snapshot_download
    import soundfile as sf
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm import LoRAConfig
    base = snapshot_download(repo_id=config['base_model'], revision=config['base_model_revision'])
    model = VoxCPM.from_pretrained(base, load_denoiser=False, optimize=False,
                                  device=args.device, lora_config=LoRAConfig(**config['lora_config']))
    loaded, skipped = model.load_lora(str(args.checkpoint))
    if len(loaded) != 384 or skipped:
        raise RuntimeError(f'Incomplete adapter load: {len(loaded)} loaded, {len(skipped)} skipped')
    audio = model.generate(text=args.text, reference_wav_path=str(args.reference), seed=args.seed,
                           cfg_value=2.0, inference_timesteps=10, max_len=1000,
                           normalize=False, denoise=False, retry_badcase=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, model.tts_model.sample_rate)
    args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps({
        'synthetic_audio': True, 'adapter': args.checkpoint.name,
        'base_model': config['base_model'], 'base_model_revision': config['base_model_revision'],
        'seed': args.seed, 'sample_rate': model.tts_model.sample_rate,
    }, indent=2) + '\n')
    print(f'Wrote synthetic audio to {args.output}')


if __name__ == '__main__':
    main()
