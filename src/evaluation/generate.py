"""Generate from saved latent vectors without reading any source episode."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.evaluation.evaluate import load_model, save_generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--latents', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resample-z', action='store_true', help='Resample saved posterior instead of reusing z')
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error('Positive samples and batch-size required')
    model, checkpoint = load_model(args.checkpoint)
    latent = torch.load(args.latents, map_location='cpu', weights_only=True)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if latent.get('checkpoint_sha256') != digest:
        raise ValueError('Latent file belongs to a different checkpoint.')
    torch.manual_seed(args.seed)
    z = (latent['mu'] + latent['scale']*torch.randn_like(latent['mu'])
         if args.resample_z else latent['z'])
    generated = torch.cat([model.generate(batch, args.samples)
                           for batch in z.split(args.batch_size)])[..., 0]
    generated = generated.numpy()*checkpoint['std']+checkpoint['mean']
    save_generated(generated, args.output)
    args.output.with_suffix('.json').write_text(json.dumps({
        'checkpoint_sha256': digest, 'latent_file': str(args.latents),
        'seed': args.seed, 'samples': args.samples, 'resample_z': args.resample_z,
        'batch_size': args.batch_size,
    }, indent=2)+'\n')
    print(f'Saved latent-only generation: {args.output}')


if __name__ == '__main__':
    main()
