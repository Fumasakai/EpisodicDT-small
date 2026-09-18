import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from src.train.train import main as train_main
from src.evaluation.evaluate import main as evaluate_main
from src.models.episodic_diffusion import LatentEpisodeDiffusion


class TrainingWorkflowTests(unittest.TestCase):
    def test_early_stopping_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, evaluation = root / 'train', root / 'evaluation'
            train.mkdir()
            evaluation.mkdir()
            values = np.linspace(-110, -80, 120, dtype=np.float32)
            pd.DataFrame({'rsrp': values}).to_csv(train / 'trace.csv', index=False)
            pd.DataFrame({'rsrp': values[::-1]}).to_csv(evaluation / 'trace.csv', index=False)
            config = {
                'data': {'input_dim': 1, 'episode_length': 8,
                         'train_path': str(train), 'evaluation_path': str(evaluation)},
                'model': {'hidden_dim': 8, 'latent_dim': 4, 'transformer_heads': 2,
                          'transformer_layers': 1, 'dropout': 0.0},
                'diffusion': {'timesteps': 4, 'denoiser': 'conv1d', 'conv_channels': 8, 'conv_blocks': 2},
                'training': {'batch_size': 16, 'epochs': 5, 'learning_rate': 0.0, 'seed': 7,
                             'checkpoint_dir': str(root / 'checkpoints'), 'validation_fraction': 0.2,
                             'validation_samples': 4, 'validation_seed': 13,
                             'min_epochs': 2, 'early_stopping_patience': 1,
                             'early_stopping_min_delta': 0.001},
                'evaluation': {'batch_size': 16,
                               'figure_dir': str(root / 'figures'),
                               'boxplot_path': str(root / 'figures/box.png'),
                               'generated_csv': str(root / 'generated.csv'),
                               'actual_csv': str(root / 'actual.csv')},
            }
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            with patch('sys.argv', ['train', '--config', str(config_path)]):
                train_main()
            checkpoint = torch.load(root / 'checkpoints/episodicdt.pt', weights_only=False)
            logs = json.loads((root / 'checkpoints/training_metrics.json').read_text())
            # With lr=0 and fixed validation noise, epoch 2 cannot improve.
            self.assertEqual(len(logs), 2)
            self.assertEqual(checkpoint['best_epoch'], 1)
            self.assertEqual(checkpoint['model_format'], LatentEpisodeDiffusion.FORMAT)
            self.assertEqual(logs[0]['reconstruction_crps_dbm'], logs[1]['reconstruction_crps_dbm'])
            self.assertAlmostEqual(checkpoint['mean'], values[:96].mean().item(), places=4)
            with patch('sys.argv', ['evaluate', '--config', str(config_path), '--samples', '4']):
                evaluate_main()
            generated = pd.read_csv(root / 'generated.csv')
            self.assertEqual(len(generated), (120 - 8 + 1) * 4 * 8)
            self.assertTrue(np.isfinite(generated.rsrp_generated_dbm).all())
            self.assertIn('episode_step', generated.columns)
            latent_path = root / 'generated.latents.pt'
            self.assertTrue(latent_path.is_file())
            (evaluation / 'trace.csv').unlink()
            from src.evaluation.generate import main as generate_main
            with patch('sys.argv', ['generate', '--checkpoint', str(root/'checkpoints/episodicdt.pt'),
                                    '--latents', str(latent_path), '--output', str(root/'from_z.csv'),
                                    '--samples', '2']):
                generate_main()
            self.assertEqual(len(pd.read_csv(root/'from_z.csv')), (120-8+1)*2*8)
            self.assertFalse((root / 'figures/episode_median_boxplot_by_step.png').exists())
            self.assertFalse((root / 'figures/episode_boxplot_by_step.png').exists())
            self.assertTrue((root / 'figures/episode_autocorrelation.png').is_file())
            self.assertTrue((root / 'figures/episode_mean_squared_change.png').is_file())


if __name__ == '__main__':
    unittest.main()
