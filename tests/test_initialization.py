import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import initialization as init


class CheckpointPreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        os.chdir(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(os.chdir, self.previous)
        self.cfg = dict(encoder_layers=24, encoder_embed_dim=1024, encoder_attention_heads=16)

    def test_extract_and_reuse_config_without_base_checkpoint(self):
        source = Path('source.pt')
        torch.save({'cfg': self.cfg, 'model': {'weight': torch.zeros(1)}}, source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with patch.object(init, 'hf_hub_download', return_value=str(source)) as download, patch.object(init, 'WAVLM_SHA256', digest):
            init._download_wavlm()
            download.assert_called_once_with(repo_id=init.WAVLM_REPO, filename='wavlm_large.pt', revision=init.WAVLM_REVISION)
        source.unlink()
        with patch.object(init, 'hf_hub_download', side_effect=AssertionError('Must reuse config')):
            init._download_wavlm()
        self.assertEqual(torch.load(init.WAVLM_CFG_PATH, weights_only=True), self.cfg)

    def test_download_failure_is_fatal(self):
        with patch.object(init, 'hf_hub_download', side_effect=OSError('download failed')):
            with self.assertRaisesRegex(OSError, 'download failed'):
                init._download_wavlm()
        self.assertFalse(Path(init.WAVLM_CFG_PATH).exists())

    def test_rejects_wrong_digest_and_missing_config(self):
        source = Path('source.pt')
        torch.save({'model': {}}, source)
        with patch.object(init, 'hf_hub_download', return_value=str(source)):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                init._download_wavlm()
            with patch.object(init, 'WAVLM_SHA256', hashlib.sha256(source.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    init._download_wavlm()
        self.assertFalse(Path(init.WAVLM_CFG_PATH).exists())

    def test_verification_catches_missing_or_empty_packaged_files(self):
        for _, filename in init.TASK_FILES['train_40ms']:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
        torch.save(self.cfg, init.WAVLM_CFG_PATH)
        with self.assertRaisesRegex(FileNotFoundError, 'wavlm_large_finetune'):
            init.verify_files('train_40ms')
        Path(init.WAVLM_FINETUNED_PATH).write_bytes(b'fixture')
        init.verify_files('train_40ms')
        Path(init.WAVLM_CFG_PATH).write_bytes(b'')
        with self.assertRaisesRegex(FileNotFoundError, 'wavlm_large_cfg'):
            init.verify_files('train_40ms')


if __name__ == '__main__':
    unittest.main()
