"""Exercise the real encoder's block timing without downloading models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F


class StreamTimingTests(unittest.TestCase):
    def test_encoder_preserves_audio_duration_across_cached_blocks(self):
        source = Path(__file__).resolve().parents[1] / 'runtime' / 'run_rt.py'
        runner = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                      if isinstance(n, ast.ClassDef) and n.name == 'VCRunner')
        method = next(n for n in runner.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_encode_chunk')
        # Match Kaldi's 25 ms window, 10 ms hop and snip_edges frame count.
        kaldi = SimpleNamespace(fbank=lambda wav, **kw:
                                torch.zeros((wav.shape[-1] - 400) // 160 + 1, 80))
        scope = dict(np=np, torch=torch, F=F, kaldi=kaldi)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
        for samples, expected_frames, calls_per_block in [(1280, 8, 1), (2560, 16, 2)]:
            with self.subTest(samples=samples):
                calls = []
                def asr(fbank, offset, required, att, cnn):
                    calls.append(int(offset))
                    return torch.ones(1, 2, 256), att, cnn
                state = SimpleNamespace(samples_cache=np.zeros(720, dtype=np.float32),
                    samples_cache_len=720, fbank_cache=None, asr_offset=4,
                    asr_att_cache=torch.zeros(6, 4, 4, 128), asr_cnn_cache=torch.zeros(6, 1, 256, 8),
                    _bn_window=11, _bn_stride=8, _required_cache_size=4,
                    _asr_offset_step=2, asr=asr, encoder_output_cache=None)
                for _ in range(4):
                    result = scope['_encode_chunk'](state, np.zeros(samples, dtype=np.float32))
                    self.assertEqual(tuple(result.shape), (1, expected_frames, 256))
                self.assertEqual(len(calls), 4 * calls_per_block)


if __name__ == '__main__':
    unittest.main()
