import array
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify import sha256, verify_checkpoint

spec = importlib.util.spec_from_file_location('training_runner', ROOT / 'training/run.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class AdapterVerificationTests(unittest.TestCase):
    def fixture(self, path, values=(1.0, 2.0)):
        data = array.array('f', values)
        if sys.byteorder != 'little':
            data.byteswap()
        raw = data.tobytes()
        header = json.dumps({'layer.lora_A.weight': {'dtype':'F32','shape':[1,len(values)],'data_offsets':[0,len(raw)]}}).encode()
        weights = path / 'lora_weights.safetensors'
        weights.write_bytes(struct.pack('<Q',len(header)) + header + raw)
        config = path / 'lora_config.json'
        config.write_text('{}')
        (path / 'provenance.json').write_text(json.dumps({
            'adapter':'test','weights_sha256':sha256(weights),
            'portable_config_sha256':sha256(config),'tensor_count':1,
            'trainable_parameter_count':len(values),
        }))

    def test_finite_adapter_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            self.fixture(path)
            result=verify_checkpoint(path)
            self.assertEqual(result['parameters'],2)
            self.assertTrue(result['all_values_finite'])

    def test_changed_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            self.fixture(path)
            with (path/'lora_weights.safetensors').open('ab') as stream:
                stream.write(b'x')
            with self.assertRaisesRegex(ValueError,'weight checksum'):
                verify_checkpoint(path)

    def test_nonfinite_tensor_with_valid_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            self.fixture(path,(float('nan'),2.0))
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                verify_checkpoint(path)

    def test_changed_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            self.fixture(path)
            (path/'lora_config.json').write_text('{"different":true}')
            with self.assertRaisesRegex(ValueError,'configuration checksum'):
                verify_checkpoint(path)


class TrainingGuardTests(unittest.TestCase):
    def test_nonfinite_training_values_are_rejected(self):
        training.check_finite([0,1.5,-4])
        for value in [float('nan'),float('inf'),float('-inf')]:
            with self.assertRaises(FloatingPointError):
                training.check_finite([value])

    def test_selected_gpu_must_match_explicit_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            config.write_text('{}')
            argv=['run.py','--source',tmp,'--config',str(config),'--expected-gpu-uuid','GPU-selected']
            fake_yaml=types.SimpleNamespace(safe_load=json.loads)
            with patch.object(sys,'argv',argv), patch.dict(sys.modules,{'yaml':fake_yaml}), patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-other'}):
                with self.assertRaisesRegex(RuntimeError,'must match --expected-gpu-uuid'):
                    training.main()

    def test_gpu_argument_cannot_be_omitted(self):
        with patch.object(sys,'argv',['run.py','--source','source','--config','config.json']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                training.main()


class DatasetMetadataTests(unittest.TestCase):
    def test_inventory_preserves_disjoint_assignments_without_audio_or_text(self):
        for variant,count in [('pilot',61),('expanded',431)]:
            root=ROOT/'datasets'/f'codsworth-{variant}-v1'
            parts={split:[json.loads(x) for x in (root/f'{split}.metadata.jsonl').read_text().splitlines()]
                   for split in ['train','development']}
            self.assertEqual(len(parts['train']),count)
            self.assertEqual(len(parts['development']),6)
            for key in ['id','group_id']:
                self.assertFalse({r[key] for r in parts['train']} & {r[key] for r in parts['development']})
            for split,items in parts.items():
                for item in items:
                    self.assertEqual(item['split'],split)
                    self.assertNotIn('text',item)
                    self.assertFalse(item['audio_path'].startswith('/'))
                    self.assertEqual(len(item['file_sha256']),64)
            summary=json.loads((root/'summary.json').read_text())
            for split in parts:
                self.assertAlmostEqual(sum(r['duration_seconds'] for r in parts[split]),summary[split]['seconds'],places=6)
                self.assertEqual(sha256(root/f'{split}.metadata.jsonl'),summary['metadata_sha256'][split])

    def test_future_huggingface_pointer_does_not_claim_publication(self):
        pointer=json.loads((ROOT/'dataset.json').read_text())
        self.assertEqual(pointer['status'],'publication_pending')
        for key in ['repository_id','repository_url','download_url']:
            self.assertIsNone(pointer[key])


if __name__ == '__main__':
    unittest.main()
