"""Verify a downloaded Codsworth adapter without importing Torch or using a GPU."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_checkpoint(path):
    path = Path(path)
    provenance = json.loads((path / 'provenance.json').read_text())
    weights = path / 'lora_weights.safetensors'
    config = path / 'lora_config.json'
    if sha256(weights) != provenance['weights_sha256']:
        raise ValueError('Adapter weight checksum mismatch')
    if sha256(config) != provenance['portable_config_sha256']:
        raise ValueError('Adapter configuration checksum mismatch')
    with weights.open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        if header_length > 16 * 1024 * 1024:
            raise ValueError('Unexpected safetensors header size')
        header = json.loads(stream.read(header_length))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        if len(tensors) != provenance['tensor_count'] or not all('lora_' in name for name in tensors):
            raise ValueError('Unexpected adapter tensor inventory')
        end = 0
        parameters = 0
        for name, tensor in sorted(tensors.items(), key=lambda item: item[1]['data_offsets'][0]):
            start, stop = tensor['data_offsets']
            count = math.prod(tensor['shape'])
            if tensor['dtype'] != 'F32' or start != end or stop - start != count * 4:
                raise ValueError(f'Unexpected tensor layout: {name}')
            values = array.array('f')
            values.frombytes(stream.read(stop - start))
            if sys.byteorder != 'little':
                values.byteswap()
            if len(values) != count or not all(math.isfinite(x) for x in values):
                raise ValueError(f'Nonfinite or incomplete tensor: {name}')
            end = stop
            parameters += count
        if stream.read(1) or parameters != provenance['trainable_parameter_count']:
            raise ValueError('Unexpected trailing data or parameter count')
    return {'adapter': provenance['adapter'], 'sha256': provenance['weights_sha256'],
            'tensors': len(tensors), 'parameters': parameters, 'all_values_finite': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_checkpoint(args.checkpoint), indent=2))
