"""Evaluate a selected checkpoint exactly once on PDF holdout; report limitations."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .config import NMCConfig
from .model import AMNNMC
from .baseline_gru import GRUByteLM
from .corpus import load_split
from .train import evaluate, _atomic_json
from .train_pdf import validate_source


def assess(data, checkpoint, output, *, max_bytes=None):
    manifest=validate_source(data)
    data_path=Path(data)
    ckpt=Path(checkpoint)
    payload=torch.load(ckpt,map_location='cpu',weights_only=True)
    if 'format_version' in payload:
        if payload.get('fingerprints',{}).get('train')!=manifest['split_files_sha256']['train']:
            raise ValueError('NMC checkpoint trained on different corpus')
        model=AMNNMC(NMCConfig(**payload['config']))
        kind='AMN-NMC'
        chunk=payload['training_settings']['chunk_size']
    else:
        model=GRUByteLM(**payload['config'])
        kind='GRU-control'
        chunk=64
    model.load_state_dict(payload['model'])
    metrics=evaluate(model,load_split(data_path,'test'),chunk,max_bytes=max_bytes)
    result={'model':kind,'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            'source_sha256':manifest['source_sha256'],'test':metrics,
            'test_is_independent_document':False,
            'limitation':manifest['split_type']}
    dest=Path(output)
    if dest.exists():raise FileExistsError('Refusing to overwrite previous test results')
    _atomic_json(result,dest)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-bytes',type=int)
    a=parser.parse_args()
    print(json.dumps(assess(a.data,a.checkpoint,a.output,max_bytes=a.max_bytes),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
