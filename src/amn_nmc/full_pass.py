"""Gate a completed *full* first pass, never mistake a pilot for finished training."""
import argparse
import json
import math
from pathlib import Path
import torch
from .corpus import load_split
from .train_pdf import validate_source


def verify(data, checkpoint):
    manifest = validate_source(data)
    docs = load_split(data, 'train')
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('format_version') != 2:
        raise ValueError('The checkpoint is not a resumable NMC v2 checkpoint')
    c = int(payload['training_settings']['chunk_size'])
    expected_steps = sum(math.ceil((len(doc)-1)/c) for doc in docs)
    expected_bytes = sum(len(doc)-1 for doc in docs)
    actual_steps = payload['step']
    seen = set(payload['visited_training_docs'])
    cursor = payload['cursor']
    result = {'source_sha256':manifest['source_sha256'],
              'expected_first_pass_steps':expected_steps,
              'expected_first_pass_target_bytes':expected_bytes,
              'actual_steps':actual_steps,'actual_training_bytes':payload['training_bytes'],
              'training_pages':len(docs),'visited_pages':len(seen),
              'cursor':cursor,
              'one_full_epoch_complete':bool(
                  actual_steps >= expected_steps and
                  payload['training_bytes'] >= expected_bytes and
                  len(seen) == len(docs) and
                  (cursor['epoch'] >= 1 or
                   (cursor['epoch']==0 and cursor['doc_position']==len(docs)
                    and cursor['byte_offset']==0)))}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=verify(a.data,a.checkpoint)
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['one_full_epoch_complete']:
        raise SystemExit('TRAINING_INCOMPLETE: full first corpus pass not finished')


if __name__=='__main__':
    main()
