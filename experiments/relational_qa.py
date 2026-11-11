"""Structured Turkish relational QA: train, development and test without shared triples.

Facts: "<entity> <relation> <value>."  Query: "<entity> <relation>?"
All inputs are tokenized (not free-form Turkish). The task supervises only
the answer. Grammar boundaries and subject/relation positions are supplied;
facts/answers and the relational address mapping are not hardcoded at inference.
The static grammar is a strong inductive prior and must be disclosed.

Train/dev/test partition by SHA-256(entity,relation,value). Dev controls
early stopping. Test measures held-out bindings, though the test design was
visible during algorithm development: not a fully blind benchmark.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

NAMES = ('Ali','Ayşe','Can','Deniz','Ece','Efe','Elif','Emre','Mert','Mina','Murat',
         'Nisa','Okan','Selin','Sena','Umut','Yaren','Yiğit','Zeynep','Arda')
VALUES = {
    'renk': ('kırmızı','mavi','yeşil','sarı','mor','beyaz','siyah','turuncu'),
    'konum': ('ev','okul','park','ofis','bahçe','kafe','salon','garaj'),
}
RELS = tuple(VALUES)
VOCAB = ('<pad>',) + NAMES + RELS + VALUES['renk'] + VALUES['konum'] + (';', 'soru', '?')
ID = {s:i for i,s in enumerate(VOCAB)}
V = len(VOCAB)


def split_of(fact):
    digest = hashlib.sha256('|'.join(fact).encode('utf-8')).digest()
    rem = int.from_bytes(digest[:8], 'big') % 10
    return 'dev' if rem == 0 else 'test' if rem == 1 else 'train'


FACTS = {split: [] for split in ('train', 'dev', 'test')}
for name in NAMES:
    for relation in RELS:
        for value in VALUES[relation]:
            triple = (name, relation, value)
            FACTS[split_of(triple)].append(triple)


def generate(split, batch, n_facts, rng):
    """All context triples and the requested answer belong to this split."""
    assert n_facts >= 3
    pool = FACTS[split]
    triples, queries, targets = [], [], []
    for _ in range(batch):
        target = rng.choice(pool)
        choices = [target]
        same_name = [f for f in pool if f[0] == target[0] and f[1] != target[1]]
        if same_name:
            choices.append(rng.choice(same_name))
        same_rel = [f for f in pool if f[1] == target[1] and f[0] != target[0]]
        if same_rel:
            choices.append(rng.choice(same_rel))
        used = {(f[0], f[1]) for f in choices}
        if len(used) != len(choices):
            raise RuntimeError('duplicate key in one prompt')
        while len(choices) < n_facts:
            f = rng.choice(pool)
            if (f[0], f[1]) in used:
                continue
            choices.append(f)
            used.add((f[0], f[1]))
        rng.shuffle(choices)
        triples.append([[ID[word] for word in fact] for fact in choices])
        queries.append([ID[target[0]], ID[target[1]]])
        targets.append(ID[target[2]])
    return (torch.tensor(triples, dtype=torch.long),
            torch.tensor(queries, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


def prompt(example):
    facts, query, label = example
    lines = [' '.join(VOCAB[i] for i in fact) + '.' for fact in facts.tolist()]
    return (' '.join(lines) + ' Soru: ' +
            ' '.join(VOCAB[i] for i in query.tolist()) + '? -> ' + VOCAB[label.item()])


class AssociativeAMN(nn.Module):
    """Original-style recurrent delta memory, with multiplicative pair keys."""
    def __init__(self, d=48):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(V, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, V))

    def address(self, pairs):
        # Multiplicative entity/relation binding avoids exact additive
        # rectangle dependencies: (A,r1)+(B,r2)=(A,r2)+(B,r1).
        words = self.embed(pairs)
        return F.normalize(self.key(words[..., 0, :] * words[..., 1, :]), dim=-1)

    def forward(self, triples, query):
        batch, n_facts, _ = triples.shape
        keys = self.address(triples[:, :, :2])
        values = F.normalize(self.value(self.embed(triples[:, :, 2])), dim=-1)
        state = keys.new_zeros((batch, self.d, self.d))
        for i in range(n_facts):
            key, value = keys[:, i], values[:, i]
            old = torch.bmm(state, key.unsqueeze(-1)).squeeze(-1)
            state = state + torch.bmm((value - old).unsqueeze(-1), key.unsqueeze(1))
        q = self.address(query)
        read = torch.bmm(state, q.unsqueeze(-1)).squeeze(-1)
        return self.out(read)


class DualAddressAMN(AssociativeAMN):
    """Protected associative state for linearly independent keys.

    K: batch x memory_slots x key_dim. G=K K^T. D=(G+eps I)^-1 K.
    S=V^T D, read S q. With full rank and eps=0, D K^T=I so
    updating/deleting a relation need not corrupt another exact-key read.
    eps=1e-5 is used for numerical stability and slightly breaks exactness.
    Memory cost depends on the number of stored keys, unlike fixed-state AMN.
    """
    def forward(self, triples, query):
        keys = self.address(triples[:, :, :2])
        values = F.normalize(self.value(self.embed(triples[:, :, 2])), dim=-1)
        gram = torch.bmm(keys, keys.transpose(1, 2))
        slots = keys.size(1)
        eye = torch.eye(slots, device=keys.device, dtype=keys.dtype).expand_as(gram)
        dual = torch.linalg.solve(gram + 1e-5 * eye, keys)
        state = torch.bmm(values.transpose(1, 2), dual)
        q = self.address(query)
        read = torch.bmm(state, q.unsqueeze(-1)).squeeze(-1)
        return self.out(read)


class TransformerQA(nn.Module):
    """Full context encoder; identical text tokens and answer supervision."""
    def __init__(self, d=48, n_facts=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(3 * n_facts + 2, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=2*d, dropout=0.0,
            batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False)
        self.out = nn.Linear(d, V)

    def forward(self, triples, query):
        batch, slots, _ = triples.shape
        ids = torch.cat((triples.reshape(batch, slots*3), query), dim=1)
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        return self.out(self.encoder(x)[:, -1])


@torch.no_grad()
def evaluate(model, seed, split='dev', n=1000, n_facts=4, batch=128):
    model.eval()
    rng = random.Random(seed)
    correct, total, loss_sum, errors = 0, 0, 0.0, []
    while total < n:
        count = min(batch, n-total)
        x, q, y = generate(split, count, n_facts, rng)
        logits = model(x, q)
        predicted = logits.argmax(-1)
        correct += int((predicted == y).sum())
        loss_sum += float(F.cross_entropy(logits, y, reduction='sum'))
        if len(errors) < 6:
            for i in range(count):
                if predicted[i] != y[i] and len(errors) < 6:
                    errors.append({'prompt': prompt((x[i],q[i],y[i])),
                                   'pred': VOCAB[predicted[i]]})
        total += count
    return {'accuracy':correct/n, 'correct':correct, 'total':n,
            'loss':loss_sum/n, 'errors':errors}


def train(kind, seed=0, steps=500, batch=128, n_facts=4, d=48,
          dev_every=50, dev_size=1000, test_size=2000,
          patience=2, lr=0.003, output_dir=None):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = (AssociativeAMN(d) if kind == 'amn' else
             DualAddressAMN(d) if kind == 'dual' else
             TransformerQA(d, n_facts))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed+100_000)
    history, streak, first = [], 0, None
    started = time.perf_counter()
    for step in range(1, steps+1):
        model.train()
        x,q,y = generate('train', batch, n_facts, rng)
        logits = model(x,q)
        loss = F.cross_entropy(logits,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % dev_every == 0 or step == steps:
            dev = evaluate(model,seed+200_000,'dev',dev_size,n_facts)
            elapsed = time.perf_counter()-started
            history.append({'step':step, 'seconds':elapsed,
                            'training_loss':float(loss.detach()),
                            'dev_acc':dev['accuracy'], 'dev_loss':dev['loss']})
            print(kind,seed,step,'loss',round(float(loss.detach()),4),
                  'dev',dev['correct'],'/',dev_size,'seconds',round(elapsed,2),flush=True)
            if dev['accuracy']==1.0:
                streak+=1
                if first is None:
                    first=step
            else:
                streak=0
            if streak>=patience:
                break
    seconds = time.perf_counter()-started
    final = evaluate(model,seed+300_000,'test',test_size,n_facts)
    result = {
        'kind':kind, 'seed':seed, 'steps':step,
        'first_dev_100_step':first, 'training_seconds':seconds,
        'train_loss':float(loss.detach()), 'dev':dev, 'test':final,
        'params':sum(p.numel() for p in model.parameters()),
        'data_splits':{name:len(facts) for name,facts in FACTS.items()},
        'settings':{'batch':batch,'n_facts':n_facts,'dim':d,'lr':lr,
                    'dev_every':dev_every,'dev_size':dev_size,
                    'test_size':test_size,'patience':patience},
        'history':history
    }
    if output_dir:
        path=Path(output_dir)
        path.mkdir(parents=True,exist_ok=True)
        (path/f'{kind}_{seed}.json').write_text(
            json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        torch.save({'state_dict':model.state_dict(),
                    'settings':result['settings'],'seed':seed},
                   path/f'{kind}_{seed}.pt')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['amn','dual','transformer','both','all'],default='both')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--dev-every',type=int,default=50)
    parser.add_argument('--dev-size',type=int,default=1000)
    parser.add_argument('--test-size',type=int,default=2000)
    parser.add_argument('--patience',type=int,default=2)
    parser.add_argument('--dim',type=int,default=48)
    parser.add_argument('--lr',type=float,default=0.003)
    parser.add_argument('--output-dir',default='results/relational_qa')
    args=parser.parse_args()
    kinds=(['amn','transformer'] if args.model=='both' else
           ['amn','dual','transformer'] if args.model=='all' else [args.model])
    for kind in kinds:
        for seed in args.seeds:
            train(kind,seed,args.steps,args.batch,d=args.dim,
                  dev_every=args.dev_every,dev_size=args.dev_size,
                  test_size=args.test_size,patience=args.patience,
                  lr=args.lr,output_dir=args.output_dir)


if __name__=='__main__':
    main()
