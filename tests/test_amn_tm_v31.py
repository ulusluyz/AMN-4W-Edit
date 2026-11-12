"""AMN-TM v3.1 scaling correction tests (original v3 is unchanged)."""
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))

from amn_tm_v3 import AMNTMv3
from amn_tm_v31 import AMNTMv31,save,restore

def test_parameter_budget_and_original_model_not_reinterpreted(tmp_path):
    model=AMNTMv31(8192,544,6,128,192,0)
    params=sum(p.numel() for p in model.parameters())
    assert 15_000_000<=params<=18_000_000
    save(tmp_path/"v31.pt",model,{"seed":3})
    copy,meta=restore(tmp_path/"v31.pt")
    assert meta["seed"]==3
    torch.save({"architecture":"AMNTMv3","config":model.config,
                "state_dict":model.state_dict()},tmp_path/"old.pt")
    try:
        restore(tmp_path/"old.pt")
    except ValueError:
        pass
    else:
        raise AssertionError("Old v3 weights must never be silently reinterpreted")

def test_initial_logits_scaled_avoids_extreme_cross_entropy():
    torch.manual_seed(42)
    old=AMNTMv3(256,32,1,8,16,0)
    new=AMNTMv31(256,32,1,8,16,0)
    new.load_state_dict(old.state_dict(),strict=True)
    data=torch.randint(4,256,(3,22))
    with torch.no_grad():
        source=old(data)
        adjusted=new(data)
    ratio=math.sqrt(32)
    assert torch.allclose(source/ratio,adjusted,atol=1e-5,rtol=1e-5)
    y=torch.randint(4,256,(3,22))
    assert F.cross_entropy(adjusted.reshape(-1,256),y.reshape(-1)).item()<12

def test_forward_step_output_parity_and_gradients():
    torch.manual_seed(4)
    model=AMNTMv31(192,40,2,8,16,0)
    x=torch.randint(4,192,(2,13))
    model.eval()
    y=model(x)
    with torch.no_grad():
        state=None;outs=[]
        for i in range(x.shape[1]):
            pred,state=model.step(x[:,i],state)
            outs.append(pred)
    assert torch.allclose(y,torch.stack(outs,1),atol=2e-5,rtol=1e-5)
    loss=F.cross_entropy(y.reshape(-1,192),x.reshape(-1))
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert model.output_bias.grad is not None
