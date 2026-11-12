"""AMN-TM v3.1: scale-corrected tied token head; preserves v3 model code/weights.

v3 had raw embedding dot products with output states of unit variance,
giving extreme initial token logits (observed first-step CE ~81). Here:
    logits = F.linear(hidden / sqrt(width), embedding.weight, bias)
This initializes the logits at O(1) rather than O(sqrt(width)), enabling
stable softmax training without changing the native AMN / GRU mechanism.

v3.1 is a DISTINCT checkpoint format; existing v3 results remain reproducible.
"""
from __future__ import annotations
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from amn_tm_v3 import AMNTMv3

class AMNTMv31(AMNTMv3):
    def output_logits(self,hidden):
        return F.linear(hidden / math.sqrt(self.config["width"]),
                        self.embedding.weight,self.output_bias)

    def forward(self,ids):
        return self.output_logits(self.features(ids))

    def step(self,token,state=None):
        previous,mat,ks=(None,None,None) if state is None else state
        x=self.embedding(token[:,None])
        h,previous=self.gru(x,previous)
        mem,mat,ks=self.memory.step(h[:,0],mat,ks)
        return self.output_logits(mem),(previous,mat,ks)

def save(path,model,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"architecture":"AMNTMv3.1","config":model.config,
                "state_dict":model.state_dict(),"metadata":metadata},path)

def restore(path,device="cpu"):
    record=torch.load(path,map_location=device,weights_only=True)
    if record.get("architecture")!="AMNTMv3.1":
        raise ValueError("Wrong model version; refusing to reinterpret v3 weights")
    model=AMNTMv31(**record["config"])
    model.load_state_dict(record["state_dict"],strict=True)
    return model.to(device),record.get("metadata",{})
