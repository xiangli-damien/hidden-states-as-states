"""Apply frozen benchmark detectors to new per-layer local state IDs."""
import json
from pathlib import Path
import joblib
import numpy as np
from .structure import aggregate_losses


def score_new_routes(root, map_name, states, device='cpu', methods=None):
    """Return label-free mean/top4 scores; never accepts ground truth.

    states: N x L local IDs assigned by the EXACT original fitted map. This
    function does not cluster raw hidden states or reinterpret global IDs.
    Only load trusted benchmark artifacts (joblib models are executable).
    """
    root=Path(root)
    encoder=joblib.load(root/'models'/map_name/'encoder.joblib')
    z=encoder.transform(states);x=encoder.onehot(z);out={}
    selected=[r for r in json.loads((root/'selected.json').read_text()) if r['map']==map_name]
    if methods is not None:selected=[r for r in selected if r['method'] in methods]
    for row in selected:
        values=[]
        for name in row['candidates']:
            path=root/name
            if (path/'model.pt').exists():
                import torch
                from .neural import RouteNet,neural_losses
                saved=torch.load(path/'model.pt',map_location=device,weights_only=True)
                model=RouteNet(saved['sizes'],saved['kind'],saved['width']).to(device)
                model.load_state_dict(saved['state_dict'])
                losses=neural_losses(model,z,device=device)
            else:
                model=joblib.load(path/'model.joblib');kind=row['method']
                if hasattr(model,'losses'):losses=model.losses(z)
                elif getattr(model,'input_encoding',None)=='local_state_ids':losses=-model.score_samples(z)[:,None]
                elif kind=='knn':losses=model.kneighbors(x)[0].mean(1)[:,None]
                elif kind=='pca':losses=np.mean((x-model.inverse_transform(model.transform(x)))**2,1)[:,None]
                else:losses=-model.score_samples(x)[:,None]
            values.append(aggregate_losses(losses))
        for agg in ['mean','top4']:
            out[row['method']+'__'+agg]=np.mean([v[agg] for v in values],axis=0)
    return out
