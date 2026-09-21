"""Small self-supervised route models; PyTorch is an optional dependency.

These are categorical-route adaptations of model families, not claims to
reproduce published anomaly benchmarks. No correctness labels enter this file.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RouteNet(nn.Module):
    def __init__(self, sizes, kind, width=64):
        super().__init__()
        self.sizes = list(map(int, sizes)); self.kind = kind; self.depth = len(sizes)
        self.width = width; self.total = sum(sizes); self.maxk = max(sizes)
        self.register_buffer('offsets', torch.tensor(np.r_[0, np.cumsum(sizes)[:-1]], dtype=torch.long))
        valid = torch.arange(self.maxk)[None, :] < torch.tensor(sizes)[:, None]
        self.register_buffer('valid', valid)
        if kind in ['gru', 'causal_transformer', 'masked_transformer']:
            self.embedding = nn.Embedding(self.total+1, width)
            self.position = nn.Parameter(torch.randn(1, self.depth, width)*.02)
            if kind == 'gru':
                self.core = nn.GRU(width, width, batch_first=True)
            else:
                layer = nn.TransformerEncoderLayer(width, 4, width*2, dropout=.1, batch_first=True)
                self.core = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(width, self.maxk)
        else:
            latent = width//4
            # Bias-free encoder is also used in the SVDD adaptation.
            self.encoder = nn.Sequential(nn.Linear(self.total, width, bias=False), nn.ReLU(), nn.Linear(width, latent, bias=False))
            self.decoder = nn.Sequential(nn.Linear(latent, width), nn.ReLU(), nn.Linear(width, self.depth*self.maxk))
            if kind == 'vae':
                self.logvar = nn.Linear(latent, latent)
            if kind == 'suffix_contrastive':
                self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(latent, 1))
            self.register_buffer('center', torch.zeros(latent))

    def onehot(self, z):
        x = torch.zeros(len(z), self.total, device=z.device)
        return x.scatter_(1, z+self.offsets, 1.)

    def ce(self, logits, z):
        logits = logits.masked_fill(~self.valid[None], -1e4)
        return F.cross_entropy(logits.transpose(1, 2), z, reduction='none')

    def sequence_logits(self, z, mask=None):
        ids = z+self.offsets
        if self.kind == 'masked_transformer':
            ids = ids.masked_fill(mask, self.total)
        else:
            ids = torch.cat([torch.full_like(ids[:, :1], self.total), ids[:, :-1]], 1)
        x = self.embedding(ids)+self.position
        if self.kind == 'gru':
            h, _ = self.core(x)
        elif self.kind == 'causal_transformer':
            mask = torch.triu(torch.ones(self.depth, self.depth, device=z.device, dtype=torch.bool), diagonal=1)
            h = self.core(x, mask=mask)
        else:
            h = self.core(x)
        return self.head(h)

    def ae_loss(self, z, corrupt=False, sample=False):
        x = self.onehot(z)
        if corrupt:
            keep = (torch.rand(z.shape, device=z.device)>.3).float()
            x = x * torch.repeat_interleave(keep, torch.tensor(self.sizes, device=z.device), dim=1)
        h = self.encoder(x)
        kl = torch.zeros(len(z), device=z.device)
        if self.kind == 'vae':
            lv = self.logvar(h).clamp(-8, 8)
            kl = .5*(h.square()+lv.exp()-1-lv).sum(1)/self.depth
            if sample:
                h = h + torch.randn_like(h)*(.5*lv).exp()
        logits = self.decoder(h).reshape(len(z), self.depth, self.maxk)
        return self.ce(logits, z), kl


def suffix_negatives(z):
    """Swap suffixes within common middle states, preserving adjacent edges."""
    pivot = int(torch.randint(1, z.shape[1]-1, ()).item())
    out = z.clone()
    for value in z[:, pivot].unique():
        ii = torch.where(z[:, pivot] == value)[0]
        jj = ii[torch.randperm(len(ii), device=z.device)]
        out[ii, pivot+1:] = z[jj, pivot+1:]
    return out, (out != z).any(1)


def objective(model, z, pretrain=False):
    kind = model.kind
    if kind in ['gru', 'causal_transformer']:
        return model.ce(model.sequence_logits(z), z).mean()
    if kind == 'masked_transformer':
        mask = torch.rand(z.shape, device=z.device)<.25
        mask[:, int(torch.randint(z.shape[1], ()).item())] = True
        loss = model.ce(model.sequence_logits(z, mask), z)
        return loss[mask].mean()
    if kind == 'suffix_contrastive':
        negative, changed = suffix_negatives(z)
        if not changed.any():
            return None
        both = torch.cat([z[changed], negative[changed]])
        target = torch.cat([torch.zeros(changed.sum(), device=z.device), torch.ones(changed.sum(), device=z.device)])
        logit = model.classifier(model.encoder(model.onehot(both))).squeeze(1)
        return F.binary_cross_entropy_with_logits(logit, target)
    if kind == 'deep_svdd' and not pretrain:
        return (model.encoder(model.onehot(z))-model.center).square().sum(1).mean()
    loss, kl = model.ae_loss(z, corrupt=(kind!='vae'), sample=(kind=='vae'))
    return loss.mean()+kl.mean()


def fit_neural(train, validation, sizes, kind, width, seed, config, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RouteNet(sizes, kind, width).to(device)
    tr = torch.tensor(train, device=device); va = torch.tensor(validation, device=device)
    batch = config['batch_size']; trace = []
    phases = [('pretrain', 15), ('svdd', 25)] if kind=='deep_svdd' else [('main', config['epochs'])]
    for phase, epochs in phases:
        pretrain = phase == 'pretrain'
        if phase == 'svdd':
            with torch.no_grad():
                center = model.encoder(model.onehot(tr)).mean(0)
                center = torch.where(center.abs()<.1, torch.where(center<0, -.1, .1), center)
                model.center.copy_(center)
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        best = float('inf'); state = None; stale = 0; best_epoch = -1
        for epoch in range(epochs):
            model.train(); values=[]
            for ii in torch.randperm(len(tr), device=device).split(batch):
                loss = objective(model, tr[ii], pretrain)
                if loss is None: continue
                if not torch.isfinite(loss): raise ValueError('Nonfinite neural objective')
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step(); values.append(loss.item())
            model.eval(); vals=[]
            # Fixed validation corruption and VAE noise; not correctness labels.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith('cuda') else []):
                torch.manual_seed(67123)
                with torch.no_grad():
                    for x in va.split(batch):
                        loss = objective(model, x, pretrain)
                        if loss is not None: vals.append((loss.item(), len(x)))
            val = float(np.average([a for a,b in vals], weights=[b for a,b in vals]))
            trace.append(dict(phase=phase, epoch=epoch, train=float(np.mean(values)), validation=val))
            if val < best-1e-5:
                best=val; state=copy.deepcopy(model.state_dict()); stale=0; best_epoch=epoch
            else: stale+=1
            if stale>=config['patience'] and phase!='svdd': break
        if phase!='svdd': model.load_state_dict(state)
        # SVDD uses fixed final epoch; validation norm alone rewards collapse.
    model.eval()
    return model, dict(trace=trace, best_validation=best, best_epoch=best_epoch,
                       selection='fixed final epoch' if kind=='deep_svdd' else 'minimum unlabeled validation objective',
                       parameters=sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def neural_losses(model, z, batch=128, device='cpu'):
    model.eval(); result=[]; torch.manual_seed(18221)
    for start in range(0, len(z), batch):
        x = torch.tensor(z[start:start+batch], device=device)
        if model.kind in ['gru', 'causal_transformer']:
            loss = model.ce(model.sequence_logits(x), x)
        elif model.kind == 'masked_transformer':
            loss = torch.empty_like(x, dtype=torch.float32)
            for l in range(x.shape[1]):
                mask = torch.zeros_like(x, dtype=torch.bool); mask[:, l]=True
                loss[:, l] = model.ce(model.sequence_logits(x, mask), x)[:, l]
        elif model.kind == 'suffix_contrastive':
            loss = model.classifier(model.encoder(model.onehot(x)))
        elif model.kind == 'deep_svdd':
            loss = (model.encoder(model.onehot(x))-model.center).square().sum(1, keepdim=True)
        else:
            parts=[]
            for _ in range(4 if model.kind=='vae' else 1):
                ce, kl = model.ae_loss(x, sample=model.kind=='vae')
                parts.append(ce+kl[:, None])
            loss = torch.stack(parts).mean(0)
        result.append(loss.cpu().numpy())
    return np.concatenate(result)
