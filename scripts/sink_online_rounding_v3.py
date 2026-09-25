"""Extend failed numerical solves only; preserve successful v2 operations exactly."""
from sink_online_rounding_v2 import MatchedTransform as PreviousTransform, selftest
from run_sink_energy_matched_v3 import MatchedTransform as ReferenceTransform


def preflight(cfg, device='cuda'):
    import numpy as np
    import torch
    from pathlib import Path
    from revision_common import sha
    base = Path('/lambda/nfs/dami/hss/sink-next-20260925')
    v = dict(np.load('/lambda/nfs/dami/hss/sink-energy-matched-20260925-v3/directions.npz'))
    args = (v['axis'], float(v['tau']), v['orthogonal_controls'][5], cfg, device)
    results = []
    for name in ['exp4', 'exp4-v2']:
        path = base/name/'rounding_diagnostic/failed_activation.npz'
        h = torch.as_tensor(np.load(path)['h'], device=device).to(torch.bfloat16)
        new = MatchedTransform(*args); after = new(h)
        if name == 'exp4':
            old = PreviousTransform(*args)
            assert torch.equal(after, old(h)), 'Previously successful operation changed'
        else:
            old = PreviousTransform(*args)
            try:
                old(h)
            except AssertionError:
                pass
            else:
                raise AssertionError('Expected v2 failure did not reproduce')
        results.append(dict(source=str(path), sha256=sha(path), metrics=new.metrics))
    h = torch.zeros_like(h)
    assert torch.equal(MatchedTransform(*args)(h), h)
    return dict(passed=True, successful_v2_preserved=True, inactive_unchanged=True, cases=results)


class MatchedTransform(PreviousTransform):
    def __call__(self, h):
        try:
            return super().__call__(h)
        except AssertionError:
            m = self.metrics
            if not (self.direction is not None and m and m['same_active_mask'] and
                    m['all_tokens_within_tolerance'] and
                    m['relative_total_energy_error'] > self.cfg['relative_total_energy_tolerance']):
                raise
            # Same rounding algorithm, objective and acceptance gates. The
            # additional iterations are attempted only after v2 has failed.
            for steps in [128, 512, 2048]:
                cfg = dict(self.cfg, internal_absolute_norm_target=0., rounding_repair_steps=steps)
                retry = ReferenceTransform(self.axis, self.tau, self.direction, cfg, h.device)
                try:
                    out = retry(h)
                except AssertionError:
                    if steps == 2048:
                        raise
                    continue
                self.arrays = retry.arrays
                self.metrics = retry.metrics
                self.metrics.update(extended_rounding_retry=True, rounding_iteration_budget=steps,
                                    original_energy_error=m['relative_total_energy_error'])
                return out
