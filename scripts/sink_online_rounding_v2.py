"""Numerical retry only when a tiny online update fails the original energy gate."""
from run_sink_energy_matched_v3 import MatchedTransform as OriginalTransform, selftest


class MatchedTransform(OriginalTransform):
    def __call__(self,h):
        try:
            out=super().__call__(h)
            self.metrics['strict_relative_retry']=False
            return out
        except AssertionError:
            m=self.metrics
            if not (self.direction is not None and m and m['same_active_mask'] and
                    m['all_tokens_within_tolerance'] and
                    m['relative_total_energy_error']>self.cfg['relative_total_energy_tolerance']):
                raise
            # Keep every scientific quantity and outer acceptance gate fixed.
            # The 1e-7 internal absolute stopping floor can dominate tiny targets.
            cfg=dict(self.cfg,internal_absolute_norm_target=0.)
            retry=OriginalTransform(self.axis,self.tau,self.direction,cfg,h.device)
            out=retry(h)
            self.arrays=retry.arrays;self.metrics=retry.metrics
            self.metrics.update(strict_relative_retry=True,original_energy_error=m['relative_total_energy_error'])
            return out
