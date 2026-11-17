"""Causal ridge propagation in the span of revealed full20 residual blocks.

Fit R_next = Y (X^T X + ridge I)^-1 X^T R_last separately per
trajectory and velocity. This low-rank, generally non-diagonal field operator
uses only actual previously issued raw forecast errors. A second causal
scalar can calibrate its reliability using earlier issued proposals.
"""
import torch


class ResidualSubspaceStream:
    def __init__(self, gated=True, ridge=.01, memory=8):
        assert ridge > 0 and memory >= 2
        self.gated, self.ridge, self.memory = gated, ridge, memory
        self.previous_raw = self.previous_proposal = None
        self.errors = []
        self.gate_xy = self.gate_xx = None

    @torch.no_grad()
    def issue(self, history, raw):
        assert history.shape == raw.shape and raw.ndim == 5 and raw.shape[2] == 20
        assert torch.isfinite(history).all() and torch.isfinite(raw).all()
        device, dtype = raw.device, raw.dtype
        h, p = history.detach().cpu().double(), raw.detach().cpu().double()
        if self.previous_raw is not None:
            assert self.previous_raw.shape == h.shape
            error = h - self.previous_raw
            self.errors.append(error)
            self.errors = self.errors[-self.memory:]
            if self.previous_proposal is not None:
                proposal = self.previous_proposal
                self.gate_xy += (proposal*error).sum((2, 3, 4))
                self.gate_xx += proposal.square().sum((2, 3, 4))
        if self.gate_xy is None:
            self.gate_xy = torch.zeros(p.shape[:2], dtype=torch.float64)
            self.gate_xx = torch.zeros_like(self.gate_xy)
        proposal = torch.zeros_like(p)
        if len(self.errors) >= 2:
            states = torch.stack(self.errors).flatten(3)  # n,B,C,space-time
            for b in range(p.shape[0]):
                for c in range(p.shape[1]):
                    x = states[:-1, b, c].T
                    y = states[1:, b, c].T
                    last = states[-1, b, c]
                    gram = x.T@x
                    energy = gram.diagonal().mean()
                    if energy <= 1e-24:
                        continue
                    weights = torch.linalg.solve(
                        gram+self.ridge*energy*torch.eye(len(gram), dtype=gram.dtype), x.T@last)
                    out = y@weights
                    # Bound extrapolation energy using the latest revealed error.
                    out *= torch.clamp(last.norm()/out.norm().clamp_min(1e-24), max=1.)
                    proposal[b, c] = out.reshape_as(p[b, c])
        gain = (self.gate_xy/((1+self.ridge)*self.gate_xx).clamp_min(1e-24)).clamp(0., 1.)
        if not self.gated:
            gain = torch.ones_like(gain)
        corrected = p + gain[:, :, None, None, None]*proposal
        # Store the unscaled proposal even when the current online gain is zero.
        self.previous_raw, self.previous_proposal = p.clone(), proposal.clone()
        return corrected.to(device=device, dtype=dtype), dict(
            revealed_error_blocks=len(self.errors), gain=gain.tolist(),
            proposal_rms=proposal.square().mean((2, 3, 4)).sqrt().tolist())
