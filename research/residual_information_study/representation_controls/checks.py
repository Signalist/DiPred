"""Small CPU checks for the representation-specific publication policies."""
import torch

from research.observed_residual_feedback.core import FIELD, MEMBERS
from research.residual_information_study.representation_controls.model import (
    ARMS, ControlPublication, spatial_mean,
)


def check():
    # Asymmetric planes demonstrate spatial replicate padding and isolation
    # across the channel and physical frame axes.
    field = torch.arange(18, dtype=torch.float64).reshape(1, 2, 3, 3)
    pooled = spatial_mean(field)
    expected = sum(field[..., iy[:, None], ix[None, :]]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   for iy, ix in [((torch.arange(3) + dy).clamp(0, 2),
                                  (torch.arange(3) + dx).clamp(0, 2))]) / 9
    torch.testing.assert_close(pooled, expected)
    initial = torch.ones(FIELD)
    noise = torch.ones((MEMBERS, *FIELD))
    for arm in ARMS:
        stream = ControlPublication(initial, arm)
        seen = []

        def mock(history, noise_array, memory):
            seen.append((history, memory))
            return noise_array * 2, noise_array * 3

        stream.issue(mock, noise)
        assert not seen[0][1].available
        assert not seen[0][1].first.count_nonzero()
        try:
            stream.issue(mock, noise)
        except RuntimeError:
            pass
        else:
            raise AssertionError('A pending publication was overwritten')
        stream.reveal(initial * 7)
        assert torch.equal(stream.memory.residual, initial * 5)
        assert torch.equal(stream.memory.correction, initial * 3)
        stream.issue(mock, noise)
        assert seen[-1][1].available
        assert torch.equal(seen[-1][0], initial * 7)
        expected_memory = {'C0': (0., 0.), 'CH': (1., 0.), 'CS': (5., 3.)}[arm]
        assert torch.equal(seen[-1][1].first, initial * expected_memory[0])
        assert torch.equal(seen[-1][1].second, initial * expected_memory[1])
        stream.reveal(initial * 11)
        stream.issue(mock, noise)
        if arm == 'CH':
            assert torch.equal(seen[-1][1].first, initial * 7)


if __name__ == '__main__':
    check()
