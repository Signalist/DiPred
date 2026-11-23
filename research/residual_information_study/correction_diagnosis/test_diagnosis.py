"""Small CPU checks of B's intervention and diagnostic calculations."""

import unittest

import torch
from torch import nn

from research.observed_residual_feedback.core import Publication
from research.residual_information_study.correction_diagnosis.stage import (
    GROUPS,
    inner_product_diagnosis,
    loss_gradient_rows,
    masked_backbone,
)


class RoutingProbe:
    """Two branches implement the documented distinction between L and J."""

    arm = "J"

    def __call__(self, history, noise, memory):
        backbone = history + noise + (memory.residual if self.arm == "J" else 0)
        correction = 2 * memory.correction + backbone
        return backbone, correction


class GradientProbe(nn.Module):
    """Provide tiny differentiable tensors for all three weight groups."""

    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.net = nn.Module()
        self.base.net.up2 = nn.Linear(2, 2)
        self.base.net.out = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2)
        self.projection = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)

    def forward(self, noise):
        features = self.projection(self.encoder(noise))
        backbone = self.base.net.out(self.base.net.up2(noise + features))
        correction = self.head(backbone + features)
        return backbone.reshape(3, 1, 2), correction.reshape(3, 1, 2)


class DiagnosisTests(unittest.TestCase):
    def test_mse_identity_and_undefined_cosine(self):
        residual = torch.tensor([1., -2., 3.], dtype=torch.float64)
        correction = -residual
        values = inner_product_diagnosis(residual, correction)
        self.assertAlmostEqual(values["direction_cosine"], -1.)
        self.assertAlmostEqual(values["mse_change"], 14.)
        self.assertAlmostEqual(values["mse_change"], values["correction_mse"] + values["alignment_contribution"])
        self.assertIsNone(inner_product_diagnosis(residual, torch.zeros_like(residual))["direction_cosine"])
        self.assertIsNone(inner_product_diagnosis(torch.zeros_like(residual), correction)["direction_cosine"])

    def test_mask_does_not_write_publication(self):
        history = torch.zeros(2, 20, 32, 64)
        noise = torch.ones(3, *history.shape)
        model = RoutingProbe()
        stream = Publication(history)
        stream.issue(model, noise)
        stream.reveal(history + 2.)
        factual_backbone, factual_correction = stream.issue(model, noise)
        pending = stream.pending
        prior = stream.memory
        masked_forecast, masked_correction = masked_backbone(model, stream, noise)
        self.assertEqual(model.arm, "J")
        self.assertIs(stream.pending, pending)
        self.assertIs(stream.memory, prior)
        torch.testing.assert_close(factual_backbone - masked_forecast, torch.ones_like(factual_backbone))
        torch.testing.assert_close(masked_correction - masked_forecast, 2 * prior.correction.expand_as(masked_forecast))
        torch.testing.assert_close(factual_correction - factual_backbone, 2 * prior.correction.expand_as(factual_backbone))

    def test_loss_gradient_disconnect_and_auxiliary_value(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = GradientProbe().double()
        noise = torch.tensor([[1., -.5], [-.3, .9], [1.3, .3]], dtype=torch.float64)
        truth = torch.tensor([[.9, -1.2]], dtype=noise.dtype)
        backbone, correction = model(noise)
        rows = {row["group"]: row for row in loss_gradient_rows(model, backbone, correction, truth, {})}
        self.assertEqual(set(rows), set(GROUPS))
        for row in rows.values():
            self.assertAlmostEqual(row["auxiliary_unweighted_value"], row["ensemble_mean_smse"])
            self.assertGreater(row["gradient_l2_norm"]["weighted_auxiliary"], 0.)
        head = rows["head"]
        self.assertEqual(head["gradient_l2_norm"]["weighted_backbone_score_u"], 0.)
        self.assertIsNone(head["gradient_cosine"]["final_score_u vs weighted_backbone_score_u"])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
