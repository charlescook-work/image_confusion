"""
tests/test_forward.py

Tests the full forward() handler with a real synapse but mocked
Bittensor network. Verifies all 6 validator zero-score gates are passed.
"""

from __future__ import annotations

import asyncio
import pytest
import torch

from perturbnet.image_io import decode_image_b64
from perturbnet.model import predict_label, normalize_prediction_label


def run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestForwardBasic:
    def test_forward_sets_perturbed_field(self, mock_miner, make_synapse):
        syn = make_synapse(epsilon=0.10)
        run(mock_miner.forward(syn))
        assert syn.perturbed_image_b64 is not None
        assert len(syn.perturbed_image_b64) > 0

    def test_forward_returns_different_image(self, mock_miner, make_synapse):
        syn = make_synapse(epsilon=0.10)
        run(mock_miner.forward(syn))
        assert syn.perturbed_image_b64 != syn.clean_image_b64

    def test_forward_increments_round_count(self, mock_miner, make_synapse):
        count_before = mock_miner._local_round_count
        syn = make_synapse(epsilon=0.10)
        run(mock_miner.forward(syn))
        assert mock_miner._local_round_count == count_before + 1


class TestValidatorGates:
    """
    Each test directly maps to one of the 6 validator zero-score gates
    in neurons/validator.py verify_and_score().
    """

    def test_gate_shape_unchanged(self, mock_miner, make_synapse):
        """Gate 1: shape mismatch → score 0."""
        syn = make_synapse(epsilon=0.10)
        clean = decode_image_b64(syn.clean_image_b64)
        run(mock_miner.forward(syn))
        adv = decode_image_b64(syn.perturbed_image_b64)
        assert adv.shape == clean.shape, (
            f"Shape changed: clean={clean.shape} adv={adv.shape}"
        )

    def test_gate_pixels_in_range(self, mock_miner, make_synapse):
        """Gate 2: pixel out of range → score 0."""
        syn = make_synapse(epsilon=0.10)
        run(mock_miner.forward(syn))
        adv = decode_image_b64(syn.perturbed_image_b64)
        assert float(adv.min()) >= 0.0 - 1e-5, f"min pixel={adv.min():.6f} < 0"
        assert float(adv.max()) <= 1.0 + 1e-5, f"max pixel={adv.max():.6f} > 1"

    def test_gate_norm_above_min_delta(self, mock_miner, make_synapse):
        """Gate 3: norm < min_delta → score 0."""
        syn = make_synapse(epsilon=0.10)
        clean = decode_image_b64(syn.clean_image_b64)
        run(mock_miner.forward(syn))
        adv = decode_image_b64(syn.perturbed_image_b64)
        norm = float((adv - clean).abs().max().item())
        assert norm >= 0.002, f"norm={norm:.6f} is below min_delta=0.002"

    def test_gate_norm_below_max_delta(self, mock_miner, make_synapse):
        """Gate 4: norm > min(epsilon, 0.12) → score 0."""
        epsilon = 0.10
        syn = make_synapse(epsilon=epsilon)
        clean = decode_image_b64(syn.clean_image_b64)
        run(mock_miner.forward(syn))
        adv = decode_image_b64(syn.perturbed_image_b64)
        norm = float((adv - clean).abs().max().item())
        effective_ceil = min(epsilon, 0.12)
        assert norm <= effective_ceil + 1e-5, (
            f"norm={norm:.6f} exceeds effective ceil={effective_ceil}"
        )

    def test_gate_label_flipped(self, mock_miner, make_synapse, device):
        """Gate 5: predicted label == true_label → score 0."""
        syn = make_synapse(epsilon=0.12)
        run(mock_miner.forward(syn))
        adv = decode_image_b64(syn.perturbed_image_b64).to(device)
        adv_label = normalize_prediction_label(
            predict_label(mock_miner.model, adv)
        )
        assert adv_label != syn.true_label, (
            f"Label was NOT flipped: adv_label={adv_label} == true_label={syn.true_label}"
        )

    def test_gate_response_not_empty(self, mock_miner, make_synapse):
        """Gate 6: empty response → validator treats as failure."""
        syn = make_synapse(epsilon=0.10)
        run(mock_miner.forward(syn))
        assert syn.perturbed_image_b64 is not None
        assert len(syn.perturbed_image_b64) > 100  # not a trivially short string


class TestEdgeCases:
    def test_unsupported_norm_type_returns_clean(self, mock_miner, make_synapse):
        """Validator only scores Linf — other norms should return clean image."""
        syn = make_synapse(epsilon=0.10)
        syn.norm_type = "L2"
        run(mock_miner.forward(syn))
        assert syn.perturbed_image_b64 == syn.clean_image_b64

    def test_high_epsilon_still_passes_gates(self, mock_miner, make_synapse):
        """
        [V1] epsilon=0.19 raw → effective cap is 0.12.
        Norm must still be within [0.002, 0.12].
        """
        syn = make_synapse(epsilon=0.19)
        clean = decode_image_b64(syn.clean_image_b64)
        run(mock_miner.forward(syn))
        adv = decode_image_b64(syn.perturbed_image_b64)
        norm = float((adv - clean).abs().max().item())
        assert norm >= 0.002
        assert norm <= 0.12 + 1e-5

    def test_low_epsilon_still_passes_gates(self, mock_miner, make_synapse):
        """epsilon=0.06 (minimum possible) must still produce a valid perturbation."""
        syn = make_synapse(epsilon=0.06)
        clean = decode_image_b64(syn.clean_image_b64)
        run(mock_miner.forward(syn))
        adv = decode_image_b64(syn.perturbed_image_b64)
        norm = float((adv - clean).abs().max().item())
        assert norm >= 0.002
        assert norm <= 0.06 + 1e-5
