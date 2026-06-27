"""
tests/test_attack.py

Unit tests for every attack primitive in neurons/miner.py.
No Bittensor dependency — tests the math directly.
"""

from __future__ import annotations

import torch
import pytest

from neurons.miner import (
    _cw_margin_loss,
    _mi_fgsm,
    _rank_target_classes,
    _fast_flip_check,
    _binary_search_min_norm,
    _multi_target_search,
    _untargeted_fallback,
    _enforce_norm_floor,
    _effective_epsilon,
    precompute_epsilon,
    MAX_LINF_DELTA,
    MIN_LINF_DELTA,
    NORM_FLOOR_SAFETY,
)
from perturbnet.image_io import decode_image_b64
from perturbnet.model import logits_for_images, predict_index, resolve_target_index


# ---------------------------------------------------------------------------
# _effective_epsilon  [V1]
# ---------------------------------------------------------------------------

class TestEffectiveEpsilon:
    def test_caps_at_max_linf_delta(self):
        assert _effective_epsilon(0.19) == MAX_LINF_DELTA

    def test_caps_at_max_exactly(self):
        assert _effective_epsilon(0.12) == MAX_LINF_DELTA

    def test_passes_through_below_cap(self):
        assert _effective_epsilon(0.08) == pytest.approx(0.08)

    def test_caps_very_large(self):
        assert _effective_epsilon(1.0) == MAX_LINF_DELTA

    def test_small_epsilon_unchanged(self):
        assert _effective_epsilon(0.06) == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# precompute_epsilon  [V2]
# ---------------------------------------------------------------------------

class TestPrecomputeEpsilon:
    def test_returns_float(self):
        eps = precompute_epsilon(netuid=26, block=1000)
        assert isinstance(eps, float)

    def test_in_valid_range(self):
        for block in [1, 100, 9999, 1_000_000]:
            eps = precompute_epsilon(netuid=26, block=block)
            assert 0.06 <= eps <= 0.199, f"epsilon={eps} out of range at block={block}"

    def test_deterministic(self):
        eps1 = precompute_epsilon(26, 5000)
        eps2 = precompute_epsilon(26, 5000)
        assert eps1 == eps2

    def test_different_blocks_different_eps(self):
        eps1 = precompute_epsilon(26, 100)
        eps2 = precompute_epsilon(26, 101)
        assert eps1 != eps2


# ---------------------------------------------------------------------------
# _cw_margin_loss
# ---------------------------------------------------------------------------

class TestCwMarginLoss:
    def test_targeted_returns_scalar(self, model, device):
        dummy = torch.zeros(1, 3, 456, 456, device=device)
        with torch.no_grad():
            logits = logits_for_images(model, dummy)
        n = logits.shape[1]
        loss = _cw_margin_loss(logits, target_idx=1, targeted=True, n_classes=n, device=device)
        assert loss.shape == torch.Size([])

    def test_untargeted_returns_scalar(self, model, device):
        dummy = torch.zeros(1, 3, 456, 456, device=device)
        with torch.no_grad():
            logits = logits_for_images(model, dummy)
        n = logits.shape[1]
        loss = _cw_margin_loss(logits, target_idx=0, targeted=False, n_classes=n, device=device)
        assert loss.shape == torch.Size([])

    def test_targeted_is_differentiable(self, model, device):
        dummy = torch.zeros(1, 3, 456, 456, device=device, requires_grad=True)
        logits = logits_for_images(model, dummy)
        n = logits.shape[1]
        loss = _cw_margin_loss(logits, target_idx=1, targeted=True, n_classes=n, device=device)
        loss.backward()
        assert dummy.grad is not None

    def test_targeted_loss_decreases_after_step(self, model, device, dog_image_b64):
        """Targeted loss should decrease after one MI-FGSM step toward target class."""
        clean = decode_image_b64(dog_image_b64).to(device)
        with torch.no_grad():
            logits_before = logits_for_images(model, clean.unsqueeze(0))
        n = logits_before.shape[1]
        true_idx = int(logits_before.argmax().item())
        # Pick a different target
        target_idx = (true_idx + 1) % n
        loss_before = _cw_margin_loss(
            logits_before, target_idx, targeted=True, n_classes=n, device=device
        ).item()

        adv = _mi_fgsm(model, clean, target_idx, epsilon=0.05, steps=5,
                       targeted=True, device=device)
        with torch.no_grad():
            logits_after = logits_for_images(model, adv.unsqueeze(0))
        loss_after = _cw_margin_loss(
            logits_after, target_idx, targeted=True, n_classes=n, device=device
        ).item()

        assert loss_after <= loss_before, (
            f"Targeted CW loss should decrease after attack step: "
            f"before={loss_before:.4f} after={loss_after:.4f}"
        )


# ---------------------------------------------------------------------------
# _mi_fgsm constraints
# ---------------------------------------------------------------------------

class TestMiFgsm:
    def test_output_shape_unchanged(self, model, device, dog_image_b64):
        clean = decode_image_b64(dog_image_b64).to(device)
        adv = _mi_fgsm(model, clean, target_idx=0, epsilon=0.05, steps=5, device=device)
        assert adv.shape == clean.shape

    def test_pixels_in_range(self, model, device, dog_image_b64):
        clean = decode_image_b64(dog_image_b64).to(device)
        adv = _mi_fgsm(model, clean, target_idx=0, epsilon=0.05, steps=5, device=device)
        assert float(adv.min()) >= 0.0 - 1e-6
        assert float(adv.max()) <= 1.0 + 1e-6

    def test_linf_norm_within_epsilon(self, model, device, dog_image_b64):
        clean = decode_image_b64(dog_image_b64).to(device)
        epsilon = 0.08
        adv = _mi_fgsm(model, clean, target_idx=0, epsilon=epsilon, steps=10, device=device)
        norm = float((adv - clean).abs().max().item())
        assert norm <= epsilon + 1e-5, f"norm={norm:.6f} exceeds epsilon={epsilon}"

    def test_produces_perturbation(self, model, device, dog_image_b64):
        clean = decode_image_b64(dog_image_b64).to(device)
        adv = _mi_fgsm(model, clean, target_idx=0, epsilon=0.05, steps=5, device=device)
        norm = float((adv - clean).abs().max().item())
        assert norm > 0.0, "MI-FGSM produced zero perturbation"

    def test_gradient_does_not_accumulate(self, model, device, dog_image_b64):
        """Output should be a detached tensor with no grad_fn."""
        clean = decode_image_b64(dog_image_b64).to(device)
        adv = _mi_fgsm(model, clean, target_idx=0, epsilon=0.05, steps=3, device=device)
        assert adv.grad_fn is None, "adv should be detached"


# ---------------------------------------------------------------------------
# _rank_target_classes  [V9]
# ---------------------------------------------------------------------------

class TestRankTargetClasses:
    def test_returns_list(self, model, device, dog_image_b64):
        clean = decode_image_b64(dog_image_b64).to(device)
        candidates = _rank_target_classes(model, clean, true_idx=0, top_k=3)
        assert isinstance(candidates, list)

    def test_correct_length(self, model, device, dog_image_b64):
        clean = decode_image_b64(dog_image_b64).to(device)
        candidates = _rank_target_classes(model, clean, true_idx=0, top_k=3)
        assert len(candidates) == 3

    def test_excludes_true_idx(self, model, device, dog_image_b64, dog_true_idx):
        clean = decode_image_b64(dog_image_b64).to(device)
        candidates = _rank_target_classes(model, clean, true_idx=dog_true_idx, top_k=5)
        assert dog_true_idx not in candidates, "true_idx must not appear in candidates"

    def test_candidates_are_valid_class_indices(self, model, device, dog_image_b64):
        clean = decode_image_b64(dog_image_b64).to(device)
        with torch.no_grad():
            from perturbnet.model import logits_for_images
            logits = logits_for_images(model, clean.unsqueeze(0))
        n_classes = logits.shape[1]
        candidates = _rank_target_classes(model, clean, true_idx=0, top_k=3)
        for c in candidates:
            assert 0 <= c < n_classes


# ---------------------------------------------------------------------------
# _enforce_norm_floor
# ---------------------------------------------------------------------------

class TestEnforceNormFloor:
    def test_nudges_below_floor(self, dog_image_b64, device):
        clean = decode_image_b64(dog_image_b64).to(device)
        # Create perturbation with norm exactly 0.001 (below min_delta=0.002)
        adv = clean.clone()
        adv[0, 0, 0] = (clean[0, 0, 0] + 0.001).clamp(0, 1)
        result = _enforce_norm_floor(adv, clean, min_delta=0.002)
        norm = float((result - clean).abs().max().item())
        assert norm >= 0.002, f"norm after floor enforcement={norm:.6f} still below min_delta"

    def test_does_not_modify_above_floor(self, dog_image_b64, device):
        clean = decode_image_b64(dog_image_b64).to(device)
        adv = clean.clone()
        adv[0, 0, 0] = (clean[0, 0, 0] + 0.01).clamp(0, 1)
        original_norm = float((adv - clean).abs().max().item())
        result = _enforce_norm_floor(adv, clean, min_delta=0.002)
        new_norm = float((result - clean).abs().max().item())
        assert abs(new_norm - original_norm) < 1e-5

    def test_output_pixels_in_range(self, dog_image_b64, device):
        clean = decode_image_b64(dog_image_b64).to(device)
        adv = clean.clone()
        adv[0, 0, 0] = (clean[0, 0, 0] + 0.001).clamp(0, 1)
        result = _enforce_norm_floor(adv, clean, min_delta=0.002)
        assert float(result.min()) >= 0.0 - 1e-6
        assert float(result.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# _binary_search_min_norm  (core score driver)
# ---------------------------------------------------------------------------

class TestBinarySearchMinNorm:
    def test_returns_valid_flip(self, model, device, dog_image_b64, dog_true_idx):
        clean = decode_image_b64(dog_image_b64).to(device)
        candidates = _rank_target_classes(model, clean, true_idx=dog_true_idx, top_k=1)
        target_idx = candidates[0]

        adv, norm, pred = _binary_search_min_norm(
            model=model, clean=clean,
            true_idx=dog_true_idx, target_idx=target_idx,
            epsilon=0.12, min_delta=0.002,
            inner_steps=10, binary_steps=8,
            decay=1.0, device=device,
        )
        # Must satisfy all three validator norm gates
        assert norm >= 0.002, f"norm={norm:.6f} below min_delta"
        assert norm <= 0.12 + 1e-5, f"norm={norm:.6f} exceeds epsilon"
        assert adv.shape == clean.shape
        assert float(adv.min()) >= 0.0 - 1e-6
        assert float(adv.max()) <= 1.0 + 1e-6

    def test_norm_smaller_than_naive_pgd(self, model, device, dog_image_b64, dog_true_idx):
        """Binary search should find a smaller norm than naive full-epsilon PGD."""
        import torch.nn.functional as F
        clean = decode_image_b64(dog_image_b64).to(device)
        candidates = _rank_target_classes(model, clean, true_idx=dog_true_idx, top_k=1)
        target_idx = candidates[0]
        epsilon = 0.12

        # Naive baseline: single-pass PGD at full epsilon
        naive_adv = _mi_fgsm(
            model, clean, target_idx, epsilon=epsilon, steps=10, device=device
        )
        naive_norm = float((naive_adv - clean).abs().max().item())

        # Binary search
        adv, norm, pred = _binary_search_min_norm(
            model=model, clean=clean,
            true_idx=dog_true_idx, target_idx=target_idx,
            epsilon=epsilon, min_delta=0.002,
            inner_steps=10, binary_steps=8,
            decay=1.0, device=device,
        )
        if pred != dog_true_idx:
            assert norm <= naive_norm + 1e-4, (
                f"Binary search norm={norm:.4f} should be ≤ naive norm={naive_norm:.4f}"
            )


# ---------------------------------------------------------------------------
# _multi_target_search  [V9]
# ---------------------------------------------------------------------------

class TestMultiTargetSearch:
    def test_returns_tuple(self, model, device, dog_image_b64, dog_true_idx):
        clean = decode_image_b64(dog_image_b64).to(device)
        candidates = _rank_target_classes(model, clean, true_idx=dog_true_idx, top_k=3)
        adv, norm, pred = _multi_target_search(
            model=model, clean=clean,
            true_idx=dog_true_idx, candidates=candidates,
            epsilon=0.12, min_delta=0.002,
            inner_steps=8, binary_steps=5,
            decay=1.0, device=device,
        )
        assert isinstance(adv, torch.Tensor)
        assert isinstance(norm, float)
        assert isinstance(pred, int)

    def test_multi_target_norm_le_single_target(self, model, device, dog_image_b64, dog_true_idx):
        """Multi-target search should find norm ≤ single-target search."""
        clean = decode_image_b64(dog_image_b64).to(device)
        candidates = _rank_target_classes(model, clean, true_idx=dog_true_idx, top_k=3)

        # Single target
        adv1, norm1, pred1 = _binary_search_min_norm(
            model=model, clean=clean,
            true_idx=dog_true_idx, target_idx=candidates[0],
            epsilon=0.12, min_delta=0.002,
            inner_steps=8, binary_steps=5,
            decay=1.0, device=device,
        )
        # Multi target
        adv3, norm3, pred3 = _multi_target_search(
            model=model, clean=clean,
            true_idx=dog_true_idx, candidates=candidates,
            epsilon=0.12, min_delta=0.002,
            inner_steps=8, binary_steps=5,
            decay=1.0, device=device,
        )
        if pred1 != dog_true_idx and pred3 != dog_true_idx:
            assert norm3 <= norm1 + 1e-4, (
                f"Multi-target norm={norm3:.4f} should be ≤ single-target norm={norm1:.4f}"
            )

    def test_output_satisfies_validator_gates(self, model, device, dog_image_b64, dog_true_idx):
        clean = decode_image_b64(dog_image_b64).to(device)
        candidates = _rank_target_classes(model, clean, true_idx=dog_true_idx, top_k=3)
        adv, norm, pred = _multi_target_search(
            model=model, clean=clean,
            true_idx=dog_true_idx, candidates=candidates,
            epsilon=0.12, min_delta=0.002,
            inner_steps=10, binary_steps=6,
            decay=1.0, device=device,
        )
        assert adv.shape == clean.shape
        assert float(adv.min()) >= 0.0 - 1e-6
        assert float(adv.max()) <= 1.0 + 1e-6
        assert norm <= 0.12 + 1e-5
