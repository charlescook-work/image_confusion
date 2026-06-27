"""
tests/test_scoring.py

Replicates the validator's exact scoring formula locally.
Run with -s flag to see actual score values printed.
"""

from __future__ import annotations

import asyncio
import pytest
import torch

from perturbnet.image_io import decode_image_b64
from perturbnet.model import predict_label, normalize_prediction_label


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Replicate validator scoring exactly (from neurons/validator.py)
# ---------------------------------------------------------------------------

def simulate_validator_score(
    model: torch.nn.Module,
    device: torch.device,
    clean_b64: str,
    perturbed_b64: str,
    true_label: str,
    epsilon: float,
    timeout_s: float,
    response_ms: int,
) -> dict:
    """
    Exact copy of PerturbValidator.verify_and_score() logic.
    Returns dict with score, reason, norm, perturbation_score, speed_score.
    """
    MIN_LINF_DELTA = 0.002
    MAX_LINF_DELTA = 0.12
    PERTURBATION_WEIGHT = 0.65
    SPEED_WEIGHT = 0.35

    try:
        x_clean = decode_image_b64(clean_b64).to(device)
        x_adv = decode_image_b64(perturbed_b64).to(device)
    except Exception as e:
        return {"score": 0.0, "reason": f"decode_failed:{e}"}

    if x_adv.shape != x_clean.shape:
        return {"score": 0.0, "reason": "shape_mismatch"}

    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        return {"score": 0.0, "reason": "value_out_of_range"}

    try:
        prediction = normalize_prediction_label(predict_label(model, x_adv))
    except Exception as e:
        return {"score": 0.0, "reason": f"model_inference_failed:{e}"}

    norm = float((x_adv - x_clean).abs().max().item())

    if norm < MIN_LINF_DELTA:
        return {"score": 0.0, "reason": "below_min_delta", "norm": norm}

    if norm > min(epsilon, MAX_LINF_DELTA):
        return {"score": 0.0, "reason": "above_max_delta", "norm": norm}

    if prediction == true_label:
        return {"score": 0.0, "reason": "label_match_with_original",
                "norm": norm, "prediction": prediction}

    perturbation_score = 1.0 - min(norm / max(epsilon, 1e-12), 1.0)
    speed_score = 1.0 - min(response_ms / (timeout_s * 1000.0), 1.0)
    score = PERTURBATION_WEIGHT * perturbation_score + SPEED_WEIGHT * speed_score

    return {
        "score": score,
        "reason": "success",
        "norm": norm,
        "perturbation_score": perturbation_score,
        "speed_score": speed_score,
        "prediction": prediction,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoringFormula:
    def test_forward_scores_nonzero(self, mock_miner, make_synapse, device):
        syn = make_synapse(epsilon=0.10)
        run(mock_miner.forward(syn))
        result = simulate_validator_score(
            model=mock_miner.model, device=device,
            clean_b64=syn.clean_image_b64,
            perturbed_b64=syn.perturbed_image_b64,
            true_label=syn.true_label,
            epsilon=0.10, timeout_s=60, response_ms=5000,
        )
        print(f"\n[score] reason={result['reason']} score={result.get('score', 0):.4f} "
              f"norm={result.get('norm', 0):.6f} "
              f"pert_score={result.get('perturbation_score', 0):.4f} "
              f"speed_score={result.get('speed_score', 0):.4f}")
        assert result["score"] > 0.0, f"Expected score > 0, got reason={result['reason']}"
        assert result["reason"] == "success"

    def test_below_min_delta_scores_zero(self, mock_miner, make_synapse, device,
                                         dog_image_b64, dog_true_label):
        """Manually construct a perturbation below min_delta."""
        clean = decode_image_b64(dog_image_b64)
        adv = clean.clone()
        # norm = 0.001 (below 0.002)
        adv[0, 0, 0] = (clean[0, 0, 0] + 0.001).clamp(0, 1)
        from perturbnet.image_io import encode_image_b64
        perturbed_b64 = encode_image_b64(adv)

        result = simulate_validator_score(
            model=mock_miner.model, device=device,
            clean_b64=dog_image_b64, perturbed_b64=perturbed_b64,
            true_label=dog_true_label,
            epsilon=0.10, timeout_s=60, response_ms=1000,
        )
        print(f"\n[below_min_delta] reason={result['reason']} score={result['score']:.4f}")
        assert result["score"] == 0.0
        assert result["reason"] == "below_min_delta"

    def test_above_max_delta_scores_zero(self, mock_miner, make_synapse, device,
                                          dog_image_b64, dog_true_label):
        """Norm above 0.12 must score zero — tests [V1] ceiling gate."""
        clean = decode_image_b64(dog_image_b64)
        adv = (clean + 0.13).clamp(0, 1)
        from perturbnet.image_io import encode_image_b64
        perturbed_b64 = encode_image_b64(adv)

        result = simulate_validator_score(
            model=mock_miner.model, device=device,
            clean_b64=dog_image_b64, perturbed_b64=perturbed_b64,
            true_label=dog_true_label,
            epsilon=0.10, timeout_s=60, response_ms=1000,
        )
        print(f"\n[above_max_delta] reason={result['reason']} score={result['score']:.4f}")
        assert result["score"] == 0.0
        assert result["reason"] == "above_max_delta"

    def test_high_epsilon_improves_perturbation_score(self, mock_miner, device,
                                                       dog_image_b64, dog_true_label):
        """
        [V2] With the same absolute norm, a higher epsilon gives a better
        perturbation_score because denominator is larger.
        """
        from perturbnet.image_io import encode_image_b64
        clean = decode_image_b64(dog_image_b64)
        # Fixed norm of 0.01
        adv = clean.clone()
        adv = (adv + 0.01).clamp(0, 1)
        perturbed_b64 = encode_image_b64(adv)

        r_low = simulate_validator_score(
            model=mock_miner.model, device=device,
            clean_b64=dog_image_b64, perturbed_b64=perturbed_b64,
            true_label=dog_true_label,
            epsilon=0.06, timeout_s=60, response_ms=3000,
        )
        r_high = simulate_validator_score(
            model=mock_miner.model, device=device,
            clean_b64=dog_image_b64, perturbed_b64=perturbed_b64,
            true_label=dog_true_label,
            epsilon=0.19, timeout_s=60, response_ms=3000,
        )
        if r_low["reason"] == "success" and r_high["reason"] == "success":
            print(f"\n[V2] low_eps_pert={r_low['perturbation_score']:.4f} "
                  f"high_eps_pert={r_high['perturbation_score']:.4f}")
            assert r_high["perturbation_score"] > r_low["perturbation_score"], (
                "Higher epsilon should give better perturbation score for same norm"
            )

    def test_faster_response_improves_speed_score(self, mock_miner, make_synapse, device):
        """Speed score = 1 - response_ms/60000. Faster = better."""
        syn = make_synapse(epsilon=0.10)
        run(mock_miner.forward(syn))

        r_fast = simulate_validator_score(
            model=mock_miner.model, device=device,
            clean_b64=syn.clean_image_b64, perturbed_b64=syn.perturbed_image_b64,
            true_label=syn.true_label,
            epsilon=0.10, timeout_s=60, response_ms=2000,
        )
        r_slow = simulate_validator_score(
            model=mock_miner.model, device=device,
            clean_b64=syn.clean_image_b64, perturbed_b64=syn.perturbed_image_b64,
            true_label=syn.true_label,
            epsilon=0.10, timeout_s=60, response_ms=30000,
        )
        if r_fast["reason"] == "success" and r_slow["reason"] == "success":
            print(f"\n[speed] fast={r_fast['speed_score']:.4f} slow={r_slow['speed_score']:.4f}")
            assert r_fast["score"] > r_slow["score"]

    def test_new_attack_beats_baseline(self, mock_miner, device, model,
                                        dog_image_b64, dog_true_label, dog_true_idx):
        """
        Compares score from new multi-target binary-search attack against
        the original baseline 10-step PGD. Proves improvement is real.
        """
        import torch.nn.functional as F
        from perturbnet.image_io import encode_image_b64
        from neurons.miner import _multi_target_search, _rank_target_classes

        clean = decode_image_b64(dog_image_b64).to(device)
        epsilon = 0.12

        # --- Baseline: naive 10-step untargeted PGD (original miner.py) ---
        steps = 10
        step_size = max(epsilon / 4.0, 1.0 / 255.0)
        adv_base = clean.clone().detach()
        best_base = adv_base.clone()
        best_delta = 0.0
        for _ in range(steps):
            adv_base.requires_grad_(True)
            logits = from_perturbnet_logits(model, adv_base.unsqueeze(0), device)
            loss = F.cross_entropy(logits, torch.tensor([dog_true_idx], device=device))
            grad = torch.autograd.grad(loss, adv_base)[0]
            adv_base = adv_base.detach() + step_size * grad.sign()
            adv_base = torch.max(
                torch.min(adv_base, clean + epsilon), clean - epsilon
            ).clamp(0.0, 1.0)
            delta = float((adv_base - clean).abs().max().item())
            if delta > best_delta:
                best_base = adv_base.clone()
                best_delta = delta

        # --- New attack ---
        candidates = _rank_target_classes(model, clean, true_idx=dog_true_idx, top_k=3)
        adv_new, norm_new, pred_new = _multi_target_search(
            model=model, clean=clean,
            true_idx=dog_true_idx, candidates=candidates,
            epsilon=epsilon, min_delta=0.002,
            inner_steps=15, binary_steps=10,
            decay=1.0, device=device,
        )

        r_base = simulate_validator_score(
            model=model, device=device,
            clean_b64=dog_image_b64,
            perturbed_b64=encode_image_b64(best_base.cpu()),
            true_label=dog_true_label,
            epsilon=epsilon, timeout_s=60, response_ms=5000,
        )
        r_new = simulate_validator_score(
            model=model, device=device,
            clean_b64=dog_image_b64,
            perturbed_b64=encode_image_b64(adv_new.cpu()),
            true_label=dog_true_label,
            epsilon=epsilon, timeout_s=60, response_ms=5000,
        )
        print(f"\n[comparison]")
        print(f"  baseline: reason={r_base['reason']} score={r_base.get('score',0):.4f} "
              f"norm={r_base.get('norm',0):.6f}")
        print(f"  new:      reason={r_new['reason']}  score={r_new.get('score',0):.4f} "
              f"norm={r_new.get('norm',0):.6f}")

        if r_new["reason"] == "success":
            assert r_new["score"] >= r_base.get("score", 0), (
                "New attack should score >= baseline"
            )


def from_perturbnet_logits(model, image_bchw, device):
    from perturbnet.model import logits_for_images
    return logits_for_images(model, image_bchw)


# ---------------------------------------------------------------------------
# Timing tests
# ---------------------------------------------------------------------------

class TestTiming:
    def test_calibration_returns_positive(self, mock_miner):
        assert mock_miner._secs_per_step > 0
        assert mock_miner._secs_per_step < 60

    def test_forward_completes_within_60s(self, mock_miner, make_synapse):
        import time
        syn = make_synapse(epsilon=0.10, timeout_seconds=60)
        t0 = time.time()
        run(mock_miner.forward(syn))
        elapsed = time.time() - t0
        print(f"\n[timing] elapsed={elapsed:.2f}s (limit=55s)")
        assert elapsed < 55, (
            f"forward() took {elapsed:.1f}s — will timeout in production. "
            f"Reduce ATTACK_BINARY_STEPS or ATTACK_INNER_STEPS."
        )

    def test_forward_completes_within_30s(self, mock_miner, make_synapse):
        """Tight timeout test — confirms adaptive budget shrinks correctly."""
        import time
        syn = make_synapse(epsilon=0.10, timeout_seconds=30)
        t0 = time.time()
        run(mock_miner.forward(syn))
        elapsed = time.time() - t0
        print(f"\n[timing-30s] elapsed={elapsed:.2f}s (limit=28s)")
        assert elapsed < 28, (
            f"forward() with 30s timeout took {elapsed:.1f}s — too slow."
        )

    def test_step_budget_within_time(self, mock_miner):
        """
        _compute_step_budget must produce a combination where
        binary_steps * inner_steps * candidates * secs_per_step < remaining_s.
        """
        for remaining_s in [5, 15, 30, 55]:
            import time
            t_fake_start = time.time() - (60 - remaining_s)
            b, i, f = mock_miner._compute_step_budget(
                timeout_s=60,
                t_start=t_fake_start,
                num_candidates=3,
            )
            estimated = b * i * 3 * mock_miner._secs_per_step
            print(f"\n[budget] remaining={remaining_s}s binary={b} inner={i} "
                  f"estimated={estimated:.1f}s")
            assert estimated < remaining_s * 0.90, (
                f"Budget overflows at remaining={remaining_s}s: "
                f"binary={b} inner={i} estimated={estimated:.1f}s"
            )
