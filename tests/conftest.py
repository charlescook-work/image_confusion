"""
Shared pytest fixtures for local miner testing.
Mocks all Bittensor network dependencies so tests run without
a live subtensor connection or registered wallet.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from perturbnet.image_io import encode_image_b64
from perturbnet.model import load_efficientnet_b5, predict_label, resolve_target_index


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def model(device) -> torch.nn.Module:
    """Real EfficientNet-B5 loaded once per test session."""
    return load_efficientnet_b5(device)


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dog_image_b64() -> str:
    """Load the project's fallback dog image as base64."""
    try:
        with open("assets/dog_1.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        # Generate a synthetic RGB image if fallback not found.
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="session")
def dog_true_label(model, device, dog_image_b64) -> str:
    """Get the actual EfficientNet label for the dog image."""
    from perturbnet.image_io import decode_image_b64
    img = decode_image_b64(dog_image_b64).to(device)
    return predict_label(model, img)


@pytest.fixture(scope="session")
def dog_true_idx(dog_true_label) -> int:
    idx = resolve_target_index(dog_true_label)
    assert idx is not None, f"Could not resolve label: {dog_true_label}"
    return idx


# ---------------------------------------------------------------------------
# Bittensor mocks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_config() -> SimpleNamespace:
    cfg = SimpleNamespace()
    cfg.netuid = 26
    cfg.log_level = "ERROR"
    cfg.wallet = SimpleNamespace(name="test", hotkey="test")
    cfg.subtensor = SimpleNamespace(network="finney", chain_endpoint="")
    cfg.logging = SimpleNamespace(logging_dir="./logs")
    cfg.axon = SimpleNamespace(port=9000)
    return cfg


@pytest.fixture(scope="session")
def fake_metagraph() -> MagicMock:
    meta = MagicMock()
    meta.n = 256
    meta.hotkeys = ["fake_validator_hk"] * 256
    meta.validator_permit = [True] * 256
    meta.S = [1.0] * 256
    meta.axons = [MagicMock(ip="1.2.3.4")] * 256
    return meta


@pytest.fixture(scope="session")
def mock_miner(mock_config, fake_metagraph):
    """
    Fully initialised PerturbMiner with all Bittensor network calls mocked.
    Loads the real EfficientNet-B5 model — this is the expensive fixture.
    """
    import sys, os
    sys.path.insert(0, os.path.abspath("."))

    fake_wallet = MagicMock()
    fake_wallet.hotkey.ss58_address = "fake_miner_hk"

    with (
        patch("neurons.miner._make_wallet", return_value=fake_wallet),
        patch("neurons.miner._make_subtensor", return_value=MagicMock()),
        patch("neurons.miner._make_axon", return_value=MagicMock()),
    ):
        from neurons.miner import PerturbMiner

        miner = object.__new__(PerturbMiner)
        miner.config = mock_config
        miner.wallet = fake_wallet
        miner.subtensor = MagicMock()
        miner.metagraph = fake_metagraph
        miner.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        import logging as pylogging
        pylogging.getLogger().setLevel(pylogging.ERROR)

        raw_model = load_efficientnet_b5(miner.device)
        try:
            miner.model = torch.compile(raw_model)
        except Exception:
            miner.model = raw_model

        # Warmup
        if miner.device.type == "cuda":
            try:
                dummy = torch.zeros(1, 3, 456, 456, device=miner.device)
                with torch.no_grad():
                    _ = miner.model(dummy)
                torch.cuda.synchronize()
            except Exception:
                pass

        # Import and call calibration
        from neurons.miner import _calibrate_step_time
        miner._secs_per_step = _calibrate_step_time(miner.model, miner.device)

        miner.binary_steps = 10
        miner.inner_steps = 15
        miner.mi_decay = 1.0
        miner.fallback_steps = 20
        miner.num_target_candidates = 3
        miner.reply_buffer_s = 4.0
        miner._local_round_count = 0

        miner.axon = MagicMock()

    return miner


# ---------------------------------------------------------------------------
# Synapse factory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def make_synapse(dog_image_b64, dog_true_label):
    """
    Returns a factory function that builds a real AttackChallenge synapse
    from the dog image — exactly what the validator would send.
    """
    from perturbnet.protocol import AttackChallenge

    def _make(epsilon: float = 0.10, timeout_seconds: int = 60) -> AttackChallenge:
        syn = AttackChallenge(
            task_id="test-local-001",
            model_name="EfficientNet-B5",
            prompt="dog",
            clean_image_b64=dog_image_b64,
            true_label=dog_true_label,
            epsilon=epsilon,
            norm_type="Linf",
            min_delta=0.002,
            timeout_seconds=timeout_seconds,
        )
        return syn

    return _make
