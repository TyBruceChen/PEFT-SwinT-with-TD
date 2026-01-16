"""Test CaRA module setup."""

import random
from typing import Any, Dict

import numpy as np
import torch as th
from timm.models import create_model

from src.cara.cara_swt import cara_swt


def _get_swt() -> th.nn.Module:
    """Create swt model.

    Returns:
        th.nn.Module: swt model.
    """
    return create_model("swin_tiny_patch4_window7_224", drop_path_rate=0.1)


def _get_cara_swt_config() -> Dict[str, Any]:
    """Create configuration dictionary.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    random.seed(0)
    th.manual_seed(0)
    np.random.seed(0)
    th.cuda.manual_seed_all(0)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    return {
        "model": _get_swt(),
        "rank": 32,
        "scale": 1.0,
        "l_mu": 1.0,
        "l_std": 0.0,
    }


def test_swt_without_cara():
    """swt test without CaRA."""
    swt = _get_swt()
    assert (
        (not hasattr(swt, "CP_A1"))
        and (not hasattr(swt, "CP_A2"))
        and (not hasattr(swt, "CP_A3"))
        and (not hasattr(swt, "CP_A4"))
    )
    assert (
        (not hasattr(swt, "CP_P1"))
        and (not hasattr(swt, "CP_P2"))
        and (not hasattr(swt, "CP_P3"))
    )
    assert not hasattr(swt, "CP_R1")
    assert not hasattr(swt, "CP_R2")


def test_swt_with_cara():
    """swt test with CaRA module."""
    swt = cara_swt(_get_cara_swt_config())
    assert (
        (hasattr(swt, "CP_A1"))
        and (hasattr(swt, "CP_A2"))
        and (hasattr(swt, "CP_A3"))
        and (hasattr(swt, "CP_A4"))
    )
    assert (
        (hasattr(swt, "CP_P1"))
        and (hasattr(swt, "CP_P2"))
        and (hasattr(swt, "CP_P3"))
    )
    assert hasattr(swt, "CP_R1")
    assert hasattr(swt, "CP_R2")

    return swt


def test_cara_swt_zero_init():
    """Check for zero initialisation in CaRA params."""
    swt = cara_swt(_get_cara_swt_config())
    assert th.allclose(swt.CP_A2, th.zeros_like(swt.CP_A2))
    assert th.allclose(swt.CP_P2, th.zeros_like(swt.CP_P2))


def test_cara_swt_lambda_init():
    """Check for Lambda initis in CaRA params."""
    swt = cara_swt(_get_cara_swt_config())
    assert th.allclose(swt.CP_R1, th.ones_like(swt.CP_R1))
    assert th.allclose(swt.CP_R2, th.ones_like(swt.CP_R2))


def test_cara_swt_forward():
    """Check for dummy forward pass."""
    swt = cara_swt(_get_cara_swt_config())
    dummy_input = th.randn((2, 3, 224, 224))
    output = swt(dummy_input)
    assert np.allclose(list(output.shape), (2, 1000))
    return output
