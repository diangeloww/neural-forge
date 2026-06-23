"""Tests for routing strategies."""

from neural_forge.routing import Router, Strategy


def test_round_robin():
    router = Router(strategy=Strategy.ROUND_ROBIN)
    assert router.strategy == Strategy.ROUND_ROBIN


def test_cost_optimized():
    router = Router(strategy=Strategy.COST_OPTIMIZED)
    assert router.strategy == Strategy.COST_OPTIMIZED


def test_failure_recording():
    router = Router(strategy=Strategy.ROUND_ROBIN)
    router.record_failure("openai")
    assert router._failures["openai"] == 1
    router.record_failure("openai")
    assert router._failures["openai"] == 2


def test_latency_recording():
    router = Router(strategy=Strategy.LATENCY)
    router.record_latency("openai", 100.0)
    router.record_latency("openai", 200.0)
    assert router._avg_latency("openai") == 150.0
