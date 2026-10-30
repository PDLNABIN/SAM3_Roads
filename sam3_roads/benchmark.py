from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def parameter_footprint(model: nn.Module) -> dict[str, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio_percent": 100 * trainable / max(total, 1),
        "parameter_reduction_percent": 100 * (1 - trainable / max(total, 1)),
    }


@torch.no_grad()
def profile_inference(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int = 5,
    iterations: int = 30,
) -> dict[str, float]:
    model.eval()
    sample = sample.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            model(pixel_values=sample)
        torch.cuda.synchronize(device)
        events = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(pixel_values=sample)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize(device)
        timings = [start.elapsed_time(end) for start, end in events]
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        for _ in range(warmup):
            model(pixel_values=sample)
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(pixel_values=sample)
            timings.append((time.perf_counter() - start) * 1000)
        peak_mb = 0.0

    timing = torch.tensor(timings, dtype=torch.float64)
    mean_ms = timing.mean().item()
    return {
        "mean_latency_ms": mean_ms,
        "std_latency_ms": timing.std(unbiased=False).item(),
        "throughput_fps": 1000 / max(mean_ms, 1e-6),
        "peak_gpu_memory_mb": peak_mb,
        "peak_gpu_memory_gb": peak_mb / 1024,
        "profile_iterations": iterations,
    }


def build_portfolio_stats(model: nn.Module, sample: torch.Tensor, device: torch.device) -> dict[str, Any]:
    return {
        "parameters": parameter_footprint(model),
        "inference": profile_inference(model, sample, device),
    }
