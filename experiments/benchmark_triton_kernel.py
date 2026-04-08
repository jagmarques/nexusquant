"""Triton kernel benchmark: E8 nearest-point and encode/decode vs CPU baseline.

Runs on Modal A100. Measures latency at 1K / 8K / 32K / 128K vectors (d=128)
and reports Triton vs CPU speedup. Also validates encode/decode round-trip RMSE.

Run with:
    modal run experiments/benchmark_triton_kernel.py
"""

import modal
import os

app = modal.App("nexusquant-triton-benchmark")

nq_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nexusquant-oss", "nexusquant")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "triton>=3.0.0", "numpy<2.0")
    .add_local_dir(nq_local, remote_path="/root/nexusquant")
)


@app.function(image=image, gpu="A100", timeout=600)
def benchmark():
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")

    # ---- Import CPU baseline (always available) ----
    from nexusquant.core.e8_lattice import E8Lattice

    # ---- Import Triton kernels (graceful fallback) ----
    try:
        from nexusquant.kernels.e8_triton import (
            e8_decode,
            e8_dequant_matmul,
            e8_encode,
            e8_nearest_point,
            e8_quantize_perhead,
        )
        triton_available = True
        print("Triton import: OK")
    except ImportError as exc:
        triton_available = False
        print(f"Triton import FAILED — falling back to CPU-only report. Reason: {exc}")

    # ------------------------------------------------------------------ #
    #  Nearest-point benchmark                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  E8 nearest-point benchmark  (d=8 groups, N = total vectors)")
    print("=" * 70)
    print(f"{'Label':35s}  {'CPU (ms)':>10}  {'Triton (ms)':>12}  {'Speedup':>8}")
    print("-" * 70)

    sizes = [
        (1_024,   128, "1K vectors, d=128"),
        (8_192,   128, "8K vectors, d=128"),
        (32_768,  128, "32K vectors, d=128"),
        (131_072, 128, "128K vectors, d=128"),
    ]

    for n, d, label in sizes:
        # Each row is one head vector; reshape to (-1, 8) for the CPU nearest_point call
        x_cpu = torch.randn(n, d, dtype=torch.float32)
        x_gpu = x_cpu.cuda() if triton_available else None

        n_groups = n * d // 8
        x_cpu_8 = x_cpu.reshape(n_groups, 8)

        # --- Warmup ---
        for _ in range(3):
            E8Lattice.nearest_point(x_cpu_8)
        if triton_available:
            e8_nearest_point(x_gpu.reshape(n_groups, 8))
            torch.cuda.synchronize()

        # --- CPU timing ---
        t0 = time.perf_counter()
        for _ in range(10):
            E8Lattice.nearest_point(x_cpu_8)
        cpu_ms = (time.perf_counter() - t0) / 10 * 1000

        if triton_available:
            # --- Triton timing ---
            x_g8 = x_gpu.reshape(n_groups, 8)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                e8_nearest_point(x_g8)
            torch.cuda.synchronize()
            triton_ms = (time.perf_counter() - t0) / 10 * 1000
            speedup = cpu_ms / triton_ms
            print(f"{label:35s}  {cpu_ms:>10.2f}  {triton_ms:>12.2f}  {speedup:>7.1f}x")
        else:
            print(f"{label:35s}  {cpu_ms:>10.2f}  {'N/A (no Triton)':>12}  {'N/A':>8}")

    # ------------------------------------------------------------------ #
    #  quantize_perhead benchmark (per-vector scale, matches pipeline use) #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  E8 quantize_perhead benchmark  (per-vector scale, levels=4)")
    print("=" * 70)
    print(f"{'Label':35s}  {'CPU (ms)':>10}  {'Triton (ms)':>12}  {'Speedup':>8}")
    print("-" * 70)

    for n, d, label in sizes:
        x_cpu = torch.randn(n, d, dtype=torch.float32)
        x_gpu = x_cpu.cuda() if triton_available else None

        # --- Warmup ---
        for _ in range(3):
            E8Lattice.quantize_perhead(x_cpu, levels=4)
        if triton_available:
            e8_quantize_perhead(x_gpu, levels=4)
            torch.cuda.synchronize()

        # --- CPU timing ---
        t0 = time.perf_counter()
        for _ in range(10):
            E8Lattice.quantize_perhead(x_cpu, levels=4)
        cpu_ms = (time.perf_counter() - t0) / 10 * 1000

        if triton_available:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                e8_quantize_perhead(x_gpu, levels=4)
            torch.cuda.synchronize()
            triton_ms = (time.perf_counter() - t0) / 10 * 1000
            speedup = cpu_ms / triton_ms
            print(f"{label:35s}  {cpu_ms:>10.2f}  {triton_ms:>12.2f}  {speedup:>7.1f}x")
        else:
            print(f"{label:35s}  {cpu_ms:>10.2f}  {'N/A (no Triton)':>12}  {'N/A':>8}")

    # ------------------------------------------------------------------ #
    #  Encode / decode round-trip (Triton only)                            #
    # ------------------------------------------------------------------ #
    if triton_available:
        print("\n" + "=" * 70)
        print("  Encode / decode round-trip  (8K vectors, d=128, levels=4)")
        print("=" * 70)

        x = torch.randn(8_192, 128, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            codes, scales = e8_encode(x, levels=4)
            _ = e8_decode(codes, scales, levels=4, original_head_dim=128)
        torch.cuda.synchronize()

        # Encode timing
        t0 = time.perf_counter()
        for _ in range(10):
            codes, scales = e8_encode(x, levels=4)
        torch.cuda.synchronize()
        encode_ms = (time.perf_counter() - t0) / 10 * 1000

        # Decode timing
        t0 = time.perf_counter()
        for _ in range(10):
            decoded = e8_decode(codes, scales, levels=4, original_head_dim=128)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) / 10 * 1000

        rmse = (x - decoded).pow(2).mean().sqrt().item()
        relative_rmse = rmse / x.pow(2).mean().sqrt().item()

        print(f"  Encode latency:        {encode_ms:.3f} ms")
        print(f"  Decode latency:        {decode_ms:.3f} ms")
        print(f"  Round-trip RMSE:       {rmse:.6f}")
        print(f"  Relative RMSE:         {relative_rmse:.4%}")
        print(f"  Codes shape:           {codes.shape}  dtype={codes.dtype}")
        print(f"  Scales shape:          {scales.shape}  dtype={scales.dtype}")

        # Storage ratio: FP16 original vs int8 codes + fp32 scales
        fp16_bytes  = x.numel() * 2
        codes_bytes = codes.numel() * 1          # int8 = 1 byte
        scale_bytes = scales.numel() * 4          # fp32 = 4 bytes
        total_compressed = codes_bytes + scale_bytes
        storage_ratio = fp16_bytes / total_compressed
        print(f"  Storage ratio vs FP16: {storage_ratio:.2f}x  "
              f"({fp16_bytes // 1024} KB -> {total_compressed // 1024} KB)")

        # ------------------------------------------------------------------ #
        #  1-bit vs 2-bit quality comparison (relevant for soft eviction)     #
        # ------------------------------------------------------------------ #
        print("\n" + "=" * 70)
        print("  1-bit vs 2-bit quantization quality  (soft eviction context)")
        print("=" * 70)

        for levels, label in [(2, "1-bit (soft evict)"), (4, "2-bit (kept tokens)"), (8, "3-bit")]:
            codes_l, scales_l = e8_encode(x, levels=levels)
            decoded_l = e8_decode(codes_l, scales_l, levels=levels, original_head_dim=128)
            rmse_l = (x - decoded_l).pow(2).mean().sqrt().item()
            rel_l  = rmse_l / x.pow(2).mean().sqrt().item()
            print(f"  {label:25s}  RMSE={rmse_l:.5f}  ({rel_l:.3%} relative)")

    # ------------------------------------------------------------------ #
    #  Fused dequant-matmul benchmark (e8_dequant_matmul)                 #
    # ------------------------------------------------------------------ #
    if triton_available:
        print("\n" + "=" * 70)
        print("  Fused dequant-matmul benchmark  (d=128, Q_cols=128, levels=4)")
        print("=" * 70)
        print(f"{'Label':35s}  {'Naive (ms)':>10}  {'Fused (ms)':>11}  {'Speedup':>8}  {'Max err':>10}")
        print("-" * 70)

        head_dim = 128
        levels = 4

        kv_sizes = [
            (1_024,   "1K KV tokens"),
            (8_192,   "8K KV tokens"),
            (32_768,  "32K KV tokens"),
            (131_072, "128K KV tokens"),
        ]

        for K, label in kv_sizes:
            # Build quantized KV cache block and query
            kv = torch.randn(K, head_dim, device="cuda", dtype=torch.float32)
            codes, scales = e8_encode(kv, levels=levels)   # int8 [K, head_dim], float32 [K]
            query = torch.randn(head_dim, head_dim, device="cuda", dtype=torch.float32)

            # --- Warmup ---
            for _ in range(3):
                decoded = e8_decode(codes, scales, levels=levels, original_head_dim=head_dim)
                _ = decoded @ query
                _ = e8_dequant_matmul(codes, scales, query, levels=levels)
            torch.cuda.synchronize()

            # --- Naive: decode then matmul ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                decoded = e8_decode(codes, scales, levels=levels, original_head_dim=head_dim)
                naive_out = decoded @ query
            torch.cuda.synchronize()
            naive_ms = (time.perf_counter() - t0) / 10 * 1000

            # --- Fused kernel ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                fused_out = e8_dequant_matmul(codes, scales, query, levels=levels)
            torch.cuda.synchronize()
            fused_ms = (time.perf_counter() - t0) / 10 * 1000

            speedup = naive_ms / fused_ms
            max_err = (fused_out - naive_out).abs().max().item()
            print(f"{label:35s}  {naive_ms:>10.2f}  {fused_ms:>11.2f}  {speedup:>7.1f}x  {max_err:>10.6f}")

    print("\nBenchmark complete.")


@app.local_entrypoint()
def main():
    benchmark.remote()
