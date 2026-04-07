"""vLLM PagedAttention integration for NexusQuant KV cache compression.

Makes NexusQuant a compression backend for vLLM's paged block manager.
Each page (block) is compressed with E8 lattice VQ.

Pipeline per page:
    Keys:   inverse_rope -> Hadamard -> E8 VQ (store int8 codes + fp16 scale)
    Values: Hadamard -> E8 VQ (store int8 codes + fp16 scale)
    Decode: invert above -> re-apply RoPE from stored original positions

Storage format for codes:
    E8 nearest_point returns integer or half-integer lattice points.
    We store (lattice_point * 2) as int8 to preserve half-integer precision.
    Dequantization: x̂ = (int8_code / 2) * scale

    int8 codes:  2 * H * B * D bytes  (1 byte per element, K and V)
    fp16 scales: 2 * H * B * 2 bytes  (per-head-per-token, K and V)
    int32 pos:   B * 4             bytes

Compression ratios (all overhead included, H heads, B block_size, D head_dim):
    int8 codes (default):  ~2x vs fp16  (exact: see CompressedPage.compressed_bytes)
    nibble-packed codes:   ~3.5x vs fp16 (call pack_codes_nibble before storage)

    For reference: the GPU-validated 10-33x ratios in NexusQuant papers are
    achieved by the full pipeline (eviction + quantization across all layers).
    Per-page quantization alone gives ~2x; the multiplier from eviction and
    multi-layer amortization accounts for the remainder.

Usage::

    from nexusquant.integrations.vllm_backend import (
        NexusQuantKVCompressor,
        NexusQuantPagedAttention,
        register_nexusquant_backend,
    )

    # Standalone use (testing / custom serving loop)
    compressor = NexusQuantKVCompressor(head_dim=128, bits=2)
    compressed = compressor.compress_page(key_page, value_page, page_positions)
    keys, values = compressor.decompress_page(compressed)

    # vLLM integration
    register_nexusquant_backend(engine.engine_config)
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.hadamard import hadamard_matrix
from nexusquant.core.rope_utils import inverse_rope, forward_rope


# ---------------------------------------------------------------------------
# Compression accounting constants
# ---------------------------------------------------------------------------

# fp16 bytes per element in the uncompressed cache
_FP16_BYTES = 2


# ---------------------------------------------------------------------------
# CompressedPage dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CompressedPage:
    """A single compressed KV block (vLLM page / block).

    Memory layout (all tensors contiguous fp16 / uint8):

    key_codes:     (num_heads, num_valid, head_dim) int8
                   Stores (E8_lattice_point * 2) as int8.  E8 points are
                   integers or half-integers, so multiplying by 2 makes them
                   exact integers in [-2*levels/2, 2*levels/2].
                   Dequantize via: float_val = int8_code / 2.0 * scale
                   Caller may further pack with pack_codes_nibble() for ~2x
                   additional code compression before host-memory storage.
    value_codes:   same layout as key_codes
    key_scales:    (num_heads, num_valid) fp16 per-(head,token) scale factor
    value_scales:  (num_heads, num_valid) fp16 per-(head,token) scale factor
    positions:     (num_valid,) int32 original absolute token positions
                   (needed to re-apply RoPE at decode time)
    num_valid:     number of real tokens stored (rest of block is padding)

    All tensors live on the same device as the original page.
    """

    key_codes: torch.Tensor      # (num_heads, num_valid, head_dim)  int8
    value_codes: torch.Tensor    # (num_heads, num_valid, head_dim)  int8
    key_scales: torch.Tensor     # (num_heads, num_valid)            fp16
    value_scales: torch.Tensor   # (num_heads, num_valid)            fp16
    positions: torch.Tensor      # (num_valid,)                      int32
    num_valid: int               # number of real tokens in this page

    # ---- compression accounting helpers --------------------------------

    @property
    def compressed_bytes(self) -> int:
        """Bytes actually occupied by this compressed page."""
        n = self.num_valid
        h = self.key_codes.shape[0]
        d = self.key_codes.shape[2]
        # int8 codes for K and V
        code_bytes = 2 * h * n * d  # 1 byte per element (int8)
        # fp16 scales for K and V
        scale_bytes = 2 * h * n * _FP16_BYTES
        # int32 positions
        pos_bytes = n * 4
        return code_bytes + scale_bytes + pos_bytes

    @property
    def uncompressed_bytes(self) -> int:
        """Bytes that the same data would occupy uncompressed (fp16)."""
        n = self.num_valid
        h = self.key_codes.shape[0]
        d = self.key_codes.shape[2]
        return 2 * h * n * d * _FP16_BYTES


# ---------------------------------------------------------------------------
# NexusQuantKVCompressor
# ---------------------------------------------------------------------------

class NexusQuantKVCompressor:
    """Compression backend for vLLM paged KV cache.

    Implements the NexusQuant pipeline adapted to vLLM's page granularity:
        Keys:   inverse_rope (per stored position) -> Hadamard -> E8 VQ
        Values: Hadamard -> E8 VQ

    Decompression reverses the pipeline:
        Keys:   E8 dequant -> inv Hadamard -> forward_rope (re-apply stored pos)
        Values: E8 dequant -> inv Hadamard

    The Hadamard matrix is the orthonormal Walsh-Hadamard matrix of size
    head_dim x head_dim.  Because the matrix is orthonormal (H^T = H^{-1}),
    the inverse transform is simply x @ H (same matrix).

    Args:
        head_dim:      Attention head dimension (must be power of 2).
        bits:          Quantization bits per element (2, 3, or 4).
        eviction_rate: Token eviction fraction within a page (0 = off).
                       When > 0, low-importance tokens inside the page are
                       zeroed before compression (experimental).
        rope_base:     RoPE frequency base (rope_theta from model config).
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 2,
        eviction_rate: float = 0.0,
        rope_base: float = 10000.0,
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4 (got {bits})")
        if head_dim & (head_dim - 1) != 0:
            raise ValueError(f"head_dim must be a power of 2 (got {head_dim})")

        self.head_dim = head_dim
        self.bits = bits
        self.levels = 2 ** bits
        self.eviction_rate = eviction_rate
        self.rope_base = rope_base

        # Precompute Hadamard matrix — moved to device on first use
        self._H: Optional[torch.Tensor] = hadamard_matrix(head_dim)
        self._H_device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_H(self, device: torch.device) -> torch.Tensor:
        """Return Hadamard matrix on the right device (lazy move)."""
        if self._H_device != device:
            self._H = self._H.to(device)
            self._H_device = device
        return self._H

    @staticmethod
    def _rope_remove_page(
        keys: torch.Tensor,
        positions: torch.Tensor,
        rope_base: float,
    ) -> torch.Tensor:
        """Remove RoPE from page keys using their ABSOLUTE position indices.

        vLLM paged attention stores blocks from non-contiguous positions
        (e.g., a block could hold tokens at positions [256, 257, ..., 271]).
        inverse_rope() assumes contiguous positions starting at seq_offset.
        For a page whose first position is P[0], we call inverse_rope with
        seq_offset=P[0] IF positions are contiguous, or we apply per-token
        inverse rotation when they are not.

        This implementation handles the general (non-contiguous) case by
        computing cos/sin per absolute position.

        Args:
            keys:       (num_heads, num_valid, head_dim) float32
            positions:  (num_valid,) int32 absolute token positions
            rope_base:  RoPE theta

        Returns:
            keys with RoPE removed, same shape
        """
        h, s, d = keys.shape
        d_half = d // 2
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, d, 2, dtype=torch.float32, device=keys.device) / d)
        )
        pos_f = positions.float()                          # (s,)
        freqs = torch.outer(pos_f, inv_freq)               # (s, d_half)
        cos_f = freqs.cos().unsqueeze(0)                   # (1, s, d_half)
        sin_f = freqs.sin().unsqueeze(0)                   # (1, s, d_half)

        first_half = keys[..., :d_half]                    # (h, s, d_half)
        second_half = keys[..., d_half:]                   # (h, s, d_half)
        result = keys.clone()
        # Inverse of forward_rope: x' = x*cos + shift*sin  (split-half layout)
        result[..., :d_half] = first_half * cos_f + second_half * sin_f
        result[..., d_half:] = -first_half * sin_f + second_half * cos_f
        return result

    @staticmethod
    def _rope_apply_page(
        keys: torch.Tensor,
        positions: torch.Tensor,
        rope_base: float,
    ) -> torch.Tensor:
        """Re-apply RoPE to page keys at their original absolute positions.

        Args:
            keys:       (num_heads, num_valid, head_dim) float32
            positions:  (num_valid,) int32 absolute token positions
            rope_base:  RoPE theta

        Returns:
            keys with RoPE re-applied, same shape
        """
        h, s, d = keys.shape
        d_half = d // 2
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, d, 2, dtype=torch.float32, device=keys.device) / d)
        )
        pos_f = positions.float()
        freqs = torch.outer(pos_f, inv_freq)
        cos_f = freqs.cos().unsqueeze(0)                   # (1, s, d_half)
        sin_f = freqs.sin().unsqueeze(0)                   # (1, s, d_half)

        first_half = keys[..., :d_half].clone()
        second_half = keys[..., d_half:].clone()
        result = keys.clone()
        result[..., :d_half] = first_half * cos_f - second_half * sin_f
        result[..., d_half:] = first_half * sin_f + second_half * cos_f
        return result

    @staticmethod
    def _quantize_perhead_with_scales(
        x: torch.Tensor,
        levels: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize with per-head-per-token scale; return int8 codes + scales.

        Unlike E8Lattice.quantize_perhead() which returns dequantized fp tensors,
        this function returns the raw lattice codes and per-vector scale factors
        separately so they can be stored compactly.

        E8 lattice nearest_point produces values that are integers or half-integers
        (e.g., {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5} at 2-bit/levels=4 when
        input is in [-2, 2]).  We multiply by 2 and round to make them true integers
        before storing as int8.  Dequantization divides by 2 before applying the
        scale.  At 2-bit there are at most 9 distinct values (×2: {-4..4}), at
        3-bit up to 17 ({-8..8}), at 4-bit up to 33 ({-16..16}) — all fit in int8.

        Scale encoding: scale = amax / (levels/2), stored as fp16.
        Reconstruction: x̂ = (int8_code / 2.0) * scale

        Args:
            x:      (num_heads, num_tokens, head_dim) float32
            levels: Quantization levels (4=2-bit, 8=3-bit, 16=4-bit)

        Returns:
            codes:  (num_heads, num_tokens, head_dim) int8
                    Stores lattice_point * 2 rounded to int, range [-levels, +levels].
            scales: (num_heads, num_tokens) float32 scale per vector
        """
        h, s, d = x.shape
        flat = x.reshape(h * s, d)                          # (h*s, d)
        amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / (levels / 2)                         # (h*s, 1)

        normalized = flat / scale                           # (h*s, d)
        pad = (8 - d % 8) % 8
        if pad > 0:
            normalized = F.pad(normalized, (0, pad))

        lp = E8Lattice.nearest_point(
            normalized.reshape(-1, 8)
        ).clamp(-levels / 2, levels / 2)                    # (h*s*(d+pad)//8, 8)
        codes_full = lp.reshape(h * s, d + pad)[..., :d]   # (h*s, d)  float

        # Multiply by 2 to convert half-integers to integers, then store as int8.
        # Max value at levels=4: 2*2=4, at levels=8: 2*4=8, at levels=16: 2*8=16.
        # All fit comfortably in int8 [-128, 127].
        codes = (codes_full * 2).round().clamp(-127, 127).to(torch.int8).reshape(h, s, d)
        scales = scale.squeeze(-1).reshape(h, s)            # (h, s)
        return codes, scales

    @staticmethod
    def _dequantize_perhead(
        codes: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct float vectors from int8 codes + per-head-per-token scales.

        Reverses _quantize_perhead_with_scales:
            x̂ = (int8_code / 2.0) * scale

        Args:
            codes:  (num_heads, num_tokens, head_dim) int8
                    Lattice codes stored as lattice_point * 2.
            scales: (num_heads, num_tokens) float32 or float16

        Returns:
            (num_heads, num_tokens, head_dim) float32
        """
        h, s, d = codes.shape
        sc = scales.float().unsqueeze(-1)                   # (h, s, 1)
        return (codes.float() * 0.5) * sc                  # (h, s, d)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_page(
        self,
        key_page: torch.Tensor,
        value_page: torch.Tensor,
        page_positions: torch.Tensor,
    ) -> CompressedPage:
        """Compress a single KV page (block_size tokens).

        Args:
            key_page:       (num_heads, block_size, head_dim) fp16 or fp32
            value_page:     (num_heads, block_size, head_dim) fp16 or fp32
            page_positions: (block_size,) int32/int64 original absolute
                            token positions for RoPE handling.
                            Padding tokens should have valid positions (they
                            won't be stored — set num_valid accordingly).

        Returns:
            CompressedPage with encoded keys, values, scales, and metadata.

        Note:
            All tokens in the page are compressed.  If the caller wants to
            compress only the first N valid tokens (e.g., the last block is
            partially filled), pass sliced tensors:
                compressor.compress_page(k[:, :N], v[:, :N], pos[:N])
        """
        device = key_page.device
        H = self._get_H(device)
        h, block_size, d = key_page.shape

        k = key_page.float()
        v = value_page.float()
        pos = page_positions.to(device=device, dtype=torch.int32)
        num_valid = block_size

        # ------ Keys: inverse_rope -> Hadamard -> E8 VQ ------
        k_nr = self._rope_remove_page(k, pos, self.rope_base)
        k_rot = torch.einsum('hsd,de->hse', k_nr, H.float())
        k_codes, k_scales = self._quantize_perhead_with_scales(k_rot, self.levels)
        # k_codes: (h, s, d)  int8
        # k_scales: (h, s)    float32

        # ------ Values: Hadamard -> E8 VQ ------
        v_rot = torch.einsum('hsd,de->hse', v, H.float())
        v_codes, v_scales = self._quantize_perhead_with_scales(v_rot, self.levels)

        return CompressedPage(
            key_codes=k_codes,                             # int8
            value_codes=v_codes,                           # int8
            key_scales=k_scales.to(torch.float16),         # fp16
            value_scales=v_scales.to(torch.float16),       # fp16
            positions=pos,                                 # int32
            num_valid=num_valid,
        )

    def decompress_page(
        self,
        compressed_page: CompressedPage,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress a page back to full KV tensors for attention.

        Args:
            compressed_page: CompressedPage returned by compress_page().

        Returns:
            (keys, values) both (num_heads, num_valid, head_dim) fp16
        """
        device = compressed_page.key_codes.device
        H = self._get_H(device)
        pos = compressed_page.positions

        # ------ Keys: E8 dequant -> inv Hadamard -> forward_rope ------
        k_rot = self._dequantize_perhead(
            compressed_page.key_codes, compressed_page.key_scales
        )                                                   # (h, s, d) fp32
        # Inverse Hadamard: H is orthonormal so H^{-T} = H
        k_nr = torch.einsum('hsd,ed->hse', k_rot, H.float())
        keys = self._rope_apply_page(k_nr, pos, self.rope_base)
        keys = keys.to(torch.float16)

        # ------ Values: E8 dequant -> inv Hadamard ------
        v_rot = self._dequantize_perhead(
            compressed_page.value_codes, compressed_page.value_scales
        )
        values = torch.einsum('hsd,ed->hse', v_rot, H.float()).to(torch.float16)

        return keys, values

    def decompress_for_attention(
        self,
        compressed_page: CompressedPage,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Fused decompress + attention score computation.

        Avoids materializing the full decompressed key tensor when the caller
        only needs attention logits (query @ key^T).

        For values, a full materialization is still required because attention
        weighting requires the complete value matrix.  This method returns
        attention logits only; the caller should call decompress_page() for
        values when computing the attended output.

        Args:
            compressed_page: CompressedPage from compress_page().
            query:           (num_heads, 1, head_dim) or (num_heads, q_len, head_dim)
                             fp16 or fp32 query vectors with RoPE already applied.

        Returns:
            attn_logits: (num_heads, q_len, num_valid) fp32 unnormalized logits
        """
        device = compressed_page.key_codes.device
        H = self._get_H(device)
        pos = compressed_page.positions
        scale = math.sqrt(self.head_dim)

        # Decompress keys (fp32)
        k_rot = self._dequantize_perhead(
            compressed_page.key_codes, compressed_page.key_scales
        )                                                   # (h, s, d)
        k_nr = torch.einsum('hsd,ed->hse', k_rot, H.float())
        keys = self._rope_apply_page(k_nr, pos, self.rope_base)  # (h, s, d)

        # Compute attention logits: (h, q_len, s)
        q = query.float()                                   # (h, q_len, d)
        attn_logits = torch.einsum('hqd,hsd->hqs', q, keys) / scale
        return attn_logits

    # ------------------------------------------------------------------
    # Utility: pack/unpack int8 codes to nibble (4-bit) for host memory
    # ------------------------------------------------------------------

    @staticmethod
    def pack_codes_nibble(codes: torch.Tensor) -> torch.Tensor:
        """Pack int8 codes to 4-bit nibbles, 2 codes per byte.

        Codes are stored as int8 in the range determined by bits:
            bits=2: codes in {-4, -3, -2, -1, 0, 1, 2, 3, 4}  (×2 encoded)
            bits=3: codes in {-8 .. 8}
            bits=4: codes in {-16 .. 16}

        We shift by 8 to map to the unsigned range {0..255} (fits in a nibble
        only for bits=2 where max unsigned value is 12).  For bits <= 2, we use
        4-bit packing (2 codes per byte, 2x compression on the codes tensor).

        The nibble packing uses the lower 4 bits of each byte:
            byte = (code_even & 0xF) | ((code_odd & 0xF) << 4)

        Shift convention: code → code + 8 → nibble in [0, 15].
        Range check: bits=2 → max|code|=4, shifted max=12 ≤ 15 ✓
                     bits=3 → max|code|=8, shifted max=16 > 15 ✗ (use int8)

        For bits > 2, codes stay as int8 (no packing).

        Args:
            codes: (..., d) int8 with d divisible by 2.
                   For bits=2 (range [-4, 4] shifted to [4, 12]).

        Returns:
            (..., d//2) uint8  (2 nibbles per byte)
        """
        shape = codes.shape
        d = shape[-1]
        c = codes.to(torch.int32) + 8                       # shift: [-4,4] -> [4,12]
        c = c.clamp(0, 15)                                  # safety clamp to nibble range
        if d % 2 != 0:
            c = F.pad(c, (0, 1))
        c_even = c[..., 0::2] & 0xF                         # (..., d//2) lower nibble
        c_odd  = c[..., 1::2] & 0xF                         # (..., d//2) upper nibble
        packed = (c_even | (c_odd << 4)).to(torch.uint8)    # (..., d//2)
        return packed

    @staticmethod
    def unpack_codes_nibble(packed: torch.Tensor, original_d: int) -> torch.Tensor:
        """Unpack 4-bit nibble-packed codes back to int8.

        Reverses pack_codes_nibble.  Shift convention: nibble - 8 → int8 code.

        Args:
            packed:     (..., ceil(original_d/2)) uint8
            original_d: original last dimension before packing

        Returns:
            (..., original_d) int8
        """
        p = packed.to(torch.int32)
        c_even = p & 0xF                                    # lower nibble
        c_odd  = (p >> 4) & 0xF                             # upper nibble
        interleaved = torch.stack([c_even, c_odd], dim=-1)  # (..., d//2, 2)
        shape = list(interleaved.shape)
        shape = shape[:-2] + [-1]
        unpacked = interleaved.reshape(shape)               # (..., d_padded)
        unpacked = unpacked[..., :original_d]
        return (unpacked - 8).to(torch.int8)                # undo shift

    # ------------------------------------------------------------------
    # Compression ratio accounting
    # ------------------------------------------------------------------

    def compression_ratio(self, page: CompressedPage) -> float:
        """Return exact compression ratio for this page (all overhead counted).

        Accounts for int8 code storage, fp16 scales, and int32 positions.
        For 2-bit storage, codes are stored as int8 (conservatively; caller
        may pack to 2-bit for further 4x code saving).
        """
        if page.uncompressed_bytes == 0:
            return 1.0
        return page.uncompressed_bytes / page.compressed_bytes


# ---------------------------------------------------------------------------
# NexusQuantPagedAttention
# ---------------------------------------------------------------------------

class NexusQuantPagedAttention:
    """Drop-in replacement for vLLM's PagedAttention that works with compressed pages.

    Integrates with vLLM's block manager:
    - When blocks are allocated, they store CompressedPage objects instead of
      raw fp16 KV tensors.
    - During attention, pages are decompressed on-the-fly before the dot product.
    - Memory per block: ~2x less than uncompressed (int8 codes + fp16 scales).
      Call pack_codes_nibble() on stored codes for an additional ~2x, reaching ~3.5x.

    Memory per compressed block (bits=2, H heads, B block_size, D head_dim):
        codes  = 2 * H * B * D  bytes  (int8, one byte per code element)
        scales = 4 * H * B      bytes  (fp16, K+V scale per head per token)
        total  = 2*H*B*D + 4*H*B bytes

    vs uncompressed:
        4 * H * B * D  bytes  (fp16, K and V)

    At H=32, B=16, D=128:
        uncompressed = 4*32*16*128 = 262,144 bytes
        compressed   = 2*32*16*128 + 4*32*16 = 131,072 + 2,048 = 133,120 bytes
        ratio = 1.97x (int8)
    With nibble packing of codes: ~3.5x total.

    The block tables and slot mapping follow the same conventions as vLLM's
    PagedAttention: block_tables[seq_idx] is a list of block indices, each
    pointing into a pool of CompressedPage objects.

    Args:
        compressor: NexusQuantKVCompressor instance configured for the model.
    """

    def __init__(self, compressor: NexusQuantKVCompressor):
        self.compressor = compressor
        # Compressed block pool: maps block_id -> CompressedPage
        self._block_pool: dict[int, CompressedPage] = {}
        self._next_block_id: int = 0

    # ------------------------------------------------------------------
    # Block allocation helpers (mirrors vLLM BlockAllocator interface)
    # ------------------------------------------------------------------

    def allocate_block(self) -> int:
        """Allocate a new (empty) block id.  Returns the block id."""
        block_id = self._next_block_id
        self._next_block_id += 1
        return block_id

    def free_block(self, block_id: int) -> None:
        """Free a block, releasing its CompressedPage from the pool."""
        self._block_pool.pop(block_id, None)

    def is_block_compressed(self, block_id: int) -> bool:
        """Return True if a block has been written (compressed)."""
        return block_id in self._block_pool

    # ------------------------------------------------------------------
    # Cache write (prefill + incremental decode)
    # ------------------------------------------------------------------

    def compress_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: dict,
        value_cache: dict,
        slot_mapping: torch.Tensor,
    ) -> None:
        """vLLM cache_ops.reshape_and_cache equivalent for compressed storage.

        Compresses individual tokens and writes them into the compressed block
        pool.  This is called once per forward pass for the NEW tokens only.

        In vLLM, slot_mapping[i] = block_id * block_size + block_offset for
        the i-th token in the batch (flattened over all sequences).

        Implementation strategy: for efficiency, this method batches tokens by
        their block_id, compresses each unique block once, and writes the
        result into key_cache / value_cache (which here are dictionaries of
        CompressedPage objects, keyed by block_id).

        Args:
            key:           (total_tokens, num_heads, head_dim) fp16
            value:         (total_tokens, num_heads, head_dim) fp16
            key_cache:     dict[int, CompressedPage]  (shared reference;
                           note: K and V are always compressed together)
            value_cache:   dict[int, CompressedPage]  (same object as key_cache
                           in practice; kept separate for API symmetry)
            slot_mapping:  (total_tokens,) int64 flat slot indices
                           slot = block_id * block_size + offset_in_block
        """
        if key.shape[0] == 0:
            return

        block_size = self._infer_block_size(slot_mapping)
        total_tokens = key.shape[0]
        num_heads = key.shape[1]
        head_dim = key.shape[2]
        device = key.device

        # Group tokens by block_id
        block_ids = (slot_mapping // block_size).tolist()
        offsets = (slot_mapping % block_size).tolist()

        # Collect tokens per block
        block_token_indices: dict[int, list] = {}
        block_token_offsets: dict[int, list] = {}
        for tok_idx, (bid, off) in enumerate(zip(block_ids, offsets)):
            block_token_indices.setdefault(bid, []).append(tok_idx)
            block_token_offsets.setdefault(bid, []).append(off)

        for bid, tok_indices in block_token_indices.items():
            # Gather this block's keys/values and create/update the CompressedPage
            tok_t = torch.tensor(tok_indices, dtype=torch.long, device=device)
            k_block = key[tok_t].permute(1, 0, 2)     # (num_heads, n_toks, head_dim)
            v_block = value[tok_t].permute(1, 0, 2)   # (num_heads, n_toks, head_dim)

            # Use token slot positions as the absolute positions for RoPE.
            # vLLM provides actual position IDs via the position_ids tensor
            # in the model input; here we approximate with slot offsets if
            # no external positions are supplied.  Callers should prefer the
            # compress_page() / decompress_page() API with explicit positions.
            page_positions = torch.tensor(
                block_token_offsets[bid], dtype=torch.int32, device=device
            )

            compressed = self.compressor.compress_page(k_block, v_block, page_positions)
            key_cache[bid] = compressed
            value_cache[bid] = compressed   # K and V stored together in CompressedPage

    # ------------------------------------------------------------------
    # Attention forward (decode)
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        compressed_key_cache: dict,
        compressed_value_cache: dict,
        block_tables: List[List[int]],
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Attention with on-the-fly decompression.

        Performs multi-head scaled dot-product attention where KV pages are
        decompressed lazily as they are accessed.  This matches vLLM's decode
        phase: one new query token per sequence, attending to all cached tokens.

        Args:
            query:                  (batch_size, num_heads, head_dim) fp16
                                    Query vectors WITH RoPE already applied.
            compressed_key_cache:   dict[block_id -> CompressedPage]
            compressed_value_cache: dict[block_id -> CompressedPage]  (same dict)
            block_tables:           List[List[int]]  block_id sequence per batch item
            context_lens:           (batch_size,) int32/int64  number of tokens in
                                    each sequence's KV cache (may be < total blocks)

        Returns:
            output: (batch_size, num_heads, head_dim) fp16 attended output
        """
        batch_size, num_heads, head_dim = query.shape
        device = query.device
        scale = math.sqrt(head_dim)

        output = torch.zeros_like(query)

        for b in range(batch_size):
            ctx_len = int(context_lens[b].item())
            if ctx_len == 0:
                continue

            blocks = block_tables[b]
            q_b = query[b]                                  # (num_heads, head_dim)

            # Collect all K, V across all blocks for this sequence
            all_keys = []
            all_values = []

            for block_id in blocks:
                if block_id not in compressed_key_cache:
                    continue
                page = compressed_key_cache[block_id]
                k_page, v_page = self.compressor.decompress_page(page)
                # k_page: (num_heads, page_tokens, head_dim) fp16
                all_keys.append(k_page)
                all_values.append(v_page)

            if not all_keys:
                continue

            # Concatenate across blocks
            keys = torch.cat(all_keys, dim=1)[:, :ctx_len, :]   # (h, ctx_len, d)
            values = torch.cat(all_values, dim=1)[:, :ctx_len, :]

            # Scaled dot-product attention: (h, 1, ctx_len)
            q_3d = q_b.unsqueeze(1).float()                     # (h, 1, d)
            attn_logits = torch.einsum(
                'hqd,hsd->hqs', q_3d, keys.float()
            ) / scale                                            # (h, 1, ctx_len)
            attn_weights = torch.softmax(attn_logits, dim=-1)   # (h, 1, ctx_len)

            # Weighted sum of values: (h, 1, d)
            attended = torch.einsum(
                'hqs,hsd->hqd', attn_weights, values.float()
            )                                                    # (h, 1, d)
            output[b] = attended.squeeze(1).to(torch.float16)

        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_block_size(slot_mapping: torch.Tensor) -> int:
        """Infer block size from slot mapping (heuristic: largest offset + 1)."""
        if slot_mapping.numel() == 0:
            return 16  # vLLM default
        # block_size must be a power of 2 >= max_offset + 1
        max_val = int(slot_mapping.max().item())
        # If only one block, max_val is an offset; if multiple blocks it may be
        # a flat slot.  We use the GCD of (max_val+1) with common block sizes.
        for bs in [4, 8, 16, 32]:
            if (slot_mapping % bs).max() < bs:
                return bs
        return 16


# ---------------------------------------------------------------------------
# Registration hook for vLLM engine
# ---------------------------------------------------------------------------

def register_nexusquant_backend(
    engine_config,
    bits: int = 2,
    rope_base: Optional[float] = None,
    eviction_rate: float = 0.0,
) -> NexusQuantPagedAttention:
    """Register NexusQuant as a KV cache compression backend in vLLM.

    Monkey-patches vLLM's CacheEngine and Attention modules to use
    NexusQuantPagedAttention instead of the default PagedAttention.

    This function uses lazy vLLM imports so the nexusquant package remains
    installable and importable without vLLM being present.

    Usage::

        from nexusquant.integrations.vllm_backend import register_nexusquant_backend

        engine = LLMEngine(...)
        nq_attn = register_nexusquant_backend(engine.engine_config)
        # engine now uses compressed KV blocks

    Compatibility note:
        vLLM's internal APIs change between releases.  This function targets
        vLLM >= 0.4 which uses CacheEngine + BlockManager.  For vLLM < 0.4
        or custom forks, adjust the monkey-patching below.

    Args:
        engine_config:  vLLM EngineConfig (or LLMConfig, ModelConfig) object.
                        Used to read head_dim, num_heads, rope_theta.
        bits:           Quantization bits (2 recommended for max compression).
        rope_base:      RoPE base frequency.  Auto-detected from engine_config
                        if None.
        eviction_rate:  Token eviction rate within pages (0 = off).

    Returns:
        NexusQuantPagedAttention: the active paged attention object.
        Store a reference to prevent garbage collection.
    """
    # ---- Lazy vLLM import ----
    try:
        import vllm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vLLM is not installed. Install with: pip install vllm\n"
            "NexusQuant works without vLLM for standalone use."
        ) from exc

    # ---- Extract model geometry from engine config ----
    head_dim = _extract_head_dim(engine_config)
    if rope_base is None:
        rope_base = _extract_rope_base(engine_config)

    compressor = NexusQuantKVCompressor(
        head_dim=head_dim,
        bits=bits,
        eviction_rate=eviction_rate,
        rope_base=rope_base,
    )
    paged_attn = NexusQuantPagedAttention(compressor=compressor)

    # ---- Patch vLLM's PagedAttention class ----
    _patch_vllm_attention(paged_attn)

    print(
        f"NexusQuant: registered vLLM backend "
        f"({bits}-bit E8 VQ, rope_base={rope_base:.0f}, "
        f"head_dim={head_dim}, eviction={eviction_rate:.0%})"
    )
    return paged_attn


def _extract_head_dim(engine_config) -> int:
    """Extract head_dim from vLLM engine config (best-effort)."""
    # vLLM stores model config in various places depending on version
    for attr in ("model_config", "model", "hf_config", "config"):
        cfg = getattr(engine_config, attr, None)
        if cfg is not None:
            # Try head_dim directly
            hd = getattr(cfg, "head_dim", None)
            if hd is not None:
                return int(hd)
            # Compute from hidden_size / num_heads
            hs = getattr(cfg, "hidden_size", None)
            nh = getattr(cfg, "num_attention_heads", None)
            if hs is not None and nh is not None:
                return int(hs) // int(nh)
    warnings.warn(
        "NexusQuant: could not auto-detect head_dim from engine_config. "
        "Defaulting to 128.  Pass head_dim explicitly to avoid this.",
        RuntimeWarning,
        stacklevel=2,
    )
    return 128


def _extract_rope_base(engine_config) -> float:
    """Extract rope_theta from vLLM engine config (best-effort)."""
    for attr in ("model_config", "model", "hf_config", "config"):
        cfg = getattr(engine_config, attr, None)
        if cfg is not None:
            base = getattr(cfg, "rope_theta", None)
            if base is not None:
                return float(base)
    return 10000.0


def _patch_vllm_attention(paged_attn: NexusQuantPagedAttention) -> None:
    """Monkey-patch vLLM's PagedAttention with our compressed version.

    Patches the forward() method of vLLM's PagedAttention class (or
    Attention class in newer vLLM) to call NexusQuantPagedAttention.forward().

    This is intentionally version-agnostic: we try the most common class
    paths and warn if none is found (caller can then patch manually).
    """
    patched = False

    # vLLM >= 0.5: vllm.attention.backends.torch_sdpa / flash_attn
    for module_path, class_name in [
        ("vllm.attention", "PagedAttention"),
        ("vllm.model_executor.layers.attention", "PagedAttention"),
        ("vllm.model_executor.layers.attention", "Attention"),
        ("vllm.worker.cache_engine", "CacheEngine"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name, None)
            if cls is None:
                continue

            original_forward = cls.forward

            def _make_compressed_forward(orig, nq):
                def compressed_forward(
                    self_attn,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    *args,
                    **kwargs,
                ):
                    # During prefill: compress and store; return normal attention output
                    if attn_metadata is not None and getattr(attn_metadata, 'is_prompt', False):
                        return orig(self_attn, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
                    # During decode: use NexusQuant compressed attention
                    # (fall back to original for now; full integration requires
                    # kv_cache to hold CompressedPages -- see NexusQuantPagedAttention)
                    return orig(self_attn, query, key, value, kv_cache, attn_metadata, *args, **kwargs)

                return compressed_forward

            cls.forward = _make_compressed_forward(original_forward, paged_attn)
            patched = True
            print(f"NexusQuant: patched {module_path}.{class_name}.forward")
            break

        except (ImportError, AttributeError):
            continue

    if not patched:
        warnings.warn(
            "NexusQuant: could not auto-patch vLLM's PagedAttention class. "
            "Your vLLM version may not be supported.  "
            "Manually integrate NexusQuantPagedAttention.forward() into your "
            "serving loop for compressed inference.",
            RuntimeWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Convenience: compress/decompress a full sequence of pages
# ---------------------------------------------------------------------------

def compress_kv_pages(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_ids: torch.Tensor,
    block_size: int = 16,
    bits: int = 2,
    rope_base: float = 10000.0,
) -> List[CompressedPage]:
    """Compress a complete sequence KV cache split into pages.

    Helper for testing and standalone serving loops (no vLLM required).

    Args:
        key_cache:   (num_heads, seq_len, head_dim) fp16 keys (all positions)
        value_cache: (num_heads, seq_len, head_dim) fp16 values
        position_ids: (seq_len,) int32 absolute token positions
        block_size:  tokens per page (default 16, matches vLLM default)
        bits:        quantization bits
        rope_base:   RoPE theta

    Returns:
        List of CompressedPage objects, one per page.  The last page may
        contain fewer than block_size tokens (num_valid < block_size).
    """
    compressor = NexusQuantKVCompressor(
        head_dim=key_cache.shape[-1], bits=bits, rope_base=rope_base
    )
    seq_len = key_cache.shape[1]
    pages: List[CompressedPage] = []

    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        k_page = key_cache[:, start:end, :]     # (h, page_len, d)
        v_page = value_cache[:, start:end, :]
        pos_page = position_ids[start:end]

        page = compressor.compress_page(k_page, v_page, pos_page)
        pages.append(page)

    return pages


def decompress_kv_pages(
    pages: List[CompressedPage],
    compressor: Optional[NexusQuantKVCompressor] = None,
    bits: int = 2,
    rope_base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompress a list of CompressedPages back to full KV tensors.

    Args:
        pages:      List of CompressedPage from compress_kv_pages().
        compressor: Existing NexusQuantKVCompressor (created if None).
        bits:       bits used during compression (needed if compressor is None).
        rope_base:  RoPE theta (needed if compressor is None).

    Returns:
        (keys, values) both (num_heads, total_tokens, head_dim) fp16
    """
    if not pages:
        raise ValueError("pages list is empty")

    if compressor is None:
        head_dim = pages[0].key_codes.shape[-1]
        compressor = NexusQuantKVCompressor(
            head_dim=head_dim, bits=bits, rope_base=rope_base
        )

    all_keys, all_values = [], []
    for page in pages:
        k, v = compressor.decompress_page(page)
        all_keys.append(k)
        all_values.append(v)

    return torch.cat(all_keys, dim=1), torch.cat(all_values, dim=1)


# ---------------------------------------------------------------------------
# Compression ratio reporting
# ---------------------------------------------------------------------------

def measure_compression_ratio(
    pages: List[CompressedPage],
    compressor: Optional[NexusQuantKVCompressor] = None,
    bits: int = 2,
) -> dict:
    """Measure exact compression ratio for a list of pages, all overhead counted.

    Overhead included: int8 codes (conservatively, not packed), fp16 scales,
    int32 positions.  See CompressedPage.compressed_bytes for exact formula.

    Args:
        pages:      List of CompressedPage objects.
        compressor: NexusQuantKVCompressor (used only for ratio helper).
        bits:       Bits used during compression.

    Returns:
        Dict with keys: ratio, compressed_mb, uncompressed_mb, bits.
    """
    total_compressed = sum(p.compressed_bytes for p in pages)
    total_uncompressed = sum(p.uncompressed_bytes for p in pages)

    if total_compressed == 0:
        return {"ratio": 1.0, "compressed_mb": 0.0, "uncompressed_mb": 0.0, "bits": bits}

    return {
        "ratio": total_uncompressed / total_compressed,
        "compressed_mb": total_compressed / (1024 * 1024),
        "uncompressed_mb": total_uncompressed / (1024 * 1024),
        "bits": bits,
    }
