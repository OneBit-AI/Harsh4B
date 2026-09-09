"""Per-(row, block) base offsets into the compacted T1c plane.

WHY THIS EXISTS
  pack_lattice.py stores T1 only for order-2 positions:

      o2_sel   = (maskid == o2_idx[0]) | (maskid == o2_idx[1])     # == maskid < 2
      T1_codes = unpack2(T1_dense, ic)[o2_sel]                     # row-major boolean gather
      T1c      = pack2_flat(T1_codes)

  Boolean indexing on a 2-D tensor yields elements in ROW-MAJOR order, so the
  index of position (r, c) inside T1c is its *rank* among all order-2 positions
  scanned row-major. There is no fixed stride: measured order-2 count per
  (row, 128-block) ranges 12..63 (mean 33) on a real layer.

  A kernel cannot afford a global running scan. Instead we precompute
  base[r, b] = rank of the first order-2 position in (row r, block b); the
  kernel then needs only an exclusive prefix sum over that block's own 128
  maskid values, which is a single tl.cumsum.

      t1_index(r, c) = base[r, c // 128] + popcount(is_o2[r, b*128 : c])

Cost: [oc, nblocks] int32 - 0.74 MiB for a 2560x9728 layer, ~0.02 bits/weight.
"""
import torch


def unpack2(c: torch.Tensor, k: int) -> torch.Tensor:
    """[n, ceil(k/4)] uint8 -> [n, k] uint8 values in 0..3, 4 per byte, LE within byte."""
    n, pk = c.shape
    t = torch.stack([c & 3, (c >> 2) & 3, (c >> 4) & 3, (c >> 6) & 3], -1).reshape(n, pk * 4)
    return t[:, :k]


def build_t1_base_offsets(maskid_packed: torch.Tensor, ic: int, nblocks: int,
                          blocksize: int, o2_idx) -> torch.Tensor:
    """-> int32 [oc, nblocks], the T1c rank of each (row, block)'s first order-2 slot."""
    mid = unpack2(maskid_packed, ic)                     # [oc, ic]
    oc = mid.shape[0]
    o2 = torch.zeros_like(mid, dtype=torch.bool)
    for m in (o2_idx.tolist() if torch.is_tensor(o2_idx) else o2_idx):
        o2 |= (mid == int(m))

    pad = nblocks * blocksize - ic
    if pad:                                              # last block may be partial
        o2 = torch.nn.functional.pad(o2, (0, pad), value=False)

    counts = o2.view(oc, nblocks, blocksize).sum(-1)     # [oc, nblocks] order-2 per block
    flat = counts.reshape(-1).to(torch.int64)
    base = torch.cumsum(flat, 0) - flat                  # exclusive prefix, row-major
    assert int(base[-1] + flat[-1]) == int(o2.sum()), "offset table does not close"
    return base.reshape(oc, nblocks).to(torch.int32)


def t1_index_naive(maskid_packed: torch.Tensor, ic: int, o2_idx) -> torch.Tensor:
    """Ground truth: rank of every position among order-2 positions, row-major.
    -1 where the position is not order-2. O(oc*ic) memory - test use only."""
    mid = unpack2(maskid_packed, ic)
    o2 = torch.zeros_like(mid, dtype=torch.bool)
    for m in (o2_idx.tolist() if torch.is_tensor(o2_idx) else o2_idx):
        o2 |= (mid == int(m))
    flat = o2.reshape(-1)
    rank = torch.cumsum(flat.to(torch.int64), 0) - 1     # inclusive-1 == exclusive rank
    rank[~flat] = -1
    return rank.reshape(mid.shape)
