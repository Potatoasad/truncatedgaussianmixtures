"""Canonical covariance-block metadata shared by the JAX kernels and the Julia backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BlockStructure:
    """Covariance blocks as static metadata.

    ``blocks`` are ordered by first appearance of their label (the same rule as
    ``canonical_blocks`` in Julia), so the packed layouts agree on both sides.
    Packed block matrices are ``[K, P]`` with ``P = sum(n_b**2)``; block ``b``
    occupies columns ``offsets[b] : offsets[b] + n_b**2`` in column-major order.
    """

    d: int
    blocks: Tuple[Tuple[int, ...], ...]
    cov: str

    @classmethod
    def from_spec(cls, d: int, cov: str = "full", block_structure: Optional[Sequence] = None) -> "BlockStructure":
        if cov == "diag":
            blocks = tuple((i,) for i in range(d))
        elif cov == "full":
            if block_structure is None or block_structure is False:
                blocks = (tuple(range(d)),)
            else:
                labels = list(block_structure)
                if len(labels) != d:
                    raise ValueError(f"block_structure has length {len(labels)} but the data has {d} dimensions")
                order, groups = [], {}
                for i, lab in enumerate(labels):
                    if lab not in groups:
                        order.append(lab)
                        groups[lab] = []
                    groups[lab].append(i)
                blocks = tuple(tuple(groups[lab]) for lab in order)
        else:
            raise ValueError(f"cov must be 'full' or 'diag', got {cov!r}")
        return cls(d=d, blocks=blocks, cov=cov)

    @property
    def sizes(self) -> Tuple[int, ...]:
        return tuple(len(b) for b in self.blocks)

    @property
    def offsets(self) -> Tuple[int, ...]:
        out, o = [], 0
        for n in self.sizes:
            out.append(o)
            o += n * n
        return tuple(out)

    @property
    def packed_length(self) -> int:
        return sum(n * n for n in self.sizes)

    @property
    def labels(self) -> list:
        """Block index of every coordinate (canonical labels for Julia)."""
        lab = [0] * self.d
        for b, idx in enumerate(self.blocks):
            for i in idx:
                lab[i] = b
        return lab

    def _pack_index(self):
        rows, cols = [], []
        for idx in self.blocks:
            for j in idx:          # column-major within the block
                for i in idx:
                    rows.append(i)
                    cols.append(j)
        return np.array(rows), np.array(cols)

    def pack(self, dense: np.ndarray) -> np.ndarray:
        """``[K, d, d] -> [K, P]`` (allowed block entries only)."""
        rows, cols = self._pack_index()
        return np.ascontiguousarray(dense[:, rows, cols])

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        """``[K, P] -> [K, d, d]`` with zero cross-block entries."""
        rows, cols = self._pack_index()
        out = np.zeros((packed.shape[0], self.d, self.d), dtype=packed.dtype)
        out[:, rows, cols] = packed
        return out

    def project(self, dense: np.ndarray) -> np.ndarray:
        """Zero the cross-block entries of ``[K, d, d]`` covariances."""
        return self.unpack(self.pack(dense))
