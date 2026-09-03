"""P100-safe exact L2 bridge for PyTabKit TabR.

FAISS GPU wheels currently published for CUDA 12 omit sm_60 kernels and abort
the process on a Tesla P100.  This class implements the tiny FAISS index API
used by TabR with chunked torch matrix multiplication.  Neural training and
retrieval both remain on CUDA; only the incompatible FAISS GPU kernel is gone.
"""
from __future__ import annotations

import numpy as np
import torch


class TorchExactL2Index:
    def __init__(self, _resources=None, dimension=None, _config=None, *, chunk_size=65536):
        self.dimension = int(dimension) if dimension is not None else None
        self.chunk_size = int(chunk_size)
        self.candidates = None

    def reset(self):
        self.candidates = None

    def add(self, candidates):
        is_numpy = isinstance(candidates, np.ndarray)
        x = torch.from_numpy(candidates) if is_numpy else candidates
        if x.ndim != 2 or (self.dimension is not None and x.shape[1] != self.dimension):
            raise ValueError("candidate matrix has the wrong shape")
        # TabR updates candidate encodings every optimizer step. Keep the tensor
        # on its existing device and detached, exactly as FAISS search does.
        self.candidates = x.detach().contiguous().float()

    @torch.no_grad()
    def search(self, queries, k):
        return_numpy = isinstance(queries, np.ndarray)
        q = torch.from_numpy(queries) if return_numpy else queries
        q = q.detach().contiguous().float()
        if self.candidates is None:
            raise RuntimeError("add must be called before search")
        if q.device != self.candidates.device:
            raise ValueError("queries and candidates must share a device")
        k = min(int(k), len(self.candidates))
        if k <= 0:
            raise ValueError("k must be positive")

        q_norm = q.square().sum(1, keepdim=True)
        best_d = torch.full((len(q), k), torch.inf, device=q.device)
        best_i = torch.full((len(q), k), -1, dtype=torch.long, device=q.device)
        for start in range(0, len(self.candidates), self.chunk_size):
            c = self.candidates[start:start + self.chunk_size]
            # Squared L2, clamped only for roundoff. This is the metric used by
            # faiss.GpuIndexFlatL2 and expected by TabR's self-neighbor removal.
            d = (q_norm + c.square().sum(1).unsqueeze(0) - 2.0 * (q @ c.T)).clamp_min_(0)
            take = min(k, d.shape[1])
            cd, ci = torch.topk(d, take, dim=1, largest=False, sorted=False)
            ci = ci + start
            merged_d = torch.cat((best_d, cd), dim=1)
            merged_i = torch.cat((best_i, ci), dim=1)
            best_d, pos = torch.topk(merged_d, k, dim=1, largest=False, sorted=True)
            best_i = merged_i.gather(1, pos)

        if return_numpy:
            return best_d.cpu().numpy(), best_i.cpu().numpy()
        return best_d, best_i


def install_faiss_gpu_replacement(faiss_module):
    """Replace only the three GPU-FAISS symbols instantiated by PyTabKit."""
    class _Resources:
        pass

    class _Config:
        device = 0

    faiss_module.StandardGpuResources = _Resources
    faiss_module.GpuIndexFlatConfig = _Config
    faiss_module.GpuIndexFlatL2 = TorchExactL2Index
    return faiss_module
