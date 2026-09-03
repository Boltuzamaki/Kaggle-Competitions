import numpy as np
import torch

from faiss_search_bridge import TorchExactL2Index


def test_bridge_matches_brute_force_across_chunks():
    gen = torch.Generator().manual_seed(608)
    candidates = torch.randn(113, 17, generator=gen)
    queries = torch.randn(9, 17, generator=gen)
    index = TorchExactL2Index(None, 17, None, chunk_size=23)
    index.add(candidates)
    distances, indices = index.search(queries, 11)
    truth = torch.cdist(queries, candidates).square()
    td, ti = torch.topk(truth, 11, largest=False, sorted=True)
    assert torch.equal(indices, ti)
    assert torch.allclose(distances, td, atol=2e-5, rtol=2e-5)
    assert distances.device == queries.device and indices.device == queries.device


def test_numpy_api_and_reset():
    c = np.asarray([[0., 0.], [2., 0.], [0., 3.]], dtype="float32")
    q = np.asarray([[1., 0.]], dtype="float32")
    index = TorchExactL2Index(None, 2, None, chunk_size=2)
    index.add(c)
    d, i = index.search(q, 2)
    assert isinstance(d, np.ndarray) and isinstance(i, np.ndarray)
    assert i.tolist() == [[0, 1]] and np.allclose(d, [[1., 1.]])
    index.reset()
    try:
        index.search(q, 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("reset index unexpectedly remained searchable")


if __name__ == "__main__":
    test_bridge_matches_brute_force_across_chunks()
    test_numpy_api_and_reset()
    if torch.cuda.is_available():
        x = torch.randn(257, 32, device="cuda")
        index = TorchExactL2Index(None, 32, None, chunk_size=61)
        index.add(x)
        d, i = index.search(x[:8], 5)
        assert d.is_cuda and i.is_cuda and torch.equal(i[:, 0], torch.arange(8, device="cuda"))
    print("P100-safe exact search bridge tests passed")
