from ytkb.embeddings import Embedder


class FakeBackend:
    def embed(self, texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]


def test_embed_uses_backend():
    emb = Embedder("fake", backend=FakeBackend())
    out = emb.embed(["ab", "abc"])
    assert out == [[2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
    assert emb.dim == 3
