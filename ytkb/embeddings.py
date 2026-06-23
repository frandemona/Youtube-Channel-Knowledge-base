class Embedder:
    def __init__(self, model_name: str, *, backend=None):
        self.model_name = model_name
        self._backend = backend
        self._dim: int | None = None

    def _ensure(self):
        if self._backend is None:
            from fastembed import TextEmbedding
            self._backend = TextEmbedding(model_name=self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        vectors = [list(map(float, v)) for v in self._backend.embed(texts)]
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["x"])[0])
        return self._dim
