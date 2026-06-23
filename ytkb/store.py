import lancedb

from .db import insert_chunks
from .embeddings import Embedder
from .models import Chunk, ChunkHit
from .paths import ChannelPaths

TABLE = "chunks"


class ChannelStore:
    def __init__(self, paths: ChannelPaths, conn, embedder: Embedder):
        self.paths = paths
        self.conn = conn
        self.embedder = embedder
        self._db = lancedb.connect(str(paths.vectors_dir))

    def _table(self):
        if TABLE in self._db.table_names():
            return self._db.open_table(TABLE)
        return None

    def add(self, chunks: list[Chunk], title_of: dict[str, str]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        rows = [
            {"vector": vectors[i], "video_id": c.video_id, "start": c.start,
             "title": title_of[c.video_id], "text": c.text}
            for i, c in enumerate(chunks)
        ]
        table = self._table()
        if table is None:
            self._db.create_table(TABLE, data=rows)
        else:
            table.add(rows)
        insert_chunks(self.conn, chunks, title_of)

    def keyword_search(self, query: str, k: int) -> list[ChunkHit]:
        rows = self.conn.execute(
            """SELECT c.video_id, c.title, c.start, c.text, bm25(chunks_fts) AS score
               FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid
               WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?""",
            (query, k),
        ).fetchall()
        return [ChunkHit(r["video_id"], r["title"], r["start"], r["text"], r["score"]) for r in rows]

    def semantic_search(self, query: str, k: int) -> list[ChunkHit]:
        table = self._table()
        if table is None:
            return []
        qv = self.embedder.embed([query])[0]
        res = table.search(qv).limit(k).to_list()
        return [
            ChunkHit(r["video_id"], r["title"], r["start"], r["text"], float(r.get("_distance", 0.0)))
            for r in res
        ]

    def read_around(self, video_id: str, around_ts: float | None, window: float = 90.0) -> str:
        text = self.paths.clean_path(video_id).read_text()
        if around_ts is None:
            return text[:4000]
        # clean.txt stores plain text; for windowing we fall back to chunk rows near the ts.
        rows = self.conn.execute(
            "SELECT text FROM chunks WHERE video_id=? AND start BETWEEN ? AND ? ORDER BY start",
            (video_id, around_ts - window, around_ts + window),
        ).fetchall()
        return "\n".join(r["text"] for r in rows) or text[:4000]

    def list_videos(self, contains: str | None = None) -> list[dict]:
        if contains:
            rows = self.conn.execute(
                "SELECT video_id, title, upload_date FROM videos WHERE title LIKE ? AND state='indexed'",
                (f"%{contains}%",),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT video_id, title, upload_date FROM videos WHERE state='indexed'"
            ).fetchall()
        return [dict(r) for r in rows]
