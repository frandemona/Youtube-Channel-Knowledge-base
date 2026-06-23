from ytkb.db import connect, upsert_video, set_state, videos_by_state, count_by_state, VideoState, insert_chunks
from ytkb.models import VideoMeta, Chunk


def make_meta(vid="v1"):
    return VideoMeta(video_id=vid, title="Title " + vid, duration=600, upload_date="20240101", url=f"https://youtu.be/{vid}")


def test_upsert_and_state_transitions(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    row = videos_by_state(conn, [VideoState.DISCOVERED])
    assert len(row) == 1 and row[0]["video_id"] == "v1"

    set_state(conn, "v1", VideoState.TRANSCRIPT_FETCHED)
    assert videos_by_state(conn, [VideoState.DISCOVERED]) == []
    assert len(videos_by_state(conn, [VideoState.TRANSCRIPT_FETCHED])) == 1


def test_upsert_does_not_reset_state(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    set_state(conn, "v1", VideoState.INDEXED)
    upsert_video(conn, make_meta("v1"))  # re-listed on a later sync
    assert len(videos_by_state(conn, [VideoState.INDEXED])) == 1


def test_set_state_records_error_and_attempts(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    set_state(conn, "v1", VideoState.FAILED_FETCH, error="boom")
    set_state(conn, "v1", VideoState.FAILED_FETCH, error="boom2")
    v = videos_by_state(conn, [VideoState.FAILED_FETCH])[0]
    assert v["attempt_count"] == 2
    assert v["last_error"] == "boom2"


def test_count_by_state(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    upsert_video(conn, make_meta("v2"))
    set_state(conn, "v2", VideoState.INDEXED)
    counts = count_by_state(conn)
    assert counts[VideoState.DISCOVERED.value] == 1
    assert counts[VideoState.INDEXED.value] == 1


def test_insert_chunks_and_fts(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    chunks = [Chunk(video_id="v1", idx=0, start=12.0, text="how to find a cofounder")]
    insert_chunks(conn, chunks, title_of={"v1": "Title v1"})
    rows = conn.execute(
        "SELECT c.video_id, c.start, c.text FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid WHERE chunks_fts MATCH ?",
        ("cofounder",),
    ).fetchall()
    assert rows[0]["video_id"] == "v1"
    assert rows[0]["start"] == 12.0


def test_clear_all_chunks_empties_content_and_fts(tmp_path):
    from ytkb.db import clear_all_chunks
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    insert_chunks(conn, [Chunk(video_id="v1", idx=0, start=1.0, text="cofounder advice")], {"v1": "T"})
    clear_all_chunks(conn)
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 0
    # FTS index is empty too -> a prior match now returns nothing
    rows = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ("cofounder",)).fetchall()
    assert rows == []
