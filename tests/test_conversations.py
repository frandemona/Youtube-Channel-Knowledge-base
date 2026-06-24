from ytkb import conversations as cv


def test_create_add_get_roundtrip(tmp_path):
    conn = cv.connect(tmp_path)
    cid = cv.create_conversation(conn, "ycombinator")
    assert isinstance(cid, int)
    cv.add_message(conn, cid, "user", "How do I find a co-founder?")
    cv.add_message(conn, cid, "assistant", "Find a technical one.",
                   citations=[{"video_id": "v1", "title": "T", "start": 1.0, "url": "u"}])
    conv = cv.get_conversation(conn, cid)
    assert conv["slug"] == "ycombinator"
    assert conv["title"] is None
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["citations"][0]["video_id"] == "v1"
    assert conv["messages"][0]["citations"] == []


def test_list_orders_by_updated_desc(tmp_path):
    conn = cv.connect(tmp_path)
    a = cv.create_conversation(conn, "c1")
    b = cv.create_conversation(conn, "c2")
    cv.add_message(conn, a, "user", "later")  # bumps a's updated_at after b
    ids = [c["id"] for c in cv.list_conversations(conn)]
    assert ids[0] == a
    assert set(ids) == {a, b}
    assert all("slug" in c and "title" in c for c in cv.list_conversations(conn))


def test_get_history_caps_and_strips_citations(tmp_path):
    conn = cv.connect(tmp_path)
    cid = cv.create_conversation(conn, "c")
    for i in range(5):
        cv.add_message(conn, cid, "user", f"q{i}")
        cv.add_message(conn, cid, "assistant", f"a{i}", citations=[{"x": 1}])
    hist = cv.get_history(conn, cid, max_turns=2)  # last 4 messages
    assert [m["content"] for m in hist] == ["q3", "a3", "q4", "a4"]
    assert all(set(m.keys()) == {"role", "content"} for m in hist)


def test_set_title_and_delete(tmp_path):
    conn = cv.connect(tmp_path)
    cid = cv.create_conversation(conn, "c")
    cv.add_message(conn, cid, "user", "hi")
    cv.set_title(conn, cid, "My chat")
    assert cv.get_conversation(conn, cid)["title"] == "My chat"
    cv.delete_conversation(conn, cid)
    assert cv.get_conversation(conn, cid) is None
    assert conn.execute("SELECT COUNT(*) c FROM messages WHERE conversation_id=?", (cid,)).fetchone()["c"] == 0
