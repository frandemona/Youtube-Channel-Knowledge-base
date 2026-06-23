from typer.testing import CliRunner
from ytkb import cli
from ytkb.channel import ChannelInfo, ChannelFilters
from ytkb.models import Answer, Citation, RunSummary

runner = CliRunner()


def test_add_invokes_sync(monkeypatch, tmp_path):
    captured = {}

    def fake_add(cfg, url, filters, name=None, **k):
        captured["url"] = url
        captured["no_shorts"] = filters.no_shorts
        return "y-combinator"

    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli.sync, "add_channel", fake_add)
    result = runner.invoke(cli.app, ["add", "https://youtu.be/x", "--no-shorts"])
    assert result.exit_code == 0
    assert captured["url"] == "https://youtu.be/x"
    assert captured["no_shorts"] is True
    assert "y-combinator" in result.stdout


def test_ask_prints_answer_and_citations(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))

    class FakeCtx:
        store = object()
        llm = object()
        cfg = None

    info = ChannelInfo("UC1", "@yc", "Y Combinator", "u")
    monkeypatch.setattr(cli.sync, "build_context", lambda cfg, slug: FakeCtx())
    monkeypatch.setattr(cli.sync, "load_channel", lambda cfg, slug: (info, ChannelFilters()))
    monkeypatch.setattr(cli.agent, "answer", lambda *a, **k: Answer(
        "Find a technical cofounder.",
        [Citation("v1", "Cofounders", 42.0, "https://youtu.be/v1?t=42")],
    ))
    result = runner.invoke(cli.app, ["ask", "y-combinator", "How do I find a cofounder?"])
    assert result.exit_code == 0
    assert "Find a technical cofounder." in result.stdout
    assert "youtu.be/v1?t=42" in result.stdout


def test_status_prints_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli.sync, "channel_status", lambda cfg, slug: {"slug": slug, "counts": {"indexed": 3}})
    result = runner.invoke(cli.app, ["status", "y-combinator"])
    assert result.exit_code == 0
    assert "indexed" in result.stdout and "3" in result.stdout


def test_reindex_invokes_sync(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    captured = {}
    def fake_reindex(cfg, slug):
        captured["slug"] = slug
        return RunSummary(done=3, failed=0, skipped=1)
    monkeypatch.setattr(cli.sync, "reindex_channel", fake_reindex)
    result = runner.invoke(cli.app, ["reindex", "y-combinator"])
    assert result.exit_code == 0
    assert captured["slug"] == "y-combinator"
    assert "reindexed=3" in result.stdout
    assert "skipped=1" in result.stdout
