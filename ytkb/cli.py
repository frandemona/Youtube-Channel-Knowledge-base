import typer

from . import sync, agent
from .channel import ChannelFilters
from .config import load_config

app = typer.Typer(help="YouTube channel knowledge base")


@app.command()
def add(url: str, max: int = typer.Option(None), since: str = typer.Option(None),
        min_length: int = typer.Option(None), no_shorts: bool = typer.Option(False),
        whisper: bool = typer.Option(True), name: str = typer.Option(None)):
    """Add a channel (does not sync)."""
    cfg = load_config()
    filters = ChannelFilters(max=max, since=since, min_length=min_length,
                             no_shorts=no_shorts, whisper=whisper)
    slug = sync.add_channel(cfg, url, filters, name=name)
    typer.echo(f"Added channel '{slug}'. Run: kb sync {slug}")


def sync_cmd(slug: str = typer.Argument(None), all: bool = typer.Option(False, "--all"),
             dry_run: bool = typer.Option(False, "--dry-run")):
    """Discover and process new/failed videos."""
    cfg = load_config()
    slugs = sync.list_channels(cfg) if all else [slug]
    for s in slugs:
        summary = sync.sync_channel(cfg, s, dry_run=dry_run)
        typer.echo(f"[{s}] new={summary.new} done={summary.done} "
                   f"failed={summary.failed} skipped={summary.skipped}")


# Typer maps function name 'sync_cmd' to command 'sync-cmd'; rename explicitly:
app.command(name="sync")(sync_cmd)


@app.command()
def ask(slug: str, question: str):
    """Ask a channel's agent a question."""
    cfg = load_config()
    info, _filters = sync.load_channel(cfg, slug)
    ctx = sync.build_context(cfg, slug)
    if ctx.llm is None:
        typer.echo("No OPENROUTER_API_KEY configured in data/.env", err=True)
        raise typer.Exit(1)
    ans = agent.answer(question, info.title, ctx.store, ctx.llm,
                       chat_model=cfg.chat_model, top_k=cfg.top_k)
    typer.echo(ans.text)
    if ans.citations:
        typer.echo("\nSources:")
        for c in ans.citations:
            typer.echo(f"  - {c.title} @ {int(c.start)}s  {c.url}")


@app.command()
def status(slug: str):
    cfg = load_config()
    st = sync.channel_status(cfg, slug)
    typer.echo(f"Channel: {st['slug']}")
    for state, n in sorted(st["counts"].items()):
        typer.echo(f"  {state}: {n}")


@app.command()
def retry(slug: str):
    cfg = load_config()
    summary = sync.retry_channel(cfg, slug)
    typer.echo(f"[{slug}] done={summary.done} failed={summary.failed} skipped={summary.skipped}")


@app.command()
def reindex(slug: str = typer.Argument(None), all: bool = typer.Option(False, "--all")):
    """Rebuild a channel's index from local transcripts (e.g. after changing the embedding model)."""
    cfg = load_config()
    slugs = sync.list_channels(cfg) if all else [slug]
    for s in slugs:
        summary = sync.reindex_channel(cfg, s)
        typer.echo(f"[{s}] reindexed={summary.done} failed={summary.failed} skipped={summary.skipped}")


@app.command(name="list")
def list_cmd():
    cfg = load_config()
    for s in sync.list_channels(cfg):
        typer.echo(s)


@app.command()
def remove(slug: str, yes: bool = typer.Option(False, "--yes")):
    import shutil
    from .paths import ChannelPaths
    cfg = load_config()
    if not yes:
        typer.confirm(f"Delete all data for '{slug}'?", abort=True)
    shutil.rmtree(ChannelPaths.for_slug(cfg.data_dir, slug).dir, ignore_errors=True)
    typer.echo(f"Removed {slug}")


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    from .web.app import create_app
    uvicorn.run(create_app(), host=host, port=port)
