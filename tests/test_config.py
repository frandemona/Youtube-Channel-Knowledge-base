from pathlib import Path
from ytkb.config import load_config, Config


def test_load_config_defaults(tmp_path: Path):
    cfg = load_config(tmp_path)
    assert isinstance(cfg, Config)
    assert cfg.data_dir == tmp_path
    assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.chat_model == "anthropic/claude-haiku-4.5"
    assert cfg.chunk_tokens == 500
    assert cfg.top_k == 6


def test_load_config_reads_toml_and_env(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        'chat_model = "openai/gpt-5-mini"\nchunk_tokens = 300\n'
    )
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-test-123\n")
    cfg = load_config(tmp_path)
    assert cfg.chat_model == "openai/gpt-5-mini"
    assert cfg.chunk_tokens == 300
    assert cfg.openrouter_api_key == "sk-test-123"
