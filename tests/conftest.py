import pytest
from pathlib import Path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path
