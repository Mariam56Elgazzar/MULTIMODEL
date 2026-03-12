from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_src_package_exists():
    assert (ROOT / "src" / "multimodel_rag").exists()


def test_vscode_config_exists():
    assert (ROOT / ".vscode" / "launch.json").exists()
    assert (ROOT / ".vscode" / "tasks.json").exists()
    assert (ROOT / ".vscode" / "settings.json").exists()
