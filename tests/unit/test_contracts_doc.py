import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_the_contracts_page_is_what_the_exporter_writes_now() -> None:
    result = subprocess.run([sys.executable, str(REPO / "tools/export_contracts_doc.py"), "--check"],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_page_covers_every_contract_and_names_the_hooks() -> None:
    from freecam.pi_cam.hooks import load_hooks

    text = (REPO / "docs/contracts.md").read_text()
    for path in sorted((REPO / "native/pi_cam/functions").glob("*.yaml")):
        assert f"`{path.relative_to(REPO).as_posix()}`" in text, path.name
    for hook in load_hooks().hooks:
        assert f"Hook `{hook.kernel}`" in text, hook.kernel
    assert "### Cloud block `macro`" in text and "### Cloud block `micro`" in text and "### Radiation branch" in text
