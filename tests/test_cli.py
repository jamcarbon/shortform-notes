from unittest.mock import AsyncMock, patch

from shortform_notes.cli import main
from shortform_notes.pipeline import ReelImportError, ReelImportResult


def test_cli_no_args_prints_help(capsys):
    assert main([]) == 2
    assert "Turn Instagram / Facebook Reels" in capsys.readouterr().out


def test_cli_json_output(tmp_path, capsys):
    result = ReelImportResult(tmp_path / "n.md", "T", "S", ("a",), ("caption",), ())
    with patch("shortform_notes.cli.import_reel", AsyncMock(return_value=result)):
        assert main(["https://youtu.be/x", "--json", "-o", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert '"title": "T"' in out and '"sources"' in out


def test_cli_failure_exit_code(tmp_path, capsys):
    with patch("shortform_notes.cli.import_reel", AsyncMock(side_effect=ReelImportError("nope"))):
        assert main(["https://youtu.be/x", "-o", str(tmp_path)]) == 1
    assert "nope" in capsys.readouterr().err
