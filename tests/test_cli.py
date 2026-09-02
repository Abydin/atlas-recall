import pytest

from atlas_recall.cli import main
from atlas_recall import __version__


def test_version_flag_prints_version_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out
