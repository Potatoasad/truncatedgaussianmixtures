"""Progress-bar selection (loaded by path: no Julia/JAX needed)."""
import importlib.util
import pathlib
import sys

import pytest

_path = pathlib.Path(__file__).parents[1] / "truncatedgaussianmixtures" / "jax_backend" / "progress.py"
_spec = importlib.util.spec_from_file_location("tgmm_progress", _path)
progress = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(progress)


def test_disabled_is_silent(capsys):
    bar = progress.progress_bar(False, 10, "TGMM EM")
    bar.update(1); bar.set_postfix_str("x"); progress.finish_bar(bar, "converged")
    assert capsys.readouterr().err == ""


def test_outside_marimo_uses_tqdm_and_finishes_at_100_percent():
    assert not progress.in_marimo()
    bar = progress.progress_bar(True, 200, "TGMM EM")
    for _ in range(7):
        bar.update(1)
    progress.finish_bar(bar, "converged")
    assert type(bar).__module__.startswith("tqdm")
    assert bar.total == bar.n == 7 and bar.desc.endswith("(converged)")


def test_marimo_bar_is_used_inside_a_marimo_notebook(monkeypatch):
    mo = pytest.importorskip("marimo")
    monkeypatch.setattr(mo, "running_in_notebook", lambda: True)
    assert progress.in_marimo()
    monkeypatch.delitem(sys.modules, "marimo")          # never imported just to check
    assert not progress.in_marimo()
