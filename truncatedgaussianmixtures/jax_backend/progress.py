"""Progress bars for the fit loops: marimo's own bar inside a marimo notebook,
``tqdm.auto`` everywhere else (the widget bar in Jupyter, a text bar in a terminal).

Kept free of JAX/Julia imports so it can be tested on its own.
"""
import sys


class _NoBar:
    """Stand-in for a tqdm bar when progress is off (``write`` still prints)."""
    def update(self, n=1): pass
    def set_postfix_str(self, s="", refresh=True): pass
    def close(self): pass
    def finish(self, outcome): pass
    @staticmethod
    def write(s): print(s)


class _MarimoBar(_NoBar):
    """marimo's ``mo.status.progress_bar`` behind the tqdm calls used here."""

    def __init__(self, total, desc):
        import marimo as mo
        self._bar = mo.status.progress_bar(total=max(int(total), 1), title=desc).progress
        self._desc = desc

    def update(self, n=1):
        self._bar.update(increment=n)

    def set_postfix_str(self, s="", refresh=True):
        self._bar.update(increment=0, subtitle=s)

    def close(self):
        if not self._bar.closed:
            self._bar.close()

    def finish(self, outcome):
        if self._bar.closed:
            return
        self._bar.total = max(self._bar.current, 1)
        self._bar.update(increment=0, title=f"{self._desc} ({outcome})" if outcome else None)
        self._bar.close()


def in_marimo():
    mo = sys.modules.get("marimo")           # never import marimo just to check
    try:
        return mo is not None and mo.running_in_notebook()
    except Exception:
        return False


def progress_bar(enabled, total, desc):
    """A marimo progress bar inside a marimo notebook, otherwise ``tqdm.auto``
    (the widget bar in Jupyter, a text bar in a terminal)."""
    if not enabled:
        return _NoBar()
    if in_marimo():
        return _MarimoBar(total, desc)
    from tqdm.auto import tqdm
    return tqdm(total=total, desc=desc, unit="it", dynamic_ncols=True)


def finish_bar(bar, outcome):
    """Close ``bar`` at 100% of the iterations actually made, with the outcome in its label."""
    if isinstance(bar, _NoBar):
        bar.finish(outcome)
        return
    bar.total = bar.n
    bar.set_description_str(f"{bar.desc} ({outcome})" if outcome else bar.desc, refresh=False)
    bar.close()
