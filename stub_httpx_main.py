# PyInstaller runtime hook: pre-seed a stub for httpx._main (the httpx CLI).
# httpx/__init__.py does `from ._main import main`, which pulls in rich/pygments.
# We never use the CLI, so replace it with a no-op to keep the exe small.
import sys
import types

_main = types.ModuleType("httpx._main")
_main.__path__ = []


def _noop(*args, **kwargs):
    return None


_main.main = _noop
sys.modules.setdefault("httpx._main", _main)
