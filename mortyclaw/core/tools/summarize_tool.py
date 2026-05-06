import sys

from . import summarize as _summarize

sys.modules[__name__] = _summarize
