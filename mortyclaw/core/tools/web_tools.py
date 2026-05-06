import sys

from . import web as _web

sys.modules[__name__] = _web
