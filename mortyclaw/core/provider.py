import sys

from .llm import provider as _provider

sys.modules[__name__] = _provider
