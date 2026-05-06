from importlib import import_module


def __getattr__(name: str):
    if name == "agent":
        module = import_module(f"{__name__}.agent")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["agent"]
