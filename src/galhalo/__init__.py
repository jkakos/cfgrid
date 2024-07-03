import importlib


def __getattr__(name):
    return importlib.import_module(f".{name}", __name__)
