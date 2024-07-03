import importlib


def __getattr__(name):
    if name in ['FIGWIDTH', 'LEGEND_FONTSIZE']:
        module = importlib.import_module(f".config", __name__)
        return getattr(module, name)

    module = importlib.import_module(f".{name}", __name__)
    return module
