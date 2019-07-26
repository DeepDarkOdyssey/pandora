from typing import Optional, Iterable, Callable


class LazyDict(dict):
    def __init__(self, load_func: Callable, iterable: Optional[Iterable] = None):
        if iterable:
            super().__init__(iterable)
        self.load_func = load_func

    def __getitem__(self, key):
        try:
            value = dict.__getitem__(self, key)
        except KeyError:
            value = self.load_func(key)
            if value is None:
                raise KeyError
        return value
