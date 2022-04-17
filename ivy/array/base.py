# global
import abc

# local
import ivy


# noinspection PyUnresolvedReferences
class ArrayBase(abc.ABC):

    def wrap_fn(self, fn):
        def new_fn(*args, **kwargs):
            return fn(self._data, *args, **kwargs)
        return new_fn

    def __init__(self, module, to_ignore=()):
        for key, val in module.__dict__.items():
            if key.startswith('_') or key[0].isupper() or not callable(val) or \
                    key in self.__dict__ or key in to_ignore or key not in ivy.__dict__:
                continue
            try:
                setattr(self, key, self.wrap_fn(val))
            except AttributeError:
                pass
