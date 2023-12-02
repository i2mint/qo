"""
Personalized and collaborative quick access to imports.
"""
from operator import methodcaller as _methodcaller
from qo.qo_utils import module_not_found_ignore, find_objects

source_module_names = ['tw']


def copy_to_clipboard(obj):
    """Copy object to clipboard"""
    import pyperclip

    pyperclip.copy(obj)


def paste_from_clipboard(obj):
    """Paste object from clipboard"""
    import pyperclip

    return pyperclip.paste(obj)


def reload_module(module):
    """Reload a module"""
    from types import ModuleType
    from importlib import reload as _reload, import_module

    if not isinstance(module, ModuleType):
        if isinstance(module, str):
            module = import_module(module)
        elif hasattr(module, '__module__'):
            module = import_module(module.__module__)
        else:
            raise TypeError(f'Expected module or module name: {module}')

    return _reload(module)


# TODO: automate the following from source_module_names specs
with module_not_found_ignore:
    from qo.tw import *
    from qo import tw

from qo.qo_utils import import_and_add_if_available, print_signatures
from functools import partial as _partial

acquire = _partial(import_and_add_if_available, scope=locals())

# TODO: automate the following from source_module_names specs
with module_not_found_ignore:
    from sklearn.utils import all_estimators

    for obj_name, obj in all_estimators():
        acquire(obj_name, obj.__module__)

    acquire('metrics', 'sklearn')
    acquire('confusion_matrix', 'sklearn.metrics')

_root_names = tuple(filter(lambda x: not x.startswith('_'), locals()))


def find_names(query: str):
    """Find names of objects exposed in ``qo`` root."""
    return list(
        find_objects(
            query, _root_names, key=[_methodcaller('span'), lambda x: len(x.string)]
        )
    )
