"""
Personalized and collaborative quick access to imports.
"""
from operator import methodcaller as _methodcaller
from qo.qo_utils import module_not_found_ignore, find_objects

source_module_names = ['tw']

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

