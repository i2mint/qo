"""
Personalized and collaborative quick access to imports.
"""

from qo.qo_utils import module_not_found_ignore

source_module_names = ['tw']

# TODO: automate the following from source_module_names specs
with module_not_found_ignore:
    from qo.tw import *
    from qo import tw

from qo.qo_utils import import_and_add_if_available
from functools import partial

acquire = partial(import_and_add_if_available, scope=locals())

# TODO: automate the following from source_module_names specs
with module_not_found_ignore:
    from sklearn.utils import all_estimators

    for obj_name, obj in all_estimators():
        acquire(obj_name, obj.__module__)

    acquire('metrics', 'sklearn')
    acquire('confusion_matrix', 'sklearn.metrics')


