"""QO Utils"""
from contextlib import suppress
from importlib import import_module
from typing import Optional, Callable

module_not_found_ignore = suppress(ModuleNotFoundError, ImportError)

not_found_sentinel = object()


def import_and_add_if_available(
    obj_name: str,
    module_name: str,
    as_name: Optional[str] = None,
    scope: dict = None,
    if_already_in_scope: Optional[Callable] = None,
    if_not_found: Optional[Callable] = None,
):
    """
    Import object named ``obj_name`` from module named ``module_name``, sticking it
    in scope (a dict, usually locals()).

    :param scope: A dict where the object will be stored (if found). Usually locals().
    :param obj_name: The name of the object to import
    :param module_name: The name of the module to import the object from
    :param as_name: The name to use in the scope. Defaults to ``obj_name``
    :param if_already_in_scope: If not None, should be a ``(scope, as_name, obj)``
        callable that will provide the return value if ``as_name`` is already in the
        scope.
        This is to be able to overwrite the default of simply doing a
        ``scope[as_name] = obj`` (having the net effect of resolving name conflicts
        by a "last one seen wins" strategy.
    :param if_not_found: If not None, should be a ``(obj_name, module_name)`` callable
        that will provide the return value if ``obj_name`` is not found in
        ``module_name``.

    :return:

    >>> from qo.qo_utils import import_and_add_if_available
    >>> from functools import partial  # not necessary, but represents use case
    >>> scope = dict()  # you'd usually put locals() here
    >>> acquire = partial(import_and_add_if_available, scope=scope)
    >>> acquire('join', 'os.path')
    True
    >>> acquire('isfile', 'os.path', as_name='check_if_file_exists')
    True
    >>> acquire('not_in_os_path', 'os.path')  # returns False: not found in os.path
    False
    >>> acquire('object', 'does.not.exist')  # returns None: the module_name not found
    >>> import os
    >>> assert (scope ==
    ...     {'join': os.path.join, 'check_if_file_exists': os.path.isfile}
    ... )

    See in the above that neither ``'not_in_os_path'`` nor ``'object'`` are in
    ``scope``. They were simply skipped.

    """
    as_name = as_name or obj_name
    assert scope is not None, 'You need to specify a scope. Usually, scope=locals()'
    with module_not_found_ignore:
        module_obj = import_module(module_name)
        obj = getattr(module_obj, obj_name, not_found_sentinel)
        if obj is not not_found_sentinel:
            if if_already_in_scope and as_name in scope:
                return if_already_in_scope(scope, as_name, obj)
            else:
                scope[as_name] = obj
                return True
        else:
            if if_not_found is None:
                return False
            else:
                return if_not_found(obj_name, module_name)
    return None
    # returns None if module couldn't be imported!
