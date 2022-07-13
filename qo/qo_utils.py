"""QO Utils"""
from contextlib import suppress
from operator import attrgetter, methodcaller
from functools import partial
from itertools import starmap
from importlib import import_module
from typing import Optional, Callable, Iterable, Any, Sequence
import re
from inspect import signature

module_not_found_ignore = suppress(ModuleNotFoundError, ImportError)

not_found_sentinel = object()

ddir = lambda o: filter(lambda x: not x.startswith('_'), dir(o))


StringIterableFactory = Callable[[Any], Iterable[str]]

_if_not_iterable_get_attributes: StringIterableFactory


def _if_not_iterable_get_attributes(x: Any) -> Iterable[str]:
    if not isinstance(x, Iterable):
        x = list(ddir(x))
    return x


def callables_and_signatures(obj, object_to_strings: StringIterableFactory = ddir):
    """A generator of strings describing the callables in obj (module, class, ...)"""
    for attr_name in object_to_strings(obj):
        attr = getattr(obj, attr_name)
        if callable(attr):
            try:
                yield f'{attr_name}{signature(attr)}'
            except Exception:
                pass


def print_callables_signatures(
    obj, object_to_strings: StringIterableFactory = ddir, sep='\n * '
):
    """Prints information on callable attributes of obj (module, class, ...)"""
    print('', *callables_and_signatures(obj, object_to_strings), sep=sep)


def find_objects(
    pattern: str,
    objects: Any,
    objects_to_strings: StringIterableFactory = _if_not_iterable_get_attributes,
    *,
    key=None,
):
    """Find strings matching a given pattern, possibly sorting them in a specific way.

    :param pattern: The pattern to search (string representing a regular expression).
    :param objects: An iterable of strings or an object that will resolve in one.
    :param objects_to_strings: The function that resolves the ``objects`` input
        (which is not even necessarily an iterable) into an iterable of strings.
    :param key: If not None, will be used to sort matched objects.
        ``key`` will be applied to objects of type ``re.Match``;
        see https://docs.python.org/3/library/re.html#match-objects to get an idea of
        how to define the right key function.
        If ``key`` is an iterable ("of callables", assumed), make an
        aggregate of these functions that will return a tuple of the sort keys.
        This is akin to sorting by one columns primarly, a second secondarily, etc.
    :return: A generator of matched strings.

    >>> kwargs = dict(
    ...     pattern='li',
    ...     objects='bob and lilabet like to play with alice',
    ...     objects_to_strings=str.split,
    ... )
    >>> list(find_objects(**kwargs))
    ['lilabet', 'like', 'alice']

    Sort the results in reverse length of string:

    >>> list(find_objects(**kwargs, key=lambda x: -len(x.string)))
    ['lilabet', 'alice', 'like']

    Sort the results according to how early in the string the match happens:

    >>> from operator import methodcaller
    >>> list(find_objects(**kwargs, key=methodcaller('span')))
    ['lilabet', 'like', 'alice']

    The same as above, but with secondary "length" sorting:

    >>> tuple(  # for a change!
    ...     find_objects(**kwargs, key=[methodcaller('span'), lambda x: len(x.string)])
    ... )
    ('like', 'lilabet', 'alice')
    """
    strings = list(objects_to_strings(objects))
    pattern = re.compile(pattern)
    match_objects = list(map(pattern.search, strings))

    # The match_objects are re.Match instances that resolve to True when there's a match,
    # and can therefore be used as selectors
    match_objects = list(bitmap_selection(match_objects, selector=match_objects))

    if key is not None:
        if isinstance(key, Iterable):
            # If match_sort_key is an iterable ("of callables", assumed), make an
            # aggregate of these functions that will return a tuple of the sort keys.
            # This is akin to sorting by one columns primarly, a second secondarily, etc.
            _key = partial(func_fanout, funcs=key)
            key = lambda x: tuple(_key(x))
        match_objects = sorted(match_objects, key=key)

    # instead of getting the strings from strings, we get them from the re.Match objects'
    # 'string' attribute, because these contain more information to be able to sort with
    return map(attrgetter('string'), match_objects)


def func_fanout(*args, funcs: Iterable[Callable], **kwargs):
    """Util to make a function that applies multiple functions to a single input.
    Note that the function returns a generator that needs to be "consumed" to actually
    get the outputs of the function.

    ``func_fanout`` is meant to be used with ``functools.partial`` to "make" the
    desired function, such as:

    >>> from functools import partial
    >>> f = partial(func_fanout, funcs=[
    ...     lambda x, y: x + y,
    ...     lambda x, y: x * y,
    ... ])
    >>> list(f(2, 3))
    [5, 6]
    >>> tuple(f(3, y=4))
    (7, 12)

    .. seealso: ``i2.multi_object.FuncFanout`` for a more involved version of this.

    """
    return (func(*args, **kwargs) for func in funcs)


def bitmap_selection(iterable: Iterable, selector: Sequence):
    """Select items of an iterable with items of a selector sequence of the same size.
    The selector items should be, or resolve to, a ``bool``.

    >>> list(bitmap_selection([2, 4, 6, 8], [True, False, True, False]))
    [2, 6]
    >>> tuple(bitmap_selection(range(5), [1, 0, None, '', 'blah']))
    (0, 4)

    """
    return (obj for i, obj in enumerate(iterable) if selector[i])


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
